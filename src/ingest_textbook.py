"""
ingest_textbook.py  (AdaptIQ)

Textbook PDF(s) -> text/chapters/sections/pages -> Qwen (Ollama) assigns each
section to one or MORE of S1-S6 -> chunk per skill -> all-MiniLM-L6-v2
embeddings -> per-skill FAISS index + chunks pickle.

Usage:
    ollama serve                      (and: ollama pull qwen2.5:3b)
    pip install pymupdf4llm ollama sentence-transformers faiss-cpu numpy pandas
    python src/ingest_textbook.py knowledge_base/ISLP_website.pdf knowledge_base/dl2.pdf

Safeguards:
    - Qwen is tested before any slow work starts.
    - If any section cannot be classified by Qwen, the run stops BEFORE any
      index is built or replaced. Progress is cached; fix the problem and
      re-run to resume.

Outputs (data/):
    s{n}_index.faiss, s{n}_chunks.pkl   only for skills that received content
    classification_report.json          per-section result + review flags
    classification_cache.json           Qwen answers, so re-runs resume fast
"""
import re
import os
import sys
import shutil
import json
import pickle
import hashlib
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "knowledge_base"

LLM_MODEL = "qwen2.5:3b"          # try qwen2.5:7b for better accuracy
EMBED_MODEL = "all-MiniLM-L6-v2"  # must match generate_questions.py

SKILLS = {
    "S1": (
        "Python and Data Handling: Python programming, data types, "
        "data structures, NumPy, pandas, data loading, manipulation "
        "and basic data analysis."
    ),
    "S2": (
        "Machine Learning Fundamentals: supervised and unsupervised "
        "learning, regression, classification, clustering, learning "
        "algorithms, overfitting, underfitting and bias-variance trade-off."
    ),
    "S3": (
        "Deep Learning: neural networks, perceptrons, neurons, "
        "activation functions, layers, backpropagation, gradient descent, "
        "CNNs, RNNs and deep learning architectures."
    ),
    "S4": (
        "Data Preprocessing and Feature Engineering: data cleaning, "
        "missing values, encoding categorical variables, normalization, "
        "standardization, feature selection, feature extraction "
        "and data transformation."
    ),
    "S5": (
        "Model Evaluation and Deployment: training and testing, "
        "cross-validation, confusion matrix, accuracy, precision, "
        "recall, F1 score, ROC/AUC, error metrics, model deployment "
        "and production monitoring."
    ),
    "S6": (
        "LLMs and Generative AI: large language models, generative AI, "
        "prompt engineering, embeddings, semantic search, retrieval "
        "augmented generation (RAG), vector databases and fine-tuning."
    ),
}

MIN_SECTION_WORDS = 80    # smaller sections are merged into the previous one
MAX_HEADING_LEVEL = 3     # '#', '##', '###' start sections
CLASSIFY_WORDS = 400      # how much of each section Qwen sees
CHUNK_WORDS = 200         # MiniLM truncates around 256 tokens
CHUNK_OVERLAP = 30
FALLBACK_PAGES_PER_SECTION = 3
CLASSIFY_ATTEMPTS = 2     # tries per section before it counts as a failure
# ----------------------------------------

SKILL_IDS = list(SKILLS)
HEAD_RE = re.compile(rf"^(#{{1,{MAX_HEADING_LEVEL}}})\s+(.*\S)\s*$")


# ---------- 0. fail fast if Qwen is not usable ----------
def check_ollama():
    """Stop immediately (before slow extraction) if Qwen cannot answer."""
    try:
        import ollama
        ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with the word OK."}],
            options={"temperature": 0, "num_predict": 5},
        )
    except Exception as ex:
        sys.exit(
            f"\nQwen check FAILED: {type(ex).__name__}: {ex}\n"
            "Fix before running ingestion:\n"
            "  1. python -m pip install ollama      (in the active environment)\n"
            "  2. Make sure the Ollama app is running (or run: ollama serve)\n"
            f"  3. ollama pull {LLM_MODEL}\n"
        )
    print(f"   Qwen check passed ({LLM_MODEL}).")


# ---------- 1. extract text with page numbers ----------
def extract_pages(path: Path):
    """Return [(pdf_page_number, markdown_text), ...]."""
    if path.suffix.lower() == ".pdf":
        import pymupdf4llm
        pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
        return [(i + 1, p["text"]) for i, p in enumerate(pages)]
    if path.suffix.lower() in {".md", ".txt"}:
        return [(1, path.read_text(encoding="utf-8"))]
    raise ValueError(f"Unsupported format: {path.suffix} (use PDF)")


# ---------- 2. chapters / sections (keeps introductory content) ----------
def extract_pdf_toc(path: Path):
    """Return PDF bookmarks as (level, title, page) entries."""
    import fitz

    with fitz.open(path) as doc:
        return doc.get_toc()


def build_sections(pages, toc=None):
    """Build sections from PDF bookmarks, or safe Markdown headings as fallback."""
    sections = []

    # PDF bookmarks are more reliable than guessing from extracted Markdown.
    if toc:
        grouped = {}
        for entry in toc:
            level, title, page = entry[:3]
            title = title.strip()
            if not title or level > MAX_HEADING_LEVEL:
                continue
            # Captions are not structural headings, even if present in the TOC.
            if re.match(r"^(?:<mark>)?t?Fig\.?\s*\d", title, re.IGNORECASE):
                continue
            if 1 <= page <= len(pages):
                grouped.setdefault(page, []).append((level, title))

        starts = sorted(grouped)
        stack = []
        for i, start_page in enumerate(starts):
            end_page = (starts[i + 1] - 1 if i + 1 < len(starts)
                        else len(pages))

            for level, title in grouped[start_page]:
                stack = stack[:level - 1] + [title]

            content = "\n".join(
                page_text for page_no, page_text in pages
                if start_page <= page_no <= end_page
            ).strip()
            if len(content.split()) < 30:
                continue

            title = grouped[start_page][-1][1]
            sections.append({
                "title": title,
                "path": " > ".join(stack),
                "chapter": stack[0] if stack else title,
                "start": start_page,
                "end": end_page,
                "text": content,
            })

    else:
        # Fallback for PDFs without bookmarks and for Markdown/TXT inputs.
        stack = []
        current = None
        intro_lines = []
        intro_start = None
        intro_end = None
        in_fence = False
        fence_marker = None

        for page_no, page_text in pages:
            for line in page_text.splitlines():
                stripped = line.strip()

                # Track fenced code blocks (``` or ~~~); headings inside them
                # are Python comments/code, not document structure.
                fence_match = re.match(r"^(```+|~~~+)", stripped)
                if fence_match:
                    marker = fence_match.group(1)[0]
                    if not in_fence:
                        in_fence = True
                        fence_marker = marker
                    elif marker == fence_marker:
                        in_fence = False
                        fence_marker = None
                    if current is not None:
                        current["lines"].append(line)
                        current["end"] = page_no
                    elif not sections:
                        if intro_start is None:
                            intro_start = page_no
                        intro_end = page_no
                        intro_lines.append(line)
                    continue

                match = HEAD_RE.match(line) if not in_fence else None
                title = None
                if match:
                    title = re.sub(r"[*_`]", "", match.group(2)).strip()
                    # Reject figure captions and obvious code/comment fragments.
                    if (
                        re.match(r"^(?:<mark>)?t?Fig\.?\s*\d", title, re.IGNORECASE)
                        or re.search(r"[=();{}\[\]]", title)
                        or title.startswith(("#", "import ", "from ", "print("))
                    ):
                        match = None

                if match:
                    level = len(match.group(1))
                    stack = stack[:level - 1] + [title]
                    current = {
                        "title": title,
                        "path": " > ".join(stack),
                        "chapter": stack[0],
                        "start": page_no,
                        "end": page_no,
                        "lines": [],
                    }
                    sections.append(current)
                elif current is not None:
                    current["lines"].append(line)
                    current["end"] = page_no
                else:
                    if intro_start is None:
                        intro_start = page_no
                    intro_end = page_no
                    intro_lines.append(line)

        for section in sections:
            section["text"] = "\n".join(section.pop("lines")).strip()

        intro_text = "\n".join(intro_lines).strip()
        if len(intro_text.split()) >= 30:
            sections.insert(0, {
                "title": "Introduction",
                "path": "Introduction",
                "chapter": "Introduction",
                "start": intro_start,
                "end": intro_end,
                "text": intro_text,
            })

    # Merge small sections into the previous section within the same chapter.
    merged = []
    for section in sections:
        if (merged
                and len(section["text"].split()) < MIN_SECTION_WORDS
                and merged[-1]["chapter"] == section["chapter"]):
            merged[-1]["text"] += (
                f"\n\n{section['title']}\n{section['text']}"
            )
            merged[-1]["end"] = section["end"]
        else:
            merged.append(section)

    result = [section for section in merged
              if len(section["text"].split()) >= 30]
    for n, section in enumerate(result, 1):
        section["section_id"] = f"SEC_{n:04d}"
    return result


def fallback_sections(pages):
    """Group pages into sections when headings are unavailable."""
    out = []
    for i in range(0, len(pages), FALLBACK_PAGES_PER_SECTION):
        group = pages[i:i + FALLBACK_PAGES_PER_SECTION]
        text = "\n".join(t for _, t in group).strip()
        if len(text.split()) >= 30:
            a, b = group[0][0], group[-1][0]
            out.append({
                "section_id": f"SEC_{len(out) + 1:04d}",
                "title": f"Pages {a}-{b}",
                "path": f"Pages {a}-{b}",
                "chapter": "Unknown",
                "start": a, "end": b, "text": text,
            })
    return out


# ---------- 3. classify with Qwen (multi-skill) ----------
def llm_classify(section, cache):
    import ollama
    snippet = " ".join(section["text"].split()[:CLASSIFY_WORDS])
    # v4 + skill definitions + source: changing the taxonomy forces fresh classification.
    skill_signature = json.dumps(SKILLS, sort_keys=True)
    cache_key = hashlib.sha1(
        f"v4|{LLM_MODEL}|{skill_signature}|{section.get('source', '')}|{section['path']}|{snippet}".encode()
    ).hexdigest()
    if cache_key in cache:
        return cache[cache_key]

    skill_list = "\n".join(f"{sid}: {desc}" for sid, desc in SKILLS.items())
    schema = {
        "type": "object",
        "properties": {
            "skills": {"type": "array", "items": {"type": "string", "enum": SKILL_IDS}},
            "confidence": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "reason": {"type": "string"},
        },
        "required": ["skills", "confidence", "reason"],
        "additionalProperties": False,
    }
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                "You classify AI/ML textbook sections into predefined learning skills.\n\n"
                f"Available skills:\n{skill_list}\n\n"
                "Instructions:\n"
                "1. Read the actual section content.\n"
                "2. Assign every skill that is substantially covered in the section.\n"
                "3. A section may belong to multiple skills.\n"
                "4. Do not assign skills based on a passing mention.\n"
                "5. Return an empty list for front matter, references, indexes "
                "or irrelevant content.\n"
                "6. Never invent new skill IDs.")},
            {"role": "user", "content": (
                f"Chapter: {section['chapter']}\n"
                f"Section: {section['title']}\n\n"
                f"Content:\n{snippet}")},
        ],
        format=schema,
        options={"temperature": 0},
    )
    result = json.loads(resp["message"]["content"])
    # Remove duplicates and reject invalid IDs.
    result["skills"] = list(dict.fromkeys(s for s in result["skills"] if s in SKILLS))
    cache[cache_key] = result
    return result


def embedding_prior(model, sections):
    """Independent single-skill cross-check by cosine similarity."""
    import pandas as pd
    texts = dict(SKILLS)
    q_path = DATA_DIR / "adaptive_questions.csv"
    if q_path.exists():
        qs = pd.read_csv(q_path)
        if {"skills", "question"}.issubset(qs.columns):
            for sid, grp in qs.groupby("skills"):
                if sid in texts:
                    texts[sid] += ". " + " ".join(
                        grp["question"].dropna().astype(str)
                    )
        else:
            print("WARNING: adaptive_questions.csv does not have the expected "
                  "columns. Using skill descriptions only.")
    enc = lambda t: model.encode(t, normalize_embeddings=True, show_progress_bar=False)
    skill_vecs = enc([texts[s] for s in SKILL_IDS])
    sec_vecs = enc([f"{s['path']}. {' '.join(s['text'].split()[:200])}" for s in sections])
    sims = sec_vecs @ skill_vecs.T
    best = sims.argmax(axis=1)
    return [(SKILL_IDS[b], float(sims[i, b])) for i, b in enumerate(best)]


def classify_sections(sections, priors, cache, cache_path):
    """Classify every section with Qwen. Returns a list of failures
    (empty list = every section was classified by Qwen)."""
    failures = []
    for i, (s, (e_skill, e_sim)) in enumerate(zip(sections, priors), 1):
        result, last_error = None, None
        for _attempt in range(CLASSIFY_ATTEMPTS):
            try:
                result = llm_classify(s, cache)
                break
            except Exception as ex:
                last_error = ex

        if result is None:
            failures.append({"index": i, "source": s.get("source", "unknown"),
                             "path": s["path"], "error": f"{type(last_error).__name__}: {last_error}"})
            if len(failures) == 1:
                print(f"   WARNING: Qwen failed at section {i} ({s['path']}): {last_error}")
            continue

        llm_skills, confidence = result["skills"], result["confidence"]
        s["reason"] = result.get("reason", "")
        s["llm_skills"] = llm_skills
        s["embed_skill"], s["embed_sim"] = e_skill, e_sim
        s["confidence"] = confidence
        s["skills"] = llm_skills

        # Disagreement is a reason to inspect, not proof Qwen is wrong.
        s["review"] = confidence == "Low" or (bool(llm_skills) and e_skill not in llm_skills)

        if i % 10 == 0 or i == len(sections):
            print(f"   {i}/{len(sections)} sections classified")
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return failures


# ---------- 4. chunk per skill ----------
def chunk_words(text):
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    for start in range(0, max(len(words), 1), step):
        piece = words[start:start + CHUNK_WORDS]
        if len(piece) >= 30 or start == 0:
            yield " ".join(piece)
        if start + CHUNK_WORDS >= len(words):
            break


def build_skill_chunks(sections):
    by_skill = {}
    for s in sections:
        # A section can belong to multiple skills.
        for skill in s.get("skills", []):
            pages_str = (f"{s['start']}" if s["start"] == s["end"]
                         else f"{s['start']}-{s['end']}")
            for c in chunk_words(s["text"]):
                lst = by_skill.setdefault(skill, [])
                cid = f"{skill}_C{len(lst) + 1}"
                lst.append({
                    "chunk_id": cid,
                    "skill_id": skill,
                    "content": (f"## {cid} --- {s['title']}\n\n"
                                f"Source: {s.get('source', 'unknown')}, {s['chapter']}, "
                                f"PDF pages {pages_str}\n\n{c}"),
                    "source": s.get("source", "unknown"),
                    "chapter": s["chapter"],
                    "section": s["path"],
                    "page_start": s["start"],
                    "page_end": s["end"],
                })
    return by_skill


def install_indexes(tmp_dir: Path, data_dir: Path, produced: set):
    """Swap freshly built files into place. Runs only after ALL embedding
    succeeded. Skills with no new content have their old files removed so
    stale indexes from a previous textbook are never loaded."""
    for sid in SKILL_IDS:
        for suffix in ("_index.faiss", "_chunks.pkl"):
            name = f"{sid.lower()}{suffix}"
            if sid in produced:
                os.replace(tmp_dir / name, data_dir / name)   # atomic overwrite
            elif (data_dir / name).exists():
                (data_dir / name).unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main(textbooks: list[str]):
    import faiss
    from sentence_transformers import SentenceTransformer

    if not textbooks:
        raise ValueError("Provide at least one textbook PDF.")

    print("0/6 Checking Qwen...")
    check_ollama()

    DATA_DIR.mkdir(exist_ok=True)
    KB_DIR.mkdir(exist_ok=True)

    print("1/6 Extracting text and pages from all textbooks...")
    all_sections = []

    for book_number, textbook in enumerate(textbooks, 1):
        path = Path(textbook).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Textbook not found: {path}")

        print(f"\n   Processing: {path.name}")
        pages = extract_pages(path)
        if not pages or sum(len(t.strip()) for _, t in pages) < 100:
            raise ValueError(
                f"Very little text was extracted from {path.name}. "
                "Check whether the PDF is scanned or requires OCR."
            )

        toc = extract_pdf_toc(path) if path.suffix.lower() == ".pdf" else []
        print(f"   PDF bookmarks found: {len(toc)}")
        sections = build_sections(pages, toc)
        if not sections:
            print("   No usable headings found - using page-window sections.")
            sections = fallback_sections(pages)

        for section in sections:
            section["source"] = path.name
            section["section_id"] = f"BOOK{book_number:02d}_{section['section_id']}"

        full_md = KB_DIR / f"{path.stem}_full.md"
        full_md.write_text(
            "\n\n".join(f"<!-- page {n} -->\n{t}" for n, t in pages),
            encoding="utf-8"
        )
        print(f"   {len(pages)} pages -> {len(sections)} sections")
        all_sections.extend(sections)

    print(f"\n   Total: {len(all_sections)} sections from {len(textbooks)} textbooks")

    print(f"2/6 Classifying sections with {LLM_MODEL} (cached results are reused)...")
    cache_path = DATA_DIR / "classification_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    embed_model = SentenceTransformer(EMBED_MODEL)
    priors = embedding_prior(embed_model, all_sections)
    failures = classify_sections(all_sections, priors, cache, cache_path)

    if failures:
        log_path = DATA_DIR / "classification_failures.json"
        log_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        sys.exit(
            f"\nSTOPPED: {len(failures)} of {len(all_sections)} sections could not be "
            "classified by Qwen.\n"
            f"Details: {log_path}\n"
            "No indexes were built or changed. Progress is cached - fix the "
            "problem (is Ollama still running?) and re-run to resume."
        )

    report = [{
        "section_id": s["section_id"],
        "source": s.get("source", "unknown"),
        "chapter": s["chapter"],
        "path": s["path"],
        "start_page": s["start"],
        "end_page": s["end"],
        "llm_skills": s.get("llm_skills", []),
        "embedding_skill": s.get("embed_skill"),
        "embedding_similarity": s.get("embed_sim"),
        "confidence": s.get("confidence"),
        "reason": s.get("reason", ""),
        "final_skills": s.get("skills", []),
        "review": s.get("review", False),
    } for s in all_sections]
    report_path = DATA_DIR / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"   Classification report saved to: {report_path}")

    print("3/6 Chunking classified content from all textbooks...")
    by_skill = build_skill_chunks(all_sections)

    # Build all new indexes in a temporary folder before replacing old indexes.
    tmp_dir = DATA_DIR / "_ingest_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()

    print("4/6 Embedding with", EMBED_MODEL, "and 5/6 building FAISS indexes...")
    for skill, chunks in sorted(by_skill.items()):
        emb = embed_model.encode(
            [c["content"] for c in chunks],
            convert_to_numpy=True
        ).astype("float32")
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(tmp_dir / f"{skill.lower()}_index.faiss"))
        with open(tmp_dir / f"{skill.lower()}_chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

    install_indexes(tmp_dir, DATA_DIR, set(by_skill))

    print("6/6 Summary")
    for sid in SKILL_IDS:
        chunks = by_skill.get(sid, [])
        if not chunks:
            print(f"   {sid}: 0 chunks   <-- NO CONTENT in these textbooks")
            continue
        per_book = {}
        for c in chunks:
            per_book[c["source"]] = per_book.get(c["source"], 0) + 1
        detail = ", ".join(f"{b}: {n}" for b, n in per_book.items())
        print(f"   {sid}: {len(chunks)} chunks  ({detail})")
    print(f"   {sum(s['review'] for s in all_sections)} sections flagged for review "
          f"in {report_path.name}")
    print(f"   Sources: {', '.join(Path(p).name for p in textbooks)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python src/ingest_textbook.py <textbook1.pdf> [textbook2.pdf ...]")
    main(sys.argv[1:])