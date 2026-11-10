"""
ingest_textbook.py  (AdaptIQ)

Textbook PDF(s) -> text/chapters/sections/pages -> Qwen (Ollama) assigns each
section to one or MORE of S1-S6 -> chunk per skill -> all-MiniLM-L6-v2
embeddings -> per-skill FAISS index + chunks pickle.

Usage:
    ollama serve                      (and: ollama pull qwen2.5:3b)
    pip install pymupdf4llm ollama sentence-transformers faiss-cpu numpy pandas
    python src/ingest_textbook.py knowledge_base/ISLP_website.pdf knowledge_base/dl2.pdf

Section detection:
    - Uses the PDF's built-in bookmarks (table of contents) when it has them -
      the most reliable source of real chapter/section titles.
    - Otherwise falls back to markdown headings, ignoring code blocks and
      figure/table captions (so Python comments are not mistaken for headings).
    - Very long sections are split so Qwen sees each part.
    - Manual corrections: data/skill_overrides.json (see --overrides-template).

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
MAX_SECTION_WORDS = 3500  # longer sections are split into parts
SPLIT_TARGET_WORDS = 2500
TOC_MAX_LEVEL = 2         # bookmark levels used as sections (1=chapter, 2=section)
MIN_TOC_ENTRIES = 10      # fewer bookmarks than this = treat PDF as having none
OVERRIDES_PATH = DATA_DIR / "skill_overrides.json"
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


# ---------- 2. chapters / sections ----------
CAPTION_RE = re.compile(r"^(t?fig(ure)?\.?|table|listing|algorithm)\s*\d", re.I)
CODE_LIKE_RE = re.compile(r"[=(){}\[\];<>]|->")


def clean_heading(raw: str) -> str:
    text = re.sub(r"<[^>]+>", "", raw)            # strip <mark>, <br>, ...
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_real_heading(title: str) -> bool:
    """Reject captions, code and Python comments that start with '#'."""
    words = title.split()
    if not (3 <= len(title) <= 100) or len(words) > 14:
        return False
    if CAPTION_RE.match(title):                    # "Fig. 1.4.1 ...", "Table 3.2"
        return False
    if CODE_LIKE_RE.search(title):                 # code or leftover HTML
        return False
    if not title[0].isalnum():
        return False
    if title[0].islower() and len(words) >= 3:     # "create a tensor" (comment)
        return False
    return True


def build_sections(pages):
    """Heading-based sections (used when the PDF has no bookmarks)."""
    sections, stack = [], []
    current = None
    intro_lines, intro_start, intro_end = [], None, None

    for page_no, text in pages:
        in_code = False                            # fences are balanced per page
        for line in text.splitlines():
            m = None
            if line.strip().startswith("```"):
                in_code = not in_code
            elif not in_code:
                m = HEAD_RE.match(line)
            title = clean_heading(m.group(2)) if m else ""

            if m and is_real_heading(title):
                level = len(m.group(1))
                stack = stack[: level - 1] + [title]
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
                # text before the first heading = introduction / front matter
                if intro_start is None:
                    intro_start = page_no
                intro_lines.append(line)
                intro_end = page_no

    for s in sections:
        s["text"] = "\n".join(s.pop("lines")).strip()

    intro_text = "\n".join(intro_lines).strip()
    if len(intro_text.split()) >= 30:
        sections.insert(0, {
            "title": "Introduction", "path": "Introduction",
            "chapter": "Introduction", "start": intro_start,
            "end": intro_end, "text": intro_text,
        })
    return finalize_sections(sections)


def get_toc(path: Path):
    """PDF bookmarks as [(level, title, page), ...]; [] if none."""
    if path.suffix.lower() != ".pdf":
        return []
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open(str(path))
        toc = doc.get_toc()
        doc.close()
        return toc
    except Exception:
        return []


def build_sections_from_toc(pages, toc):
    """Sections from PDF bookmarks. A section spans from its own start page to
    the page where the next section starts (that boundary page appears in both,
    which acts like chunk overlap)."""
    page_text = dict(pages)
    last_page = pages[-1][0]
    entries = []
    for level, title, page in toc:
        title = clean_heading(title)
        if level <= TOC_MAX_LEVEL and 1 <= page <= last_page and title:
            entries.append((level, title, page))

    sections, stack = [], []
    for i, (level, title, page) in enumerate(entries):
        stack = stack[: level - 1] + [title]
        end = entries[i + 1][2] if i + 1 < len(entries) else last_page
        end = max(end, page)
        text = "\n".join(page_text.get(n, "") for n in range(page, end + 1)).strip()
        sections.append({
            "title": title, "path": " > ".join(stack), "chapter": stack[0],
            "start": page, "end": end, "text": text,
        })
    return finalize_sections(sections)


def split_oversized(sections):
    """Split very long sections so Qwen (which reads the start) sees each part."""
    out = []
    for s in sections:
        words = s["text"].split()
        if len(words) <= MAX_SECTION_WORDS:
            out.append(s)
            continue
        n = -(-len(words) // SPLIT_TARGET_WORDS)
        size = -(-len(words) // n)
        span = s["end"] - s["start"]
        for k in range(n):
            part = " ".join(words[k * size:(k + 1) * size])
            if not part:
                continue
            ns = dict(s)
            ns["text"] = part
            ns["title"] = f"{s['title']} (part {k + 1}/{n})"
            ns["path"] = f"{s['path']} (part {k + 1}/{n})"
            ns["start"] = s["start"] + round(span * k / n)       # approximate pages
            ns["end"] = s["end"] if k == n - 1 else s["start"] + round(span * (k + 1) / n)
            out.append(ns)
    return out


def finalize_sections(sections):
    # Merge very small sections into the previous one (same chapter only).
    merged = []
    for s in sections:
        if (merged and len(s["text"].split()) < MIN_SECTION_WORDS
                and merged[-1]["chapter"] == s["chapter"]):
            merged[-1]["text"] += f"\n\n{s['title']}\n{s['text']}"
            merged[-1]["end"] = s["end"]
        else:
            merged.append(s)
    result = split_oversized([s for s in merged if len(s["text"].split()) >= 30])
    for n, s in enumerate(result, 1):
        s["section_id"] = f"SEC_{n:04d}"
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


# ---------- manual corrections ----------
def section_key(s):
    return f"{s['source']} | {s['path']} | p{s['start']}"


def load_overrides():
    if not OVERRIDES_PATH.exists():
        return {}
    raw = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    out = {}
    for key, skills in raw.items():
        if key.startswith("_"):                    # "_help" / comment entries
            continue
        valid = [x for x in skills if x in SKILLS]
        if len(valid) != len(skills):
            print(f"   WARNING: invalid skill ID in override: {key}")
        out[key] = valid
    return out


def apply_overrides(sections, overrides):
    """Your manual skill assignments win over Qwen. Returns unmatched keys."""
    used = set()
    for s in sections:
        for key in (section_key(s), s["section_id"]):
            if key in overrides:
                s["skills"] = overrides[key]
                s["overridden"] = True
                s["review"] = False
                used.add(key)
                break
    return [k for k in overrides if k not in used]


def make_overrides_template():
    """Write data/skill_overrides.template.json from the last report: one entry
    per flagged section, pre-filled with the current skills. Edit the skills,
    delete entries you agree with, and save as skill_overrides.json."""
    report = json.loads((DATA_DIR / "classification_report.json").read_text(encoding="utf-8"))
    template = {"_help": "Edit skill lists (e.g. [\"S5\"]). Delete entries that are fine. "
                         "Save as skill_overrides.json, then re-run ingestion."}
    for r in report:
        if r.get("review") and not r.get("overridden"):
            key = f"{r['source']} | {r['path']} | p{r['start_page']}"
            template[key] = r["final_skills"]
    out = DATA_DIR / "skill_overrides.template.json"
    out.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(template) - 1} flagged sections to {out}")


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

        toc = get_toc(path)
        sections, method = [], "detected headings"
        if len(toc) >= MIN_TOC_ENTRIES:
            sections = build_sections_from_toc(pages, toc)
            method = f"PDF bookmarks ({len(toc)} entries)"
        if not sections:
            sections, method = build_sections(pages), "detected headings"
        if not sections:
            print("   No usable headings found - using page-window sections.")
            sections, method = fallback_sections(pages), "page windows"
        print(f"   Section method: {method}")

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

    overrides = load_overrides()
    if overrides:
        unmatched = apply_overrides(all_sections, overrides)
        print(f"   Applied {len(overrides) - len(unmatched)} manual overrides "
              f"from {OVERRIDES_PATH.name}")
        if unmatched:
            print(f"   WARNING: {len(unmatched)} override keys matched no section "
                  "(section titles/pages changed?):")
            for k in unmatched[:5]:
                print("     ", k)

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
        "overridden": s.get("overridden", False),
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
        sys.exit("Usage: python src/ingest_textbook.py <textbook1.pdf> [textbook2.pdf ...]\n"
                 "       python src/ingest_textbook.py --overrides-template")
    if sys.argv[1] == "--overrides-template":
        make_overrides_template()
    else:
        main(sys.argv[1:])