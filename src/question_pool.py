"""
question_pool.py  (AdaptIQ) - SHARED CONTRACT FILE

STANDARD QUESTION FORMAT (used everywhere: bank, generated, UI, selector):
    {
      "question_id":    "G_S2_0001",
      "question":       "stem only - no answer choices inside",
      "option_a" .. "option_d": "full answer text, no 'A)' prefix",
      "correct_answer": "A",                 # letter
      "difficulty":     "Easy" | "Medium" | "Hard",
      "skills":         ["S2"],              # list
      "explanation":    "why the answer is right",
      "source":         "ISLP_website.pdf, pages 45-47",
      "origin":         "generated" | "bank"
    }
Generated questions also carry "source_chunk_id" (extra field, safe to ignore).

PUBLIC FUNCTIONS (signatures are a contract - do not change):

    generate_pool(skill_id: str, n: int) -> list[dict]
        Up to n questions in the standard format, mixed difficulty.
        Reads the skill's textbook chunks, asks the local LLM (Ollama) to write
        questions, validates every one, caches the good ones in
        data/generated_pool/<skill>.json and returns them. Never raises for a
        skill with no textbook content or when Ollama is unavailable: it falls
        back to bank_pool(). May return fewer than n.
    bank_pool(skill_id, n)      # hand-written questions (fallback)
    validate_question(q)        # raises ValueError if q breaks the format
    check_question(q, ...)      # (ok, reason): format + quality checks

Try it from the project root (needs Ollama running with qwen2.5:3b):
    python src/question_pool.py S2 6
"""
from pathlib import Path
import csv
import difflib
import hashlib
import json
import logging
import os
import pickle
import random
import re
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated_pool"

DIFFICULTIES = ("Easy", "Medium", "Hard")

LLM_MODEL = "qwen2.5:3b"          # same local model as the rest of the project
MAX_ATTEMPT_FACTOR = 3            # at most 3 LLM calls per question we still need
VERIFY_BLIND_SOLVE = False        # True: the model must re-solve each question
                                  # without the key (slower, stricter)

log = logging.getLogger(__name__)

# Filled by the last generation run, handy for measuring quality.
last_run_stats = {"attempts": 0, "accepted": 0, "rejected": Counter()}


# ------------------------------------------------------------------ validation

def validate_question(q: dict) -> dict:
    for key in ("question_id", "question", "option_a", "option_b", "option_c",
                "option_d", "correct_answer", "difficulty", "skills"):
        if key not in q or q[key] in (None, "", []):
            raise ValueError(f"Missing field: {key}")
    if q["correct_answer"] not in ("A", "B", "C", "D"):
        raise ValueError("correct_answer must be A, B, C or D")
    if q["difficulty"] not in DIFFICULTIES:
        raise ValueError("difficulty must be Easy, Medium or Hard")
    if not isinstance(q["skills"], list):
        raise ValueError("skills must be a list")
    options = [q[f"option_{c}"].strip() for c in "abcd"]
    if len(set(options)) != 4:
        raise ValueError("options must be four different answers")
    if any(o.rstrip(".)").upper() in ("A", "B", "C", "D") for o in options):
        raise ValueError("options look like bare letters, not answer text")
    return q


_CHOICE_IN_STEM = re.compile(r"(?:^|\s)\(?[A-D][\).]\s")
_BAD_OPTIONS = ("all of the above", "none of the above", "both a and b",
                "a and b", "b and c")
_PASSAGE_WORDS = ("the passage", "the text", "the excerpt", "the source",
                  "the material", "the chunk", "as mentioned above",
                  "the given context", "the supplied")


# Pointers to numbered parts of the book ("Figure 11.7", "Section 3.2") or to
# a picture the student cannot see ("as shown in the figure").
_BOOK_REFERENCE = re.compile(
    r"\b(?:figure|fig\.|table|equation|eq\.|algorithm|listing|section|chapter"
    r"|exercise|example)s?\s*\(?\d", re.I)
_UNSEEN_VISUAL = re.compile(
    r"\b(?:in|from|of|see|shown in|illustrated in|above|below|depicted in) "
    r"the (?:figure|diagram|table|plot|graph|image|picture|chart)\b", re.I)


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4}


def check_question(q: dict, chunk_text: str = None, accepted_stems=()):
    """Quality gate for one question. Returns (ok, reason).

    Structural checks (no AI): required fields, four different options, valid
    letter and difficulty, explanation present, no choices inside the stem,
    near-identical options, answer leaked in the stem, 'the passage' wording,
    all/none-of-the-above, duplicates of already accepted questions.
    If chunk_text is given, also checks that the correct answer is supported
    by the source text (lexical overlap).
    """
    try:
        validate_question(q)
    except ValueError as exc:
        return False, str(exc)

    stem = q["question"].strip()
    options = [q[f"option_{c}"].strip() for c in "abcd"]
    correct = q[f"option_{q['correct_answer'].lower()}"].strip()

    if not q.get("explanation", "").strip():
        return False, "missing explanation"
    if len(stem) < 15:
        return False, "question is too short"
    if _CHOICE_IN_STEM.search(stem):
        return False, "answer choices written inside the question"
    low_stem = stem.lower()
    if any(p in low_stem for p in _PASSAGE_WORDS):
        return False, "question refers to 'the passage' the student cannot see"
    explanation = q["explanation"].strip()
    for text in (stem, explanation):
        if _BOOK_REFERENCE.search(text) or _UNSEEN_VISUAL.search(text):
            return False, "refers to a figure, table or section the student cannot see"
    if any(o.lower().rstrip(".") in _BAD_OPTIONS for o in options):
        return False, "uses an 'all/none of the above' style option"
    for i in range(4):
        for j in range(i + 1, 4):
            if difflib.SequenceMatcher(
                    None, options[i].lower(), options[j].lower()).ratio() > 0.9:
                return False, "two options are nearly identical"
    if len(correct) >= 6 and correct.lower() in low_stem:
        return False, "correct answer is given away in the question"
    for other in accepted_stems:
        if difflib.SequenceMatcher(
                None, low_stem, other.lower()).ratio() > 0.85:
            return False, "duplicate of another question"
    if chunk_text is not None:
        needed = _words(correct)
        if needed:
            found = len(needed & _words(chunk_text)) / len(needed)
            if found < 0.4:
                return False, "correct answer is not supported by the source text"
    return True, ""


# ------------------------------------------------------------------- the bank

def _load_bank():
    with open(DATA_DIR / "adaptive_q_matrix.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    skill_ids = list(rows[0].keys())[1:]
    matrix = {r["question_id"]: [s for s in skill_ids if int(r[s]) == 1] for r in rows}

    bank = []
    with open(DATA_DIR / "adaptive_questions.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            bank.append({
                "question_id": r["question_id"],
                "question": r["question"],
                "option_a": r["option_a"], "option_b": r["option_b"],
                "option_c": r["option_c"], "option_d": r["option_d"],
                "correct_answer": r["correct_answer"].strip().upper(),
                "difficulty": r["difficulty"],
                "skills": matrix.get(r["question_id"]) or [r["skills"]],
                "explanation": r.get("explanation", ""),
                "source": "question bank",
                "origin": "bank",
            })
    return bank


def _mix(questions: list, n: int) -> list:
    """Take n questions, rotating Easy -> Medium -> Hard so difficulty is mixed."""
    buckets = {d: [q for q in questions if q["difficulty"] == d] for d in DIFFICULTIES}
    out = []
    while len(out) < n and any(buckets.values()):
        for d in DIFFICULTIES:
            if buckets[d] and len(out) < n:
                out.append(buckets[d].pop(0))
    return out


def bank_pool(skill_id: str, n: int) -> list:
    pool = [q for q in _load_bank() if skill_id in q["skills"]]
    return _mix(pool, n)


# --------------------------------------------------------- textbook generation

_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "option_a": {"type": "string"},
        "option_b": {"type": "string"},
        "option_c": {"type": "string"},
        "option_d": {"type": "string"},
        "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "explanation": {"type": "string"},
    },
    "required": ["question", "option_a", "option_b", "option_c", "option_d",
                 "correct_answer", "explanation"],
    "additionalProperties": False,
}

_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["A", "B", "C", "D"]}},
    "required": ["answer"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You write one multiple-choice question for a university machine-learning "
    "course, using ONLY the passage you are given.\n"
    "Rules:\n"
    "- The question must make sense on its own. Never say 'the passage', "
    "'the text' or 'the source'.\n"
    "- Exactly four different, plausible options; exactly one is correct.\n"
    "- Options are plain answer text. Do not start them with A), B), 1. etc. "
    "Do not use 'all of the above' or 'none of the above'.\n"
    "- Never mention figures, tables, equations, sections, chapters or "
    "exercise numbers, and never write 'the passage' - not even in the "
    "explanation. The student cannot see them.\n"
    "- Do not put the answer inside the question.\n"
    "- The explanation is one or two sentences saying why the correct option "
    "is right.\n"
    "- Do not use facts that are not in the passage."
)

_DIFFICULTY_HINT = {
    "Easy": "Easy: recall a definition or a basic fact.",
    "Medium": "Medium: apply the idea to a small example or compare two ideas.",
    "Hard": "Hard: reason about a scenario, a trade-off or a cause and effect.",
}


def _call_llm(system: str, user: str, schema: dict) -> dict:
    """One structured call to the local model. Tests replace this function."""
    import ollama  # imported lazily so the rest of the module works without it

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        format=schema,
        options={"temperature": 0.2},
    )
    return json.loads(response["message"]["content"])


def _chunks_path(skill_id: str) -> Path:
    return DATA_DIR / f"{skill_id.lower()}_chunks.pkl"


def _load_chunks(skill_id: str):
    path = _chunks_path(skill_id)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            chunks = pickle.load(f)
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    return chunks or None


def _clean_text(chunk: dict) -> str:
    """Drop the '## S2_C41 --- title' and 'Source: ...' header lines."""
    parts = chunk["content"].split("\n\n")
    if parts and parts[0].startswith("##"):
        parts = parts[1:]
    if parts and parts[0].startswith("Source:"):
        parts = parts[1:]
    return "\n\n".join(parts).strip()


_JUNK_SECTION = re.compile(
    r"contents|index|preface|references|bibliograph|exercise|acknowledg", re.I)


def _usable(chunk: dict) -> bool:
    """Skip tables of contents, indexes, short fragments, code and equation soup."""
    if _JUNK_SECTION.search(chunk.get("section", "")):
        return False
    text = _clean_text(chunk)
    if len(text) < 400 or text.count("`") > 12:
        return False
    # Raw code and equation soup have many symbol characters per letter.
    symbols = sum(ch in "(){}[]=_;\\<>*" for ch in text)
    return symbols / len(text) <= 0.06


def _chunk_order(chunks: list, rng: random.Random, exclude_ids: set) -> list:
    """Chunks in an order that spreads questions over different sections."""
    by_section = {}
    for c in chunks:
        if c["chunk_id"] not in exclude_ids and _usable(c):
            by_section.setdefault(c.get("section", ""), []).append(c)
    groups = list(by_section.values())
    rng.shuffle(groups)
    for g in groups:
        rng.shuffle(g)
    order = []
    while any(groups):
        for g in groups:
            if g:
                order.append(g.pop())
    return order


def _strip_letter(option: str) -> str:
    return re.sub(r"^\(?[A-Da-d][\).:]\s+", "", str(option).strip())


def _make_question(raw: dict, skill_id: str, chunk: dict, number: int,
                   difficulty: str) -> dict:
    pages = (f", pages {chunk['page_start']}-{chunk['page_end']}"
             if chunk.get("page_start") else "")
    return {
        "question_id": f"G_{skill_id}_{number:04d}",
        "question": str(raw["question"]).strip(),
        "option_a": _strip_letter(raw["option_a"]),
        "option_b": _strip_letter(raw["option_b"]),
        "option_c": _strip_letter(raw["option_c"]),
        "option_d": _strip_letter(raw["option_d"]),
        "correct_answer": str(raw["correct_answer"]).strip().upper()[:1],
        "difficulty": difficulty,
        "skills": [skill_id],
        "explanation": str(raw["explanation"]).strip(),
        "source": f"{chunk.get('source', 'textbook')}{pages}",
        "origin": "generated",
        "source_chunk_id": chunk["chunk_id"],
    }


def _blind_solve(q: dict) -> bool:
    """Ask the model to answer WITHOUT the key; True if it agrees with us."""
    prompt = (f"{q['question']}\nA) {q['option_a']}\nB) {q['option_b']}\n"
              f"C) {q['option_c']}\nD) {q['option_d']}\nAnswer with one letter.")
    try:
        answer = _call_llm("You are a careful machine-learning student.",
                           prompt, _ANSWER_SCHEMA)
        return str(answer["answer"]).strip().upper() == q["correct_answer"]
    except Exception:
        return False


# ------------------------------------------------------------------- the cache

def _signature(skill_id: str) -> str:
    """Changes whenever the skill's chunk file is rebuilt (re-ingestion).

    A content hash (not a timestamp) so a cache committed to Git stays valid
    on every teammate's machine after cloning.
    """
    return hashlib.sha1(_chunks_path(skill_id).read_bytes()).hexdigest()


def _cache_path(skill_id: str) -> Path:
    return GENERATED_DIR / f"{skill_id}.json"


def _load_cache(skill_id: str) -> list:
    try:
        data = json.loads(_cache_path(skill_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    if data.get("signature") != _signature(skill_id):
        return []  # textbooks were re-ingested: old questions may be stale
    return data.get("questions", [])


def _save_cache(skill_id: str, questions: list) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(skill_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"signature": _signature(skill_id), "model": LLM_MODEL,
         "questions": questions}, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ generation

def _generate_more(skill_id: str, chunks: list, existing: list, need: int) -> list:
    """Return existing + up to `need` new valid questions."""
    stats = {"attempts": 0, "accepted": 0, "rejected": Counter()}
    global last_run_stats
    last_run_stats = stats

    used = {q.get("source_chunk_id") for q in existing}
    order = _chunk_order(chunks, random.Random(f"{skill_id}-{len(existing)}"), used)

    numbers = [int(q["question_id"].rsplit("_", 1)[1]) for q in existing]
    next_number = max(numbers, default=0) + 1

    accepted = list(existing)
    new = []
    max_attempts = need * MAX_ATTEMPT_FACTOR

    for chunk in order:
        if len(new) >= need or stats["attempts"] >= max_attempts:
            break
        stats["attempts"] += 1
        # Ask for whichever difficulty we have the fewest of so far.
        difficulty = min(DIFFICULTIES,
                         key=lambda d: sum(q["difficulty"] == d for q in accepted))
        text = _clean_text(chunk)
        user = (f"Difficulty - {_DIFFICULTY_HINT[difficulty]}\n"
                f"Topic area: {chunk.get('section', '')}\n\n"
                f"PASSAGE:\n{text}")
        try:
            raw = _call_llm(_SYSTEM_PROMPT, user, _QUESTION_SCHEMA)
        except Exception as exc:  # Ollama stopped, model missing, bad JSON...
            log.warning("Question generation stopped: %s", exc)
            stats["rejected"]["llm error"] += 1
            break
        try:
            q = _make_question(raw, skill_id, chunk, next_number, difficulty)
        except (KeyError, TypeError, AttributeError):
            stats["rejected"]["malformed model output"] += 1
            continue

        ok, reason = check_question(q, text, [a["question"] for a in accepted])
        if ok and VERIFY_BLIND_SOLVE and not _blind_solve(q):
            ok, reason = False, "model could not re-solve it without the key"
        if ok:
            new.append(q)
            accepted.append(q)
            next_number += 1
            stats["accepted"] += 1
        else:
            stats["rejected"][reason] += 1

    return list(existing) + new


def generate_pool(skill_id: str, n: int) -> list:
    """Up to n validated questions for a skill (see the module docstring)."""
    skill_id = skill_id.strip().upper()
    if n <= 0:
        return []

    chunks = _load_chunks(skill_id)
    if chunks is None:                      # no textbook content for this skill
        return bank_pool(skill_id, n)

    cached = _load_cache(skill_id)
    # Re-check old questions with today's rules, so a stricter inspector also
    # cleans a cache that was written earlier.
    questions = []
    for q in cached:
        if check_question(q, accepted_stems=[a["question"] for a in questions])[0]:
            questions.append(q)
    cleaned = len(questions) != len(cached)

    if len(questions) < n:
        before = len(questions)
        questions = _generate_more(skill_id, chunks, questions, n - len(questions))
        if len(questions) > before:
            cleaned = True
    if cleaned:
        _save_cache(skill_id, questions)

    pool = _mix(questions, n)
    if len(pool) < n:                       # top up from the hand-written bank
        have = {q["question_id"] for q in pool}
        extra = [q for q in _load_bank()
                 if skill_id in q["skills"] and q["question_id"] not in have]
        pool += _mix(extra, n - len(pool))
    return pool


if __name__ == "__main__":
    import sys

    skill = sys.argv[1] if len(sys.argv) > 1 else "S2"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    logging.basicConfig(level=logging.INFO)
    result = generate_pool(skill, count)
    for item in result:
        print(f"\n[{item['origin']}] {item['question_id']} ({item['difficulty']})")
        print(item["question"])
        for letter in "abcd":
            print(f"  {letter.upper()}. {item['option_' + letter]}")
        print("  Answer:", item["correct_answer"], "|", item["source"])
        print("  Why:", item["explanation"])
    print(f"\n{len(result)} questions. Last run: "
          f"{last_run_stats['accepted']} accepted of "
          f"{last_run_stats['attempts']} attempts; "
          f"rejected: {dict(last_run_stats['rejected'])}")
