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

v0 (this file): returns questions from the hand-written bank only, so the UI
can be built TODAY. Teammate 1 replaces the body of generate_pool() with real
textbook-based generation, KEEPING these signatures:

    generate_pool(skill_id: str, n: int) -> list[dict]
        Up to n questions in the standard format, mixed difficulty.
        Never raises for a skill with no textbook content: falls back to
        bank_pool(). May return fewer than n.
    bank_pool(skill_id, n)      # hand-written questions (fallback)
    validate_question(q)        # raises ValueError if q breaks the format
"""
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DIFFICULTIES = ("Easy", "Medium", "Hard")


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


def bank_pool(skill_id: str, n: int) -> list:
    pool = [q for q in _load_bank() if skill_id in q["skills"]]
    return pool[:n]


def generate_pool(skill_id: str, n: int) -> list:
    """v0 stub - bank questions only. Teammate 1: replace with real generation."""
    return bank_pool(skill_id, n)
