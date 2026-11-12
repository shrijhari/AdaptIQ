"""Student-specific JSON persistence for AdaptIQ."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
STUDENTS_DIR = ROOT / "data" / "students"
STUDENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

def student_path(student_id: str) -> Path:
    student_id = (student_id or "").strip()
    if not STUDENT_ID_RE.fullmatch(student_id):
        raise ValueError(
            "Student ID must contain 1–40 letters, numbers, hyphens or underscores."
        )
    return STUDENTS_DIR / f"{student_id}.json"

def new_student_record(student_id: str) -> dict[str, Any]:
    # Validate before using the ID in a path.
    student_path(student_id)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "student_id": student_id,
        "created_at": now,
        "updated_at": now,
        "current_stage": "Student identification",
        "uploaded_textbooks": [],
        "diagnostic": {
            "completed": False,
            "responses": [],
            "initial_mastery": {},
        },
        "practice": {
            "completed": False,
            "requested_questions": 0,
            "attempts": [],
            "final_mastery": {},
        },
        "results": {},
    }

def load_student(student_id: str) -> dict[str, Any] | None:
    path = student_path(student_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Could not read the saved student record: {exc}") from exc
    if record.get("student_id") != student_id:
        raise ValueError("The saved student record does not match this student ID.")
    return record

def save_student(record: dict[str, Any]) -> Path:
    student_id = str(record.get("student_id", "")).strip()
    path = student_path(student_id)
    STUDENTS_DIR.mkdir(parents=True, exist_ok=True)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)
    return path
