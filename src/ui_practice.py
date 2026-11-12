"""Adaptive practice UI integrated with the existing BKT and AdaptiveSession."""
from __future__ import annotations

import csv
from pathlib import Path
import streamlit as st

from bkt import BKT
from adaptive_engine import AdaptiveQuestionSelector
from adaptive_session import AdaptiveSession
from student_store import save_student

ROOT = Path(__file__).resolve().parent.parent
SKILLS_PATH = ROOT / "data" / "skills.csv"
ADAPTIVE_QUESTIONS_PATH = ROOT / "data" / "adaptive_questions.csv"
ADAPTIVE_Q_PATH = ROOT / "data" / "adaptive_q_matrix.csv"

class QuotaAwareSelector:
    """Adapter: lets the existing selector choose among skills with quota left.

    The selector—not the Streamlit page—still chooses the weakest eligible skill
    and the next question.
    """
    def __init__(self, base_selector, allocation):
        self.base = base_selector
        self.allocation = {k: int(v) for k, v in allocation.items()}
        self.used = {k: 0 for k in self.allocation}

    def select_next_question(self, mastery, asked_question_ids=None, allow_repeat=False):
        asked = set(asked_question_ids or set())
        skill_candidates = []
        for skill, quota in self.allocation.items():
            used = sum(
                1 for q in self.base.questions
                if q["question_id"] in asked and skill in q.get("skills_list", [])
            )
            if used < quota:
                has_unseen = any(
                    skill in q.get("skills_list", []) and
                    (allow_repeat or q["question_id"] not in asked)
                    for q in self.base.questions
                )
                if has_unseen and skill in mastery:
                    skill_candidates.append(skill)
        if not skill_candidates:
            raise ValueError("No allocated unseen questions remain.")
        eligible_mastery = {sid: mastery[sid] for sid in skill_candidates}
        return self.base.select_next_question(
            eligible_mastery,
            asked_question_ids=asked,
            allow_repeat=allow_repeat,
        )

def _read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def _build_adaptive_session(record):
    mastery = record["diagnostic"]["initial_mastery"]
    allocation = record.get("practice", {}).get("allocation") or st.session_state.practice_allocation
    if not allocation:
        raise ValueError("Calculate a question allocation on the mastery dashboard first.")
    selector = AdaptiveQuestionSelector(str(ADAPTIVE_QUESTIONS_PATH), str(ADAPTIVE_Q_PATH))
    quota_selector = QuotaAwareSelector(selector, allocation)
    bkt = BKT({sid: float(value) for sid, value in mastery.items()})
    session = AdaptiveSession(bkt=bkt, selector=quota_selector, asked_question_ids=set(), allow_repeat=False)

    # Rebuild the in-memory adaptive session from saved attempts. This keeps
    # BKT mastery and the selector's asked-question set consistent after a
    # Streamlit session restart or page refresh.
    for saved_attempt in record.get("practice", {}).get("attempts", []):
        replay_question = session.next_question()
        if replay_question["question_id"] != saved_attempt["question_id"]:
            raise RuntimeError(
                "Saved practice history no longer matches the adaptive question bank. "
                "Please contact the project administrator."
            )
        session.submit_answer(saved_attempt["answer"])
    return session

def _get_session(record):
    session = st.session_state.get("practice_session")
    if session is None:
        session = _build_adaptive_session(record)
        st.session_state.practice_session = session
    return session

def _get_question(session):
    if st.session_state.get("practice_question") is None:
        st.session_state.practice_question = session.next_question()
    return st.session_state.practice_question

def render_practice_page():
    st.header("5. Adaptive practice")
    record = st.session_state.student
    if not record.get("diagnostic", {}).get("completed"):
        st.warning("Complete the diagnostic assessment before starting practice.")
        return
    allocation = record.get("practice", {}).get("allocation") or st.session_state.get("practice_allocation", {})
    if not allocation:
        st.warning("Choose a practice length and calculate your allocation on the mastery dashboard.")
        if st.button("Go to mastery dashboard"):
            st.session_state.stage = "Mastery dashboard"
            st.rerun()
        return

    practice = record.setdefault("practice", {})
    attempts = practice.setdefault("attempts", [])
    total = int(practice.get("requested_questions") or sum(allocation.values()))
    if practice.get("completed") or len(attempts) >= total:
        st.success("Adaptive practice is complete.")
        if st.button("View results"):
            st.session_state.stage = "Results & recommendations"
            st.rerun()
        return

    try:
        session = _get_session(record)
        question = _get_question(session)
    except Exception as exc:
        st.warning(f"Practice has stopped because no eligible question remains: {exc}")
        if st.button("View current results"):
            practice["completed"] = True
            practice["final_mastery"] = dict(
                st.session_state.practice_session.bkt.mastery
            ) if st.session_state.get("practice_session") else practice.get("final_mastery", {})
            save_student(record)
            st.session_state.stage = "Results & recommendations"
            st.rerun()
        return

    st.progress(len(attempts) / max(total, 1), text=f"Question {len(attempts) + 1} of {total}")
    st.caption(f"Target skill: {question['target_skill']} · Difficulty: {question['difficulty']}")
    st.subheader(question["question"])
    options = {
        "A": question["option_a"], "B": question["option_b"],
        "C": question["option_c"], "D": question["option_d"],
    }

    feedback = st.session_state.get("practice_feedback")
    if feedback:
        st.info(feedback["message"])
        if feedback.get("correct"):
            st.success("Correct answer!")
        else:
            st.error(f"Incorrect. Correct answer: {feedback['correct_answer']}")
        st.write(feedback["explanation"])
        if st.button("Next question", type="primary"):
            st.session_state.practice_feedback = None
            st.session_state.practice_question = None
            if len(attempts) >= total:
                practice["completed"] = True
                practice["final_mastery"] = dict(session.bkt.mastery)
                save_student(record)
                st.session_state.stage = "Results & recommendations"
            st.rerun()
        return

    with st.form(f"practice_answer_{question['question_id']}_{len(attempts)}"):
        selected = st.radio(
            "Choose an answer",
            list(options.keys()),
            format_func=lambda letter: f"{letter}. {options[letter]}",
        )
        submitted = st.form_submit_button("Submit answer", type="primary")
    if submitted:
        if selected not in options:
            st.warning("Select an answer before submitting.")
            return
        try:
            result = session.submit_answer(selected)
        except Exception as exc:
            st.error(f"Could not submit this answer: {exc}")
            return
        correct_letter = question["correct_answer"]
        correct_text = options[correct_letter]
        attempt = {
            "question_id": result["question_id"],
            "skill": question["target_skill"],
            "skills": question["skills"],
            "answer": selected,
            "correct_answer": correct_letter,
            "correct": bool(result["correct"]),
            "updates": result["updates"],
        }
        attempts.append(attempt)
        practice["final_mastery"] = dict(session.bkt.mastery)
        save_student(record)
        st.session_state.practice_feedback = {
            "correct": bool(result["correct"]),
            "correct_answer": f"{correct_letter}. {correct_text}",
            "message": "Your answer has been evaluated and your mastery estimate has been updated.",
            "explanation": (
                f"The correct option is {correct_letter}: {correct_text}. "
                "Review this concept in the relevant textbook section if you need a refresher."
            ),
        }
        st.rerun()
