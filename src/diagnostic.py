"""Fixed 30-item diagnostic assessment and DINA integration."""
from __future__ import annotations

import csv
from pathlib import Path
import streamlit as st

from cdm import dina
from student_store import save_student

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "questions.csv"
Q_MATRIX_PATH = ROOT / "data" / "q_matrix.csv"
SKILLS_PATH = ROOT / "data" / "skills.csv"
LETTERS = ("A", "B", "C", "D")

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required project file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def load_diagnostic_data():
    questions = _read_csv(QUESTIONS_PATH)
    q_rows = _read_csv(Q_MATRIX_PATH)
    skill_rows = _read_csv(SKILLS_PATH)
    if len(questions) != 30:
        raise ValueError(f"Expected 30 fixed diagnostic questions; found {len(questions)}.")
    q_by_id = {row["question_id"]: row for row in q_rows}
    skill_ids = [row["skill_id"] for row in skill_rows]
    aligned_q = []
    for question in questions:
        qid = question["question_id"]
        if qid not in q_by_id:
            raise ValueError(f"Question {qid} is missing from data/q_matrix.csv.")
        aligned_q.append([int(q_by_id[qid][skill]) for skill in skill_ids])
    return questions, skill_ids, aligned_q, {
        row["skill_id"]: row["skill_name"] for row in skill_rows
    }

def render_diagnostic_page():
    st.header("3. Diagnostic assessment")
    st.write("Answer the existing 30-question assessment. It estimates your starting mastery across six skills.")
    record = st.session_state.student
    try:
        questions, skill_ids, q_matrix, skill_names = load_diagnostic_data()
    except Exception as exc:
        st.error(f"Diagnostic assessment could not be loaded: {exc}")
        return

    diagnostic = record.setdefault("diagnostic", {})
    if diagnostic.get("completed"):
        st.success("Your diagnostic assessment is complete.")
        mastery = diagnostic.get("initial_mastery", {})
        for sid in skill_ids:
            st.progress(float(mastery.get(sid, 0.0)), text=f"{sid} — {skill_names[sid]}: {mastery.get(sid, 0.0):.1%}")
        if st.button("Retake diagnostic assessment"):
            diagnostic.update({"completed": False, "responses": [], "initial_mastery": {}})
            st.session_state.diagnostic_index = 0
            st.session_state.diagnostic_answers = {}
            save_student(record)
            st.rerun()
        return

    # Restore in-progress answers from the student's JSON record after a rerun
    # or a browser refresh/new Streamlit session.
    if not st.session_state.diagnostic_answers and diagnostic.get("in_progress_answers"):
        st.session_state.diagnostic_answers = dict(diagnostic["in_progress_answers"])
        st.session_state.diagnostic_index = int(diagnostic.get("in_progress_index", 0))

    answers = st.session_state.diagnostic_answers
    index = int(st.session_state.diagnostic_index)
    index = min(max(index, 0), len(questions) - 1)
    question = questions[index]
    st.progress((index + 1) / len(questions), text=f"Question {index + 1} of {len(questions)}")

    st.subheader(f"{question['question_id']}. {question['question']}")
    options = {
        "A": question["option_a"],
        "B": question["option_b"],
        "C": question["option_c"],
        "D": question["option_d"],
    }
    with st.form(f"diagnostic_form_{index}"):
        selected = st.radio(
            "Choose one answer",
            list(options.keys()),
            format_func=lambda letter: f"{letter}. {options[letter]}",
            index=(list(options).index(answers[question["question_id"]])
                   if question["question_id"] in answers else None),
        )
        submitted = st.form_submit_button(
            "Submit answer" if index < 29 else "Finish assessment",
            type="primary",
        )
    if submitted:
        if selected not in LETTERS:
            st.warning("Please select an answer before continuing.")
            return
        answers[question["question_id"]] = selected
        st.session_state.diagnostic_answers = answers
        diagnostic["in_progress_answers"] = dict(answers)
        if index < len(questions) - 1:
            st.session_state.diagnostic_index = index + 1
            diagnostic["in_progress_index"] = index + 1
            save_student(record)
            st.rerun()
        else:
            responses = []
            for item in questions:
                answer = answers.get(item["question_id"])
                if answer not in LETTERS:
                    st.error("One or more questions are unanswered. Please complete the assessment.")
                    return
                responses.append({
                    "question_id": item["question_id"],
                    "answer": answer,
                    "correct": int(answer == item["correct_answer"]),
                })
            try:
                model_result = dina(
                    [item["correct"] for item in responses],
                    q_matrix,
                )
                initial_mastery = {
                    sid: float(prob)
                    for sid, prob in zip(skill_ids, model_result["mastery_probabilities"])
                }
            except Exception as exc:
                st.error(f"DINA could not estimate mastery: {exc}")
                return
            diagnostic.update({
                "completed": True,
                "responses": responses,
                "initial_mastery": initial_mastery,
                "map_profile": list(model_result["map_profile"]),
                "in_progress_answers": {},
                "in_progress_index": 0,
            })
            record["current_stage"] = "Mastery dashboard"
            save_student(record)
            st.session_state.diagnostic_index = 0
            st.session_state.diagnostic_answers = {}
            st.success("Assessment complete. Your initial mastery estimates have been saved.")
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Previous", disabled=index == 0):
            st.session_state.diagnostic_index = index - 1
            diagnostic["in_progress_index"] = index - 1
            save_student(record)
            st.rerun()
    with col2:
        st.caption("Your answers are retained when the page reruns.")
