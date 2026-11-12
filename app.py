"""AdaptIQ main Streamlit application.

Run from the repository root:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from student_store import load_student, save_student, new_student_record
from ui_upload import render_upload_page
from diagnostic import render_diagnostic_page
from ui_dashboard import render_dashboard_page
from ui_practice import render_practice_page
from ui_results import render_results_page

st.set_page_config(
    page_title="AdaptIQ | Adaptive Learning",
    page_icon="🧠",
    layout="wide",
)

STAGES = [
    "Student identification",
    "Textbook upload",
    "Diagnostic assessment",
    "Mastery dashboard",
    "Adaptive practice",
    "Results & recommendations",
]

def initialize_state() -> None:
    defaults = {
        "student_id": "",
        "student": None,
        "stage": STAGES[0],
        "diagnostic_index": 0,
        "diagnostic_answers": {},
        "practice_session": None,
        "practice_question": None,
        "practice_feedback": None,
        "practice_limit": 10,
        "practice_allocation": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def switch_stage(stage: str) -> None:
    st.session_state.stage = stage
    st.rerun()

initialize_state()

st.title("🧠 AdaptIQ")
st.caption("An adaptive learning journey powered by diagnostic assessment and knowledge tracing.")

with st.sidebar:
    st.header("Learning journey")
    if st.session_state.student:
        st.success(f"Student: {st.session_state.student_id}")
    else:
        st.info("Identify yourself to begin.")

    selected_stage = st.radio(
        "Go to stage",
        STAGES,
        index=STAGES.index(st.session_state.stage),
        key=f"stage_navigation_{st.session_state.stage}",
    )
    if selected_stage != st.session_state.stage:
        st.session_state.stage = selected_stage
        if st.session_state.get("student"):
            st.session_state.student["current_stage"] = selected_stage
            save_student(st.session_state.student)
        st.rerun()

    st.divider()
    if st.session_state.student:
        record = st.session_state.student
        completed = []
        if record.get("diagnostic", {}).get("completed"):
            completed.append("Diagnostic")
        if record.get("practice", {}).get("completed"):
            completed.append("Practice")
        st.caption("Completed: " + (", ".join(completed) if completed else "No stages yet"))

stage = st.session_state.stage
if st.session_state.get("student"):
    st.session_state.student["current_stage"] = stage

if stage == STAGES[0]:
    st.header("1. Student identification")
    st.write("Enter your student ID. AdaptIQ keeps a separate progress record for each student.")
    with st.form("student_identification"):
        student_id = st.text_input(
            "Student ID",
            value=st.session_state.student_id,
            placeholder="e.g. STU2026A",
            help="Use letters, numbers, hyphens or underscores.",
        ).strip()
        submitted = st.form_submit_button("Continue", type="primary")
    if submitted:
        try:
            record = load_student(student_id)
            if record is None:
                record = new_student_record(student_id)
                save_student(record)
            st.session_state.student_id = student_id
            st.session_state.student = record
            st.session_state.stage = STAGES[1]
            st.success("Student record ready. Your progress will be saved automatically.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

elif not st.session_state.student:
    st.warning("Please enter your student ID first.")
    if st.button("Go to student identification"):
        switch_stage(STAGES[0])

else:
    if stage == STAGES[1]:
        render_upload_page()
    elif stage == STAGES[2]:
        render_diagnostic_page()
    elif stage == STAGES[3]:
        render_dashboard_page()
    elif stage == STAGES[4]:
        render_practice_page()
    elif stage == STAGES[5]:
        render_results_page()

# Keep the in-memory record and JSON file synchronized after each stage render.
if st.session_state.get("student"):
    save_student(st.session_state.student)
