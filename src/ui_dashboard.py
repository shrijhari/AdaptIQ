"""Student mastery dashboard and adaptive question allocation."""
from __future__ import annotations

import csv
from pathlib import Path
import streamlit as st

from student_store import save_student

ROOT = Path(__file__).resolve().parent.parent
SKILLS_PATH = ROOT / "data" / "skills.csv"
ADAPTIVE_QUESTIONS = ROOT / "data" / "adaptive_questions.csv"

def _skills():
    with SKILLS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def available_practice_skills() -> dict[str, int]:
    """Return skills with at least one item in the existing adaptive bank."""
    counts = {}
    with ADAPTIVE_QUESTIONS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            skill = row.get("skills", "").strip()
            if skill:
                counts[skill] = counts.get(skill, 0) + 1
    return counts

def allocate_questions(mastery: dict[str, float], total: int, available: dict[str, int]) -> dict[str, int]:
    """Fallback allocator because this repository currently has no allocation.py.

    Allocate proportionally to skill weakness, while respecting question-bank
    capacity. Remaining questions are redistributed to skills with capacity.
    """
    allocation = {sid: 0 for sid in available}
    eligible = [sid for sid, count in available.items() if count > 0 and sid in mastery]
    remaining = min(max(int(total), 0), sum(available[sid] for sid in eligible))
    while remaining and eligible:
        weights = {sid: max(0.05, 1.0 - float(mastery.get(sid, 0.0))) for sid in eligible}
        weight_sum = sum(weights.values())
        ranked = sorted(eligible, key=lambda sid: (-weights[sid], sid))
        assigned_this_round = 0
        for sid in ranked:
            if remaining <= 0:
                break
            if allocation[sid] >= available[sid]:
                continue
            share = max(1, round(remaining * weights[sid] / weight_sum))
            take = min(share, available[sid] - allocation[sid], remaining)
            allocation[sid] += take
            remaining -= take
            assigned_this_round += take
        eligible = [sid for sid in eligible if allocation[sid] < available[sid]]
        if assigned_this_round == 0:
            break
    return allocation

def render_dashboard_page():
    st.header("4. Mastery dashboard")
    record = st.session_state.student
    diagnostic = record.get("diagnostic", {})
    if not diagnostic.get("completed"):
        st.warning("Complete the diagnostic assessment to see your initial mastery.")
        if st.button("Go to diagnostic"):
            st.session_state.stage = "Diagnostic assessment"
            st.rerun()
        return

    mastery = diagnostic.get("initial_mastery", {})
    skills = _skills()
    st.write("Your initial mastery probabilities:")
    cols = st.columns(2)
    for i, skill in enumerate(skills):
        sid = skill["skill_id"]
        value = float(mastery.get(sid, 0.0))
        with cols[i % 2]:
            st.metric(skill["skill_name"], f"{value:.1%}")
            st.progress(min(max(value, 0.0), 1.0))

    st.divider()
    st.subheader("Plan your adaptive practice")
    available = available_practice_skills()
    unavailable = [s["skill_id"] for s in skills if available.get(s["skill_id"], 0) == 0]
    if unavailable:
        st.warning("No adaptive questions are currently available for: " + ", ".join(unavailable))
    if not available:
        st.error("The adaptive question bank could not be read or contains no questions.")
        return

    choice = st.selectbox("How many questions would you like to attempt?", [10, 15, 20], index=0)
    if st.button("Calculate my question allocation", type="primary"):
        allocation = allocate_questions(mastery, choice, available)
        actual_total = sum(allocation.values())
        st.session_state.practice_allocation = allocation
        st.session_state.practice_limit = actual_total
        record["practice"]["requested_questions"] = actual_total
        record["practice"]["allocation"] = allocation
        record["current_stage"] = "Mastery dashboard"
        save_student(record)

    allocation = st.session_state.get("practice_allocation", {})
    if allocation:
        st.subheader("Your question distribution")
        for skill in skills:
            sid = skill["skill_id"]
            count = allocation.get(sid, 0)
            if count:
                st.write(f"**{sid} — {skill['skill_name']}:** {count} question(s)")
            elif sid in unavailable:
                st.caption(f"{sid} — {skill['skill_name']}: unavailable (no questions)")
        st.info(f"Total allocated: {sum(allocation.values())} question(s)")
        if st.button("Start adaptive practice", type="primary"):
            st.session_state.stage = "Adaptive practice"
            st.session_state.student["current_stage"] = "Adaptive practice"
            st.rerun()
