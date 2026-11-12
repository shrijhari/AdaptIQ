"""Final results, mastery comparison charts and recommendations."""
from __future__ import annotations

import csv
from pathlib import Path
import streamlit as st

from student_store import save_student

ROOT = Path(__file__).resolve().parent.parent
SKILLS_PATH = ROOT / "data" / "skills.csv"

def _skills():
    with SKILLS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def render_results_page():
    st.header("6. Results & learning recommendations")
    record = st.session_state.student
    diagnostic = record.get("diagnostic", {})
    practice = record.get("practice", {})
    if not diagnostic.get("completed"):
        st.warning("Complete the diagnostic assessment first.")
        return

    before = diagnostic.get("initial_mastery", {})
    after = practice.get("final_mastery") or before
    attempts = practice.get("attempts", [])
    total = len(attempts)
    correct = sum(1 for item in attempts if item.get("correct"))
    incorrect = total - correct
    accuracy = (correct / total * 100) if total else 0.0

    st.subheader("Practice performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Attempted", total)
    c2.metric("Correct", correct)
    c3.metric("Incorrect", incorrect)
    c4.metric("Accuracy", f"{accuracy:.1f}%")

    skills = _skills()
    skill_ids = [s["skill_id"] for s in skills]
    st.subheader("Mastery before and after practice")
    chart_rows = []
    for skill in skills:
        sid = skill["skill_id"]
        chart_rows.append({
            "Skill": sid,
            "Before diagnostic": float(before.get(sid, 0.0)),
            "After practice": float(after.get(sid, before.get(sid, 0.0))),
        })
    st.bar_chart(chart_rows, x="Skill", y=["Before diagnostic", "After practice"])

    st.subheader("Skill-wise progress")
    recommendations = []
    for skill in skills:
        sid = skill["skill_id"]
        start = float(before.get(sid, 0.0))
        end = float(after.get(sid, start))
        delta = end - start
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.write(f"**{sid} — {skill['skill_name']}**")
                st.progress(min(max(end, 0.0), 1.0), text=f"Current mastery: {end:.1%}")
            with right:
                st.metric("Change", f"{delta:+.1%}")
        if end < 0.60:
            recommendations.append(f"{sid} ({skill['skill_name']}) — continue practising this skill; estimated mastery is below 60%.")
        elif delta < 0.03:
            recommendations.append(f"{sid} ({skill['skill_name']}) — revisit the topic and attempt more questions to reinforce learning.")

    if recommendations:
        st.subheader("Recommended next steps")
        for recommendation in recommendations:
            st.write("• " + recommendation)
    else:
        st.success("Your current mastery estimates are at least 60% across all skills. Keep practising to maintain them.")

    record["results"] = {
        "total_attempted": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy_percent": round(accuracy, 2),
        "initial_mastery": before,
        "final_mastery": after,
        "mastery_change": {
            sid: round(float(after.get(sid, before.get(sid, 0.0))) - float(before.get(sid, 0.0)), 6)
            for sid in skill_ids
        },
        "recommendations": recommendations,
    }
    save_student(record)

    st.caption("Your results are saved in data/students/<student_id>.json.")
    if st.button("Return to dashboard"):
        st.session_state.stage = "Mastery dashboard"
        st.rerun()
