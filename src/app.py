"""Simple Streamlit UI for the AdaptIQ Person 3 prototype."""

import streamlit as st

from cdm import load_csv_data, dina
from bkt import BKT
from adaptive_engine import AdaptiveQuestionSelector
from adaptive_session import AdaptiveSession


st.set_page_config(page_title="AdaptIQ", page_icon="🧠")
st.title("AdaptIQ — Adaptive Learning Prototype")
st.write("DINA → BKT → Adaptive Question Selection")


@st.cache_resource
def build_session():
    skill_ids, question_ids, q_matrix, responses = load_csv_data(
        "data/q_matrix.csv",
        "data/student_responses.csv",
        "S001",
    )

    result = dina(responses, q_matrix)
    initial_mastery = dict(zip(skill_ids, result["mastery_probabilities"]))

    bkt = BKT(initial_mastery)
    for question_id, response in zip(question_ids, responses):
        index = question_ids.index(question_id)
        for i, value in enumerate(q_matrix[index]):
            if value == 1:
                bkt.update_skill(skill_ids[i], response)

    selector = AdaptiveQuestionSelector(
    "data/adaptive_questions.csv",
    "data/adaptive_q_matrix.csv",
)

   # Use a separate adaptive question bank so diagnostic
# questions are not repeated during the adaptive stage.
    return AdaptiveSession(
        bkt=bkt,
        selector=selector,
        asked_question_ids=set(question_ids),
        allow_repeat=False,
    )


if "session" not in st.session_state:
    st.session_state.session = build_session()

session = st.session_state.session

st.subheader("Current skill mastery")
for skill, value in session.bkt.mastery.items():
    st.progress(float(value), text=f"{skill}: {value:.3f}")

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if st.session_state.current_question is None:
    if st.button("Select next question"):
        st.session_state.current_question = session.next_question()
        st.rerun()
else:
    q = st.session_state.current_question

    st.subheader(f"Question {q['question_id']}")
    st.caption(
        f"Target skill: {q['target_skill']} · Difficulty: {q['difficulty']}"
    )
    st.write(q["question"])

    answer = st.radio(
        "Choose an answer:",
        [
            f"A. {q['option_a']}",
            f"B. {q['option_b']}",
            f"C. {q['option_c']}",
            f"D. {q['option_d']}",
        ],
        key=f"answer_{q['question_id']}_{len(session.asked_question_ids)}",
    )

    if st.button("Submit answer"):
        letter = answer[0]
        session.current_question = q
        result = session.submit_answer(letter)
        st.session_state.current_question = None

        if result["correct"]:
            st.success("Correct!")
        else:
            st.error("Incorrect.")

        for update in result["updates"]:
            st.write(
                f"{update['skill']}: "
                f"{update['mastery_before']:.3f} → "
                f"{update['mastery_after']:.3f}"
            )

        st.rerun()

st.info(
    "Prototype note: the current repository uses all 30 questions for the "
    "diagnostic history. A production adaptive loop should use a separate "
    "pool of unseen adaptive questions instead of repeating diagnostic items."
)
