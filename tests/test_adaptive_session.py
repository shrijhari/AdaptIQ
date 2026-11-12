import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_engine import AdaptiveQuestionSelector, NoQuestionAvailable
from adaptive_session import AdaptiveSession
from bkt import BKT

SKILLS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def make_selector():
    return AdaptiveQuestionSelector(
        str(ROOT / "data/questions.csv"),
        str(ROOT / "data/q_matrix.csv"),
    )


def make_session(quota=None, mastery=None):
    mastery = mastery or {s: 0.5 for s in SKILLS}
    return AdaptiveSession(BKT(mastery), make_selector(), quota=quota)


def wrong_letter(question):
    return "A" if question["correct_answer"] != "A" else "B"


# ---- selector: quota behaviour -------------------------------------------

def test_selector_skips_skill_with_no_quota():
    selector = make_selector()
    mastery = {"S1": 1.0, "S2": 1.0, "S3": 0.20, "S4": 1.0, "S5": 1.0, "S6": 1.0}
    quota = {"S1": 1, "S2": 1, "S3": 0, "S4": 1, "S5": 1, "S6": 1}
    q = selector.select_next_question(mastery, remaining_quota=quota)
    assert q["target_skill"] != "S3"  # weakest skill, but its quota is 0


def test_selector_raises_when_all_quotas_used():
    selector = make_selector()
    mastery = {s: 0.5 for s in SKILLS}
    with pytest.raises(NoQuestionAvailable):
        selector.select_next_question(mastery, remaining_quota={s: 0 for s in SKILLS})


def test_selector_without_quota_is_unchanged():
    selector = make_selector()
    mastery = {"S1": 1.0, "S2": 1.0, "S3": 0.20, "S4": 1.0, "S5": 0.99, "S6": 0.65}
    assert selector.select_next_question(mastery)["target_skill"] == "S3"


# ---- session: quota, termination, history --------------------------------

def test_session_never_exceeds_quota_and_terminates():
    quota = {"S1": 2, "S2": 1, "S3": 2, "S4": 0, "S5": 0, "S6": 0}
    session = make_session(quota=quota)
    used = {s: 0 for s in SKILLS}
    seen = set()

    guard = 0
    while not session.is_finished():
        guard += 1
        assert guard <= 20, "session did not terminate"
        q = session.next_question()
        if q is None:
            break
        assert q["question_id"] not in seen  # never repeats
        seen.add(q["question_id"])
        used[q["target_skill"]] += 1
        session.submit_answer("A")

    assert all(used[s] <= quota[s] for s in SKILLS)
    assert len(session.history) == sum(quota.values())
    assert session.is_finished()
    assert session.next_question() is None


def test_session_without_quota_never_finishes_by_itself():
    session = make_session()
    assert session.is_finished() is False
    assert session.next_question() is not None


def test_submit_answer_returns_correct_answer_and_explanation():
    session = make_session(quota={s: 1 for s in SKILLS})
    q = session.next_question()
    result = session.submit_answer(q["correct_answer"])
    assert result["correct_answer"] == q["correct_answer"]
    assert "explanation" in result
    assert result["correct"] == 1


def test_bkt_goes_up_after_correct_and_down_after_wrong():
    session = make_session(quota={s: 2 for s in SKILLS})

    q = session.next_question()
    up = session.submit_answer(q["correct_answer"])["updates"][0]
    assert up["mastery_after"] > up["mastery_before"]

    q = session.next_question()
    down = session.submit_answer(wrong_letter(q))["updates"][0]
    assert down["mastery_after"] < down["mastery_before"]


def test_history_records_every_answer():
    session = make_session(quota={"S1": 2, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0})
    for _ in range(2):
        session.next_question()
        session.submit_answer("A")
    assert [h["target_skill"] for h in session.history] == ["S1", "S1"]


def test_submit_before_next_question_raises():
    session = make_session(quota={s: 1 for s in SKILLS})
    with pytest.raises(RuntimeError):
        session.submit_answer("A")
