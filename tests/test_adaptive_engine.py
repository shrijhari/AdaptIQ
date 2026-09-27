import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_engine import AdaptiveQuestionSelector


ROOT = Path(__file__).resolve().parents[1]


def make_selector():
    return AdaptiveQuestionSelector(
        str(ROOT / "data/questions.csv"),
        str(ROOT / "data/q_matrix.csv"),
    )


def test_weakest_skill_is_selected():
    selector = make_selector()
    mastery = {
        "S1": 1.0,
        "S2": 1.0,
        "S3": 0.20,
        "S4": 1.0,
        "S5": 0.99,
        "S6": 0.65,
    }
    question = selector.select_next_question(mastery)
    assert question["target_skill"] == "S3"
    assert "S3" in question["skills"]


def test_previous_questions_are_excluded():
    selector = make_selector()
    mastery = {
        "S1": 1.0,
        "S2": 1.0,
        "S3": 0.20,
        "S4": 1.0,
        "S5": 0.99,
        "S6": 0.65,
    }
    question = selector.select_next_question(mastery, asked_question_ids={"Q11"})
    assert question["question_id"] != "Q11"
