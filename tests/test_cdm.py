import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdm import dina, mastery_indicator


def test_mastery_indicator_requires_all_skills():
    assert mastery_indicator([1, 1], [1, 1]) == 1
    assert mastery_indicator([1, 0], [1, 1]) == 0
    assert mastery_indicator([1, 0], [1, 0]) == 1


def test_dina_returns_valid_probabilities():
    q = [[1,0], [0,1], [1,1]]
    responses = [1, 0, 0]
    result = dina(responses, q, slip=0.05, guess=0.20)

    assert len(result["profiles"]) == 4
    assert abs(sum(result["posterior"]) - 1.0) < 1e-9
    assert all(0 <= p <= 1 for p in result["mastery_probabilities"])


def test_strong_evidence_for_first_skill():
    q = [[1,0], [1,0], [1,0], [0,1]]
    responses = [1,1,1,0]
    result = dina(responses, q, slip=0.05, guess=0.20)

    assert result["mastery_probabilities"][0] > result["mastery_probabilities"][1]

def test_perfect_student_has_high_mastery():
    q = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ]

    responses = [1, 1, 1, 1, 1]

    result = dina(
        responses,
        q,
        slip=0.05,
        guess=0.20
    )

    assert all(
        p > 0.90
        for p in result["mastery_probabilities"]
    )


def test_completely_wrong_student_has_low_mastery():
    q = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ]

    responses = [0, 0, 0, 0, 0]

    result = dina(
        responses,
        q,
        slip=0.05,
        guess=0.20
    )

    assert all(
        p < 0.20
        for p in result["mastery_probabilities"]
    )


def test_s1_specialist_is_highest():
    q = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    responses = [1, 1, 1, 0, 0]

    result = dina(
        responses,
        q,
        slip=0.05,
        guess=0.20
    )

    p = result["mastery_probabilities"]

    assert p[0] > p[1]
    assert p[0] > p[2]
