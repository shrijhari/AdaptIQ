"""
DINA-style Cognitive Diagnostic Model for AdaptIQ.

Inputs:
  - Q-matrix: questions x skills, where 1 means a question requires a skill.
  - Binary student responses: 1 correct, 0 incorrect.

Outputs:
  - Posterior probability for each of the 2^K mastery profiles.
  - Marginal mastery probability for each skill.
  - MAP (most probable) binary mastery profile.

This is a compact baseline implementation suitable for the project demo.
"""

from __future__ import annotations
from itertools import product
from math import log, exp
from typing import Sequence


def mastery_indicator(alpha: Sequence[int], q_row: Sequence[int]) -> int:
    """Return 1 when the learner has every skill required by the item."""
    for a, q in zip(alpha, q_row):
        if q == 1 and a == 0:
            return 0
    return 1


def dina_log_likelihood(
    responses: Sequence[int],
    q_matrix: Sequence[Sequence[int]],
    alpha: Sequence[int],
    slip: float = 0.05,
    guess: float = 0.20,
) -> float:
    """Log-likelihood P(responses | alpha) under the DINA model."""
    if not (0 < slip < 1 and 0 < guess < 1):
        raise ValueError("slip and guess must be strictly between 0 and 1.")
    if len(responses) != len(q_matrix):
        raise ValueError("Number of responses must equal number of Q-matrix rows.")

    ll = 0.0
    for x, q_row in zip(responses, q_matrix):
        eta = mastery_indicator(alpha, q_row)
        p_correct = (1.0 - slip) if eta else guess
        p = p_correct if int(x) == 1 else (1.0 - p_correct)
        ll += log(max(p, 1e-15))
    return ll


def dina(
    responses: Sequence[int],
    q_matrix: Sequence[Sequence[int]],
    slip: float = 0.05,
    guess: float = 0.20,
    prior: Sequence[float] | None = None,
):
    """
    Enumerate all mastery profiles and return posterior results.

    A uniform prior over profiles is used by default.
    """
    if not q_matrix:
        raise ValueError("Q-matrix cannot be empty.")
    k = len(q_matrix[0])
    if any(len(row) != k for row in q_matrix):
        raise ValueError("All Q-matrix rows must have the same number of skills.")
    if len(responses) != len(q_matrix):
        raise ValueError("Number of responses must equal number of questions.")
    if any(int(x) not in (0, 1) for x in responses):
        raise ValueError("Responses must be binary: 0 or 1.")

    profiles = list(product([0, 1], repeat=k))
    if prior is None:
        prior = [1.0 / len(profiles)] * len(profiles)
    if len(prior) != len(profiles) or any(p <= 0 for p in prior):
        raise ValueError("Prior must contain a positive probability for every profile.")

    log_weights = [
        dina_log_likelihood(responses, q_matrix, alpha, slip, guess) + log(prior[i])
        for i, alpha in enumerate(profiles)
    ]

    # Stable normalization (log-sum-exp)
    m = max(log_weights)
    weights = [exp(v - m) for v in log_weights]
    total = sum(weights)
    posterior = [w / total for w in weights]

    best_idx = max(range(len(profiles)), key=lambda i: posterior[i])
    map_profile = profiles[best_idx]

    # Marginal P(skill_j is mastered | responses)
    mastery_probabilities = [
        sum(p for profile, p in zip(profiles, posterior) if profile[j] == 1)
        for j in range(k)
    ]

    return {
        "profiles": profiles,
        "posterior": posterior,
        "map_profile": map_profile,
        "mastery_probabilities": mastery_probabilities,
    }


def load_csv_data(q_matrix_path: str, responses_path: str, student_id: str):
    """Load one student's Q-matrix and binary responses from CSV files."""
    import csv

    with open(q_matrix_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    skill_ids = list(rows[0].keys())[1:]
    q_matrix = [[int(row[s]) for s in skill_ids] for row in rows]
    question_ids = [row["question_id"] for row in rows]

    with open(responses_path, newline="", encoding="utf-8") as f:
        response_rows = [
            row for row in csv.DictReader(f) if row["student_id"] == student_id
        ]

    by_qid = {row["question_id"]: int(row["correct"]) for row in response_rows}
    missing = [qid for qid in question_ids if qid not in by_qid]
    if missing:
        raise ValueError(f"Missing responses for: {missing}")

    responses = [by_qid[qid] for qid in question_ids]
    return skill_ids, question_ids, q_matrix, responses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--q-matrix", default="data/q_matrix.csv")
    parser.add_argument("--responses", default="data/student_responses.csv")
    parser.add_argument("--student", default="S001")
    parser.add_argument("--slip", type=float, default=0.05)
    parser.add_argument("--guess", type=float, default=0.20)
    args = parser.parse_args()

    skill_ids, _, q_matrix, responses = load_csv_data(
        args.q_matrix, args.responses, args.student
    )
    result = dina(responses, q_matrix, args.slip, args.guess)

    print(f"Student: {args.student}")
    print("\nMAP mastery profile:")
    for sid, value in zip(skill_ids, result["map_profile"]):
        print(f"  {sid}: {'Mastered' if value else 'Not mastered'}")

    print("\nMarginal mastery probabilities:")
    for sid, p in zip(skill_ids, result["mastery_probabilities"]):
        print(f"  {sid}: {p:.3f}")
