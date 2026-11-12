"""Adaptive question selection for AdaptIQ (Person 3).

Consumes the mastery state produced by Person 2's BKT and selects the next
question from data/questions.csv using the project's Q-matrix.
"""

from __future__ import annotations

import csv
from typing import Iterable


class NoQuestionAvailable(ValueError):
    """Raised when no suitable unseen question can be found.

    It subclasses ValueError, so older code that catches ValueError still works.
    """


class AdaptiveQuestionSelector:
    def __init__(self, questions_path: str, q_matrix_path: str):
        self.questions = self._load_questions(questions_path)
        self.q_matrix = self._load_q_matrix(q_matrix_path)

        # Attach Q-matrix skills to every question.
        for question in self.questions:
            qid = question["question_id"]
            if qid not in self.q_matrix:
                raise ValueError(f"Question {qid} is missing from the Q-matrix.")
            question["skills_list"] = self.q_matrix[qid]

    @staticmethod
    def _load_questions(path: str) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _load_q_matrix(path: str) -> dict[str, list[str]]:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        skill_ids = list(rows[0].keys())[1:]
        q_matrix = {}

        for row in rows:
            q_matrix[row["question_id"]] = [
                skill for skill in skill_ids if int(row[skill]) == 1
            ]

        return q_matrix

    @staticmethod
    def _target_difficulty(mastery: float) -> str:
        """Choose a difficulty appropriate for the target skill."""
        if mastery < 0.40:
            return "Easy"
        if mastery < 0.70:
            return "Medium"
        return "Hard"

    @staticmethod
    def _difficulty_score(actual: str, target: str) -> float:
        if actual == target:
            return 1.0

        # Adjacent difficulty is still useful.
        order = {"Easy": 0, "Medium": 1, "Hard": 2}
        if abs(order[actual] - order[target]) == 1:
            return 0.5
        return 0.0

    @staticmethod
    def _predicted_correctness(
        required_skills: Iterable[str], mastery: dict[str, float]
    ) -> float:
        values = [mastery[s] for s in required_skills]
        return sum(values) / len(values)

    def _score_question(
        self,
        question: dict,
        target_skill: str,
        mastery: dict[str, float],
    ) -> float:
        skills = question["skills_list"]

        # 1. Skill relevance: directly tests the weakest target skill.
        skill_relevance = 1.0 if target_skill in skills else 0.0

        # 2. Weakness: prefer questions connected to skills that need work.
        weakness = sum(1.0 - mastery[s] for s in skills) / len(skills)

        # 3. Predicted correctness: avoid questions that are either trivial
        #    or overwhelmingly difficult. A target of 0.60 is used as a
        #    simple demo-level balance point.
        predicted = self._predicted_correctness(skills, mastery)
        correctness_fit = max(0.0, 1.0 - abs(predicted - 0.60) / 0.60)

        # 4. Difficulty should match the target skill's mastery level.
        target_difficulty = self._target_difficulty(mastery[target_skill])
        difficulty_fit = self._difficulty_score(
            question["difficulty"], target_difficulty
        )

        # Transparent weighted score for the prototype.
        return (
            0.40 * skill_relevance
            + 0.25 * weakness
            + 0.20 * correctness_fit
            + 0.15 * difficulty_fit
        )

    def find_weakest_skill(self, mastery: dict[str, float]) -> str:
        """Return the skill with the lowest current mastery probability."""
        return min(mastery, key=mastery.get)

    def select_next_question(
        self,
        mastery: dict[str, float],
        asked_question_ids: set[str] | None = None,
        allow_repeat: bool = False,
        remaining_quota: dict[str, int] | None = None,
    ) -> dict:
        """Select the next adaptive question.

        By default, previously answered questions are excluded. If the current
        question bank has no unseen question for the weakest skill, a clear
        error is raised rather than silently repeating a question.

        remaining_quota (optional): {"S1": 2, "S2": 0, ...}. When given, only
        skills with quota left are considered, the weakest of those is the
        target, and a skill with no unseen question left is skipped in favour
        of the next weakest one. When omitted, behaviour is unchanged.
        """
        missing = [s for s in mastery if not isinstance(mastery[s], (int, float))]
        if missing:
            raise ValueError(f"Invalid mastery values for: {missing}")

        asked_question_ids = asked_question_ids or set()

        if remaining_quota is None:
            # Old behaviour: only the single weakest skill is tried.
            target_skills = [self.find_weakest_skill(mastery)]
        else:
            # Weakest first, skipping skills whose quota is used up.
            with_quota = [s for s in mastery if remaining_quota.get(s, 0) > 0]
            if not with_quota:
                raise NoQuestionAvailable("All skill quotas are used up.")
            target_skills = sorted(with_quota, key=mastery.get)

        target_skill, candidates = None, []
        for skill in target_skills:
            candidates = [
                q for q in self.questions
                if skill in q["skills_list"]
                and (allow_repeat or q["question_id"] not in asked_question_ids)
            ]
            if candidates:
                target_skill = skill
                break

        if not candidates:
            if remaining_quota is not None:
                raise NoQuestionAvailable(
                    "No unseen questions remain for any skill with quota left."
                )
            weakest = target_skills[0]
            if allow_repeat:
                raise NoQuestionAvailable(
                    f"No questions in the bank are mapped to target skill {weakest}."
                )
            raise NoQuestionAvailable(
                f"No unseen questions remain for target skill {weakest}. "
                "Add/reserve adaptive questions or explicitly allow repeats."
            )

        scored = [
            (self._score_question(q, target_skill, mastery), q)
            for q in candidates
        ]

        # Deterministic tie-break: higher score, then lower question number.
        scored.sort(key=lambda item: (-item[0], item[1]["question_id"]))
        score, selected = scored[0]

        return {
            "question_id": selected["question_id"],
            "question": selected["question"],
            "option_a": selected["option_a"],
            "option_b": selected["option_b"],
            "option_c": selected["option_c"],
            "option_d": selected["option_d"],
            "correct_answer": selected["correct_answer"],
            "difficulty": selected["difficulty"],
            "skills": selected["skills_list"],
            "explanation": selected.get("explanation", ""),
            "target_skill": target_skill,
            "score": score,
        }
