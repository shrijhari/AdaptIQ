"""Integration layer for Person 3.

Connects the adaptive selector to Person 2's BKT model. The session can
select a question, score a submitted answer, update every skill mapped to
that question, and then select the next question.

Optional quota mode: pass quota={"S1": 2, "S2": 3, ...} (for example from
allocation.allocate_questions). The session never asks more than that many
questions per skill, keeps a history of answers, and reports when it is
finished. Without a quota it behaves exactly as before.
"""

from __future__ import annotations

from bkt import BKT
from adaptive_engine import AdaptiveQuestionSelector, NoQuestionAvailable


class AdaptiveSession:
    def __init__(
        self,
        bkt: BKT,
        selector: AdaptiveQuestionSelector,
        asked_question_ids: set[str] | None = None,
        allow_repeat: bool = False,
        quota: dict[str, int] | None = None,
    ):
        self.bkt = bkt
        self.selector = selector
        self.asked_question_ids = set(asked_question_ids or set())
        self.allow_repeat = allow_repeat
        self.current_question = None
        self.quota = dict(quota) if quota is not None else None
        self.history: list[dict] = []
        self._exhausted = False  # quota mode: no unseen questions left

    def is_finished(self) -> bool:
        """True when every quota is used up, or no question is left to ask."""
        if self.quota is None:
            return False
        return self._exhausted or all(n <= 0 for n in self.quota.values())

    def next_question(self) -> dict | None:
        """Return the next question, or None once the session is finished."""
        if self.is_finished():
            return None
        try:
            self.current_question = self.selector.select_next_question(
                self.bkt.mastery,
                asked_question_ids=self.asked_question_ids,
                allow_repeat=self.allow_repeat,
                remaining_quota=self.quota,
            )
        except NoQuestionAvailable:
            if self.quota is None:
                raise  # old behaviour: a clear error
            self._exhausted = True
            self.current_question = None
            return None
        return self.current_question

    def submit_answer(self, answer: str) -> dict:
        if self.current_question is None:
            raise RuntimeError("Call next_question() before submit_answer().")

        answer = answer.upper().strip()
        question = self.current_question
        question_id = question["question_id"]
        target_skill = question.get("target_skill")

        correct_answer = question["correct_answer"]
        is_correct = int(answer == correct_answer)

        updates = []
        for skill in question["skills"]:
            before = self.bkt.mastery[skill]
            after = self.bkt.update_skill(skill, is_correct)
            updates.append(
                {
                    "skill": skill,
                    "mastery_before": before,
                    "mastery_after": after,
                }
            )

        self.asked_question_ids.add(question_id)

        # A question counts against its target skill's quota.
        if self.quota is not None and target_skill in self.quota:
            self.quota[target_skill] -= 1

        self.current_question = None

        result = {
            "question_id": question_id,
            "answer": answer,
            "correct": is_correct,
            "correct_answer": correct_answer,
            "explanation": question.get("explanation", ""),
            "target_skill": target_skill,
            "updates": updates,
            "finished": self.is_finished(),
        }
        self.history.append(result)
        return result
