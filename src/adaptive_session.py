"""Integration layer for Person 3.

Connects the adaptive selector to Person 2's BKT model. The session can
select a question, score a submitted answer, update every skill mapped to
that question, and then select the next question.
"""

from __future__ import annotations

from bkt import BKT
from adaptive_engine import AdaptiveQuestionSelector


class AdaptiveSession:
    def __init__(
        self,
        bkt: BKT,
        selector: AdaptiveQuestionSelector,
        asked_question_ids: set[str] | None = None,
        allow_repeat: bool = False,
    ):
        self.bkt = bkt
        self.selector = selector
        self.asked_question_ids = set(asked_question_ids or set())
        self.allow_repeat = allow_repeat
        self.current_question = None

    def next_question(self) -> dict:
        self.current_question = self.selector.select_next_question(
            self.bkt.mastery,
            asked_question_ids=self.asked_question_ids,
            allow_repeat=self.allow_repeat,
        )
        return self.current_question

    def submit_answer(self, answer: str) -> dict:
        if self.current_question is None:
            raise RuntimeError("Call next_question() before submit_answer().")

        answer = answer.upper().strip()
        question_id = self.current_question["question_id"]

        # questions.csv stores the correct answer in this field.
        # The selector deliberately did not expose it to the UI.
        correct_answer = self.current_question["correct_answer"]
        is_correct = int(answer == correct_answer)

        updates = []
        for skill in self.current_question["skills"]:
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
        self.current_question = None

        return {
            "question_id": question_id,
            "answer": answer,
            "correct": is_correct,
            "updates": updates,
        }
