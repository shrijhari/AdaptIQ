"""End-to-end Person 3 demonstration.

Person 1: DINA -> initial mastery
Person 2: BKT -> updated mastery
Person 3: adaptive selector -> next question -> answer -> BKT update
"""

from cdm import load_csv_data, dina
from bkt import BKT
from adaptive_engine import AdaptiveQuestionSelector
from adaptive_session import AdaptiveSession


STUDENT_ID = "S001"
Q_MATRIX = "data/q_matrix.csv"
RESPONSES = "data/student_responses.csv"
QUESTIONS = "data/questions.csv"


# 1. Run Person 1's DINA model.
skill_ids, question_ids, q_matrix, responses = load_csv_data(
    Q_MATRIX, RESPONSES, STUDENT_ID
)
result = dina(responses, q_matrix)
initial_mastery = dict(zip(skill_ids, result["mastery_probabilities"]))


# 2. Run Person 2's BKT over the observed response sequence.
bkt = BKT(initial_mastery)
for question_id, response in zip(question_ids, responses):
    index = question_ids.index(question_id)
    for i, value in enumerate(q_matrix[index]):
        if value == 1:
            bkt.update_skill(skill_ids[i], response)


# 3. Create Person 3's adaptive session.
selector = AdaptiveQuestionSelector(QUESTIONS, Q_MATRIX)

# IMPORTANT: all 30 current questions are already in the diagnostic history.
# allow_repeat=True is therefore used ONLY for this repository's demo.
session = AdaptiveSession(
    bkt=bkt,
    selector=selector,
    asked_question_ids=set(question_ids),
    allow_repeat=True,
)

print("Updated mastery:")
for skill, value in bkt.mastery.items():
    print(f"  {skill}: {value:.4f}")

question = session.next_question()
print("\nSelected question:")
print(question["question_id"], "->", question["question"])
print("Target skill:", question["target_skill"])
print("Difficulty:", question["difficulty"])

# Simulate an answer only to demonstrate the feedback loop.
# Replace this with the student's real UI selection in the final app.
result = session.submit_answer("A")
print("\nSimulated response:")
print("Correct:", bool(result["correct"]))
for update in result["updates"]:
    print(
        f"{update['skill']}: "
        f"{update['mastery_before']:.4f} -> "
        f"{update['mastery_after']:.4f}"
    )
