import csv

from cdm import load_csv_data, dina
from bkt import BKT


# 1. Load the student's data
skill_ids, question_ids, q_matrix, responses = load_csv_data(
    "data/q_matrix.csv",
    "data/student_responses.csv",
    "S001"
)


# 2. Run DINA to get initial mastery probabilities
result = dina(
    responses,
    q_matrix
)


# 3. Convert DINA probabilities into BKT initial mastery
initial_mastery = {
    skill: probability
    for skill, probability
    in zip(skill_ids, result["mastery_probabilities"])
}


# 4. Create the BKT model
bkt = BKT(initial_mastery)


# 5. Print DINA's initial mastery
print("Initial mastery from DINA:")

for skill, probability in initial_mastery.items():
    print(f"{skill}: {probability:.4f}")


# 6. Run BKT and save the history
print("\nBKT updates:")

history = []


for question_id, response in zip(question_ids, responses):

    print(f"\n{question_id} - Response: {response}")

    # Find which skills this question requires
    question_index = question_ids.index(question_id)
    q_row = q_matrix[question_index]

    skills_for_question = [
        skill_ids[i]
        for i, value in enumerate(q_row)
        if value == 1
    ]


    # Update each required skill
    for skill in skills_for_question:

        old_mastery = bkt.mastery[skill]

        new_mastery = bkt.update_skill(
            skill,
            response
        )

        print(
            f"{skill}: "
            f"{old_mastery:.4f} -> {new_mastery:.4f}"
        )


        # Save this update
        history.append({
            "question_id": question_id,
            "skill": skill,
            "correct": response,
            "mastery_before": old_mastery,
            "mastery_after": new_mastery
        })


# 7. Save BKT history to CSV
with open(
    "data/bkt_history.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "question_id",
            "skill",
            "correct",
            "mastery_before",
            "mastery_after"
        ]
    )

    writer.writeheader()
    writer.writerows(history)


print("\nBKT history saved to data/bkt_history.csv")