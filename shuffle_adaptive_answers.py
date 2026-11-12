
import csv
import random
import shutil
from pathlib import Path
from collections import Counter

CSV_PATH = Path("data/adaptive_questions.csv")
BACKUP_PATH = Path("data/adaptive_questions.csv.bak")

LETTERS = ["A", "B", "C", "D"]
OPTION_FIELDS = [f"option_{x.lower()}" for x in LETTERS]

if not CSV_PATH.exists():
    raise FileNotFoundError(f"File not found: {CSV_PATH}")

if BACKUP_PATH.exists():
    raise FileExistsError(
        "Backup already exists. Do not shuffle again. "
        "Restore the original CSV backup first if needed."
    )

with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

required = set(OPTION_FIELDS + ["correct_answer"])

if not fieldnames or not required.issubset(fieldnames):
    raise ValueError(
        "Expected option_a, option_b, option_c, option_d "
        "and correct_answer columns. Check the CSV headers."
    )

# Preserve the original question bank.
shutil.copy2(CSV_PATH, BACKUP_PATH)

rng = random.Random(20261113)
counts = Counter()

for row in rows:
    old_correct = row["correct_answer"].strip().upper()

    if old_correct not in LETTERS:
        raise ValueError(f"Invalid answer: {old_correct}")

    options = [
        (letter, row[f"option_{letter.lower()}"])
        for letter in LETTERS
    ]

    rng.shuffle(options)

    new_correct = None

    for new_letter, (old_letter, option_text) in zip(LETTERS, options):
        row[f"option_{new_letter.lower()}"] = option_text

        if old_letter == old_correct:
            new_correct = new_letter

    row["correct_answer"] = new_correct
    counts[new_correct] += 1

temp_path = CSV_PATH.with_suffix(".tmp")

with open(temp_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

temp_path.replace(CSV_PATH)

print(f"Successfully shuffled {len(rows)} questions!")
print("Correct-answer distribution:")

for letter in LETTERS:
    print(f"{letter}: {counts[letter]}")

print(f"Original backup saved at: {BACKUP_PATH}")
