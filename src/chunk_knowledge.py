
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "knowledge_base" / "S2_machine_learning_fundamentals.md"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Split the document using section headings
pattern = r"(?=## S2_C\d+)"
sections = re.split(pattern, text)

chunks = []

for section in sections:
    section = section.strip()

    if not section.startswith("## S2_C"):
        continue

    heading = section.splitlines()[0]

    chunk_id = re.search(r"S2_C\d+", heading).group()

    chunks.append({
        "chunk_id": chunk_id,
        "skill_id": "S2",
        "content": section
    })

# Display the chunks
print(f"Total chunks created: {len(chunks)}")

for chunk in chunks:
    print("\n" + "=" * 60)
    print("Chunk ID:", chunk["chunk_id"])
    print("Skill ID:", chunk["skill_id"])
    print("Content:\n", chunk["content"])
