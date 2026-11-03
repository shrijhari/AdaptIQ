
from pathlib import Path
import re
import pickle

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "knowledge_base" / "S2_machine_learning_fundamentals.md"

# Read the Markdown file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Split into six sections
sections = re.split(r"(?=## S2_C\d+)", text)

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

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text chunks into vectors
texts = [chunk["content"] for chunk in chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

print("Embedding shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

index.add(embeddings.astype("float32"))

# Save FAISS index
output_dir = BASE_DIR / "data"
output_dir.mkdir(exist_ok=True)

faiss.write_index(index, str(output_dir / "s2_index.faiss"))

# Save chunk information
with open(output_dir / "s2_chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("FAISS index created successfully!")
print("Total indexed chunks:", index.ntotal)
