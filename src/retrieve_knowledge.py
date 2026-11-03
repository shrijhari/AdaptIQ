
from pathlib import Path
import pickle

import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(str(DATA_DIR / "s2_index.faiss"))

# Load corresponding chunks
with open(DATA_DIR / "s2_chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


def retrieve(query, top_k=3):

    # Convert query into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Search for nearest chunks
    distances, indices = index.search(query_embedding, top_k)

    print("\nUser query:", query)
    print("\nRetrieved knowledge:\n")

    for rank, idx in enumerate(indices[0]):
        chunk = chunks[idx]

        print("=" * 60)
        print("Rank:", rank + 1)
        print("Chunk ID:", chunk["chunk_id"])
        print("Distance:", distances[0][rank])
        print("Content:\n", chunk["content"])


# Test retrieval
retrieve("What happens when a machine learning model overfits?")

