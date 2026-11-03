
from pathlib import Path
import pickle
import json

import faiss
import ollama
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(str(DATA_DIR / "s2_index.faiss"))

# Load chunks
with open(DATA_DIR / "s2_chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


# MCQ output structure
MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "correct_option": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"]
                    },
                    "explanation": {"type": "string"},
                    "difficulty": {
                        "type": "string",
                        "enum": ["Easy", "Medium", "Hard"]
                    },
                    "skill_id": {"type": "string"},
                    "source_chunk_id": {"type": "string"}
                },
                "required": [
                    "question",
                    "options",
                    "correct_option",
                    "explanation",
                    "difficulty",
                    "skill_id",
                    "source_chunk_id"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["questions"],
    "additionalProperties": False
}


def retrieve_knowledge(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    k = min(top_k, index.ntotal)

    distances, indices = index.search(query_embedding, k)

    retrieved = []

    for idx in indices[0]:
        if idx != -1:
            retrieved.append(chunks[idx])

    return retrieved


def generate_mcqs(topic, number_of_questions=3):

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_knowledge(topic)

    if not retrieved_chunks:
        raise ValueError("No relevant knowledge was retrieved.")

    context = "\n\n".join(
        f"CHUNK ID: {chunk['chunk_id']}\n"
        f"{chunk['content']}"
        for chunk in retrieved_chunks
    )

    valid_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]

    # Step 2: Generate MCQs using local Ollama model
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": """
                You are an educational MCQ question generator.

                Generate questions ONLY from the supplied source material.

                Do not introduce facts that are not supported by the source.

                Each question must:
                - Have exactly four options.
                - Have exactly one correct answer.
                - Include a clear explanation.
                - Have difficulty Easy, Medium, or Hard.
                - Use skill_id S2.
                - Include the source chunk ID supporting the question.

                Use A, B, C, D as option labels.
                Generate only the requested number of questions.
                """
            },
            {
                "role": "user",
                "content": f"""
                Topic: {topic}

                Generate {number_of_questions} MCQs from this material:

                {context}

                Valid source chunk IDs: {valid_chunk_ids}
                """
            }
        ],
        format=MCQ_SCHEMA,
        options={"temperature": 0.2}
    )

    # Step 3: Parse JSON response
    result = json.loads(response["message"]["content"])

    questions = result["questions"]

    # Step 4: Basic validation
    if len(questions) != number_of_questions:
        raise ValueError(
            f"Expected {number_of_questions} questions, "
            f"but received {len(questions)}."
        )

    for q in questions:

        if len(q["options"]) != 4:
            raise ValueError("Question must have exactly four options.")

        if q["skill_id"] != "S2":
            raise ValueError("Incorrect skill tag.")

        if q["source_chunk_id"] not in valid_chunk_ids:
            raise ValueError("Invalid source chunk ID.")

    return questions


# Step 5: Test the complete RAG + Ollama pipeline
if __name__ == "__main__":

    topic = "Overfitting in machine learning"

    questions = generate_mcqs(
        topic,
        number_of_questions=3
    )

    for i, q in enumerate(questions, 1):

        print("\n" + "=" * 60)
        print(f"QUESTION {i}")
        print(q["question"])

        for j, option in enumerate(q["options"]):
            print(f"{chr(65+j)}. {option}")

        print("Correct answer:", q["correct_option"])
        print("Explanation:", q["explanation"])
        print("Difficulty:", q["difficulty"])
        print("Skill:", q["skill_id"])
        print("Source:", q["source_chunk_id"])

    # Save generated questions
    output_file = DATA_DIR / "generated_s2_questions.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(questions, file, indent=4, ensure_ascii=False)

    print("\nQuestions saved to:", output_file)
