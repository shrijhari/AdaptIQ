
from pathlib import Path
import pickle
import json
import re
import os

import faiss
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Load API key
load_dotenv(BASE_DIR / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embedding model and FAISS index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(DATA_DIR / "s2_index.faiss"))

with open(DATA_DIR / "s2_chunks.pkl", "rb") as file:
    chunks = pickle.load(file)


def retrieve_knowledge(query, top_k=2):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    retrieved = []

    for idx in indices[0]:
        if idx == -1:
            continue
        retrieved.append(chunks[idx])

    return retrieved


def generate_mcqs(topic, number_of_questions=3):

    # Step 1: Retrieve relevant knowledge
    retrieved_chunks = retrieve_knowledge(topic)

    context = "\n\n".join(
        f"CHUNK ID: {chunk['chunk_id']}\n{chunk['content']}"
        for chunk in retrieved_chunks
    )

    valid_chunk_ids = [c["chunk_id"] for c in retrieved_chunks]

    # Step 2: Ask the LLM to generate questions
    response = client.responses.create(
        model="gpt-4o-mini",

        instructions="""
        You are an educational MCQ question generator.

        Generate questions ONLY from the supplied source material.

        Do not introduce facts that are not supported by the source.

        Each question must:
        - Have exactly four options.
        - Have exactly one correct answer.
        - Include a clear explanation.
        - Have a difficulty of Easy, Medium, or Hard.
        - Use skill_id S2.
        - Include the source chunk ID supporting the question.

        If the source does not contain enough information,
        do not invent questions.
        """,

        input=f"""
        Topic: {topic}

        Generate {number_of_questions} MCQs using this material:

        {context}

        Valid source chunk IDs: {valid_chunk_ids}
        """,

        text={
            "format": {
                "type": "json_schema",
                "name": "mcq_set",
                "strict": True,
                "schema": {
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
            }
        }
    )

    # Step 3: Parse generated JSON
    result = json.loads(response.output_text)

    # Step 4: Basic validation
    for q in result["questions"]:
        if len(q["options"]) != 4:
            raise ValueError("A question does not have exactly four options.")

        if q["skill_id"] != "S2":
            raise ValueError("Incorrect skill tag generated.")

        if q["source_chunk_id"] not in valid_chunk_ids:
            raise ValueError("Invalid source chunk ID generated.")

    return result["questions"]


# Test the complete RAG + LLM pipeline
if __name__ == "__main__":

    topic = "Overfitting in machine learning"

    questions = generate_mcqs(topic, number_of_questions=3)

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
