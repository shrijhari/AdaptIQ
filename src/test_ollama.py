
import ollama

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain overfitting in machine learning in three sentences."
        }
    ]
)

print(response["message"]["content"])

