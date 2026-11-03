
from pathlib import Path

# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Locate the Markdown file
file_path = BASE_DIR / "knowledge_base" / "S2_machine_learning_fundamentals.md"

# Read the file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("File loaded successfully!")
print("\nExtracted content:\n")
print(text)
