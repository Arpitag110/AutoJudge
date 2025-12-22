import json
import pandas as pd

# Load JSONL dataset
file_path = "../data/problems_data.jsonl"

records = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Combine all text fields into one input
df["full_text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)

# Rename targets clearly
df["difficulty_class"] = df["problem_class"]
df["difficulty_score"] = df["problem_score"]

# Keep only what we need
df = df[["full_text", "difficulty_class", "difficulty_score"]]

# Basic inspection
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["difficulty_class"].value_counts())

print("\nScore statistics:")
print(df["difficulty_score"].describe())
