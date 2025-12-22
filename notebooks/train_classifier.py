import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
file_path = "../data/problems_data.jsonl"
records = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Combine text fields
df["full_text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("")
)

X = df["full_text"]
y = df["problem_class"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )),
    ("clf", MultinomialNB())
])

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring="accuracy")

print("Cross-validation accuracies:", scores)
print("Average accuracy:", scores.mean())

# Train final model on full data
pipeline.fit(X, y_encoded)

# Save model & encoder
joblib.dump(pipeline, "../models/classifier_pipeline.pkl")
joblib.dump(label_encoder, "../models/classifier_labels.pkl")

print("\nClassification model saved.")
