import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
import numpy as np
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
y = df["problem_score"]

# Build pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=7000
    )),
    ("reg", Ridge(alpha=1.0))
])

# Cross-validation (RMSE)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring="neg_mean_squared_error"
)

rmse_scores = np.sqrt(-neg_mse_scores)

print("RMSE scores:", rmse_scores)
print("Average RMSE:", rmse_scores.mean())

# Train final model
pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "../models/regressor_pipeline.pkl")

print("\nRegression model saved.")
