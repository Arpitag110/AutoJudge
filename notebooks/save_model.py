import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("../data/problems.csv")

X_text = df["problem_statement"]
y = df["difficulty"]

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Model
model = MultinomialNB()
model.fit(X, y)

# Save vectorizer and model
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(model, "../models/auto_judge_model.pkl")

print("Model and vectorizer saved successfully!")
