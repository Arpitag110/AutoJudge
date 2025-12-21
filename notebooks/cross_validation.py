import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Load dataset
df = pd.read_csv("../data/problems.csv")

X_text = df["problem_statement"]
y = df["difficulty"]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Model
model = MultinomialNB()

# Stratified K-Fold (preserves class balance)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation accuracy
scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

print("Cross-validation accuracies:", scores)
print("Average accuracy:", scores.mean())
