import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("../data/problems.csv")

# Input text (problem statements)
texts = df["problem_statement"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform text
X = vectorizer.fit_transform(texts)

print("TF-IDF matrix shape:")
print(X.shape)

print("\nVocabulary size:")
print(len(vectorizer.vocabulary_))

print("\nVocabulary words:")
print(vectorizer.get_feature_names_out())

