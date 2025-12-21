import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("../data/problems.csv")

# 2. Input (X) and Output (y)
X_text = df["problem_statement"]
y = df["difficulty"]

# 3. Convert text to TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 5. Train the model
model = LogisticRegression(
    max_iter=1000
)
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Predict difficulty for custom input
def predict_difficulty(problem_text):
    vector = vectorizer.transform([problem_text])
    prediction = model.predict(vector)
    return prediction[0]


# Test with custom problems
custom_problems = [
    "Find the sum of all elements in an array",
    "Find shortest path in graph using BFS",
    "Implement segment tree for range queries"
]

print("\nCustom Predictions:")
for problem in custom_problems:
    print(f"Problem: {problem}")
    print(f"Predicted Difficulty: {predict_difficulty(problem)}\n")
