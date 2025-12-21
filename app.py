from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, send_from_directory
import joblib

# Load saved model and vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/auto_judge_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "problem_statement" not in data:
        return jsonify({"error": "Problem statement missing"}), 400

    problem_text = data["problem_statement"]

    # Vectorize input
    vector = vectorizer.transform([problem_text])

    # Predict difficulty
    prediction = model.predict(vector)[0]

    return jsonify({
        "problem_statement": problem_text,
        "predicted_difficulty": prediction
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)