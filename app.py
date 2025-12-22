from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, send_from_directory
import joblib

# Load saved model and vectorizer
classifier = joblib.load("models/classifier_pipeline.pkl")
label_encoder = joblib.load("models/classifier_labels.pkl")
regressor = joblib.load("models/regressor_pipeline.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return send_from_directory("static", "index.html")

def rule_based_easy_override(text):
    easy_phrases = [
        "sum of elements",
        "find sum",
        "reverse an array",
        "maximum element",
        "minimum element",
        "check if prime",
        "count digits"
    ]
    text = text.lower()
    return any(phrase in text for phrase in easy_phrases)


def score_to_class(score):
    if score < 3.5:
        return "easy"
    elif score < 6.5:
        return "medium"
    else:
        return "hard"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("problem_statement", "")

    if not text.strip():
        return jsonify({"error": "Empty input"}), 400

    # Classification model
    class_pred_encoded = classifier.predict([text])[0]
    class_pred = label_encoder.inverse_transform([class_pred_encoded])[0]

    # Regression model
    score_pred = float(regressor.predict([text])[0])
    score_based_class = score_to_class(score_pred)

    # Final decision strategy:
    # Prefer score-based class (more fine-grained)
    if rule_based_easy_override(text):
       final_class = "easy"
    else:
       final_class = score_based_class


    return jsonify({
        "classifier_prediction": class_pred,
        "predicted_score": round(score_pred, 2),
        "final_difficulty": final_class
    })



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)