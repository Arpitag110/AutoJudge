# AutoJudge — Programming Problem Difficulty Predictor

AutoJudge is a machine learning–based system that predicts the difficulty level (Easy / Medium / Hard) of a programming problem based solely on its problem statement text.

The project was built as part of an ACM open project to explore Natural Language Processing (NLP) and machine learning pipelines, while also understanding real-world challenges such as data quality, model evaluation, and deployment.

# Features

Predicts difficulty of unseen programming problems
Uses TF-IDF for text feature extraction
Trained using classical ML models (Naive Bayes, Logistic Regression)
Evaluated properly using Stratified Cross-Validation
Exposed via a Flask REST API
Simple and clean web-based UI

# Tech Stack

Language: Python
ML Library: scikit-learn
NLP: TF-IDF Vectorization
Backend: Flask
Frontend: HTML, CSS, JavaScript
Model Persistence: joblib

# Project Motivation

In competitive programming platforms, problem difficulty is usually assigned manually.
AutoJudge explores whether textual cues in a problem statement (keywords, algorithms, data structures) are sufficient to automatically predict difficulty.
Examples:
“sum of elements in an array” → Easy
“shortest path in graph using BFS” → Medium
“segment tree with lazy propagation” → Hard

# Dataset

Custom dataset created manually
~90 problem statements
Balanced across 3 classes:
Easy
Medium
Hard
Each entry contains:
problem_statement,difficulty

This manual construction helped in understanding:
Class overlap (especially Medium)
Vocabulary patterns
Realistic ML limitations

# ML Pipeline
1. Text Vectorization (TF-IDF)

We convert problem statements into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency).
Why TF-IDF?
Emphasizes important keywords like graph, dp, segment tree
Reduces impact of common words
Well-suited for classical text classification

2. Baseline Model — Multinomial Naive Bayes

We first trained a Multinomial Naive Bayes classifier.
Why Naive Bayes?
Fast
Works well with text data
Strong baseline for NLP tasks

Result:
Accuracy around 65–70%
Easy and Hard classes predicted well
Medium class showed overlap issues (expected)

3. Experiment — Logistic Regression

We experimented with Logistic Regression to handle overlapping classes better.
Observation:
Accuracy fluctuated and sometimes decreased
Medium class predictions were unstable on small data

Conclusion:
Increasing model complexity without enough data does not guarantee better performance.
This reinforced the importance of data-first ML, not algorithm-first ML.

4. Proper Evaluation — Cross-Validation

Instead of relying on a single train-test split, we applied:
Stratified 5-Fold Cross-Validation
Why?
Reduces randomness due to small dataset
Ensures all classes appear in each fold
Gives a more reliable estimate of performance

Result:
Average cross-validated accuracy ≈ 68%
Stable across folds

5. Final Model Choice

After evaluation:
Multinomial Naive Bayes + TF-IDF was selected as the final model
Trained on 100% of the dataset
Saved using joblib for reuse

This follows the standard ML workflow:
Evaluate with cross-validation → Train final model on all data → Deploy

# Deployment Architecture
User (Browser)
   ↓
HTML + JavaScript UI
   ↓
Flask API (/predict)
   ↓
TF-IDF Vectorizer
   ↓
ML Model
   ↓
Predicted Difficulty (JSON)

# Web Interface

Simple, clean UI
Text area for problem statement
Predict button
Displays difficulty instantly
The UI is served directly from the Flask backend at the root route (/).

# How to Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Start the server
python app.py

3. Open in browser
http://127.0.0.1:5000/