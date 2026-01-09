# AutoJudge
Predicting Programming Problem Difficulty using Machine Learning

# Overview
AutoJudge is a machine learning–based system that predicts the difficulty of programming problems from their textual descriptions. Given a problem statement, the system outputs:

A difficulty category: Easy / Medium / Hard

A numerical difficulty score on a continuous scale

The goal of this project is to explore whether problem difficulty can be inferred purely from natural language descriptions, without access to constraints, test cases, or editorial information.

# Features
Text-based difficulty prediction

Dual-model approach:

Classification (Easy / Medium / Hard)

Regression (continuous difficulty score)

Cross-validated evaluation

Web interface using Flask

Hybrid ML + rule-based refinement for better reliability

# Motivation
Difficulty labels on competitive programming platforms are often:

Subjective

Platform-dependent

Inconsistent across users

This project investigates whether patterns in problem text (keywords, structure, terminology) can be used to estimate difficulty in an automated way.

# Dataset
Source: Competitive programming problem descriptions

Size: 4112 problems

Fields used:

Title

Description

Input description

Output description

Labels:

easy, medium, hard

Numerical difficulty score (1–10 scale)

Class Distribution

Hard: ~47%

Medium: ~34%

Easy: ~19%

# Methodology
1. Text Preprocessing

Combined all textual fields into a single input

Used TF-IDF vectorization

Stopword removal

Unigrams + bigrams (ngram_range=(1,2))

Vocabulary size capped for stability

2. Initial Classification Attempts

Model 1: Multinomial Naive Bayes

Simple baseline model

Resulted in low accuracy due to overlapping vocabulary between difficulty levels

Model 2: Linear SVM

Better suited for high-dimensional sparse text

Used class balancing to address dataset imbalance

Cross-validated accuracy (5-fold): ~47–50%

This performance is consistent with expectations for text-only difficulty classification, where semantic overlap is high.

3. Regression Model (Numerical Difficulty)

Instead of forcing hard class boundaries, a regression model was trained to predict a continuous difficulty score.

Model: Linear regression (Ridge)

Evaluation: Cross-validated RMSE

Average RMSE: ~2.06

This means predictions are, on average, within ~2 points of the true difficulty score — a reasonable result given the subjective nature of difficulty.

4. Score-to-Class Mapping

The regression output is mapped to difficulty categories using interpretable thresholds:
Score Range	Difficulty

< 3.5	Easy

3.5 – 6.5	Medium

≥ 6.5	Hard

The regression-based class is used as the final decision, as it is smoother and more informative than direct classification.

5. Hybrid Rule-Based Refinement

During testing, some obviously simple problems (e.g., “Find sum of elements in an array”) were occasionally predicted as Medium.
This occurs because:
The model only sees text
It does not know constraints or intended trickiness
Words like “array” and “sum” appear across difficulties
To address this, a light rule-based override was added for universally trivial patterns (e.g., sum, max, reverse array).
This creates a hybrid ML + rules system, commonly used in real-world NLP applications.

# Web Application
Backend: Flask

Endpoint: /predict

Input: 

JSON with problem statement

Output:

Final difficulty

Predicted score

A clean, minimal UI allows users to interactively test problem statements.

# Limitations (Important)
Text-Only Input

The model does not see constraints, time limits, or required optimizations

Many problems become hard only due to constraints

Subjective Labels

Difficulty labels vary across platforms and users

No universally correct “ground truth”

Vocabulary Overlap

Terms like array, graph, sum appear in problems of all difficulties

Leads to ambiguity in classification

Not a Judge Replacement

AutoJudge estimates difficulty, it does not evaluate correctness or solutions

These limitations are inherent to the problem, not implementation flaws.

# Key Takeaways
Difficulty prediction from text is possible but imperfect

Regression provides more meaningful insight than pure classification

Hybrid systems outperform purely ML-based ones in edge cases

Honest evaluation and limitation analysis are crucial in applied ML

# Technologies Used
Python

Pandas

Scikit-learn

TF-IDF Vectorization

Flask

HTML / CSS / JavaScript

# Conclusion
AutoJudge demonstrates that while programming problem difficulty cannot be perfectly inferred from text alone, statistical patterns do exist and can be leveraged effectively using machine learning. The project highlights both the potential and limitations of automated difficulty estimation and provides a realistic, end-to-end applied ML system.

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

# Report PDF
Here is the Google drive link for the project report-
https://drive.google.com/file/d/1VAlCp7iZnmBQHYOJ45MSU2coOWzZ1e30/view?usp=drivesdk

# Demo Video
Here is the Google drive link for demo video-
https://drive.google.com/file/d/1YM7322yj1v0P1yKhIOYeplRwL8rsqmX5/view?usp=drivesdk