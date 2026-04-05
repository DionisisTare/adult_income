# Adult Income Prediction — Machine Learning Project

This project builds a machine learning pipeline to predict whether an individual's annual income exceeds $50K using the Adult Census Income dataset.

Full machine learning pipeline on the UCI Adult Census dataset (48,842 records): EDA, preprocessing, feature engineering, model training, and evaluation — with Random Forest achieving AUC 0.912.


 Key Results at a Glance
<img width="397" height="205" alt="image" src="https://github.com/user-attachments/assets/b87e21bb-56d1-4dfc-a907-4ff98f2d14a1" />


Baseline (Majority Classifier): Accuracy = 0.759 — both models substantially outperform the naive baseline.


 Problem Statement
Binary classification: predict whether an individual's annual income exceeds $50K based on demographic and employment features from the 1994 US Census.
The dataset has a class imbalance (~76% ≤$50K vs ~24% >$50K), making raw accuracy a misleading metric — F1 and AUC are the relevant measures of performance.

 Pipeline Overview
Raw Data (48,842 rows × 15 cols)
    ↓
EDA → Missing value handling ('?' → NaN → drop, 2,399 rows removed)
    ↓
Feature encoding (One-Hot, 88 dummy columns created)
    ↓
Target binarization (income → 0/1)
    ↓
Train/Test Split (80/20, stratified)
    ↓
Model Training → Logistic Regression | Random Forest (Grid Search)
    ↓
Evaluation → Confusion Matrix · ROC Curve · Feature Importance
Key preprocessing steps

Replaced ? with NaN → dropped 2,399 rows (5.0% of data)
Dropped education (redundant with education_num) and fnlwgt (sampling weight, not predictive)
One-Hot encoded 6 categorical features → 88 total features after encoding
Stratified train/test split to preserve class ratio

Model training

Logistic Regression — StandardScaler + L2, max_iter=1000, random_state=0
Random Forest — Grid Search over n_estimators [100, 200], max_depth [10, 20, None], min_samples_split [2, 5] → best: max_depth=None, min_samples_split=2, n_estimators=200


 Model Evaluation
Confusion Matrices
Logistic Regression          Random Forest
  Predicted                    Predicted
  ≤50K  >50K                   ≤50K  >50K
┌──────┬──────┐             ┌──────┬──────┐
│ 3622 │  369 │  ≤50K True  │ 3046 │  542 │
│  266 │ 1250 │  >50K True  │  361 │ 1159 │  ← better recall
└──────┴──────┘             └──────┴──────┘
ROC Curves

LR: AUC = 0.902
RF: AUC = 0.912


 Top 15 Feature Importances (Random Forest)
RankFeatureImportance1age0.1642marital_status_Married-civ-spouse0.1293education_num0.1194capital_gain0.1015hours_per_week0.0846marital_status_Never-married0.0687sex_Male0.043

What I Learned

Class imbalance changes everything: accuracy of 0.850 sounds impressive until you see the baseline is 0.759. AUC and F1 are the honest metrics for imbalanced classification.
Random Forest improved Recall significantly (0.622 → 0.764) — meaning it catches far more >$50K earners, at a small cost to precision. Depending on the use case (e.g. targeted marketing), this tradeoff is often worth it.
Grid Search on Random Forest confirmed that deeper trees (max_depth=None) with more estimators (200) gave the best generalization — but with diminishing returns beyond a point.
Age dominates all other features (importance = 0.164), followed by marital status and education — a finding that aligns with labor economics literature and validates the model's real-world interpretability.
Preprocessing decisions (dropping vs imputing missing values, handling redundant features like education vs education_num) had measurable impact on final performance.

 Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook.

The workflow includes exploratory data analysis (EDA), data preprocessing, feature encoding, model training, and model evaluation.

Two models were implemented and compared:
Logistic Regression and Random Forest.

Model performance was evaluated using common classification metrics including Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

Technologies used in this project:
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Jupyter Notebook.
