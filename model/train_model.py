"""
Trains a recovery-probability classifier on the synthetic failed-payments
dataset and saves the model + encoders for the FastAPI backend to load.

Run from project root: python model/train_model.py
Output: model/recovery_model.pkl, model/encoders.pkl
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report

df = pd.read_csv("data/failed_payments.csv")

cat_cols = ["plan_type", "failure_reason"]
num_cols = ["amount", "tenure_months", "past_failures_90d",
            "day_of_week", "hour_of_day", "is_annual_plan"]

encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    df[c + "_enc"] = le.fit_transform(df[c])
    encoders[c] = le

feature_cols = [c + "_enc" for c in cat_cols] + num_cols
X = df[feature_cols]
y = df["recovered"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)
print("AUC:", round(roc_auc_score(y_test, probs), 4))
print(classification_report(y_test, preds))

joblib.dump(model, "model/recovery_model.pkl")
joblib.dump({"encoders": encoders, "feature_cols": feature_cols}, "model/encoders.pkl")
print("Saved model/recovery_model.pkl and model/encoders.pkl")
