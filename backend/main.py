"""
Reverra — FastAPI backend.

Endpoints:
  GET  /health
  GET  /at-risk               -> list of simulated at-risk customers + predictions
  GET  /metrics                -> summary metrics for dashboard cards
  POST /predict                -> recovery probability for a single failed payment
  POST /generate-message       -> personalized dunning email draft
  POST /simulate                -> simulate a brand-new failed payment end-to-end

Run from project root:
  uvicorn backend.main:app --reload --port 8000
"""
import json
import os
import random
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "failed_payments.csv"
MODEL_PATH = ROOT / "model" / "recovery_model.pkl"
ENCODERS_PATH = ROOT / "model" / "encoders.pkl"

CAT_COLS = ["plan_type", "failure_reason"]
NUM_COLS = ["amount", "tenure_months", "past_failures_90d",
            "day_of_week", "hour_of_day", "is_annual_plan"]
REQUIRED_COLS = set(CAT_COLS + NUM_COLS + ["recovered", "customer_id"])

app = FastAPI(title="Reverra")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load model + data at startup ----
model = joblib.load(MODEL_PATH)
enc_bundle = joblib.load(ENCODERS_PATH)
encoders = enc_bundle["encoders"]
feature_cols = enc_bundle["feature_cols"]
df = pd.read_csv(DATA_PATH)


def _train_on_dataframe(data: pd.DataFrame):
    """Trains a fresh model on the given dataframe, saves it to disk, and
    returns (model, encoders, feature_cols, auc)."""
    data = data.copy()
    new_encoders = {}
    for c in CAT_COLS:
        le = LabelEncoder()
        data[c + "_enc"] = le.fit_transform(data[c].astype(str))
        new_encoders[c] = le

    new_feature_cols = [c + "_enc" for c in CAT_COLS] + NUM_COLS
    X = data[new_feature_cols]
    y = data["recovered"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    new_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42
    )
    new_model.fit(X_train, y_train)

    auc = None
    if y_test.nunique() > 1:
        probs = new_model.predict_proba(X_test)[:, 1]
        auc = round(float(roc_auc_score(y_test, probs)), 4)

    joblib.dump(new_model, MODEL_PATH)
    joblib.dump({"encoders": new_encoders, "feature_cols": new_feature_cols}, ENCODERS_PATH)

    return new_model, new_encoders, new_feature_cols, auc


def _generate_synthetic_data(n: int = 6000) -> pd.DataFrame:
    """Generates a fresh batch of synthetic failed-payment data (no fixed
    seed — a different sample every time). Mirrors data/generate_data.py."""
    import numpy as np

    failure_reasons = ["insufficient_funds", "expired_card", "card_declined",
                        "bank_error", "fraud_flag", "processing_error"]
    reason_base_recovery = {
        "insufficient_funds": 0.55, "expired_card": 0.35, "card_declined": 0.45,
        "bank_error": 0.65, "fraud_flag": 0.15, "processing_error": 0.70,
    }
    plan_types = ["basic", "standard", "premium", "enterprise"]
    plan_amount_range = {
        "basic": (400, 1600), "standard": (1600, 5000),
        "premium": (5000, 12000), "enterprise": (12000, 40000),
    }

    rows = []
    for i in range(n):
        reason = np.random.choice(failure_reasons, p=[0.30, 0.20, 0.20, 0.10, 0.05, 0.15])
        plan = np.random.choice(plan_types, p=[0.35, 0.35, 0.20, 0.10])
        amount = round(np.random.uniform(*plan_amount_range[plan]), 2)
        tenure_months = max(0, int(np.random.exponential(scale=14)))
        past_failures_90d = np.random.poisson(0.6)
        day_of_week = np.random.randint(0, 7)
        hour_of_day = np.random.randint(0, 24)
        is_annual_plan = np.random.choice([0, 1], p=[0.8, 0.2])

        p = reason_base_recovery[reason]
        p += 0.15 * np.tanh(tenure_months / 24)
        p -= 0.08 * min(past_failures_90d, 4)
        p -= 0.05 if amount > 12000 else 0
        p += 0.05 if is_annual_plan else 0
        p = np.clip(p, 0.02, 0.95)
        recovered = np.random.rand() < p

        rows.append(dict(
            customer_id=f"CUST-{i:05d}", plan_type=plan, amount=amount,
            tenure_months=tenure_months, failure_reason=reason,
            past_failures_90d=past_failures_90d, day_of_week=day_of_week,
            hour_of_day=hour_of_day, is_annual_plan=is_annual_plan,
            recovered=int(recovered),
        ))
    return pd.DataFrame(rows)


class FailedPayment(BaseModel):
    plan_type: str
    amount: float
    tenure_months: int
    failure_reason: str
    past_failures_90d: int
    day_of_week: int
    hour_of_day: int
    is_annual_plan: int
    customer_id: Optional[str] = "SIM-CUSTOMER"


class NewRecord(FailedPayment):
    """Same shape as FailedPayment, plus the actual outcome — used when a
    user manually adds a real historical record to the training set."""
    recovered: int  # 0 or 1 — did this payment actually get recovered?


def _encode(payload: FailedPayment) -> pd.DataFrame:
    row = payload.model_dump()
    for c in ["plan_type", "failure_reason"]:
        le = encoders[c]
        val = row[c]
        row[c + "_enc"] = int(le.transform([val])[0]) if val in le.classes_ else 0
    X = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
    return X


def _predict_prob(payload: FailedPayment) -> float:
    X = _encode(payload)
    return float(model.predict_proba(X)[0, 1])


REASON_COPY = {
    "insufficient_funds": "it looks like your card didn't have enough funds available",
    "expired_card": "your card on file has expired",
    "card_declined": "your bank declined the charge",
    "bank_error": "your bank reported a temporary processing error",
    "fraud_flag": "your bank flagged the charge for extra verification",
    "processing_error": "we hit a temporary processing error on our end",
}


ACTION_LABELS = {
    "retry_now": "Retry now",
    "wait_3_days": "Wait 3 days, then retry",
    "escalate_to_human": "Escalate to human agent",
}


def _decide_action(prob: float) -> str:
    """Rule-based action decision — the fallback, and also what the LLM's
    judgment is compared against."""
    if prob >= 0.6:
        return "retry_now"
    elif prob >= 0.35:
        return "wait_3_days"
    else:
        return "escalate_to_human"


def _generate_message_template(payload: FailedPayment, prob: float) -> dict:
    """Rule-based strategy + templated message — no API key needed. Used
    as the always-available fallback."""
    reason_text = REASON_COPY.get(payload.failure_reason, "your recent payment didn't go through")
    urgency = "high" if prob < 0.4 else ("medium" if prob < 0.7 else "low")
    action = _decide_action(prob)

    reasoning_bits = [f"recovery probability is {prob:.0%}"]
    if payload.past_failures_90d >= 2:
        reasoning_bits.append(f"{payload.past_failures_90d} prior failures in 90 days lowers confidence")
    if payload.tenure_months >= 12:
        reasoning_bits.append(f"{payload.tenure_months}-month tenure is a positive signal")
    reasoning = "Based on rule-based thresholds: " + "; ".join(reasoning_bits) + "."

    if payload.tenure_months >= 12:
        opener = f"Hi there — as a valued customer of {payload.tenure_months} months,"
    else:
        opener = "Hi there,"

    subject = f"Action needed: your {payload.plan_type} plan payment didn't go through"
    body = (
        f"{opener} we tried to process your payment of ₹{payload.amount:,.2f} "
        f"for your {payload.plan_type} plan, but {reason_text}.\n\n"
        f"To avoid any interruption to your service, please update your payment "
        f"method or retry the charge at your earliest convenience.\n\n"
        f"[Update payment method]\n\n"
        f"If you have questions, just reply to this email — we're happy to help."
    )
    return {
        "action": action, "reasoning": reasoning,
        "subject": subject, "body": body,
        "urgency": urgency, "generated_by": "rules+template",
    }


_gemini_client = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY and genai is not None:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        _gemini_client = None

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["retry_now", "wait_3_days", "escalate_to_human"]},
        "reasoning": {"type": "string"},
        "subject": {"type": "string"},
        "body": {"type": "string"},
    },
    "required": ["action", "reasoning", "subject", "body"],
}


def _generate_message_llm(payload: FailedPayment, prob: float) -> Optional[dict]:
    """Calls Gemini with structured output — it decides the recovery
    *action* (not just drafts text) and gives its reasoning. Returns None
    (so the caller falls back to rules+template) if no key is configured
    or the call fails for any reason — this must never break the demo."""
    if _gemini_client is None:
        return None

    urgency = "high" if prob < 0.4 else ("medium" if prob < 0.7 else "low")
    reason_text = REASON_COPY.get(payload.failure_reason, "the payment didn't go through")

    prompt = (
        "You are a revenue-recovery agent for a SaaS company. Given this failed "
        "payment, decide the best recovery action and draft the customer email.\n\n"
        f"- Plan: {payload.plan_type}\n"
        f"- Amount due: ₹{payload.amount:,.2f}\n"
        f"- Customer tenure: {payload.tenure_months} months\n"
        f"- Failure reason: {reason_text}\n"
        f"- Past failures in last 90 days: {payload.past_failures_90d}\n"
        f"- Model-predicted recovery probability: {prob:.0%}\n"
        f"- Urgency tier: {urgency}\n\n"
        "Choose one action: 'retry_now' (high confidence, retry the charge "
        "immediately), 'wait_3_days' (moderate confidence, give the customer "
        "time to fix their payment method first), or 'escalate_to_human' (low "
        "confidence or risky pattern, a human agent should handle this one). "
        "Give a one-sentence reasoning for your choice referencing the specific "
        "numbers above. Then draft a subject line and a short, warm, "
        "professional email body (plain text, no markdown, 3-5 short "
        "paragraphs, a call to action to update payment method, greeting "
        "'Hi there' with no [Name] placeholder)."
    )
    try:
        resp = _gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": STRATEGY_SCHEMA,
            },
        )
        data = json.loads(resp.text)
        if data.get("action") not in ACTION_LABELS:
            return None
        return {
            "action": data["action"],
            "reasoning": data["reasoning"].strip(),
            "subject": data["subject"].strip(),
            "body": data["body"].strip(),
            "urgency": urgency,
            "generated_by": "gemini",
        }
    except Exception:
        return None


def _generate_message(payload: FailedPayment, prob: float) -> dict:
    """Tries the LLM first (if a key is configured); falls back to the
    reliable rule-based strategy on any failure, so the demo never breaks."""
    llm_result = _generate_message_llm(payload, prob)
    return llm_result if llm_result is not None else _generate_message_template(payload, prob)


@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": _gemini_client is not None}


@app.get("/metrics")
def metrics():
    total_at_risk = df[df["recovered"] == 0].shape[0] + df[df["recovered"] == 1].shape[0]
    revenue_at_risk = round(df["amount"].sum(), 2)
    revenue_recovered = round(df.loc[df["recovered"] == 1, "amount"].sum(), 2)
    recovery_rate = round(df["recovered"].mean() * 100, 1)
    return {
        "total_failed_payments": int(total_at_risk),
        "revenue_at_risk": revenue_at_risk,
        "revenue_recovered": revenue_recovered,
        "recovery_rate_pct": recovery_rate,
    }


@app.get("/revenue-breakdown")
def revenue_breakdown():
    """Revenue recovered vs lost, grouped by failure reason — powers the
    dashboard's trend chart."""
    rows = []
    for reason, group in df.groupby("failure_reason"):
        recovered = round(group.loc[group["recovered"] == 1, "amount"].sum(), 2)
        lost = round(group.loc[group["recovered"] == 0, "amount"].sum(), 2)
        rows.append({
            "failure_reason": reason,
            "recovered_revenue": recovered,
            "lost_revenue": lost,
        })
    rows.sort(key=lambda r: r["recovered_revenue"] + r["lost_revenue"], reverse=True)
    return rows


@app.get("/at-risk")
def at_risk(limit: int = 25):
    """Returns a sample of currently-failed (unrecovered) payments with live
    model predictions, sorted by recovery probability descending (best bets first)."""
    sample = df[df["recovered"] == 0].sample(min(limit, len(df)), random_state=random.randint(0, 9999))
    results = []
    for _, r in sample.iterrows():
        payload = FailedPayment(
            plan_type=r["plan_type"], amount=r["amount"], tenure_months=int(r["tenure_months"]),
            failure_reason=r["failure_reason"], past_failures_90d=int(r["past_failures_90d"]),
            day_of_week=int(r["day_of_week"]), hour_of_day=int(r["hour_of_day"]),
            is_annual_plan=int(r["is_annual_plan"]), customer_id=r["customer_id"],
        )
        prob = _predict_prob(payload)
        results.append({**payload.model_dump(), "recovery_probability": round(prob, 3)})
    results.sort(key=lambda x: x["recovery_probability"], reverse=True)
    return results


@app.post("/predict")
def predict(payload: FailedPayment):
    prob = _predict_prob(payload)
    return {"customer_id": payload.customer_id, "recovery_probability": round(prob, 3)}


@app.post("/generate-message")
def generate_message(payload: FailedPayment):
    prob = _predict_prob(payload)
    msg = _generate_message(payload, prob)
    return {"recovery_probability": round(prob, 3), **msg}


@app.post("/simulate")
def simulate(payload: FailedPayment):
    """One-shot: predict + generate message together, for the live demo button."""
    prob = _predict_prob(payload)
    msg = _generate_message(payload, prob)
    return {
        "customer_id": payload.customer_id,
        "recovery_probability": round(prob, 3),
        "message": msg,
    }


@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Accepts a CSV of failed payments, replaces the dataset, retrains the
    model immediately, and swaps it into the live server."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    try:
        new_df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't parse CSV: {e}")

    missing = REQUIRED_COLS - set(new_df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV is missing required columns: {sorted(missing)}"
        )
    if new_df["recovered"].nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="The 'recovered' column needs both 0 and 1 values to train on."
        )

    new_df.to_csv(DATA_PATH, index=False)

    global model, encoders, feature_cols, df
    try:
        new_model, new_encoders, new_feature_cols, auc = _train_on_dataframe(new_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")

    model, encoders, feature_cols = new_model, new_encoders, new_feature_cols
    df = new_df

    return {
        "status": "retrained",
        "rows": len(new_df),
        "auc": auc,
        "recovery_rate_pct": round(new_df["recovered"].mean() * 100, 1),
    }


@app.post("/regenerate-data")
def regenerate_data(n: int = 6000):
    """Generates a brand-new random batch of synthetic failed-payment data
    (no fixed seed), replaces the dataset, and retrains immediately."""
    new_df = _generate_synthetic_data(n)
    new_df.to_csv(DATA_PATH, index=False)

    global model, encoders, feature_cols, df
    new_model, new_encoders, new_feature_cols, auc = _train_on_dataframe(new_df)
    model, encoders, feature_cols = new_model, new_encoders, new_feature_cols
    df = new_df

    return {
        "status": "regenerated",
        "rows": len(new_df),
        "auc": auc,
        "recovery_rate_pct": round(new_df["recovered"].mean() * 100, 1),
    }


@app.post("/add-record")
def add_record(record: NewRecord):
    """Appends one manually-entered payment record to the dataset and
    retrains immediately — for adding a real known outcome by hand,
    one row at a time, instead of uploading a whole CSV."""
    if record.recovered not in (0, 1):
        raise HTTPException(status_code=400, detail="'recovered' must be 0 or 1")

    new_row = record.model_dump()
    global model, encoders, feature_cols, df

    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if updated_df["recovered"].nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Dataset needs both recovered and not-recovered examples to train on."
        )

    updated_df.to_csv(DATA_PATH, index=False)
    try:
        new_model, new_encoders, new_feature_cols, auc = _train_on_dataframe(updated_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")

    model, encoders, feature_cols = new_model, new_encoders, new_feature_cols
    df = updated_df

    return {
        "status": "added_and_retrained",
        "total_rows": len(updated_df),
        "auc": auc,
        "recovery_rate_pct": round(updated_df["recovered"].mean() * 100, 1),
    }


# ---- Tool functions for the /ask endpoint ----
# These are plain Python functions with type hints and docstrings so that
# Gemini's automatic function calling can call them directly — the model
# reads the docstring to decide which one(s) it needs for a given question.

def get_summary_metrics() -> dict:
    """Returns overall dashboard metrics: total number of failed payments,
    total revenue at risk in INR, total revenue recovered in INR, and the
    overall recovery rate as a percentage."""
    return {
        "total_failed_payments": int(len(df)),
        "revenue_at_risk_inr": round(float(df["amount"].sum()), 2),
        "revenue_recovered_inr": round(float(df.loc[df["recovered"] == 1, "amount"].sum()), 2),
        "recovery_rate_pct": round(float(df["recovered"].mean() * 100), 1),
    }


def get_revenue_by_failure_reason() -> list:
    """Returns revenue recovered and lost, broken down by why the payment
    failed (e.g. expired_card, insufficient_funds, fraud_flag). Each item
    has failure_reason, recovered_revenue_inr, and lost_revenue_inr."""
    rows = []
    for reason, group in df.groupby("failure_reason"):
        rows.append({
            "failure_reason": reason,
            "recovered_revenue_inr": round(float(group.loc[group["recovered"] == 1, "amount"].sum()), 2),
            "lost_revenue_inr": round(float(group.loc[group["recovered"] == 0, "amount"].sum()), 2),
        })
    return rows


def get_top_at_risk_customers(limit: int = 10) -> list:
    """Returns the top currently at-risk (unrecovered) customers, ranked by
    model-predicted recovery probability, highest first. limit controls how
    many to return (default 10). Each item has customer_id, plan_type,
    amount, failure_reason, tenure_months, and recovery_probability."""
    sample = df[df["recovered"] == 0].sample(min(limit * 3, len(df)), random_state=1)
    scored = []
    for _, r in sample.iterrows():
        payload = FailedPayment(
            plan_type=r["plan_type"], amount=r["amount"], tenure_months=int(r["tenure_months"]),
            failure_reason=r["failure_reason"], past_failures_90d=int(r["past_failures_90d"]),
            day_of_week=int(r["day_of_week"]), hour_of_day=int(r["hour_of_day"]),
            is_annual_plan=int(r["is_annual_plan"]), customer_id=r["customer_id"],
        )
        scored.append({
            "customer_id": r["customer_id"], "plan_type": r["plan_type"],
            "amount_inr": round(float(r["amount"]), 2), "failure_reason": r["failure_reason"],
            "tenure_months": int(r["tenure_months"]),
            "recovery_probability": round(_predict_prob(payload), 3),
        })
    scored.sort(key=lambda x: x["recovery_probability"], reverse=True)
    return scored[:limit]


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest):
    """Answers a natural-language question about the dashboard data by
    letting Gemini call the tool functions above as needed (function
    calling / tool use), then returns its final text answer."""
    if _gemini_client is None or genai_types is None:
        return {
            "answer": "Q&A needs a GEMINI_API_KEY set — without one, this "
                       "feature has no reliable rule-based fallback, unlike "
                       "the rest of the app.",
            "generated_by": "unavailable",
        }

    try:
        chat = _gemini_client.chats.create(
            model="gemini-3.8-flash",
            config=genai_types.GenerateContentConfig(
                tools=[get_summary_metrics, get_revenue_by_failure_reason, get_top_at_risk_customers],
                system_instruction=(
                    "You are an analyst assistant for a payment-recovery dashboard. "
                    "Answer the user's question using the provided tools to fetch real "
                    "data — never guess numbers. Be concise (2-4 sentences unless asked "
                    "for a list). Use ₹ for currency, formatted with commas."
                ),
            ),
        )
        resp = chat.send_message(req.question)
        answer = (resp.text or "").strip()
        if not answer:
            return {"answer": "Gemini didn't return an answer — try rephrasing the question.",
                    "generated_by": "gemini"}
        return {"answer": answer, "generated_by": "gemini"}
    except Exception as e:
        return {"answer": f"Something went wrong calling Gemini: {e}", "generated_by": "error"}
