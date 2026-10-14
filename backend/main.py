"""
Reverra — FastAPI backend.

One server does everything: the JSON API, the landing page (/) and the
dashboard (/dashboard).

Run from project root:
  uvicorn backend.main:app --reload --port 8000
then open http://localhost:8000

Public deployment: set ADMIN_TOKEN (see the settings block below). That turns on
read-only mode for visitors; endpoints that change data then need the token.
"""
import io
import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so `ml` imports no matter where uvicorn is started from

from ml.pipeline import (CAT_COLS, NUM_COLS, MODEL_LABELS, known_categories,  # noqa: E402
                         load_report, predict_proba, save_artifacts,
                         train_and_evaluate)
from ml.simulate import generate as _simulate  # noqa: E402
from backend import rivera as rv  # noqa: E402
from backend.security import (DailyCap, SlidingWindowLimiter, client_ip, env_int,  # noqa: E402
                              parse_origins, token_ok)

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

DATA_PATH = ROOT / "data" / "failed_payments.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "recovery_model.pkl"

# failed_at is optional; when present, evaluation uses a time-based split
REQUIRED_COLS = set(CAT_COLS + NUM_COLS + ["recovered", "customer_id"])

# ---- deployment settings (all via environment variables) ----
#   ADMIN_TOKEN        set => PUBLIC MODE: every endpoint that changes data needs the
#                      X-Admin-Token header, and the auto-generated /docs pages are
#                      hidden. Unset => local development, everything open.
#                      Must be at least 16 characters.
#   CORS_ORIGINS       unset => any origin in local mode, same-origin only in public mode.
#                      "none" => same-origin only. Or a comma-separated list of origins.
#   LLM_RATE_PER_MIN   Gemini calls allowed per visitor per minute (default 8)
#   LLM_DAILY_CAP      Gemini calls allowed per day across all visitors (default 300)
#   RIVERA_MODEL       Gemini model for ledger questions (default gemini-flash-latest)
#   RIVERA_FAST_MODEL  lighter Gemini model for general / finance questions (default
#                      gemini-flash-lite-latest). Set both if Google retires an alias.
#   MAX_UPLOAD_MB      largest CSV /upload-data accepts (default 10)
#   TRUSTED_PROXY_HOPS how many proxies sit in front of the app (default 1: Render)
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()
PUBLIC_MODE = bool(ADMIN_TOKEN)
if PUBLIC_MODE and len(ADMIN_TOKEN) < 16:
    raise RuntimeError("ADMIN_TOKEN is too short — use at least 16 random characters.")
MAX_UPLOAD_BYTES = env_int("MAX_UPLOAD_MB", 10, 1) * 1024 * 1024
MAX_UPLOAD_ROWS = 200_000
TRUSTED_PROXY_HOPS = env_int("TRUSTED_PROXY_HOPS", 1, 0)
logger = logging.getLogger("reverra")

app = FastAPI(
    title="Reverra",
    docs_url=None if PUBLIC_MODE else "/docs",
    redoc_url=None if PUBLIC_MODE else "/redoc",
    openapi_url=None if PUBLIC_MODE else "/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_origins(os.environ.get("CORS_ORIGINS"), PUBLIC_MODE),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return response


@app.middleware("http")
async def _limit_request_size(request: Request, call_next):
    """Rejects oversized request bodies up front, from the Content-Length header, before
    anything is read. (FastAPI parses an upload body before it checks the admin token, so
    without this an anonymous visitor could make the server buffer a huge file.)"""
    if request.method in ("POST", "PUT", "PATCH"):
        length = request.headers.get("content-length")
        if length is None:
            if "chunked" in request.headers.get("transfer-encoding", "").lower():
                return JSONResponse(status_code=411, content={"detail": "Content-Length header required."})
        else:
            try:
                size = int(length)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
            limit = (MAX_UPLOAD_BYTES + 1_048_576) if request.url.path == "/upload-data" else 64 * 1024
            if size > limit:
                return JSONResponse(status_code=413, content={"detail": "Request too large."})
    return await call_next(request)


_admin_failures = SlidingWindowLimiter(10, 600)                        # 10 wrong tokens / 10 min / visitor
_llm_per_visitor = SlidingWindowLimiter(env_int("LLM_RATE_PER_MIN", 8, 1), 60)
_llm_daily = DailyCap(env_int("LLM_DAILY_CAP", 300, 1))
_rivera_instant_limiter = SlidingWindowLimiter(env_int("RIVERA_INSTANT_PER_MIN", 60, 1), 60)  # instant answers cost no Gemini call, only a little CPU


def _visitor(request: Request) -> str:
    return client_ip(request.headers.get("x-forwarded-for"),
                     request.client.host if request.client else None,
                     TRUSTED_PROXY_HOPS)


def require_admin(request: Request, x_admin_token: Optional[str] = Header(default=None)):
    """Guards every endpoint that changes data. Does nothing when ADMIN_TOKEN is unset."""
    if not ADMIN_TOKEN:
        return
    who = _visitor(request)
    allowed, retry = _admin_failures.peek(who)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many failed attempts — try again later.",
                            headers={"Retry-After": str(int(retry) + 1)})
    if not token_ok(x_admin_token, ADMIN_TOKEN):
        if x_admin_token:                      # a wrong guess counts; simply sending no token doesn't
            _admin_failures.record(who)
        raise HTTPException(status_code=401,
                            detail="Admin token required — this public demo is read-only.")


def _llm_allowed(request: Request) -> bool:
    """May this visitor trigger a Gemini call right now? With no Gemini key there is
    nothing to limit (callers fall back to templates anyway)."""
    if _gemini_client is None:
        return True
    ok, _ = _llm_per_visitor.allow(_visitor(request))
    return ok and _llm_daily.allow()


# ---- load model + data at startup ----
def _load_or_create_dataset() -> pd.DataFrame:
    """Loads data/failed_payments.csv. On a fresh deploy that skipped the build-time
    generate step the file doesn't exist, so create it from the simulator instead of
    crashing. (An existing file is never overwritten.)"""
    if not DATA_PATH.exists():
        logger.warning("%s not found — generating simulated data", DATA_PATH.name)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        fresh, _meta = _simulate(6000, seed=42, return_truth=True)
        fresh.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


MODEL_DIR.mkdir(parents=True, exist_ok=True)
df = _load_or_create_dataset()


def _train_on_dataframe(data: pd.DataFrame, oracle_auc: Optional[float] = None):
    """Trains + evaluates on `data`. Returns (fitted_model, report). Nothing is
    saved or swapped in here; see _install()."""
    return train_and_evaluate(data, oracle_auc=oracle_auc)


def _install(new_model, new_report: dict, new_df: pd.DataFrame):
    """Saves artifacts + dataset to disk and swaps them into the live server.
    Only called after training succeeded, so a failed upload never clobbers anything."""
    global model, model_report, df
    save_artifacts(new_model, new_report, MODEL_DIR, plots=False)
    new_df.to_csv(DATA_PATH, index=False)
    model, model_report, df = new_model, new_report, new_df


def _auc_of(report: dict) -> Optional[float]:
    """Held-out AUC of the deployed model, or None if it couldn't be evaluated."""
    if not report.get("evaluated"):
        return None
    return report["models"][report["selected_model"]]["auc"]


def _load_or_train():
    """Loads the saved model; if it is missing or from an older version of Reverra
    (not a sklearn Pipeline), retrains from the dataset instead of crashing."""
    try:
        m = joblib.load(MODEL_PATH)
        if isinstance(m, Pipeline):
            return m, (load_report(MODEL_DIR) or {"evaluated": False, "selected_model": "gradient_boosting",
                                                  "warnings": ["No saved evaluation found; re-run model/train_model.py."]})
    except Exception:
        pass
    m, rep = _train_on_dataframe(df)
    save_artifacts(m, rep, MODEL_DIR, plots=False)
    return m, rep


model, model_report = _load_or_train()


def _generate_synthetic_data(n: int = 6000):
    """Fresh batch of simulated failed payments (no fixed seed, so a different
    sample each time). Returns (dataframe, meta) from ml/simulate.py."""
    return _simulate(n, seed=None, return_truth=True)


# Free-text fields end up inside the Gemini prompt, so they are restricted to a short,
# plain character set (no newlines or markup). Numbers get sensible bounds.
_LABEL = r"^[\w .\-/&()+]{1,40}$"


class FailedPayment(BaseModel):
    plan_type: str = Field(pattern=_LABEL)
    amount: float = Field(gt=0, le=10_000_000)
    tenure_months: int = Field(ge=0, le=600)
    failure_reason: str = Field(pattern=_LABEL)
    past_failures_90d: int = Field(ge=0, le=100)
    day_of_week: int = Field(ge=0, le=6)
    hour_of_day: int = Field(ge=0, le=23)
    is_annual_plan: int = Field(ge=0, le=1)
    customer_id: Optional[str] = Field(default="SIM-CUSTOMER", max_length=64)


class NewRecord(FailedPayment):
    """Same shape as FailedPayment, plus the actual outcome — used when a
    user manually adds a real historical record to the training set."""
    recovered: int = Field(ge=0, le=1)  # did this payment actually get recovered?


def _predict_prob(payload: FailedPayment) -> float:
    return float(predict_proba(model, pd.DataFrame([payload.model_dump()]))[0])


REASON_COPY = {
    "insufficient_funds": "it looks like your card didn't have enough funds available",
    "expired_card": "your card on file has expired",
    "card_declined": "your bank declined the charge",
    "bank_error": "your bank reported a temporary processing error",
    "fraud_flag": "your bank flagged the charge for extra verification",
    "processing_error": "we hit a temporary processing error on our end",
}


def _generate_message_template(payload: FailedPayment, prob: float) -> dict:
    """Template-based dunning message generator — no API key needed. Used
    as the always-available fallback."""
    reason_text = REASON_COPY.get(payload.failure_reason, "your recent payment didn't go through")
    urgency = "high" if prob < 0.4 else ("medium" if prob < 0.7 else "low")

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
    return {"subject": subject, "body": body, "urgency": urgency, "generated_by": "template"}


_gemini_client = None
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY and genai is not None:
    try:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        _gemini_client = None


def _generate_message_llm(payload: FailedPayment, prob: float) -> Optional[dict]:
    """Calls Gemini to draft a personalized dunning message. Returns None
    (so the caller falls back to the template) if no key is configured or
    the call fails for any reason — this must never break the demo."""
    if _gemini_client is None:
        return None

    urgency = "high" if prob < 0.4 else ("medium" if prob < 0.7 else "low")
    reason_text = REASON_COPY.get(payload.failure_reason, "the payment didn't go through")

    prompt = (
        "Write a short, warm but professional payment-recovery email for a SaaS customer.\n"
        f"- Plan: {payload.plan_type}\n"
        f"- Amount due: ₹{payload.amount:,.2f}\n"
        f"- Customer tenure: {payload.tenure_months} months\n"
        f"- Failure reason: {reason_text}\n"
        f"- Urgency: {urgency}\n\n"
        "Respond with exactly two parts, separated by a line containing only ---\n"
        "Part 1: a subject line (one line, no prefix like 'Subject:')\n"
        "Part 2: the email body (plain text, no markdown, 3-5 short paragraphs, "
        "include a call to action to update payment method). Do not include a "
        "greeting placeholder like [Name] — use 'Hi there'."
    )
    try:
        resp = _gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        text = (resp.text or "").strip()
        if "---" not in text:
            return None
        subject, body = text.split("---", 1)
        return {
            "subject": subject.strip(),
            "body": body.strip(),
            "urgency": urgency,
            "generated_by": "gemini",
        }
    except Exception:
        return None


def _generate_message(payload: FailedPayment, prob: float, use_llm: bool = True) -> dict:
    """Tries the LLM first (if a key is configured and use_llm is set); falls back to
    the reliable template on any failure, so the demo never breaks."""
    llm_result = _generate_message_llm(payload, prob) if use_llm else None
    return llm_result if llm_result is not None else _generate_message_template(payload, prob)


@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": _gemini_client is not None, "read_only": PUBLIC_MODE}


@app.get("/admin-check", dependencies=[Depends(require_admin)])
def admin_check():
    """Lets the dashboard verify an admin token. Always ok when not in public mode."""
    return {"ok": True}


@app.get("/metrics")
def metrics():
    total_at_risk = len(df)
    revenue_at_risk = round(df.loc[df["recovered"] == 0, "amount"].sum(), 2)  # still unrecovered
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
    limit = max(1, min(int(limit), 100))
    unrec = df[df["recovered"] == 0]
    sample = unrec.sample(min(limit, len(unrec)), random_state=random.randint(0, 9999))
    results = []
    for _, r in sample.iterrows():
        # model_construct: these rows are already stored data (e.g. from an uploaded CSV),
        # so don't re-validate them against the stricter rules for new input.
        payload = FailedPayment.model_construct(
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
def generate_message(payload: FailedPayment, request: Request):
    prob = _predict_prob(payload)
    msg = _generate_message(payload, prob, use_llm=_llm_allowed(request))
    return {"recovery_probability": round(prob, 3), **msg}


@app.post("/simulate")
def simulate(payload: FailedPayment, request: Request):
    """One-shot: predict + generate message together, for the live demo button."""
    prob = _predict_prob(payload)
    msg = _generate_message(payload, prob, use_llm=_llm_allowed(request))
    return {
        "customer_id": payload.customer_id,
        "recovery_probability": round(prob, 3),
        "message": msg,
    }


@app.post("/upload-data", dependencies=[Depends(require_admin)])
async def upload_data(file: UploadFile = File(...)):
    """Accepts a CSV of failed payments, replaces the dataset, retrains the
    model immediately, and swaps it into the live server."""
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)      # never read more than the limit into memory
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"File too large — the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        new_df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't parse CSV: {e}")
    if len(new_df) > MAX_UPLOAD_ROWS:
        raise HTTPException(status_code=400, detail=f"Too many rows — the limit is {MAX_UPLOAD_ROWS:,}.")

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

    try:
        new_model, new_report = _train_on_dataframe(new_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")
    _install(new_model, new_report, new_df)

    return {
        "status": "retrained",
        "rows": len(new_df),
        "auc": _auc_of(new_report),
        "model": new_report["selected_model"],
        "evaluation_split": new_report["split"]["kind"] if new_report.get("evaluated") else None,
        "warnings": new_report["warnings"],
        "recovery_rate_pct": round(new_df["recovered"].mean() * 100, 1),
    }


@app.post("/regenerate-data", dependencies=[Depends(require_admin)])
def regenerate_data(n: int = 6000):
    """Generates a brand-new random batch of simulated failed-payment data
    (no fixed seed), replaces the dataset, and retrains immediately."""
    if not 500 <= n <= 100000:
        raise HTTPException(status_code=400, detail="n must be between 500 and 100000")
    new_df, meta = _generate_synthetic_data(n)
    new_model, new_report = _train_on_dataframe(new_df, oracle_auc=meta["oracle_auc"])
    _install(new_model, new_report, new_df)

    return {
        "status": "regenerated",
        "rows": len(new_df),
        "auc": _auc_of(new_report),
        "model": new_report["selected_model"],
        "evaluation_split": new_report["split"]["kind"] if new_report.get("evaluated") else None,
        "warnings": new_report["warnings"],
        "recovery_rate_pct": round(new_df["recovered"].mean() * 100, 1),
    }


def _append_and_retrain(row: dict) -> dict:
    """Shared helper: appends one row (which must include 'recovered') to
    the dataset, retrains, and swaps the live model in. Raises
    HTTPException on any validation/training failure."""
    updated_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    if updated_df["recovered"].nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Dataset needs both recovered and not-recovered examples to train on."
        )

    try:
        new_model, new_report = _train_on_dataframe(updated_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Training failed: {e}")
    _install(new_model, new_report, updated_df)
    return {
        "total_rows": len(updated_df),
        "auc": _auc_of(new_report),
        "recovery_rate_pct": round(updated_df["recovered"].mean() * 100, 1),
    }


# ---- Pending → confirm workflow ----
# A newly-entered payment doesn't have a known outcome yet in real life —
# you log the failure, then find out later whether it recovered. Pending
# records are held here and only join the training data once confirmed.

PENDING_PATH = ROOT / "data" / "pending_payments.csv"


def _load_pending() -> list:
    if PENDING_PATH.exists():
        try:
            frame = pd.read_csv(PENDING_PATH)
            frame = frame.astype(object).where(frame.notna(), None)   # NaN is not valid JSON
            return frame.to_dict(orient="records")
        except Exception:
            return []
    return []


def _save_pending(records: list):
    pd.DataFrame(records).to_csv(PENDING_PATH, index=False) if records else PENDING_PATH.write_text(
        "pending_id,customer_id,plan_type,amount,tenure_months,failure_reason,"
        "past_failures_90d,day_of_week,hour_of_day,is_annual_plan,recovery_probability,created_at,failed_at\n"
    )


pending_records = _load_pending()


@app.post("/add-pending", dependencies=[Depends(require_admin)])
def add_pending(payload: FailedPayment):
    """Logs a newly-failed payment as pending — not yet added to the
    training data, since the real outcome isn't known yet."""
    prob = _predict_prob(payload)
    record = payload.model_dump()
    record["pending_id"] = uuid.uuid4().hex[:8]
    record["recovery_probability"] = round(prob, 3)
    now = pd.Timestamp.now(tz="UTC")
    record["created_at"] = now.isoformat()
    record["failed_at"] = now.strftime("%Y-%m-%d %H:%M:%S")   # becomes the time-split column once confirmed

    pending_records.append(record)
    _save_pending(pending_records)
    return record


@app.get("/pending")
def get_pending():
    """Lists payments awaiting outcome confirmation."""
    return pending_records


class ConfirmPending(BaseModel):
    pending_id: str = Field(max_length=32)
    recovered: int


@app.post("/confirm-pending", dependencies=[Depends(require_admin)])
def confirm_pending(req: ConfirmPending):
    """Confirms the real outcome for a pending payment — only now does it
    join the training dataset and trigger a retrain."""
    if req.recovered not in (0, 1):
        raise HTTPException(status_code=400, detail="'recovered' must be 0 or 1")

    idx = next((i for i, r in enumerate(pending_records) if r["pending_id"] == req.pending_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Pending record not found")

    record = dict(pending_records[idx])
    record.pop("pending_id", None)
    record.pop("recovery_probability", None)
    record.pop("created_at", None)
    record["recovered"] = req.recovered

    result = _append_and_retrain(record)

    pending_records.pop(idx)
    _save_pending(pending_records)

    return {"status": "confirmed_and_retrained", **result}


@app.delete("/pending/{pending_id}", dependencies=[Depends(require_admin)])
def discard_pending(pending_id: str):
    """Discards a pending entry without confirming an outcome (e.g. it was
    a test, or a duplicate)."""
    idx = next((i for i, r in enumerate(pending_records) if r["pending_id"] == pending_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Pending record not found")
    pending_records.pop(idx)
    _save_pending(pending_records)
    return {"status": "discarded", "pending_id": pending_id}


# ---- Tool functions for the /ask endpoint ----
# Plain Python functions with type hints and docstrings so Gemini's
# automatic function calling can call them directly — the model reads the
# docstring to decide which one(s) it needs for a given question.

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


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GROUPABLE = ["plan_type", "failure_reason", "is_annual_plan", "day_of_week",
             "hour_of_day", "tenure_band", "amount_band", "past_failures_90d"]


def _score_frame(frame: pd.DataFrame):
    """Batch-scores a dataframe of payments with the live model."""
    return predict_proba(model, frame)


def _with_bands(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["tenure_band"] = pd.cut(f["tenure_months"], [-1, 5, 11, 23, 10**6],
                              labels=["0-5 months", "6-11 months", "12-23 months", "24+ months"]).astype(str)
    f["amount_band"] = pd.cut(f["amount"], [-1, 1600, 5000, 12000, 10**9],
                              labels=["under ₹1,600", "₹1,600-5,000", "₹5,000-12,000", "over ₹12,000"]).astype(str)
    return f


def _customer_row(r, prob) -> dict:
    return {
        "customer_id": r["customer_id"], "plan_type": r["plan_type"],
        "amount_inr": round(float(r["amount"]), 2), "failure_reason": r["failure_reason"],
        "tenure_months": int(r["tenure_months"]), "past_failures_90d": int(r["past_failures_90d"]),
        "is_annual_plan": bool(r["is_annual_plan"]),
        "outcome": "recovered" if int(r["recovered"]) == 1 else "not recovered",
        "recovery_probability": round(float(prob), 3),
    }


def get_dataset_overview() -> dict:
    """Describes the dataset: number of rows, the valid values for plan_type and
    failure_reason, ranges for amount and tenure, and how day_of_week is coded.
    Call this first if unsure which filter values or columns exist."""
    return {
        "rows": int(len(df)),
        "plan_types": sorted(df["plan_type"].astype(str).unique().tolist()),
        "failure_reasons": sorted(df["failure_reason"].astype(str).unique().tolist()),
        "amount_inr_range": [round(float(df["amount"].min()), 2), round(float(df["amount"].max()), 2)],
        "tenure_months_range": [int(df["tenure_months"].min()), int(df["tenure_months"].max())],
        "day_of_week_coding": "0=Monday ... 6=Sunday",
        "groupable_dimensions": GROUPABLE,
        "note": "Each row is one failed payment; customers are unique per row, so a "
                "per-customer 'recovery rate' doesn't exist — use recovery_probability "
                "for individuals and recovery rate for groups.",
    }


def get_recovery_rate_by(dimension: str) -> dict:
    """Recovery rate and revenue broken down by one dimension. dimension must be one of:
    plan_type, failure_reason, is_annual_plan, day_of_week (0=Mon), hour_of_day,
    tenure_band, amount_band, past_failures_90d. Returns each group with its payment
    count, recovery_rate_pct, revenue recovered/lost in INR and average amount, sorted
    lowest recovery rate first, plus the lowest and highest groups. Use for questions
    like 'which plan recovers worst', 'best day to retry', 'do loyal customers recover more'."""
    if dimension not in GROUPABLE:
        return {"error": f"Unknown dimension '{dimension}'", "valid_dimensions": GROUPABLE}
    f = _with_bands(df)
    rows = []
    for key, g in f.groupby(dimension):
        key = key.item() if hasattr(key, "item") else key
        label = DAY_NAMES[int(key)] if dimension == "day_of_week" else key
        if dimension == "is_annual_plan":
            label = "annual" if int(key) == 1 else "monthly"
        rows.append({
            "group": label, "payments": int(len(g)),
            "recovery_rate_pct": round(float(g["recovered"].mean() * 100), 1),
            "revenue_recovered_inr": round(float(g.loc[g["recovered"] == 1, "amount"].sum()), 2),
            "revenue_lost_inr": round(float(g.loc[g["recovered"] == 0, "amount"].sum()), 2),
            "avg_amount_inr": round(float(g["amount"].mean()), 2),
        })
    rows.sort(key=lambda x: x["recovery_rate_pct"])
    return {"dimension": dimension, "groups": rows, "lowest": rows[0], "highest": rows[-1]}


def find_customers(status: str = "unrecovered", plan_type: Optional[str] = None,
                   failure_reason: Optional[str] = None, min_amount: Optional[float] = None,
                   max_amount: Optional[float] = None, min_tenure_months: Optional[int] = None,
                   max_tenure_months: Optional[int] = None, min_past_failures: Optional[int] = None,
                   is_annual_plan: Optional[int] = None,
                   sort_by: str = "recovery_probability", ascending: bool = False,
                   limit: int = 10) -> dict:
    """Finds customers/payments matching filters, with model-predicted recovery
    probability. status: 'unrecovered' (default), 'recovered' or 'all'. sort_by:
    recovery_probability, amount, tenure_months or past_failures_90d. ascending=True
    gives lowest first. is_annual_plan: 1 or 0. limit max 50. Returns matched_count and
    the top rows. Use for 'biggest unrecovered invoices', 'which customers are least likely
    to recover', 'expired-card customers on premium', 'longest-tenure customers who failed'."""
    if status not in ("unrecovered", "recovered", "all"):
        return {"error": "status must be 'unrecovered', 'recovered' or 'all'"}
    if sort_by not in ("recovery_probability", "amount", "tenure_months", "past_failures_90d"):
        return {"error": "sort_by must be recovery_probability, amount, tenure_months or past_failures_90d"}
    f = df
    if status == "unrecovered":
        f = f[f["recovered"] == 0]
    elif status == "recovered":
        f = f[f["recovered"] == 1]
    if plan_type: f = f[f["plan_type"] == plan_type]
    if failure_reason: f = f[f["failure_reason"] == failure_reason]
    if min_amount is not None: f = f[f["amount"] >= min_amount]
    if max_amount is not None: f = f[f["amount"] <= max_amount]
    if min_tenure_months is not None: f = f[f["tenure_months"] >= min_tenure_months]
    if max_tenure_months is not None: f = f[f["tenure_months"] <= max_tenure_months]
    if min_past_failures is not None: f = f[f["past_failures_90d"] >= min_past_failures]
    if is_annual_plan is not None: f = f[f["is_annual_plan"] == int(is_annual_plan)]
    if f.empty:
        return {"matched_count": 0, "customers": []}
    f = f.copy()
    matched = int(len(f))
    f["recovery_probability"] = _score_frame(f)
    f = f.sort_values(sort_by, ascending=ascending).head(max(1, min(int(limit), 50)))
    return {"matched_count": matched,
            "customers": [_customer_row(r, r["recovery_probability"]) for _, r in f.iterrows()]}


def get_top_at_risk_customers(limit: int = 10) -> list:
    """Unrecovered customers MOST likely to recover if retried (best bets first),
    ranked by model-predicted recovery probability, highest first."""
    return find_customers(status="unrecovered", sort_by="recovery_probability",
                          ascending=False, limit=limit).get("customers", [])


def get_lowest_recovery_customers(limit: int = 10) -> list:
    """Unrecovered customers LEAST likely to recover, lowest predicted recovery
    probability first. Use for 'which customer has the worst chance of recovering'."""
    return find_customers(status="unrecovered", sort_by="recovery_probability",
                          ascending=True, limit=limit).get("customers", [])


def get_customer(customer_id: str) -> dict:
    """Looks up one customer by id (e.g. 'CUST-00042'): their payment details, actual
    outcome, the model's recovery probability, and any pending entry for them."""
    match = df[df["customer_id"] == customer_id]
    pend = [r for r in pending_records if r.get("customer_id") == customer_id]
    if match.empty and not pend:
        return {"error": f"No customer '{customer_id}' found"}
    out = {}
    if not match.empty:
        r = match.iloc[0]
        out = _customer_row(r, _score_frame(match)[0])
    if pend:
        out["pending_entries"] = [{k: v for k, v in p.items()
                                   if k in ("pending_id", "amount", "failure_reason",
                                            "recovery_probability", "created_at")} for p in pend]
    return out


def predict_recovery(plan_type: str, amount: float, tenure_months: int, failure_reason: str,
                     past_failures_90d: int = 0, day_of_week: int = 0, hour_of_day: int = 12,
                     is_annual_plan: int = 0) -> dict:
    """What-if prediction: the model's recovery probability for a hypothetical failed
    payment. day_of_week 0=Mon..6=Sun. Call get_dataset_overview first if unsure of
    valid plan_type / failure_reason values."""
    cats = known_categories(model)
    if plan_type not in cats["plan_type"]:
        return {"error": f"Unknown plan_type '{plan_type}'", "valid": cats["plan_type"]}
    if failure_reason not in cats["failure_reason"]:
        return {"error": f"Unknown failure_reason '{failure_reason}'", "valid": cats["failure_reason"]}
    try:
        p = FailedPayment(plan_type=plan_type, amount=amount, tenure_months=tenure_months,
                          failure_reason=failure_reason, past_failures_90d=past_failures_90d,
                          day_of_week=day_of_week, hour_of_day=hour_of_day,
                          is_annual_plan=is_annual_plan)
    except Exception as e:
        return {"error": f"Invalid input: {e}"}
    return {"recovery_probability": round(_predict_prob(p), 3)}


def get_expected_recovery_value(failure_reason: Optional[str] = None,
                                plan_type: Optional[str] = None) -> dict:
    """Among currently unrecovered payments (optionally filtered by failure_reason and/or
    plan_type): how many there are, their total value in INR, and the expected INR value
    that would come back if they were retried (sum of probability x amount). Use for
    'how much could we win back', 'where is the biggest opportunity'."""
    f = df[df["recovered"] == 0]
    if failure_reason: f = f[f["failure_reason"] == failure_reason]
    if plan_type: f = f[f["plan_type"] == plan_type]
    if f.empty:
        return {"unrecovered_payments": 0}
    probs = _score_frame(f)
    return {
        "unrecovered_payments": int(len(f)),
        "total_value_inr": round(float(f["amount"].sum()), 2),
        "expected_recoverable_inr": round(float((probs * f["amount"].values).sum()), 2),
        "avg_recovery_probability": round(float(probs.mean()), 3),
    }


def get_pending_payments_summary() -> dict:
    """Payments logged as pending (outcome not yet confirmed): count, total value in INR,
    expected number recovered, and up to 10 entries."""
    if not pending_records:
        return {"pending_count": 0}
    amounts = [float(r.get("amount", 0)) for r in pending_records]
    probs = [float(r.get("recovery_probability", 0)) for r in pending_records]
    return {
        "pending_count": len(pending_records),
        "total_value_inr": round(sum(amounts), 2),
        "expected_recovered": round(sum(probs), 1),
        "entries": [{k: r.get(k) for k in ("pending_id", "customer_id", "plan_type", "amount",
                                           "failure_reason", "recovery_probability", "created_at")}
                    for r in pending_records[:10]],
    }


_model_info_cache = {}


def get_model_info() -> dict:
    """About the prediction model: which algorithm is deployed, training rows, held-out AUC
    on the most recent payments (how well it ranks recoverable vs unrecoverable payments;
    0.5 = random, 1.0 = perfect), comparison with simpler models, and which features matter
    most (drop in AUC when shuffled). Use for 'how accurate is the model', 'what drives recovery'."""
    key = (id(model), len(df))
    if key in _model_info_cache:
        return _model_info_cache[key]
    r = model_report
    info = {
        "algorithm": MODEL_LABELS.get(r.get("selected_model"), "unknown"),
        "training_rows": int(len(df)),
    }
    if r.get("evaluated"):
        info.update({
            "holdout_auc": r["models"][r["selected_model"]]["auc"],
            "holdout_auc_95_interval": r.get("auc_ci_95"),
            "evaluation_split": r["split"]["kind"],
            "all_models_holdout_auc": {MODEL_LABELS[k]: v["auc"] for k, v in r["models"].items()},
            "calibration_error": r.get("ece"),
            "feature_importance_auc_drop": {k: v["auc_drop"] for k, v in r["feature_importance"].items()},
            "top_30pct_recovery_rate": r["lift"]["recovery_rate_in_top"],
            "overall_recovery_rate": r["lift"]["recovery_rate_overall"],
        })
        if "oracle_auc" in r:
            info["simulated_data_ceiling_auc"] = r["oracle_auc"]
    else:
        info["holdout_auc"] = None
    if r.get("warnings"):
        info["warnings"] = r["warnings"]
    _model_info_cache[key] = info
    return info


ASK_TOOLS = [
    get_dataset_overview, get_summary_metrics, get_revenue_by_failure_reason,
    get_recovery_rate_by, find_customers, get_top_at_risk_customers,
    get_lowest_recovery_customers, get_customer, predict_recovery,
    get_expected_recovery_value, get_pending_payments_summary, get_model_info,
    *rv.CALC_TOOLS,     # exact finance arithmetic (NPV, IRR, EMI, CAGR, LTV ...)
]

# ---- Rivera, the assistant ----
# Three tiers, fastest first (see backend/rivera.py):
#   1. instant   common ledger questions, computed from the data and phrased locally
#   2. fast LLM  general-knowledge and finance questions: light model, calculators only
#   3. full LLM  anything about the ledger's own numbers: full model + the data tools
RIVERA_MODEL = os.environ.get("RIVERA_MODEL", "gemini-flash-latest")
RIVERA_FAST_MODEL = os.environ.get("RIVERA_FAST_MODEL", "gemini-flash-lite-latest")


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=1500)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=600)
    page: Literal["website", "dashboard"] = "dashboard"
    history: list[HistoryTurn] = Field(default_factory=list, max_length=12)


def _rivera_instant(question: str, page: str) -> Optional[dict]:
    """Answers the common questions straight from the data, with no model call. Returns
    None when the question isn't one of them (or anything goes wrong), so the caller
    carries on to Gemini."""
    intent = rv.match_intent(question)
    if intent is None:
        return None
    try:
        if intent == "greeting":
            return rv.answer_greeting(page)
        if intent == "thanks":
            return rv.answer_thanks()
        if intent == "help":
            return rv.answer_help(page)
        if intent == "top_loss":
            return rv.answer_top_loss(get_revenue_by_failure_reason())
        if intent == "worst_plan":
            return rv.answer_worst_plan(get_recovery_rate_by("plan_type"))
        if intent == "best_plan":
            return rv.answer_best_plan(get_recovery_rate_by("plan_type"))
        if intent == "biggest_unrecovered":
            return rv.answer_biggest_unrecovered(
                find_customers("unrecovered", sort_by="amount", ascending=False, limit=5))
        if intent == "model_accuracy":
            return rv.answer_model_accuracy(get_model_info())
        if intent == "best_day":
            return rv.answer_best_day(get_recovery_rate_by("day_of_week"))
        if intent == "headline":
            return rv.answer_headline(get_summary_metrics())
    except Exception:
        logger.exception("Rivera instant answer failed; falling back to the model")
    return None


_caps_by_model: dict = {}     # model name -> rv.Capabilities (which optional extras it accepts)


def _gemini_text(model_name: str, system: str, tools: list, prompt: str) -> str:
    """One Gemini round trip; Gemini runs any tool calls itself. Quick by design: thinking is
    switched off where the model allows it, and a temporary 503 is retried within about two
    seconds rather than ten. If the model rejects an optional extra (a bare 400), rv.call_with_
    fallbacks drops it and remembers. Raises if the call ultimately fails."""
    caps = _caps_by_model.setdefault(model_name, rv.Capabilities())

    def call(thinking_off: bool, tool_list: list) -> str:
        kwargs = {"system_instruction": system}
        if tool_list:
            kwargs["tools"] = tool_list
        if thinking_off and hasattr(genai_types, "ThinkingConfig"):
            kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
        for attempt in range(3):
            try:
                resp = _gemini_client.models.generate_content(
                    model=model_name, contents=prompt,
                    config=genai_types.GenerateContentConfig(**kwargs))
                return (resp.text or "").strip()
            except Exception as e:
                if attempt < 2 and ("503" in str(e) or "UNAVAILABLE" in str(e)):
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise

    return rv.call_with_fallbacks(call, tools, caps, log=logger.warning)


_ask_cache: dict = {}
_ASK_CACHE_TTL = 120
_ASK_CACHE_MAX = 200


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    """Rivera. Answers a question about the ledger, the website, finance or general
    knowledge. Returns {"answer", "generated_by", "actions", "ms"}."""
    started = time.perf_counter()

    def reply(body: dict, status: int = 200):
        body.setdefault("actions", [])
        body["ms"] = int((time.perf_counter() - started) * 1000)
        return JSONResponse(status_code=status, content=body)

    question = " ".join(req.question.split())
    if not question:
        return reply({"answer": "Ask me something and I'll look into it.", "generated_by": "instant"})

    # 1. instant: no model, no key, no waiting
    if _rivera_instant_limiter.allow(_visitor(request))[0]:
        instant = _rivera_instant(question, req.page)
        if instant is not None:
            return reply({**instant, "generated_by": "instant"})
    else:
        return reply({"answer": "That's a lot of questions at once. Give me a few seconds.",
                      "generated_by": "rate_limited"}, 429)

    if _gemini_client is None or genai_types is None:
        return reply({
            "answer": ("I can answer the common ledger questions instantly, but anything open-ended "
                       "needs a Gemini key on the server (GEMINI_API_KEY). Try one of the suggested "
                       "questions."),
            "generated_by": "unavailable"})

    cache_key = (req.page, question.lower(), id(model), len(df))
    if not req.history:
        hit = _ask_cache.get(cache_key)
        if hit and hit[0] > time.time():
            return reply({"answer": hit[1], "generated_by": "cache"})

    if not _llm_allowed(request):
        return reply({"answer": "Rivera is rate-limited on this public demo. Please wait a minute and try again.",
                      "generated_by": "rate_limited"}, 429)

    prompt = rv.build_prompt(question, [t.model_dump() for t in req.history])
    try:
        text, used = "", "gemini"
        if not rv.needs_data(question, req.page):
            try:                                   # 2. fast: light model, no ledger tools
                text = _gemini_text(RIVERA_FAST_MODEL, rv.system_prompt_general(req.page),
                                    rv.CALC_TOOLS, prompt)
                used = "gemini-fast"
            except Exception as e:
                if not any(c in str(e) for c in ("404", "NOT_FOUND", "not found")):
                    raise
                logger.warning("Rivera fast model unavailable (%s); using the full model", RIVERA_FAST_MODEL)
                text = rv.NEEDS_DATA              # the alias is gone: let the full model handle it
        if not text or text.startswith(rv.NEEDS_DATA):     # 3. full: ledger tools
            text = _gemini_text(RIVERA_MODEL, rv.system_prompt_data(req.page), ASK_TOOLS, prompt)
            used = "gemini"
        text = text.replace(rv.NEEDS_DATA, "").strip()
        if not text:
            return reply({"answer": "I didn't get an answer back. Try rephrasing the question.",
                          "generated_by": "gemini"})
        if not req.history:
            if len(_ask_cache) >= _ASK_CACHE_MAX:
                _ask_cache.clear()
            _ask_cache[cache_key] = (time.time() + _ASK_CACHE_TTL, text)
        return reply({"answer": text, "generated_by": used})
    except Exception as e:
        logger.warning("Rivera /ask failed: %s", e)
        busy = any(code in str(e) for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
        if busy:
            msg = "Gemini is busy or over its quota right now. Please try again in a moment."
        elif PUBLIC_MODE:
            msg = "Something went wrong answering that. Please try again or rephrase."
        else:
            msg = f"Something went wrong calling Gemini: {e}"
        return reply({"answer": msg, "generated_by": "error"})


# ---- static sites: dashboard at /dashboard, landing page at / ----
# Mounted last so every API route above takes priority. Only these two folders are
# served; data/, model/ and backend/ are never exposed.
DASHBOARD_DIR = ROOT / "frontend"
WEBSITE_DIR = ROOT / "website"
if DASHBOARD_DIR.is_dir():
    app.mount("/dashboard", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
if WEBSITE_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEBSITE_DIR, html=True), name="website")
