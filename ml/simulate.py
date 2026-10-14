"""
Calibrated simulator for failed subscription payments.

IMPORTANT: this is SIMULATED data, not real transactions. It is calibrated to
published industry figures (well-run dunning typically recovers roughly 20-40%
of failed payments), and is deliberately messy:

  * overall recovery rate lands around 30%, not ~50%
  * effects overlap and interact (e.g. expired cards recover better on annual plans)
  * a hidden "customer engagement" factor the model can NOT see, partly
    correlated with tenure, which caps how accurate any model can be
  * mild drift over time (recovery slowly improves), so a time-based split
    is meaningful
  * the label comes from a random draw, so even a perfect model is imperfect

Because we know the true probability behind each label, `generate()` also
returns the "oracle AUC": the best AUC any model could possibly reach on this
sample. A trained model's AUC should be read against that ceiling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

END_DATE = pd.Timestamp("2026-09-30")   # fixed so seeded runs are reproducible
DAYS_SPAN = 365

FAILURE_REASONS = ["insufficient_funds", "expired_card", "card_declined",
                   "bank_error", "fraud_flag", "processing_error"]
REASON_PROBS = [0.30, 0.20, 0.20, 0.10, 0.05, 0.15]
PLAN_TYPES = ["basic", "standard", "premium", "enterprise"]
PLAN_PROBS = [0.35, 0.35, 0.20, 0.10]
# amounts in INR
PLAN_AMOUNT_RANGE = {
    "basic": (400, 1600), "standard": (1600, 5000),
    "premium": (5000, 12000), "enterprise": (12000, 40000),
}
# baseline log-odds of recovery per failure reason. Transient/technical failures
# come back easily; fraud flags and expired cards rarely do.
REASON_LOGIT = {
    "processing_error": -0.15,
    "bank_error": -0.35,
    "insufficient_funds": -0.85,
    "card_declined": -1.15,
    "expired_card": -1.55,
    "fraud_flag": -2.30,
}

FEATURE_ORDER = ["customer_id", "failed_at", "plan_type", "amount", "tenure_months",
                 "failure_reason", "past_failures_90d", "day_of_week",
                 "hour_of_day", "is_annual_plan", "recovered"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _oracle_auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def generate(n: int = 6000, seed: int | None = None, return_truth: bool = False):
    """Generate `n` simulated failed payments.

    seed=None gives a different sample every call (used by the dashboard's
    "regenerate" button). Returns a DataFrame, or (DataFrame, meta dict) when
    return_truth=True; meta holds the base rate and the oracle-AUC ceiling.
    """
    rng = np.random.default_rng(seed)

    reason = rng.choice(FAILURE_REASONS, size=n, p=REASON_PROBS)
    plan = rng.choice(PLAN_TYPES, size=n, p=PLAN_PROBS)
    lo = np.array([PLAN_AMOUNT_RANGE[p][0] for p in plan], dtype=float)
    hi = np.array([PLAN_AMOUNT_RANGE[p][1] for p in plan], dtype=float)
    amount = np.round(rng.uniform(lo, hi), 2)

    tenure = np.maximum(0, rng.exponential(scale=14, size=n)).astype(int)
    past_failures = rng.poisson(0.6, size=n)
    is_annual = rng.choice([0, 1], size=n, p=[0.8, 0.2])

    # timestamps: uniform over the last year; billing batches cluster in the early hours
    day_offsets = rng.integers(0, DAYS_SPAN, size=n)
    early = rng.random(n) < 0.6
    hours = np.where(early, rng.integers(0, 6, size=n), rng.integers(0, 24, size=n))
    failed_at = (END_DATE - pd.to_timedelta(DAYS_SPAN - 1 - day_offsets, unit="D")
                 + pd.to_timedelta(hours, unit="h"))
    failed_at = pd.DatetimeIndex(failed_at)
    day_of_week = failed_at.dayofweek.to_numpy()
    hour_of_day = failed_at.hour.to_numpy()
    progress = day_offsets / (DAYS_SPAN - 1)          # 0 = a year ago, 1 = today

    # hidden factor the model never sees: how engaged / attentive the customer is.
    # Partly predictable from tenure, but mostly not.
    engagement = 0.35 * np.tanh(tenure / 24) + rng.normal(0, 0.9, size=n)

    logit = np.array([REASON_LOGIT[r] for r in reason])
    logit += 0.80 * np.tanh(tenure / 30)                     # loyal customers recover more
    logit -= 0.40 * np.minimum(past_failures, 4)             # repeat failures recover less
    logit += 0.20 * is_annual                                # annual customers are more invested
    logit -= 0.25 * (amount > 12000)                         # big invoices are harder
    # interactions
    logit += 0.55 * ((reason == "expired_card") & (is_annual == 1))   # annual plans update cards
    logit += 0.30 * ((reason == "insufficient_funds") & (day_of_week >= 5))  # weekend/payday timing
    logit -= 0.45 * ((reason == "fraud_flag") & (amount > 5000))      # big flagged charges stay blocked
    logit += 0.20 * ((reason == "bank_error") & (past_failures == 0))
    logit += 0.50 * engagement                               # hidden driver
    logit += 0.35 * (progress - 0.5)                         # recovery slowly improves over the year

    p_true = _sigmoid(logit)
    recovered = (rng.random(n) < p_true).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:05d}" for i in range(n)],
        "failed_at": failed_at,
        "plan_type": plan,
        "amount": amount,
        "tenure_months": tenure,
        "failure_reason": reason,
        "past_failures_90d": past_failures,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "is_annual_plan": is_annual,
        "recovered": recovered,
    })[FEATURE_ORDER]
    df["failed_at"] = df["failed_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if not return_truth:
        return df
    meta = {
        "rows": int(n),
        "seed": seed,
        "base_rate": round(float(recovered.mean()), 4),
        "oracle_auc": round(_oracle_auc(recovered, p_true), 4),
        "note": "Simulated data. oracle_auc = best AUC any model could reach on this sample.",
    }
    return df, meta
