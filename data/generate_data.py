"""
Generates a synthetic dataset of failed subscription payments for the
Reverra portfolio project.

Each row = one failed payment event. Target = whether it was eventually
recovered (via retry, dunning email, etc).

Run: python generate_data.py
Output: data/failed_payments.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

failure_reasons = [
    "insufficient_funds", "expired_card", "card_declined",
    "bank_error", "fraud_flag", "processing_error"
]
# base recovery likelihood per failure reason (before other factors)
reason_base_recovery = {
    "insufficient_funds": 0.55,
    "expired_card": 0.35,
    "card_declined": 0.45,
    "bank_error": 0.65,
    "fraud_flag": 0.15,
    "processing_error": 0.70,
}

plan_types = ["basic", "standard", "premium", "enterprise"]
# amounts in INR (₹)
plan_amount_range = {
    "basic": (400, 1600),
    "standard": (1600, 5000),
    "premium": (5000, 12000),
    "enterprise": (12000, 40000),
}

rows = []
for i in range(N):
    reason = np.random.choice(failure_reasons, p=[0.30, 0.20, 0.20, 0.10, 0.05, 0.15])
    plan = np.random.choice(plan_types, p=[0.35, 0.35, 0.20, 0.10])
    amount = round(np.random.uniform(*plan_amount_range[plan]), 2)
    tenure_months = max(0, int(np.random.exponential(scale=14)))
    past_failures_90d = np.random.poisson(0.6)
    day_of_week = np.random.randint(0, 7)  # 0=Mon
    hour_of_day = np.random.randint(0, 24)
    is_annual_plan = np.random.choice([0, 1], p=[0.8, 0.2])

    # --- simulate recovery probability from realistic factors ---
    p = reason_base_recovery[reason]
    p += 0.15 * np.tanh(tenure_months / 24)          # loyal customers recover more
    p -= 0.08 * min(past_failures_90d, 4)            # repeat failures = less recoverable
    p -= 0.05 if amount > 12000 else 0                # big invoices harder to recover
    p += 0.05 if is_annual_plan else 0                # annual customers more invested
    p = np.clip(p, 0.02, 0.95)

    recovered = np.random.rand() < p

    rows.append(dict(
        customer_id=f"CUST-{i:05d}",
        plan_type=plan,
        amount=amount,
        tenure_months=tenure_months,
        failure_reason=reason,
        past_failures_90d=past_failures_90d,
        day_of_week=day_of_week,
        hour_of_day=hour_of_day,
        is_annual_plan=is_annual_plan,
        recovered=int(recovered),
    ))

df = pd.DataFrame(rows)
df.to_csv("data/failed_payments.csv", index=False)
print(f"Generated {len(df)} rows -> data/failed_payments.csv")
print(f"Recovery rate: {df['recovered'].mean():.2%}")
print(df.head())
