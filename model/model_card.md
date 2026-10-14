# Reverra recovery model: model card

*Trained: 2026-10-13T04:41:22+00:00*  
*Rows: 6,000 | recovery rate: 33.6%*

## What it does
Predicts the probability that a failed subscription payment is eventually recovered, from the plan, amount, tenure, failure reason, past failures and timing.

## How it was evaluated
Time-based split: trained on 4,800 older payments (2025-10-01 to 2026-07-21), tested on the 1,200 most recent (2026-07-21 to 2026-09-30).

| Model | AUC | PR-AUC | Brier | Log loss |
|---|---|---|---|---|
| Baseline (always predicts the average) | 0.500 | 0.355 | 0.230 | 0.652 |
| Logistic regression **(deployed)** | 0.691 | 0.531 | 0.207 | 0.601 |
| Gradient boosting | 0.669 | 0.503 | 0.212 | 0.611 |

The deployed model was chosen on a validation slice of the training data (never the test set): logistic regression 0.685 vs gradient boosting 0.673; boosting must win by more than 0.005 to be picked.

95% bootstrap interval for the deployed model's test AUC: 0.661 to 0.720.
Simulated-data ceiling (a model that knew the true probabilities): AUC 0.712.

## Business view
Retrying only the top 30% of payments by score would reach a 53.3% recovery rate (overall: 35.5%) and capture 45% of all recoveries.

## Calibration
Expected calibration error: 0.043 (0 = predicted probabilities match reality).

| Score band | Payments | Mean predicted | Actually recovered |
|---|---|---|---|
| 0.0-0.1 | 34 | 0.07 | 0.09 |
| 0.1-0.2 | 171 | 0.16 | 0.12 |
| 0.2-0.3 | 374 | 0.25 | 0.29 |
| 0.3-0.4 | 292 | 0.35 | 0.39 |
| 0.4-0.5 | 180 | 0.45 | 0.49 |
| 0.5-0.6 | 110 | 0.54 | 0.59 |
| 0.6-0.7 | 30 | 0.63 | 0.73 |
| 0.7-0.8 | 8 | 0.73 | 0.38 |
| 0.8-0.9 | 1 | 0.87 | 1.00 |

## What drives predictions
Permutation importance on held-out data (drop in AUC when a feature is shuffled; ~0 means it doesn't help):

- `failure_reason`: +0.1169 (+/-0.0113)
- `plan_type`: +0.0343 (+/-0.0047)
- `tenure_months`: +0.0324 (+/-0.0069)
- `past_failures_90d`: +0.0308 (+/-0.0066)
- `amount`: +0.0106 (+/-0.0029)
- `is_annual_plan`: +0.0072 (+/-0.0020)
- `day_of_week`: +0.0012 (+/-0.0004)
- `hour_of_day`: -0.0007 (+/-0.0009)

## Limits
- Trained on whatever dataset was loaded. If that is the built-in simulation, results describe the simulation, not real customers.
- Predicts recovery probability; it does not prove that any action (an email, a retry time) causes recovery.
