# Reverra

An end-to-end ML project that predicts which failed subscription payments
are recoverable, and drafts a personalized recovery email for each one.

Built as a portfolio project to demonstrate a full pipeline: simulated data
generation → honest model evaluation → a FastAPI backend → a live dashboard,
with retraining and LLM integration wired in end-to-end.

> **About the data:** no public dataset of real failed-payment outcomes exists
> (that data lives inside payment processors). Reverra therefore trains on
> **simulated** data, calibrated to published industry recovery rates (roughly
> 20-40%) and deliberately noisy. Every number below describes that simulation,
> not real customers. The column layout is designed so real data can be dropped in.

## Features
- **Recovery prediction** — a scikit-learn pipeline predicting the probability a failed payment is recovered. Logistic regression and gradient boosting are compared on a validation slice and the better one is deployed; unseen categories and missing values are handled safely
- **Honest evaluation** — time-based train/test split, baseline and model comparison, bootstrap confidence interval, calibration check, permutation feature importance, and an auto-generated model card (`model/model_card.md`)
- **Live dashboard** — revenue-at-risk / recovered / recovery-rate metrics, an at-risk customer table ranked by recovery probability, and a revenue-by-failure-reason breakdown chart
- **AI-drafted recovery emails** — generated via Google's Gemini API when a key is configured, with an automatic fallback to a template generator if no key is set or the API call fails
- **Rivera, the floating assistant** — a chat button on both the website and the dashboard (bottom-right, not a section at the bottom of the page). Rivera answers in three tiers, fastest first:
  1. **Instant** — common ledger questions (which failure reason loses the most, which plan recovers worst, biggest unrecovered invoices, model accuracy, best day to retry) are computed straight from the data and phrased locally: no model call, no API key needed, tens of milliseconds
  2. **Fast model** — general-knowledge and finance questions go to a light Gemini model with no ledger tools (and a set of exact calculators for NPV, IRR, EMI, CAGR, compound growth, LTV/CAC and break-even, so the arithmetic is never guessed)
  3. **Full model** — anything about the ledger's own numbers goes to the full model with the data tools, so figures are looked up, never invented

  If Gemini rejects an optional extra in a request (the "thinking off" setting or the finance calculators) with a bare `400 INVALID_ARGUMENT`, Rivera retries without it, remembers, and logs which one in the server console, so a model or SDK change degrades gracefully instead of breaking the chat.

  Rivera also helps people **get around**: questions like "how do I upload data?" or "show me the dashboard" are answered in the browser and come with buttons that scroll to the right section or open the right page. Instant answers and navigation work without Gemini; open-ended questions need `GEMINI_API_KEY`.
- **Pending → confirm loop** — log a new failed payment as pending; only once its real outcome is confirmed does it join the training data and trigger a retrain
- **Upload & retrain** — drop in your own CSV of failed payments from the dashboard and the model retrains on it immediately, live
- **Regenerate demo data** — one click to refresh with a new synthetic batch
- **Light/dark theme toggle**

## Model results (simulated data, seed 42)

Trained on the oldest 80% of payments and tested on the most recent 20%, the way a
model is used in production (a random split can leak information about the future).

| Model | AUC (held-out) |
|---|---|
| Baseline (always predicts the average) | 0.500 |
| Logistic regression (deployed) | 0.691 (95% CI 0.661-0.720) |
| Gradient boosting | 0.669 |
| Ceiling: a model that knew the true probabilities | 0.712 |

- The model lands close to the ceiling. The simulation includes a hidden customer
  factor and random outcomes, so no model can do much better.
- Boosting did **not** beat logistic regression here, so the simpler model is deployed.
  The effects in the simulation are mostly additive, which a linear model captures well.
- Retrying only the top 30% of payments by score reaches a ~53% recovery rate
  (overall: ~36%) and captures ~45% of all recoveries.
- Calibration error is ~0.04. Across most score bands the actual recovery rate runs about
  0.04 above the prediction, consistent with the mild upward drift built into the
  simulation (the test period is the most recent). The top bands have very few payments,
  so they are noisy.
- `failure_reason` is by far the most important feature; `day_of_week` and
  `hour_of_day` contribute almost nothing.

Full details: `model/model_card.md`, `model/metrics.json`, `model/calibration.png`, `model/importance.png`.

## Tech stack
Python · FastAPI · scikit-learn · pandas · vanilla JS/HTML/CSS (no frontend build step) · Google Gemini API

## Setup

1. Clone the repo and open it in your editor.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Generate the simulated data and train the model (the CSV and `.pkl` are gitignored, so this step is required after cloning):
   ```bash
   python data/generate_data.py
   python model/train_model.py
   ```
   Training prints the evaluation and writes `model/model_card.md`, `metrics.json` and two plots.
   (Optional) `pip install -r requirements-dev.txt`, then run the tests with `pytest -q`. The widget's own logic has a dependency-free test: `node tests/rivera_widget.test.js`.
4. (Optional) Enable real AI-generated messages — get a free key at
   [aistudio.google.com/apikey](https://aistudio.google.com/apikey), no card required:
   ```bash
   export GEMINI_API_KEY="your-key-here"    # Windows PowerShell: $env:GEMINI_API_KEY = "your-key-here"
   ```
   Without this set, the app works identically but uses template-generated messages.
5. Start the backend:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
6. Open **http://localhost:8000** — the same server serves the landing page (`/`) and the dashboard (`/dashboard`). (Opening `website/index.html` or `frontend/index.html` straight from disk also works; the dashboard then talks to `http://localhost:8000`.)

## API endpoints
| Endpoint | Description |
|---|---|
| `GET /health` | Status, whether Gemini is on, and whether the server is in read-only (public) mode |
| `GET /metrics` | Dashboard summary cards |
| `GET /revenue-breakdown` | Revenue recovered vs lost, by failure reason |
| `GET /at-risk?limit=20` | At-risk customers ranked by recovery probability |
| `POST /predict` | Recovery probability for a single failed payment |
| `POST /generate-message` | Dunning email for a single failed payment |
| `POST /simulate` | Predict + generate message together |
| `POST /upload-data` | Upload a CSV, retrain immediately. Optional `failed_at` column enables the time-based split. Returns held-out AUC, the chosen model, and any warnings. A failed upload never overwrites your data |
| `POST /regenerate-data` | Generate a fresh simulated batch and retrain |
| `POST /ask` | Rivera. Body: `{question, page: "website"\|"dashboard", history: [{role, text}]}` (last 6 turns kept). Returns `{answer, generated_by, actions, ms}` where `generated_by` is `instant`, `gemini-fast`, `gemini`, `cache`, `unavailable`, `rate_limited` or `error` |
| `POST /add-pending` | Log a failed payment as pending (outcome unknown) |
| `GET /pending` | List payments awaiting outcome confirmation |
| `POST /confirm-pending` | Confirm the outcome, add to training data, retrain |
| `DELETE /pending/{id}` | Discard a pending entry |

Endpoints that change data (`/upload-data`, `/regenerate-data`, `/add-pending`, `/confirm-pending`, `DELETE /pending/{id}`) need the `X-Admin-Token` header when the server runs in public mode (see below). `/ask`, `/simulate` and `/generate-message` are rate-limited when Gemini is on.

## Deploying publicly
One deployment serves everything: landing page at `/`, dashboard at `/dashboard`, API at the same origin (no CORS setup).

**Render (free tier):** push the repo to GitHub, then in Render choose *New → Blueprint* and select the repo. `render.yaml` does the rest: installs dependencies, generates the simulated data, trains the model, starts the server, and creates a random `ADMIN_TOKEN`. When it asks for `GEMINI_API_KEY`, paste your key (or leave it blank to use template messages only). If your repo has this project in a subfolder, add `rootDir: <subfolder>` under the service in `render.yaml`.

**Anywhere else:** build with `pip install -r requirements.txt && python data/generate_data.py && python model/train_model.py`, start with `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`, and set the variables below. (If the data or model files are missing at start-up the server creates them itself, just more slowly.)

Run **one** worker: the dataset and pending list are held in this process.

| Variable | Purpose |
|---|---|
| `ADMIN_TOKEN` | Setting it turns on **public mode**: visitors get a read-only demo, data-changing endpoints require the token, `/docs` is hidden. Use 16+ random characters. Unset = everything open (local dev) |
| `GEMINI_API_KEY` | Enables Gemini emails and Q&A. Set it as an environment variable, never in the code |
| `CORS_ORIGINS` | Default: same-origin only in public mode, any origin locally. Comma-separated list to allow others |
| `LLM_RATE_PER_MIN` / `LLM_DAILY_CAP` | Gemini calls per visitor per minute (8) and per day across everyone (300) |
| `RIVERA_MODEL` / `RIVERA_FAST_MODEL` | Gemini models for ledger questions (`gemini-flash-latest`) and for general / finance questions (`gemini-flash-lite-latest`). Override both if Google retires an alias |
| `RIVERA_INSTANT_PER_MIN` | Instant (no-model) answers per visitor per minute (60) |
| `MAX_UPLOAD_MB` | Largest CSV `/upload-data` accepts (10) |
| `TRUSTED_PROXY_HOPS` | Proxies in front of the app, used to identify visitors for rate limits (1) |

To use the admin actions on the live site, open the dashboard and click **Read-only demo — unlock**, then paste the token (kept in that browser tab only).

**Before you go public:** this setup is meant for the *simulated* data. Anyone can read the dashboard, so don't load real customer records into a public instance. On free hosts the server sleeps when idle (first visit can take up to a minute) and the disk is wiped on restart, so the pending list resets and the simulated dataset is regenerated.

## Project structure
```
ml/         simulator (ml/simulate.py) + training/evaluation pipeline (ml/pipeline.py)
data/       data generator script
model/      training script; model card, metrics and plots (the .pkl is gitignored)
backend/    FastAPI app (main.py) + public-deployment helpers (security.py)
frontend/   static dashboard (HTML/CSS/JS)
website/    static marketing/docs site, plus rivera.js (the assistant widget, loaded by the dashboard too)
tests/      pytest tests for the simulator, pipeline, API and Rivera (+ one node test for the widget)
```

## Possible next steps
- Retry-timing recommender (best day/hour to retry, using the `day_of_week`/`hour_of_day` features already in the data)
- Flexible mode: map the columns of any CSV with a yes/no outcome, with an automatic data check, so the pipeline runs on other datasets (e.g. churn)
- Per-prediction explanations (e.g. SHAP); today the model card shows global permutation importance
- Swap simulated data for a real dataset — the pipeline only needs the same columns (a `failed_at` timestamp is optional but recommended)
- Persistent storage (a database or disk) so pending payments and uploaded data survive restarts on a host
