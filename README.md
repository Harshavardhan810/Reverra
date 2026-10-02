# Reverra

An end-to-end ML project that predicts which failed subscription payments
are recoverable, and drafts a personalized recovery email for each one.

Built as a portfolio project to demonstrate a full pipeline: synthetic data
generation → model training → a FastAPI backend → a live dashboard, with
retraining and LLM integration wired in end-to-end.

## Features
- **Recovery prediction** — a gradient-boosted classifier (scikit-learn) trained on failed-payment data, predicting the probability a payment is recoverable if retried
- **Live dashboard** — revenue-at-risk / recovered / recovery-rate metrics, an at-risk customer table ranked by recovery probability, and a revenue-by-failure-reason breakdown chart
- **AI-drafted recovery emails** — generated via Google's Gemini API when a key is configured, with an automatic fallback to a template generator if no key is set or the API call fails
- **Upload & retrain** — drop in your own CSV of failed payments from the dashboard and the model retrains on it immediately, live
- **Regenerate demo data** — one click to refresh with a new synthetic batch
- **Light/dark theme toggle**

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
3. Generate synthetic data and train the model (data/ and model/ files are gitignored, so this step is required after cloning):
   ```bash
   python data/generate_data.py
   python model/train_model.py
   ```
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
6. Open `frontend/index.html` in your browser (static file, no build step — calls `http://localhost:8000`).

## API endpoints
| Endpoint | Description |
|---|---|
| `GET /metrics` | Dashboard summary cards |
| `GET /revenue-breakdown` | Revenue recovered vs lost, by failure reason |
| `GET /at-risk?limit=20` | At-risk customers ranked by recovery probability |
| `POST /predict` | Recovery probability for a single failed payment |
| `POST /generate-message` | Dunning email for a single failed payment |
| `POST /simulate` | Predict + generate message together |
| `POST /upload-data` | Upload a CSV, retrain immediately |
| `POST /regenerate-data` | Generate a fresh synthetic batch and retrain |

## Project structure
```
data/       synthetic data generator
model/      training script + saved model (gitignored)
backend/    FastAPI app
frontend/   static dashboard (HTML/CSS/JS)
```

## Possible next steps
- Retry-timing recommender (best day/hour to retry, using the `day_of_week`/`hour_of_day` features already in the data)
- Model explainability (feature importances / SHAP) to show why a prediction was made
- Swap synthetic data for a real dataset — the training script is dataset-agnostic as long as column names match
- Deploy the backend (Render/Railway/Fly.io) so the dashboard isn't localhost-only
