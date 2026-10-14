"""
Training + evaluation pipeline for Reverra's recovery model.

Design goals
  * One sklearn Pipeline (preprocessing + classifier) is the saved model, so the
    exact same transformations run in training and in the API. No hand-rolled
    label encoders; unseen categories and missing values don't crash anything.
  * Honest evaluation: a TIME-based split when timestamps exist (train on the
    past, test on the future), a simple baseline, a logistic-regression
    comparison, calibration, a bootstrap confidence interval, and permutation
    importance on the held-out set.
  * The model that is deployed is refit on all rows AFTER evaluation; the
    reported numbers always come from data the evaluated model never saw.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["plan_type", "failure_reason"]
NUM_COLS = ["amount", "tenure_months", "past_failures_90d",
            "day_of_week", "hour_of_day", "is_annual_plan"]
FEATURE_COLS = CAT_COLS + NUM_COLS
TARGET = "recovered"
TIME_COL = "failed_at"

GBM_PARAMS = dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=30, random_state=42)

MIN_ROWS_FOR_EVAL = 200      # below this we train but skip the held-out evaluation
MIN_CLASS_FOR_EVAL = 20      # need at least this many of each class overall


# --------------------------------------------------------------------------- features
def prepare_features(frame: pd.DataFrame, cat_cols=CAT_COLS, num_cols=NUM_COLS) -> pd.DataFrame:
    """Returns a clean copy with exactly the feature columns: categoricals as
    strings, numerics coerced to numbers (bad values become NaN and get imputed)."""
    missing = [c for c in list(cat_cols) + list(num_cols) if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = frame[list(cat_cols) + list(num_cols)].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def _preprocessor(cat_cols, num_cols, scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)),
        ("num", Pipeline(num_steps), list(num_cols)),
    ])


def make_gbm(cat_cols=CAT_COLS, num_cols=NUM_COLS) -> Pipeline:
    return Pipeline([("prep", _preprocessor(cat_cols, num_cols, scale=False)),
                     ("clf", GradientBoostingClassifier(**GBM_PARAMS))])


def make_logreg(cat_cols=CAT_COLS, num_cols=NUM_COLS) -> Pipeline:
    return Pipeline([("prep", _preprocessor(cat_cols, num_cols, scale=True)),
                     ("clf", LogisticRegression(max_iter=1000))])


def make_baseline(cat_cols=CAT_COLS, num_cols=NUM_COLS) -> Pipeline:
    # always predicts the training recovery rate; AUC is 0.5 by construction
    return Pipeline([("prep", _preprocessor(cat_cols, num_cols, scale=False)),
                     ("clf", DummyClassifier(strategy="prior"))])


# --------------------------------------------------------------------------- inference
def _cols_of(model: Pipeline):
    """(cat_cols, num_cols) a fitted pipeline was trained with."""
    spec = {name: cols for name, _, cols in model.named_steps["prep"].transformers_
            if name in ("cat", "num")}
    return list(spec["cat"]), list(spec["num"])


def predict_proba(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """P(recovered) for each row of a raw dataframe."""
    cat_cols, num_cols = _cols_of(model)
    return model.predict_proba(prepare_features(frame, cat_cols, num_cols))[:, 1]


def known_categories(model: Pipeline) -> dict:
    """Category values the model saw in training, e.g. {'plan_type': [...]}."""
    cat_cols, _ = _cols_of(model)
    enc = model.named_steps["prep"].named_transformers_["cat"]
    return {c: [str(v) for v in cats] for c, cats in zip(cat_cols, enc.categories_)}


def _parse_times(series: pd.Series) -> pd.Series:
    """Parses timestamps leniently to naive UTC; unparseable values become NaT."""
    try:
        t = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):          # older pandas without format="mixed"
        t = pd.to_datetime(series, errors="coerce", utc=True)
    return t.dt.tz_convert(None)


# --------------------------------------------------------------------------- splitting
def split_train_test(df: pd.DataFrame, target=TARGET, time_col=TIME_COL,
                     test_frac: float = 0.2, random_state: int = 42):
    """Returns (train_positions, test_positions, kind).

    kind == "time": test set is the most recent `test_frac` of rows. Rows with an
    unknown timestamp are treated as the newest (they are fresh confirmations).
    kind == "random": no usable timestamps, so a stratified random split.
    """
    n = len(df)
    n_test = max(1, int(round(n * test_frac)))
    if time_col in df.columns:
        t = _parse_times(df[time_col])
        if t.notna().mean() >= 0.5:
            t = t.fillna(t.max())
            order = np.argsort(t.to_numpy(), kind="stable")
            return order[:-n_test], order[-n_test:], "time"
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=n_test, random_state=random_state,
                              stratify=df[target].to_numpy())
    return tr, te, "random"


# --------------------------------------------------------------------------- metrics
def _metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "log_loss": round(float(log_loss(y, p)), 4),
    }


def _bootstrap_auc_ci(y, p, n_boot: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    if not aucs:
        return None
    return [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]


def _calibration(y, p, n_bins: int = 10):
    """Reliability table + expected calibration error (ECE)."""
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = which == b
        if not m.any():
            continue
        mp, ob = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mp - ob)
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                     "mean_predicted": round(mp, 3), "observed_rate": round(ob, 3)})
    return rows, round(float(ece), 4)


def _lift(y, p, frac: float = 0.3) -> dict:
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-p, kind="stable")[:k]
    total = y.sum()
    return {
        "top_fraction": frac,
        "recovery_rate_in_top": round(float(y[top].mean()), 4),
        "recovery_rate_overall": round(float(y.mean()), 4),
        "share_of_all_recoveries_captured": round(float(y[top].sum() / total), 4) if total else None,
    }


MODEL_LABELS = {"baseline_rate": "Baseline (always predicts the average)",
                "logistic_regression": "Logistic regression",
                "gradient_boosting": "Gradient boosting"}
SELECTION_MARGIN = 0.005     # boosting must beat logistic regression by this much on validation


def make_model(name: str, cat_cols=CAT_COLS, num_cols=NUM_COLS) -> Pipeline:
    return {"baseline_rate": make_baseline, "logistic_regression": make_logreg,
            "gradient_boosting": make_gbm}[name](cat_cols, num_cols)


def _select_model(X_tr, y_tr, kind: str, cat_cols, num_cols):
    """Picks logistic regression vs gradient boosting using a validation slice carved
    from the TRAINING data only (the test set is never used to choose).
    Returns (name, {model: validation_auc}) or (default, None) if it can't be done."""
    n = len(y_tr)
    n_val = max(1, int(round(n * 0.25)))
    if kind == "time":                                   # tr is chronological
        fit_pos, val_pos = np.arange(n - n_val), np.arange(n - n_val, n)
    else:
        fit_pos, val_pos = train_test_split(np.arange(n), test_size=n_val, random_state=42,
                                            stratify=y_tr)
    if len(set(y_tr[fit_pos])) < 2 or len(set(y_tr[val_pos])) < 2:
        return "gradient_boosting", None
    val_auc = {}
    for name in ("logistic_regression", "gradient_boosting"):
        m = make_model(name, cat_cols, num_cols).fit(X_tr.iloc[fit_pos], y_tr[fit_pos])
        val_auc[name] = round(float(roc_auc_score(y_tr[val_pos], m.predict_proba(X_tr.iloc[val_pos])[:, 1])), 4)
    pick = ("gradient_boosting"
            if val_auc["gradient_boosting"] > val_auc["logistic_regression"] + SELECTION_MARGIN
            else "logistic_regression")
    return pick, val_auc


# --------------------------------------------------------------------------- main entry
def train_and_evaluate(df: pd.DataFrame, cat_cols=CAT_COLS, num_cols=NUM_COLS,
                       target: str = TARGET, time_col: str = TIME_COL,
                       oracle_auc: Optional[float] = None):
    """Evaluate honestly on a held-out slice, then refit the deployed model on all rows.

    Returns (final_model, report). `report` is JSON-serialisable.
    """
    if target not in df.columns:
        raise ValueError(f"Missing target column '{target}'")
    data = df.reset_index(drop=True)
    y_all = pd.to_numeric(data[target], errors="coerce")
    if y_all.isna().any() or not set(y_all.unique()) <= {0, 1}:
        raise ValueError(f"Target column '{target}' must contain only 0/1 values")
    y_all = y_all.astype(int).to_numpy()
    if len(set(y_all)) < 2:
        raise ValueError("Need both recovered and not-recovered examples to train on")

    X_all = prepare_features(data, cat_cols, num_cols)
    warnings: list = []
    report: dict = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(data)),
        "base_rate": round(float(y_all.mean()), 4),
        "features": {"categorical": list(cat_cols), "numeric": list(num_cols)},
        "warnings": warnings,
        "evaluated": False,
    }

    enough = (len(data) >= MIN_ROWS_FOR_EVAL
              and min(y_all.sum(), len(y_all) - y_all.sum()) >= MIN_CLASS_FOR_EVAL)
    if enough:
        tr, te, kind = split_train_test(data, target, time_col)
        y_tr, y_te = y_all[tr], y_all[te]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            enough = False
            warnings.append("Train or test slice had only one outcome class, so evaluation was skipped.")
    else:
        warnings.append(f"Too few rows or too few examples of one outcome for a reliable evaluation "
                        f"(need >= {MIN_ROWS_FOR_EVAL} rows and >= {MIN_CLASS_FOR_EVAL} of each class). "
                        "The model was trained, but its quality is unmeasured.")

    selected = "gradient_boosting"
    if enough:
        X_tr, X_te = X_all.iloc[tr], X_all.iloc[te]
        selected, val_auc = _select_model(X_tr, y_tr, kind, cat_cols, num_cols)
        results, probs, fitted = {}, {}, {}
        for name in ("baseline_rate", "logistic_regression", "gradient_boosting"):
            fitted[name] = make_model(name, cat_cols, num_cols).fit(X_tr, y_tr)
            probs[name] = fitted[name].predict_proba(X_te)[:, 1]
            results[name] = _metrics(y_te, probs[name])
        p_sel = probs[selected]
        cal_rows, ece = _calibration(y_te, p_sel)

        imp = permutation_importance(fitted[selected], X_te, y_te, scoring="roc_auc",
                                     n_repeats=5, random_state=42)
        importance = {c: {"auc_drop": round(float(m), 4), "std": round(float(s_), 4)}
                      for c, m, s_ in sorted(zip(X_te.columns, imp.importances_mean, imp.importances_std),
                                             key=lambda t: -t[1])}

        split = {"kind": kind, "train_rows": int(len(tr)), "test_rows": int(len(te))}
        if kind == "time":
            t = _parse_times(data[time_col])
            split["train_period"] = [str(t.iloc[tr].min().date()), str(t.iloc[tr].max().date())]
            split["test_period"] = [str(t.iloc[te].min().date()), str(t.iloc[te].max().date())]
        else:
            warnings.append("No usable timestamp column, so a random split was used. Random splits can "
                            "look slightly optimistic for data that changes over time.")

        if results[selected]["auc"] < 0.55:
            warnings.append("Held-out AUC is close to 0.5: the features carry little signal about recovery.")

        report.update({
            "evaluated": True,
            "split": split,
            "selected_model": selected,
            "selection_validation_auc": val_auc,
            "models": results,
            "auc_ci_95": _bootstrap_auc_ci(y_te, p_sel),
            "calibration": cal_rows,
            "ece": ece,
            "lift": _lift(y_te, p_sel),
            "feature_importance": importance,
        })
        if oracle_auc is not None:
            report["oracle_auc"] = round(float(oracle_auc), 4)
    report.setdefault("selected_model", selected)

    final = make_model(selected, cat_cols, num_cols).fit(X_all, y_all)
    return final, report


# --------------------------------------------------------------------------- artifacts
def save_artifacts(model: Pipeline, report: dict, model_dir: Path, plots: bool = True) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "recovery_model.pkl")
    (model_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (model_dir / "model_card.md").write_text(render_model_card(report))
    if plots and report.get("evaluated"):
        try:
            _save_plots(report, model_dir)
        except Exception as e:  # plots are a nice-to-have; never break training for them
            print(f"(skipped plots: {e})")


def load_report(model_dir: Path) -> Optional[dict]:
    p = Path(model_dir) / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def render_model_card(r: dict) -> str:
    L = ["# Reverra recovery model: model card", ""]
    L += [f"*Trained: {r['trained_at']}*  ", f"*Rows: {r['rows']:,} | recovery rate: {r['base_rate']:.1%}*", ""]
    L += ["## What it does",
          "Predicts the probability that a failed subscription payment is eventually recovered, "
          "from the plan, amount, tenure, failure reason, past failures and timing.", ""]
    if not r.get("evaluated"):
        L += ["## Evaluation", "Not evaluated (see warnings)."]
    else:
        s, m = r["split"], r["models"]
        L += ["## How it was evaluated"]
        if s["kind"] == "time":
            L += [f"Time-based split: trained on {s['train_rows']:,} older payments "
                  f"({s['train_period'][0]} to {s['train_period'][1]}), tested on the {s['test_rows']:,} "
                  f"most recent ({s['test_period'][0]} to {s['test_period'][1]})."]
        else:
            L += [f"Stratified random split: {s['train_rows']:,} train / {s['test_rows']:,} test."]
        sel = r["selected_model"]
        L += ["", "| Model | AUC | PR-AUC | Brier | Log loss |", "|---|---|---|---|---|"]
        for name in ("baseline_rate", "logistic_regression", "gradient_boosting"):
            x = m[name]
            label = MODEL_LABELS[name] + (" **(deployed)**" if name == sel else "")
            L.append(f"| {label} | {x['auc']:.3f} | {x['pr_auc']:.3f} | {x['brier']:.3f} | {x['log_loss']:.3f} |")
        if r.get("selection_validation_auc"):
            v = r["selection_validation_auc"]
            L += ["", f"The deployed model was chosen on a validation slice of the training data (never the test set): "
                      f"logistic regression {v['logistic_regression']:.3f} vs gradient boosting "
                      f"{v['gradient_boosting']:.3f}; boosting must win by more than {SELECTION_MARGIN} to be picked."]
        ci = r.get("auc_ci_95")
        if ci:
            L += ["", f"95% bootstrap interval for the deployed model's test AUC: {ci[0]:.3f} to {ci[1]:.3f}."]
        if "oracle_auc" in r:
            L += [f"Simulated-data ceiling (a model that knew the true probabilities): AUC {r['oracle_auc']:.3f}."]
        lf = r["lift"]
        L += ["", "## Business view",
              f"Retrying only the top {lf['top_fraction']:.0%} of payments by score would reach a "
              f"{lf['recovery_rate_in_top']:.1%} recovery rate (overall: {lf['recovery_rate_overall']:.1%}) and "
              f"capture {lf['share_of_all_recoveries_captured']:.0%} of all recoveries."]
        L += ["", "## Calibration", f"Expected calibration error: {r['ece']:.3f} (0 = predicted probabilities match reality).",
              "", "| Score band | Payments | Mean predicted | Actually recovered |", "|---|---|---|---|"]
        for c in r["calibration"]:
            L.append(f"| {c['bin']} | {c['n']} | {c['mean_predicted']:.2f} | {c['observed_rate']:.2f} |")
        L += ["", "## What drives predictions",
              "Permutation importance on held-out data (drop in AUC when a feature is shuffled; ~0 means it doesn't help):", ""]
        for k, v in r["feature_importance"].items():
            L.append(f"- `{k}`: {v['auc_drop']:+.4f} (+/-{v['std']:.4f})")
    if r.get("warnings"):
        L += ["", "## Warnings"] + [f"- {w}" for w in r["warnings"]]
    L += ["", "## Limits",
          "- Trained on whatever dataset was loaded. If that is the built-in simulation, results describe the "
          "simulation, not real customers.",
          "- Predicts recovery probability; it does not prove that any action (an email, a retry time) causes recovery.", ""]
    return "\n".join(L)


def _save_plots(r: dict, model_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cal = [c for c in r["calibration"] if c["n"] >= 20]     # tiny bins are just noise
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot([c["mean_predicted"] for c in cal], [c["observed_rate"] for c in cal], "-", color="C0", alpha=0.6)
    ax.scatter([c["mean_predicted"] for c in cal], [c["observed_rate"] for c in cal],
               s=[max(15, min(160, c["n"] / 2)) for c in cal], color="C0", label="model (dot size = payments)")
    ax.set_xlabel("Predicted recovery probability")
    ax.set_ylabel("Actual recovery rate")
    ax.set_title("Calibration (held-out data; bands with <20 payments hidden)", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir / "calibration.png", dpi=130)
    plt.close(fig)

    imp = r["feature_importance"]
    names = list(imp.keys())[::-1]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.barh(names, [imp[n]["auc_drop"] for n in names], xerr=[imp[n]["std"] for n in names])
    ax.set_xlabel("Drop in AUC when shuffled")
    ax.set_title("Feature importance (held-out data)")
    fig.tight_layout()
    fig.savefig(model_dir / "importance.png", dpi=130)
    plt.close(fig)
