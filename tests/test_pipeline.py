"""Sanity tests for the simulator and the training pipeline.
Run from the project root:  pytest -q      (or: python -m pytest -q)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.pipeline import (_parse_times, known_categories, predict_proba,  # noqa: E402
                         split_train_test, train_and_evaluate)
from ml.simulate import generate  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    return generate(3000, seed=1, return_truth=True)


def test_simulation_is_calibrated_and_reproducible(sim):
    df, meta = sim
    assert 0.20 <= meta["base_rate"] <= 0.45          # within the published 20-40% range (+ sampling noise)
    assert 0.55 < meta["oracle_auc"] < 0.95            # real signal, but far from perfect
    assert generate(3000, seed=1).equals(df)           # same seed -> same data
    assert not generate(3000, seed=2).equals(df)


def test_time_split_never_trains_on_the_future(sim):
    df, _ = sim
    tr, te, kind = split_train_test(df)
    assert kind == "time"
    t = _parse_times(df["failed_at"])
    assert t.iloc[tr].max() <= t.iloc[te].min()
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(df)


def test_falls_back_to_random_split_without_timestamps(sim):
    df, _ = sim
    _, _, kind = split_train_test(df.drop(columns=["failed_at"]))
    assert kind == "random"


def test_model_beats_baseline_and_reports_honestly(sim):
    df, meta = sim
    model, rep = train_and_evaluate(df, oracle_auc=meta["oracle_auc"])
    assert rep["evaluated"] and rep["split"]["kind"] == "time"
    auc = rep["models"][rep["selected_model"]]["auc"]
    assert rep["models"]["baseline_rate"]["auc"] == 0.5
    assert auc > 0.6
    assert auc <= rep["oracle_auc"] + 0.03             # can't meaningfully beat the ceiling
    lo, hi = rep["auc_ci_95"]
    assert lo <= auc <= hi


def test_unseen_categories_and_missing_values_do_not_crash(sim):
    df, _ = sim
    messy = df.copy()
    messy.loc[:30, "failure_reason"] = "brand_new_reason"
    messy.loc[10:60, "tenure_months"] = np.nan
    model, rep = train_and_evaluate(messy)
    row = df.iloc[[0]].copy()
    row["failure_reason"] = "never_seen"
    row["amount"] = np.nan
    p = predict_proba(model, row)
    assert 0 <= p[0] <= 1
    assert "plan_type" in known_categories(model)


def test_tiny_dataset_trains_but_says_it_is_unmeasured(sim):
    df, _ = sim
    model, rep = train_and_evaluate(df.sample(60, random_state=0))
    assert rep["evaluated"] is False and rep["warnings"]
    assert 0 <= predict_proba(model, df.iloc[:5]).max() <= 1


def test_bad_labels_are_rejected(sim):
    df, _ = sim
    with pytest.raises(ValueError):
        train_and_evaluate(df.assign(recovered=1))
    with pytest.raises(ValueError):
        train_and_evaluate(df.assign(recovered=df["recovered"] * 2))
