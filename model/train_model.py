"""
Trains and honestly evaluates the recovery model, then saves it.

Run from anywhere:  python model/train_model.py
Reads:   data/failed_payments.csv   (and data/sim_meta.json if present)
Writes:  model/recovery_model.pkl   the deployed sklearn Pipeline
         model/metrics.json         every number in the evaluation
         model/model_card.md        human-readable evaluation summary
         model/calibration.png, model/importance.png
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.pipeline import save_artifacts, train_and_evaluate  # noqa: E402

df = pd.read_csv(ROOT / "data" / "failed_payments.csv")

# The simulator records the best AUC any model could reach on its own sample.
# Only use it if it belongs to this exact file (same number of rows).
oracle = None
meta_path = ROOT / "data" / "sim_meta.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    if meta.get("rows") == len(df):
        oracle = meta.get("oracle_auc")

model, report = train_and_evaluate(df, oracle_auc=oracle)
save_artifacts(model, report, ROOT / "model")

print(f"Rows: {report['rows']:,}   recovery rate: {report['base_rate']:.1%}")
if report["evaluated"]:
    s = report["split"]
    print(f"Split: {s['kind']}  (train {s['train_rows']:,} / test {s['test_rows']:,})")
    for name, m in report["models"].items():
        mark = "  <- deployed" if name == report["selected_model"] else ""
        print(f"  {name:20s} AUC {m['auc']:.3f}  PR-AUC {m['pr_auc']:.3f}  Brier {m['brier']:.3f}{mark}")
    print(f"  95% CI for deployed AUC: {report['auc_ci_95']}   calibration error (ECE): {report['ece']}")
    if "oracle_auc" in report:
        print(f"  simulated ceiling AUC: {report['oracle_auc']}")
    lf = report["lift"]
    print(f"  top {lf['top_fraction']:.0%} by score: {lf['recovery_rate_in_top']:.1%} recovered "
          f"(overall {lf['recovery_rate_overall']:.1%}), captures {lf['share_of_all_recoveries_captured']:.0%} of recoveries")
for w in report["warnings"]:
    print("WARNING:", w)
print("Saved model/recovery_model.pkl, metrics.json, model_card.md")
