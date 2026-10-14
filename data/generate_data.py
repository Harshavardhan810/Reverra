"""
Generates the SIMULATED failed-payments dataset for Reverra.

Each row = one failed subscription payment; `recovered` = whether it was
eventually collected. The simulator (ml/simulate.py) is calibrated to
published industry recovery rates and is deliberately noisy, with a hidden
factor and mild drift over time. It is NOT real transaction data.

Run from anywhere:  python data/generate_data.py
Output: data/failed_payments.csv, data/sim_meta.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.simulate import generate  # noqa: E402

N, SEED = 6000, 42

df, meta = generate(N, seed=SEED, return_truth=True)
df.to_csv(ROOT / "data" / "failed_payments.csv", index=False)
(ROOT / "data" / "sim_meta.json").write_text(json.dumps(meta, indent=2))

print(f"Generated {len(df)} rows -> data/failed_payments.csv")
print(f"Recovery rate: {meta['base_rate']:.1%}  (industry range for good dunning: roughly 20-40%)")
print(f"Best possible AUC on this sample (oracle): {meta['oracle_auc']}")
print(df.head())
