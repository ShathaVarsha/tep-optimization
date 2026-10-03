from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.data.tep_loader import load_tep_dataset

ROOT = Path.cwd()
OUT = ROOT / "results" / "optimization" / "xmv11_experiment"

FEATURES = [
    "XMV_01", "XMV_02", "XMV_03", "XMV_04",
    "XMV_05", "XMV_06", "XMV_09", "XMV_11"
]
TARGETS = ["XMEAS_17", "XMEAS_07", "XMEAS_10", "XMEAS_19", "XMEAS_20"]

df = load_tep_dataset("d00.dat")
X = df[FEATURES]
Y = df[TARGETS]

models = {}
for target in TARGETS:
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    model.fit(X, Y[target])
    models[target] = model

saved = json.loads(
    (OUT / "optimization_results_8vars.json").read_text(encoding="utf-8")
)

rows = []
for problem in ["production_maximization", "cost_minimization"]:
    x0 = saved[problem]["optimized_inputs"]
    base_x = pd.DataFrame([x0], columns=FEATURES)
    base_pred = {t: float(models[t].predict(base_x)[0]) for t in TARGETS}

    for feature in FEATURES:
        center = float(x0[feature])
        lo, hi = float(df[feature].min()), float(df[feature].max())
        delta = max((hi - lo) * 0.01, 0.01)

        for direction in [-1, 1]:
            changed = dict(x0)
            changed[feature] = float(np.clip(center + direction * delta, lo, hi))
            actual_delta = changed[feature] - center
            if abs(actual_delta) < 1e-12:
                continue

            changed_x = pd.DataFrame([changed], columns=FEATURES)
            for target in TARGETS:
                after = float(models[target].predict(changed_x)[0])
                before = base_pred[target]
                rows.append({
                    "problem": problem,
                    "input_variable": feature,
                    "direction": "decrease" if direction < 0 else "increase",
                    "input_baseline": center,
                    "input_perturbed": changed[feature],
                    "input_change": actual_delta,
                    "response": target,
                    "predicted_baseline": before,
                    "predicted_perturbed": after,
                    "response_change": after - before,
                    "local_slope": (after - before) / actual_delta,
                })

result = pd.DataFrame(rows)
result.to_csv(OUT / "local_what_if_sensitivity.csv", index=False)

summary = (
    result.assign(abs_change=result["response_change"].abs())
    .sort_values("abs_change", ascending=False)
    .groupby(["problem", "response"], as_index=False)
    .first()
)
summary.to_csv(OUT / "local_what_if_sensitivity_summary.csv", index=False)

print("Local what-if sensitivity analysis completed.")
print(f"Detailed results: {OUT / 'local_what_if_sensitivity.csv'}")
print(f"Summary results:  {OUT / 'local_what_if_sensitivity_summary.csv'}")
print(f"Scenarios generated: {len(result)}")
print("Note: These are local Ridge-surrogate estimates, not formal LP shadow prices.")
