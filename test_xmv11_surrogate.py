import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data.tep_loader import load_tep_dataset

data = load_tep_dataset("d00")
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

targets = ["XMEAS_17", "XMEAS_07", "XMEAS_10", "XMEAS_19", "XMEAS_20"]
seven = ["XMV_01", "XMV_02", "XMV_03", "XMV_04", "XMV_05", "XMV_06", "XMV_09"]
eight = seven + ["XMV_11"]

rows = []
for label, features in [("Current 7 inputs", seven), ("7 + XMV_11", eight)]:
    X = data[features].to_numpy()
    print(f"\nEvaluating {label}...")
    for target in targets:
        y = data[target].to_numpy()
        scores = []
        for train, test in TimeSeriesSplit(n_splits=5).split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10)),
            ])
            model.fit(X[train], y[train])
            pred = model.predict(X[test])
            scores.append((
                mean_absolute_error(y[test], pred),
                np.sqrt(mean_squared_error(y[test], pred)),
                r2_score(y[test], pred),
            ))
        rows.append({
            "input_set": label,
            "target": target,
            "CV_MAE": np.mean([s[0] for s in scores]),
            "CV_RMSE": np.mean([s[1] for s in scores]),
            "CV_R2": np.mean([s[2] for s in scores]),
        })

result = pd.DataFrame(rows)
print("\n=== CROSS-VALIDATION COMPARISON ===")
print(result.round(4).to_string(index=False))

out = "results/optimization/xmv11_surrogate_comparison.csv"
result.to_csv(out, index=False)
print(f"\nSaved: {out}")
