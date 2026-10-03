from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.tep_loader import load_tep_dataset

FEATURES_7 = [
    "XMV_01", "XMV_02", "XMV_03", "XMV_04",
    "XMV_05", "XMV_06", "XMV_09"
]
FEATURES_8 = FEATURES_7 + ["XMV_11"]
TARGETS = ["XMEAS_17", "XMEAS_07", "XMEAS_10", "XMEAS_19", "XMEAS_20"]

out = Path("results/optimization/xmv11_experiment")
out.mkdir(parents=True, exist_ok=True)

df = load_tep_dataset("d00")
if not isinstance(df, pd.DataFrame):
    raise TypeError(f"Expected DataFrame from loader, got {type(df).__name__}")

missing = [c for c in FEATURES_8 + TARGETS if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

df = df[FEATURES_8 + TARGETS].apply(pd.to_numeric, errors="coerce").dropna()
if len(df) < 100:
    raise ValueError(f"Only {len(df)} usable rows; refusing to run an unreliable comparison.")

tscv = TimeSeriesSplit(n_splits=5)
rows = []

for feature_name, features in [("Current 7", FEATURES_7), ("With XMV_11", FEATURES_8)]:
    X = df[features].to_numpy(dtype=float)

    for target in TARGETS:
        y = df[target].to_numpy(dtype=float)
        fold_mae, fold_rmse, fold_r2 = [], [], []

        for train_idx, test_idx in tscv.split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=10.0)),
            ])
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            fold_mae.append(mean_absolute_error(y[test_idx], pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
            fold_r2.append(r2_score(y[test_idx], pred))

        rows.append({
            "input_set": feature_name,
            "target": target,
            "rows": len(df),
            "CV_MAE": float(np.mean(fold_mae)),
            "CV_RMSE": float(np.mean(fold_rmse)),
            "CV_R2": float(np.mean(fold_r2)),
        })

metrics = pd.DataFrame(rows)
metrics.to_csv(out / "surrogate_comparison.csv", index=False)
print("\n=== Chronological cross-validation comparison ===")
print(metrics.round(4).to_string(index=False))

print("\n=== XMV_11/product-flow relationship ===")
print(f"Correlation: {df['XMV_11'].corr(df['XMEAS_17']):.5f}")
print(f"XMV_11 range: {df['XMV_11'].min():.3f} to {df['XMV_11'].max():.3f}")
print(f"Product-flow range: {df['XMEAS_17'].min():.3f} to {df['XMEAS_17'].max():.3f}")

print(f"\nSaved results to: {out.resolve()}")
print("This experiment evaluates prediction quality only; it does not change or run the LP optimizer.")
