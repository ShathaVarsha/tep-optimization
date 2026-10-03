import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.tep_loader import load_tep_dataset

FEATURES = [
    "XMV_01", "XMV_02", "XMV_03", "XMV_04",
    "XMV_05", "XMV_06", "XMV_09", "XMV_11",
]
TARGETS = ["XMEAS_17", "XMEAS_07", "XMEAS_10", "XMEAS_19", "XMEAS_20"]
OUT = Path("results/optimization/xmv11_experiment")
OUT.mkdir(parents=True, exist_ok=True)

df = load_tep_dataset("d00")
X = df[FEATURES].to_numpy(float)
baseline = X.mean(axis=0)
models = {}
metric_rows = []

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=10.0)),
    ])

def original_scale_coefficients(model):
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["regressor"]
    coef = ridge.coef_ / scaler.scale_
    intercept = ridge.intercept_ - np.dot(coef, scaler.mean_)
    return np.asarray(coef, float), float(intercept)

tscv = TimeSeriesSplit(n_splits=5)
for target in TARGETS:
    y = df[target].to_numpy(float)
    fold_metrics = []
    for train_idx, test_idx in tscv.split(X):
        m = make_model()
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        fold_metrics.append({
            "MAE": mean_absolute_error(y[test_idx], pred),
            "RMSE": np.sqrt(mean_squared_error(y[test_idx], pred)),
            "R2": r2_score(y[test_idx], pred),
        })
    final_model = make_model()
    final_model.fit(X, y)
    models[target] = final_model
    metric_rows.append({
        "target": target,
        "mean": float(y.mean()),
        "std": float(y.std()),
        "CV_MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
        "CV_RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
        "CV_R2": float(np.mean([m["R2"] for m in fold_metrics])),
    })

pd.DataFrame(metric_rows).to_csv(OUT / "surrogate_metrics_8vars.csv", index=False)

def predict(u):
    return {name: float(model.predict(np.asarray(u).reshape(1, -1))[0])
            for name, model in models.items()}

def solve(kind):
    prod_c, prod_b = original_scale_coefficients(models["XMEAS_17"])
    press_c, press_b = original_scale_coefficients(models["XMEAS_07"])
    purge_c, purge_b = original_scale_coefficients(models["XMEAS_10"])
    steam_c, steam_b = original_scale_coefficients(models["XMEAS_19"])
    comp_c, comp_b = original_scale_coefficients(models["XMEAS_20"])

    # Approximate operating-cost model, consistent with the existing LP.
    cost_c = purge_c * 44.61937 * 5.80 + steam_c * 0.00318 + comp_c * 0.0536
    cost_b = purge_b * 44.61937 * 5.80 + steam_b * 0.00318 + comp_b * 0.0536

    # Constrain all modeled responses to the observed normal-operation ranges.
    response_specs = [
        ("XMEAS_17", prod_c, prod_b),
        ("XMEAS_07", press_c, press_b),
        ("XMEAS_10", purge_c, purge_b),
        ("XMEAS_19", steam_c, steam_b),
        ("XMEAS_20", comp_c, comp_b),
    ]

    A_rows = []
    b_rows = []
    for target_name, coef, intercept in response_specs:
        lower = float(df[target_name].min())
        upper = float(df[target_name].max())
        A_rows.append(coef)
        b_rows.append(upper - intercept)
        A_rows.append(-coef)
        b_rows.append(intercept - lower)

    if kind == "production_maximization":
        objective = -prod_c
    else:
        objective = cost_c

    if kind == "cost_minimization":
            target_production = float(df["XMEAS_17"].mean())
            A_rows.append(-prod_c)
            b_rows.append(prod_b - target_production)

    A = np.vstack(A_rows)
    b = np.asarray(b_rows, dtype=float)

    bounds = [
        (
            max(float(df[feature].min()), float(v - 10.0), 0.0),
            min(float(df[feature].max()), float(v + 10.0), 100.0),
        )
        for feature, v in zip(FEATURES, baseline)
    ]
    result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds")
    if not result.success:
        return {"success": False, "message": result.message}

    opt = result.x
    opt_pred = predict(opt)
    base_pred = predict(baseline)
    if kind == "production_maximization":
        base_value = base_pred["XMEAS_17"]
        opt_value = opt_pred["XMEAS_17"]
        objective_name = "Predicted product flow (m3/hr)"
        improvement = (opt_value - base_value) / abs(base_value) * 100
    else:
        base_value = float(np.dot(cost_c, baseline) + cost_b)
        opt_value = float(np.dot(cost_c, opt) + cost_b)
        objective_name = "Approximate operating cost ($/hr)"
        improvement = (base_value - opt_value) / abs(base_value) * 100 if base_value else 0.0

    return {
        "success": True,
        "problem": kind,
        "objective_name": objective_name,
        "baseline_value": float(base_value),
        "optimized_value": float(opt_value),
        "improvement_percent": float(improvement),
        "baseline_inputs": dict(zip(FEATURES, baseline.tolist())),
        "optimized_inputs": dict(zip(FEATURES, opt.tolist())),
        "predicted_responses": opt_pred,
        "baseline_predictions": base_pred,
        "solver_message": result.message,
        "iterations": int(result.nit),
        "constraints": {
            "pressure_limit": 2800,
            "compressor_limit": 355,
            "trust_region": "baseline +/- 10 percentage points, clipped to [0,100]",
        },
    }

results = {
    "dataset": "d00",
    "dataset_shape": list(df.shape),
    "features": FEATURES,
    "note": "Experimental surrogate-based estimates only; not validated plant-level savings. Cost uses the existing approximate average purge price.",
    "production_maximization": solve("production_maximization"),
    "cost_minimization": solve("cost_minimization"),
}
(OUT / "optimization_results_8vars.json").write_text(json.dumps(results, indent=2))
summary = []
for key in ["production_maximization", "cost_minimization"]:
    r = results[key]
    summary.append({
        "problem": key,
        "success": r.get("success"),
        "baseline": r.get("baseline_value"),
        "optimized": r.get("optimized_value"),
        "improvement_percent": r.get("improvement_percent"),
    })
pd.DataFrame(summary).to_csv(OUT / "optimization_summary_8vars.csv", index=False)

print("\n=== 8-VARIABLE SURROGATE VALIDATION ===")
print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
print("\n=== 8-VARIABLE OPTIMIZATION ===")
print(pd.DataFrame(summary).round(4).to_string(index=False))
print(f"\nSaved experiment to: {OUT.resolve()}")
