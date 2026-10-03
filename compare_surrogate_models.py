import pandas as pd
from src.data.tep_loader import load_tep_dataset
from src.models.surrogate_models import ProcessSurrogateModel, RESPONSE_TARGETS

df = load_tep_dataset("d00")

print("Dataset shape:", df.shape)
print("\nComparing surrogate models...")

rows = []
for model_type in ("linear", "ridge", "poly"):
    model = ProcessSurrogateModel(
        model_type=model_type,
        alpha=10.0,
        poly_degree=2,
    )
    model.fit(df)

    for target in RESPONSE_TARGETS:
        m = model.metrics[target]
        rows.append({
            "model": model_type,
            "target": target,
            "CV_MAE": m["cv_mae"],
            "CV_RMSE": m["cv_rmse"],
            "CV_R2": m["cv_r2"],
            "CV_Relative_Error_%": m["cv_rel_error_pct"],
        })

results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))

from pathlib import Path
out = Path("results/optimization")
out.mkdir(parents=True, exist_ok=True)
results.to_csv(out / "surrogate_model_comparison.csv", index=False)
print("\nSaved: results/optimization/surrogate_model_comparison.csv")
