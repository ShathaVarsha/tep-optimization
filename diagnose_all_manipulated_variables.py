import pandas as pd
from pathlib import Path
from src.data.tep_loader import load_tep_dataset

df = load_tep_dataset("d00")
out = Path("results/optimization")
out.mkdir(parents=True, exist_ok=True)

inputs = [f"XMV_{i:02d}" for i in range(1, 12)]
inputs = [c for c in inputs if c in df.columns]

report = df[inputs].agg(["mean", "std", "min", "max"]).T
report["corr_with_product_flow"] = [
    df[c].corr(df["XMEAS_17"]) for c in inputs
]
report["abs_correlation"] = report["corr_with_product_flow"].abs()
report = report.sort_values("abs_correlation", ascending=False)

print("\n=== ALL MANIPULATED VARIABLES ===")
print(report.round(4).to_string())

report.to_csv(out / "all_manipulated_variable_diagnostics.csv")
print("\nSaved: results/optimization/all_manipulated_variable_diagnostics.csv")
