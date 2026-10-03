import pandas as pd
from pathlib import Path
from src.data.tep_loader import load_tep_dataset
from src.models.surrogate_models import DECISION_FEATURES, RESPONSE_TARGETS

df = load_tep_dataset("d00")
out = Path("results/optimization")
out.mkdir(parents=True, exist_ok=True)

print("\n=== INPUT VARIATION ===")
print(df[DECISION_FEATURES].describe().loc[["mean", "std", "min", "max"]].round(4))

print("\n=== RESPONSE VARIATION ===")
print(df[RESPONSE_TARGETS].describe().loc[["mean", "std", "min", "max"]].round(4))

print("\n=== INPUT CORRELATION WITH PRODUCT FLOW ===")
corr = df[DECISION_FEATURES + ["XMEAS_17"]].corr()["XMEAS_17"].drop("XMEAS_17")
print(corr.sort_values(key=abs, ascending=False).round(4).to_string())

print("\n=== PRODUCT FLOW BY CHRONOLOGICAL BLOCK ===")
df["block"] = pd.qcut(
    range(len(df)), q=5, labels=["Block 1", "Block 2", "Block 3", "Block 4", "Block 5"]
)
block_stats = df.groupby("block", observed=False)["XMEAS_17"].agg(
    ["mean", "std", "min", "max"]
)
print(block_stats.round(4).to_string())

corr.rename("correlation_with_product_flow").to_csv(
    out / "product_flow_input_correlations.csv"
)
block_stats.to_csv(out / "product_flow_time_blocks.csv")
print("\nSaved diagnostic CSV files in results/optimization/")
