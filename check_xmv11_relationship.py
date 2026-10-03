from pathlib import Path
import pandas as pd
from src.data.tep_loader import load_tep_dataset

# Inspect the project's variable dictionary
p = Path("src/process/variable_dictionary.py")
print("=== VARIABLE DICTIONARY REFERENCES ===")
if p.exists():
    lines = p.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if "XMV_11" in line or "XMEAS_17" in line:
            print(f"{i}: {line}")
else:
    print("Variable dictionary file not found.")

# Check whether the relationship is consistent over time
df = load_tep_dataset("d00")
df = df[["XMV_11", "XMEAS_17"]].copy()
df["block"] = pd.qcut(
    range(len(df)), q=5,
    labels=["Block 1", "Block 2", "Block 3", "Block 4", "Block 5"]
)

print("\n=== XMV_11 VS PRODUCT FLOW BY TIME BLOCK ===")
print(
    df.groupby("block", observed=False)
      .agg(
          XMV11_mean=("XMV_11", "mean"),
          XMV11_std=("XMV_11", "std"),
          product_mean=("XMEAS_17", "mean"),
          product_std=("XMEAS_17", "std"),
          correlation=("XMV_11", lambda x: x.corr(df.loc[x.index, "XMEAS_17"]))
      )
      .round(4)
      .to_string()
)
