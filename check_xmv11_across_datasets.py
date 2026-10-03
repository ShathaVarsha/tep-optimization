import numpy as np
import pandas as pd
from src.data.tep_loader import load_tep_dataset

rows = []
for name in ["d00", "d01", "d02", "d03", "d04", "d05"]:
    try:
        df = load_tep_dataset(name)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        corr = df["XMV_11"].corr(df["XMEAS_17"])
        rows.append({
            "dataset": name,
            "rows": len(df),
            "XMV11_mean": df["XMV_11"].mean(),
            "product_mean": df["XMEAS_17"].mean(),
            "correlation": corr,
        })
    except Exception as e:
        print(f"{name}: unavailable ({e})")

result = pd.DataFrame(rows)
print("\n=== XMV_11 VS PRODUCT FLOW ACROSS DATASETS ===")
print(result.round(4).to_string(index=False))

out = "results/optimization/xmv11_cross_dataset_check.csv"
result.to_csv(out, index=False)
print(f"\nSaved: {out}")
