import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from src.data.tep_loader import load_tep_dataset

df = load_tep_dataset("d00")
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame(df)

X = df[["XMV_11"]].to_numpy()
y = df["XMEAS_17"].to_numpy()
rows = []

for fold, (train, test) in enumerate(
    TimeSeriesSplit(n_splits=5).split(X), start=1
):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=10)),
    ])
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    rows.append({
        "fold": fold,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_XMV11_min": X[test].min(),
        "test_XMV11_max": X[test].max(),
        "test_product_min": y[test].min(),
        "test_product_max": y[test].max(),
        "MAE": mean_absolute_error(y[test], pred),
        "R2": r2_score(y[test], pred),
    })

result = pd.DataFrame(rows)
print("=== XMV_11-ONLY CHRONOLOGICAL VALIDATION ===")
print(result.round(4).to_string(index=False))
print("\nOverall product-flow correlation:",
      round(float(np.corrcoef(X[:, 0], y)[0, 1]), 5))
