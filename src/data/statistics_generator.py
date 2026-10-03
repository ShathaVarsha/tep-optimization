"""On-demand descriptive statistics and exploratory data analytics for TEP variables.

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive descriptive statistics on demand for all process variables."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    
    # Enrich with additional OR/engineering metrics: skewness, kurtosis, IQR, coefficient of variation
    skewness = numeric_df.skew()
    kurtosis = numeric_df.kurtosis()
    iqr = stats["75%"] - stats["25%"]
    cov = (stats["std"] / stats["mean"]).abs()

    stats["skewness"] = skewness
    stats["kurtosis"] = kurtosis
    stats["iqr"] = iqr
    stats["coef_of_variation"] = cov
    return stats


def compute_correlation_matrix(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute cross-variable correlation matrix for selected or all process variables."""
    target_df = df[variables] if variables else df
    return target_df.corr(method=method)


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Identify outlier counts and percentages per variable using the interquartile range (IQR) method."""
    q25 = df.quantile(0.25)
    q75 = df.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr

    outlier_counts = ((df < lower_bound) | (df > upper_bound)).sum()
    outlier_pct = (outlier_counts / len(df)) * 100.0

    return pd.DataFrame({
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_counts,
        "outlier_pct": outlier_pct,
    })


def get_variable_profile(df: pd.DataFrame, var_name: str) -> Dict[str, float]:
    """Return descriptive metric profile for a specific variable."""
    if var_name not in df.columns:
        raise KeyError(f"Variable '{var_name}' not present in dataframe columns.")
    series = df[var_name]
    return {
        "variable": var_name,
        "count": float(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": float(series.skew()),
        "kurtosis": float(series.kurtosis()),
    }
