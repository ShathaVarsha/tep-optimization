"""Tennessee Eastman Process Baseline Stationarity and Operating State Analysis.

Combines:
  1. Augmented Dickey-Fuller (ADF) unit-root hypothesis testing.
  2. Rolling window mean drift and variance stability metrics.
  3. Chemical process domain inventory and equilibrium verification.

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

DEFAULT_KEY_VARS = [
    "XMEAS_01",  # A Feed
    "XMEAS_02",  # D Feed
    "XMEAS_03",  # E Feed
    "XMEAS_04",  # A+C Feed
    "XMEAS_05",  # Recycle Flow
    "XMEAS_07",  # Reactor Pressure
    "XMEAS_08",  # Reactor Level
    "XMEAS_09",  # Reactor Temperature
    "XMEAS_10",  # Purge Flow
    "XMEAS_17",  # Product Rate
    "XMEAS_19",  # Stripper Steam Flow
    "XMEAS_20",  # Compressor Work
]


def test_series_stationarity(
    series: pd.Series,
    window_size: int = 50,
) -> Dict[str, Union[float, bool, int]]:
    """Perform multi-criteria stationarity assessment on a single process variable series."""
    clean_series = series.dropna()
    n_pts = len(clean_series)
    if n_pts < window_size * 2:
        raise ValueError(f"Series length ({n_pts}) insufficient for window size ({window_size})")

    # 1. Augmented Dickey-Fuller unit-root test
    # Note: suppress warning by handling tuple output
    adf_result = adfuller(clean_series, autolag="AIC", result_object=False)
    adf_stat = float(adf_result[0])
    adf_pvalue = float(adf_result[1])
    used_lags = int(adf_result[2])
    crit_5pct = float(adf_result[4]["5%"])
    is_stationary_adf = bool(adf_pvalue < 0.05 and adf_stat < crit_5pct)

    # 2. Rolling window statistics
    rolling_mean = clean_series.rolling(window=window_size).mean().dropna()
    rolling_std = clean_series.rolling(window=window_size).std().dropna()

    mean_overall = float(clean_series.mean())
    std_overall = float(clean_series.std())

    # Normalized drift of rolling mean across the dataset
    abs_drift = float(abs(rolling_mean.iloc[-1] - rolling_mean.iloc[0]))
    if abs(mean_overall) > 1e-6:
        mean_drift_pct = float(abs_drift / abs(mean_overall)) * 100.0
    else:
        mean_drift_pct = abs_drift * 100.0

    # Ratio of max rolling standard deviation to overall standard deviation
    std_stability_ratio = float(rolling_std.max() / std_overall) if std_overall > 1e-6 else 1.0

    # Stationarity holds if ADF rejects unit root AND (rolling mean drift is under 10% or absolute drift < 0.05)
    is_steady_state = bool(is_stationary_adf and (mean_drift_pct < 10.0 or abs_drift < 0.05))

    return {
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "used_lags": used_lags,
        "crit_5pct": crit_5pct,
        "is_stationary_adf": is_stationary_adf,
        "mean_overall": mean_overall,
        "std_overall": std_overall,
        "mean_drift_pct": mean_drift_pct,
        "std_stability_ratio": std_stability_ratio,
        "is_steady_state": is_steady_state,
    }


def perform_stationarity_analysis(
    df: pd.DataFrame,
    variables: Optional[List[str]] = None,
    window_size: int = 50,
) -> Dict[str, object]:
    """Run comprehensive stationarity assessment across all specified TEP variables."""
    target_vars = variables if variables is not None else DEFAULT_KEY_VARS
    results = {}
    steady_state_votes = []

    for var in target_vars:
        if var not in df.columns:
            continue
        stat_dict = test_series_stationarity(df[var], window_size=window_size)
        results[var] = stat_dict
        steady_state_votes.append(stat_dict["is_steady_state"])

    overall_steady_state = bool(all(steady_state_votes)) if steady_state_votes else False

    # Process domain verification checks
    p_reactor_mean = float(df["XMEAS_07"].mean()) if "XMEAS_07" in df.columns else 2705.0
    t_reactor_mean = float(df["XMEAS_09"].mean()) if "XMEAS_09" in df.columns else 120.4
    prod_rate_mean = float(df["XMEAS_17"].mean()) if "XMEAS_17" in df.columns else 22.89

    domain_checks = {
        "reactor_pressure_in_range": bool(2680.0 <= p_reactor_mean <= 2730.0),
        "reactor_temp_in_range": bool(118.0 <= t_reactor_mean <= 123.0),
        "product_flow_in_range": bool(22.0 <= prod_rate_mean <= 24.0),
    }

    return {
        "variable_results": results,
        "overall_statistical_steady_state": overall_steady_state,
        "domain_checks": domain_checks,
        "is_validated_baseline": bool(overall_steady_state and all(domain_checks.values())),
    }


def extract_baseline_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Extract baseline mean, std, median, min, max, and quantiles for all 52 variables."""
    numeric_df = df.select_dtypes(include=[np.number])
    stats = pd.DataFrame({
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "median": numeric_df.median(),
        "min": numeric_df.min(),
        "q05": numeric_df.quantile(0.05),
        "q95": numeric_df.quantile(0.95),
        "max": numeric_df.max(),
    })
    return stats


def get_baseline_point(df: pd.DataFrame) -> pd.Series:
    """Return nominal operating baseline point as a pandas Series of 52 mean values."""
    return df.mean()
