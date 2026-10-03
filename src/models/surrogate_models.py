"""Interpretable Physics-Bounded Process Surrogate Models for TEP.

Maps manipulated decision variables (u) to state and performance responses (y):
  - u = [XMV_01 (D feed), XMV_02 (E feed), XMV_03 (A feed), XMV_04 (A+C feed),
         XMV_05 (Recycle valve), XMV_06 (Purge valve), XMV_09 (Steam valve)]
  - y = [XMEAS_17 (Product Flow), XMEAS_07 (Reactor Pressure),
         XMEAS_10 (Purge Rate), XMEAS_19 (Steam Flow), XMEAS_20 (Compressor Work)]

Includes:
  1. Interpretable baseline models (OLS, Ridge Regression, Polynomial Ridge).
  2. Chronological TimeSeriesSplit validation (k=5).
  3. Strict domain boundary / Mahalanobis distance extrapolation detector.

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

DECISION_FEATURES_7: List[str] = [
    "XMV_01",  # D Feed Flow Valve (%)
    "XMV_02",  # E Feed Flow Valve (%)
    "XMV_03",  # A Feed Flow Valve (%)
    "XMV_04",  # A and C Feed Flow Valve (%)
    "XMV_05",  # Compressor Recycle Valve (%)
    "XMV_06",  # Purge Valve (%)
    "XMV_09",  # Stripper Steam Valve (%)
]

DECISION_FEATURES_8: List[str] = DECISION_FEATURES_7 + [
    "XMV_11",  # Condenser Cooling Water Flow Valve (%)
]

DECISION_FEATURES: List[str] = DECISION_FEATURES_7

RESPONSE_TARGETS: List[str] = [
    "XMEAS_17",  # Stripper Underflow / Product Flow (m3/hr)
    "XMEAS_07",  # Reactor Pressure (kPa gauge)
    "XMEAS_10",  # Purge Rate (kscmh)
    "XMEAS_19",  # Stripper Steam Flow (kg/hr)
    "XMEAS_20",  # Compressor Work (kW)
]


class ExtrapolationDetector:
    """Detects whether an optimization candidate point lies within the validated empirical domain."""

    def __init__(self, confidence: float = 0.99, box_expansion: float = 0.20):
        self.confidence = confidence
        self.box_expansion = box_expansion
        self.mean_u: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: float = 0.0
        self.u_min: Optional[np.ndarray] = None
        self.u_max: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> ExtrapolationDetector:
        """Fit empirical mean, regularized covariance inverse, and box limits from baseline data."""
        self.mean_u = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        
        # Add slight Tikhonov regularization to covariance matrix to ensure well-conditioned inverse
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-4
        self.cov_inv = np.linalg.pinv(cov_reg)
        
        # Chi-squared critical threshold for p dimensions at specified confidence
        p = X.shape[1]
        self.threshold = float(chi2.ppf(self.confidence, df=p))

        # Box boundaries with expansion tolerance
        u_span = np.ptp(X, axis=0)
        self.u_min = np.min(X, axis=0) - self.box_expansion * u_span
        self.u_max = np.max(X, axis=0) + self.box_expansion * u_span
        return self

    def compute_mahalanobis_distance(self, u: np.ndarray) -> float:
        """Compute squared Mahalanobis distance D_M^2 for a candidate vector u."""
        diff = u - self.mean_u
        d2 = float(diff.T @ self.cov_inv @ diff)
        return max(0.0, d2)

    def is_within_domain(self, u: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """Check whether candidate point u is within statistical and box domain limits."""
        d2 = self.compute_mahalanobis_distance(u)
        mahalanobis_valid = bool(d2 <= self.threshold)
        
        # Box check
        box_valid = bool(np.all(u >= self.u_min) and np.all(u <= self.u_max))
        overall_valid = mahalanobis_valid and box_valid

        return overall_valid, {
            "mahalanobis_dist_sq": d2,
            "critical_threshold": self.threshold,
            "mahalanobis_valid": mahalanobis_valid,
            "box_valid": box_valid,
            "overall_valid": overall_valid,
        }


class ProcessSurrogateModel:
    """Manages individual linear and polynomial regularized surrogates for key TEP responses."""

    def __init__(
        self,
        model_type: str = "ridge",
        alpha: float = 10.0,
        poly_degree: int = 2,
        features: Optional[List[str]] = None,
    ):
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.poly_degree = poly_degree
        self.features = list(features) if features is not None else list(DECISION_FEATURES)
        self.models: Dict[str, Union[Pipeline, LinearRegression, Ridge]] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}
        self.extrapolation_detector = ExtrapolationDetector()
        self.is_fitted = False

    def _build_estimator(self):
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=self.alpha)),
            ])
        elif self.model_type == "poly":
            return Pipeline([
                ("poly", PolynomialFeatures(degree=self.poly_degree, include_bias=False)),
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=self.alpha)),
            ])
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose 'linear', 'ridge', or 'poly'.")

    def fit(self, df: pd.DataFrame, cv_splits: int = 5) -> ProcessSurrogateModel:
        """Fit surrogates using TimeSeriesSplit chronological cross-validation."""
        X = df[self.features].to_numpy(dtype=float)
        self.extrapolation_detector.fit(X)

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        for target in RESPONSE_TARGETS:
            y = df[target].to_numpy(dtype=float)
            cv_mae, cv_rmse, cv_r2, cv_rel_err = [], [], [], []

            for train_idx, test_idx in tscv.split(X):
                estimator = self._build_estimator()
                estimator.fit(X[train_idx], y[train_idx])
                y_pred = estimator.predict(X[test_idx])

                mae = float(mean_absolute_error(y[test_idx], y_pred))
                rmse = float(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
                r2 = float(r2_score(y[test_idx], y_pred))
                rel_err = float((mae / np.mean(y[test_idx])) * 100.0) if np.mean(y[test_idx]) != 0 else 0.0

                cv_mae.append(mae)
                cv_rmse.append(rmse)
                cv_r2.append(r2)
                cv_rel_err.append(rel_err)

            # Fit final model on all baseline data
            final_estimator = self._build_estimator()
            final_estimator.fit(X, y)
            self.models[target] = final_estimator

            y_train_pred = final_estimator.predict(X)
            train_mae = float(mean_absolute_error(y, y_train_pred))
            train_rmse = float(np.sqrt(mean_squared_error(y, y_train_pred)))
            train_r2 = float(r2_score(y, y_train_pred))

            self.metrics[target] = {
                "mean_val": float(np.mean(y)),
                "std_val": float(np.std(y)),
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_r2": train_r2,
                "cv_mae": float(np.mean(cv_mae)),
                "cv_rmse": float(np.mean(cv_rmse)),
                "cv_r2": float(np.mean(cv_r2)),
                "cv_rel_error_pct": float(np.mean(cv_rel_err)),
            }

        self.is_fitted = True
        return self

    def predict(self, u: Union[np.ndarray, List[float], pd.Series]) -> Dict[str, float]:
        """Predict key process responses given decision vector u."""
        if not self.is_fitted:
            raise RuntimeError("Surrogate models have not been fitted yet.")
        
        u_arr = np.asarray(u, dtype=float).reshape(1, -1)
        preds = {}
        for target, estimator in self.models.items():
            preds[target] = float(estimator.predict(u_arr)[0])
        return preds

    def check_extrapolation(self, u: Union[np.ndarray, List[float]]) -> Tuple[bool, Dict[str, float]]:
        """Verify whether decision vector u is within the validated operating domain."""
        u_arr = np.asarray(u, dtype=float).flatten()
        return self.extrapolation_detector.is_within_domain(u_arr)
