"""Automated test suite for Stationarity Verification and Process Surrogate Models.

Author: Team B4 (23MNG336: Operational Research)
"""

import pytest
import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.process.baseline_analyzer import (
    perform_stationarity_analysis,
    extract_baseline_vector,
    get_baseline_point,
)
from src.models.surrogate_models import (
    ProcessSurrogateModel,
    ExtrapolationDetector,
    DECISION_FEATURES,
    DECISION_FEATURES_8,
    RESPONSE_TARGETS,
)


@pytest.fixture(scope="module")
def baseline_data():
    """Load d00.dat once for the test module."""
    return load_tep_dataset("d00")


def test_stationarity_analysis_on_d00(baseline_data):
    """Verify that d00.dat satisfies multi-criteria stationarity and domain sanity checks."""
    result = perform_stationarity_analysis(baseline_data)

    assert "variable_results" in result
    assert "domain_checks" in result
    assert "is_validated_baseline" in result

    # Check that critical variables pass ADF stationarity
    var_res = result["variable_results"]
    for v in ["XMEAS_07", "XMEAS_08", "XMEAS_09", "XMEAS_17", "XMEAS_20"]:
        assert var_res[v]["adf_pvalue"] < 0.05, f"{v} failed ADF stationarity with p={var_res[v]['adf_pvalue']}"
        assert var_res[v]["mean_drift_pct"] < 5.0, f"{v} experienced excessive rolling mean drift"

    # Domain sanity checks
    assert result["domain_checks"]["reactor_pressure_in_range"] is True
    assert result["domain_checks"]["reactor_temp_in_range"] is True
    assert result["domain_checks"]["product_flow_in_range"] is True
    assert result["is_validated_baseline"] is True


def test_baseline_vector_extraction(baseline_data):
    """Verify that full 52-variable baseline vectors and statistics are correctly extracted."""
    baseline_stats = extract_baseline_vector(baseline_data)
    assert len(baseline_stats) == 52
    assert "mean" in baseline_stats.columns
    assert "std" in baseline_stats.columns
    assert "q05" in baseline_stats.columns
    assert "q95" in baseline_stats.columns

    nominal_point = get_baseline_point(baseline_data)
    assert len(nominal_point) == 52
    assert nominal_point["XMEAS_07"] == pytest.approx(2705.4, abs=5.0)


def test_surrogate_model_training_and_prediction(baseline_data):
    """Verify Ridge surrogate training, cross-validation metrics, and prediction on nominal point."""
    surrogate = ProcessSurrogateModel(model_type="ridge", alpha=10.0)
    surrogate.fit(baseline_data, cv_splits=5)

    assert surrogate.is_fitted is True
    assert set(surrogate.models.keys()) == set(RESPONSE_TARGETS)

    # Evaluate prediction on nominal operating point
    u_base = baseline_data[DECISION_FEATURES].mean().to_numpy()
    preds = surrogate.predict(u_base)

    assert "XMEAS_17" in preds
    assert "XMEAS_07" in preds
    assert "XMEAS_10" in preds
    assert "XMEAS_19" in preds
    assert "XMEAS_20" in preds

    # Check nominal predictions match empirical data within tight margin (< 2%)
    assert preds["XMEAS_17"] == pytest.approx(float(baseline_data["XMEAS_17"].mean()), rel=0.03)
    assert preds["XMEAS_07"] == pytest.approx(float(baseline_data["XMEAS_07"].mean()), rel=0.01)
    assert preds["XMEAS_20"] == pytest.approx(float(baseline_data["XMEAS_20"].mean()), rel=0.01)

    # Verify that metrics are recorded transparently
    for target in RESPONSE_TARGETS:
        metric = surrogate.metrics[target]
        assert "cv_mae" in metric
        assert "cv_rmse" in metric
        assert "cv_r2" in metric
        assert "cv_rel_error_pct" in metric
        assert metric["cv_rel_error_pct"] < 5.0, f"{target} relative error ({metric['cv_rel_error_pct']}%) exceeded 5%"


def test_extrapolation_detection(baseline_data):
    """Verify Mahalanobis distance domain boundary detection."""
    X = baseline_data[DECISION_FEATURES].to_numpy()
    detector = ExtrapolationDetector(confidence=0.99).fit(X)

    # Baseline mean should be strictly inside domain
    u_mean = np.mean(X, axis=0)
    is_valid, details = detector.is_within_domain(u_mean)
    assert is_valid is True
    assert details["mahalanobis_valid"] is True
    assert details["box_valid"] is True

    # Extreme unphysical point should be flagged as extrapolation
    u_outlier = u_mean.copy()
    u_outlier[0] = 500.0  # D feed valve = 500% (impossible)
    is_valid_out, details_out = detector.is_within_domain(u_outlier)
    assert is_valid_out is False
    assert details_out["overall_valid"] is False


def test_linear_and_poly_surrogates(baseline_data):
    """Verify that OLS and Polynomial Ridge surrogates also fit and predict stably."""
    ols = ProcessSurrogateModel(model_type="linear").fit(baseline_data, cv_splits=3)
    assert ols.is_fitted is True

    poly = ProcessSurrogateModel(model_type="poly", alpha=50.0).fit(baseline_data, cv_splits=3)
    assert poly.is_fitted is True

    u_sample = baseline_data[DECISION_FEATURES].iloc[0].to_numpy()
    pred_ols = ols.predict(u_sample)
    pred_poly = poly.predict(u_sample)
    assert np.isfinite(pred_ols["XMEAS_17"])
    assert np.isfinite(pred_poly["XMEAS_17"])


def test_8variable_surrogate_performance(baseline_data):
    """Verify that adding XMV_11 boosts product flow CV R2 from near-zero to > 0.95."""
    model_7 = ProcessSurrogateModel(model_type="ridge", alpha=10.0, features=DECISION_FEATURES).fit(baseline_data)
    model_8 = ProcessSurrogateModel(model_type="ridge", alpha=10.0, features=DECISION_FEATURES_8).fit(baseline_data)

    r2_7 = model_7.metrics["XMEAS_17"]["cv_r2"]
    r2_8 = model_8.metrics["XMEAS_17"]["cv_r2"]

    assert r2_7 < 0.1, f"Expected 7-feature model to have poor product flow R2, got {r2_7}"
    assert r2_8 > 0.95, f"Expected 8-feature model to have high product flow R2 (>0.95), got {r2_8}"
    assert model_8.metrics["XMEAS_17"]["train_r2"] > 0.95
    assert model_8.metrics["XMEAS_17"]["cv_mae"] < 0.05

