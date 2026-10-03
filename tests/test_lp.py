"""Automated test suite for Linear Programming and Dual Simplex Sensitivity Analysis.

Author: Team B4 (23MNG336: Operational Research)
"""

import pytest
import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.models.surrogate_models import (
    ProcessSurrogateModel,
    DECISION_FEATURES,
    DECISION_FEATURES_8,
    RESPONSE_TARGETS,
)
from src.optimization.lp_optimizer import (
    solve_lp_production_maximization,
    solve_lp_cost_minimization,
    solve_lp_energy_minimization,
    LPResult,
)
from src.analysis.lp_sensitivity import (
    generate_variable_sensitivity_table,
    generate_constraint_sensitivity_table,
    format_full_sensitivity_report,
    compute_local_what_if_sensitivity,
    summarize_largest_sensitivities,
)


@pytest.fixture(scope="module")
def lp_setup():
    """Train surrogate model and extract baseline once for LP tests."""
    df = load_tep_dataset("d00")
    model = ProcessSurrogateModel(model_type="linear").fit(df)
    u_base = df[DECISION_FEATURES].mean().to_numpy()
    return model, u_base


def test_lp_production_maximization(lp_setup):
    """Verify LP Problem 1: Production Maximization solves to feasibility with Dual Simplex."""
    model, u_base = lp_setup
    result = solve_lp_production_maximization(
        surrogate_model=model,
        u_baseline=u_base,
        max_pressure=2800.0,
        max_compressor_power=355.0,
        trust_delta=10.0,
    )

    assert isinstance(result, LPResult)
    assert result.success is True
    assert result.objective_value >= result.baseline_objective_value
    assert result.predicted_responses["XMEAS_07"] <= 2800.0 + 1e-3
    assert result.predicted_responses["XMEAS_20"] <= 355.0 + 1e-3
    assert len(result.shadow_prices) == 2
    assert len(result.reduced_costs) == len(DECISION_FEATURES)
    assert result.iterations >= 0


def test_lp_cost_minimization(lp_setup):
    """Verify LP Problem 2: Cost Minimization satisfies demand target within energy/safety limits."""
    model, u_base = lp_setup
    target_demand = 22.89
    result = solve_lp_cost_minimization(
        surrogate_model=model,
        u_baseline=u_base,
        target_production=target_demand,
        max_pressure=2800.0,
        max_compressor_power=355.0,
        trust_delta=10.0,
    )

    assert isinstance(result, LPResult)
    assert result.success is True
    # Production target must be satisfied
    assert result.predicted_responses["XMEAS_17"] >= target_demand - 1e-3
    assert result.predicted_responses["XMEAS_07"] <= 2800.0 + 1e-3
    assert result.predicted_responses["XMEAS_20"] <= 355.0 + 1e-3
    assert "Production Quota Target (m3/hr)" in result.shadow_prices


def test_lp_energy_minimization_total(lp_setup):
    """Verify LP Problem 3: Total Energy Minimization (Electrical + Steam Thermal)."""
    model, u_base = lp_setup
    target_demand = 22.89
    result = solve_lp_energy_minimization(
        surrogate_model=model,
        u_baseline=u_base,
        target_production=target_demand,
        max_pressure=2800.0,
        max_compressor_power=355.0,
        trust_delta=10.0,
        energy_mode="total",
    )

    assert isinstance(result, LPResult)
    assert result.success is True
    assert result.objective_value <= result.baseline_objective_value
    assert result.predicted_responses["XMEAS_17"] >= target_demand - 1e-3
    assert result.predicted_responses["XMEAS_07"] <= 2800.0 + 1e-3
    assert result.predicted_responses["XMEAS_20"] <= 355.0 + 1e-3
    assert "Total Equivalent Process Power" in result.objective_name


def test_lp_energy_minimization_compressor(lp_setup):
    """Verify LP Problem 3: Compressor Power Minimization (Electrical Work Only)."""
    model, u_base = lp_setup
    target_demand = 22.89
    result = solve_lp_energy_minimization(
        surrogate_model=model,
        u_baseline=u_base,
        target_production=target_demand,
        max_pressure=2800.0,
        max_compressor_power=355.0,
        trust_delta=10.0,
        energy_mode="compressor",
    )

    assert isinstance(result, LPResult)
    assert result.success is True
    assert result.objective_value <= result.baseline_objective_value
    assert "Compressor Electrical Power" in result.objective_name


def test_lp_8variable_response_bounded_optimization():
    """Verify that the 8-variable model (+XMV_11) with empirical response bounds reproduces benchmark improvements."""
    df = load_tep_dataset("d00")
    model = ProcessSurrogateModel(model_type="ridge", alpha=10.0, features=DECISION_FEATURES_8).fit(df)
    u_base = df[DECISION_FEATURES_8].mean().to_numpy()

    resp_bounds = {t: (float(df[t].min()), float(df[t].max())) for t in RESPONSE_TARGETS}
    feat_bounds = {f: (float(df[f].min()), float(df[f].max())) for f in DECISION_FEATURES_8}

    res_prod = solve_lp_production_maximization(
        surrogate_model=model,
        u_baseline=u_base,
        response_bounds=resp_bounds,
        feature_bounds=feat_bounds,
    )
    assert res_prod.success is True
    # Verifies the documented ~7.73% increase (22.91 -> 24.68 m3/hr)
    assert res_prod.objective_value == pytest.approx(24.681, abs=0.05)
    assert res_prod.improvement_pct == pytest.approx(7.73, abs=0.2)

    res_cost = solve_lp_cost_minimization(
        surrogate_model=model,
        u_baseline=u_base,
        target_production=float(df["XMEAS_17"].mean()),
        response_bounds=resp_bounds,
        feature_bounds=feat_bounds,
    )
    assert res_cost.success is True
    # Verifies the documented ~8.15% cost reduction ($106.40 -> $97.73/hr)
    assert res_cost.objective_value == pytest.approx(97.73, abs=0.1)
    assert res_cost.improvement_pct == pytest.approx(8.15, abs=0.2)


def test_lp_sensitivity_report_generation(lp_setup):
    """Verify generation of tabular and text Simplex Sensitivity Reports."""
    model, u_base = lp_setup
    res = solve_lp_production_maximization(model, u_base)

    var_df = generate_variable_sensitivity_table(res)
    assert isinstance(var_df, pd.DataFrame)
    assert len(var_df) == len(DECISION_FEATURES)
    assert "Reduced Cost" in var_df.columns
    assert "Status" in var_df.columns

    const_df = generate_constraint_sensitivity_table(res)
    assert isinstance(const_df, pd.DataFrame)
    assert len(const_df) == len(res.shadow_prices)
    assert "Shadow Price (Dual Value)" in const_df.columns
    assert "Binding Status" in const_df.columns

    report_text = format_full_sensitivity_report(res)
    assert "OPERATIONS RESEARCH SIMPLEX SENSITIVITY REPORT" in report_text
    assert "VARIABLE CELLS SENSITIVITY" in report_text
    assert "CONSTRAINTS & SHADOW PRICES" in report_text


def test_local_what_if_sensitivity():
    """Verify local what-if finite difference sensitivity computations and summaries."""
    df = load_tep_dataset("d00")
    model = ProcessSurrogateModel(model_type="ridge", alpha=10.0, features=DECISION_FEATURES_8).fit(df)
    u_base = df[DECISION_FEATURES_8].mean().to_numpy()

    sens_df = compute_local_what_if_sensitivity(model, u_base, df, perturbation_frac=0.01)
    assert isinstance(sens_df, pd.DataFrame)
    assert not sens_df.empty
    assert "input_variable" in sens_df.columns
    assert "response_change" in sens_df.columns
    assert "local_slope" in sens_df.columns

    summary = summarize_largest_sensitivities(sens_df)
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == len(RESPONSE_TARGETS)
    assert "response" in summary.columns

