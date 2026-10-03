"""Automated end-to-end test suite for Streamlit UI workflow and backend integration.

Author: Team B4 (23MNG336: Operational Research)
"""

import pytest
import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.data.statistics_generator import generate_summary_statistics, compute_correlation_matrix
from src.process.variable_dictionary import VARIABLE_CATALOG
from src.process.baseline_analyzer import perform_stationarity_analysis, extract_baseline_vector
from src.models.surrogate_models import (
    ProcessSurrogateModel,
    DECISION_FEATURES_7,
    DECISION_FEATURES_8,
    RESPONSE_TARGETS,
)
from src.optimization.cost_model import evaluate_operating_cost_breakdown
from src.optimization.energy_model import evaluate_energy_breakdown
from src.optimization.lp_optimizer import (
    solve_lp_production_maximization,
    solve_lp_cost_minimization,
    solve_lp_energy_minimization,
)
from src.optimization.goal_programming import (
    GoalSpec,
    solve_weighted_sum_gp,
    solve_preemptive_gp,
)
from src.analysis.lp_sensitivity import (
    generate_variable_sensitivity_table,
    generate_constraint_sensitivity_table,
    compute_local_what_if_sensitivity,
    summarize_largest_sensitivities,
)


@pytest.fixture(scope="module")
def d00_data():
    return load_tep_dataset("d00")


def test_ui_data_and_statistics_flow(d00_data):
    """Verify data preview and statistical summary generation for UI tables."""
    assert d00_data.shape == (500, 52)
    stats = generate_summary_statistics(d00_data)
    assert stats.shape[0] == 52

    key_tags = ["XMEAS_17", "XMEAS_07", "XMV_01", "XMV_11"]
    corr = compute_correlation_matrix(d00_data, variables=key_tags)
    assert corr.shape == (4, 4)


def test_ui_baseline_and_utility_flow(d00_data):
    """Verify baseline energy and cost calculations used in UI summary metrics."""
    cost = evaluate_operating_cost_breakdown(d00_data)
    assert cost["total_operating_cost_usd_hr"] > 100.0

    energy = evaluate_energy_breakdown(d00_data)
    assert energy["total_energy_power_kw"] > 400.0


def test_ui_lp_all_objectives_flow(d00_data):
    """Verify that all LP objectives run without error and produce exportable dataframes."""
    model = ProcessSurrogateModel(model_type="ridge", alpha=10.0, features=DECISION_FEATURES_8).fit(d00_data)
    u_base = d00_data[DECISION_FEATURES_8].mean().to_numpy()

    resp_bounds = {t: (float(d00_data[t].min()), float(d00_data[t].max())) for t in RESPONSE_TARGETS}
    feat_bounds = {f: (float(d00_data[f].min()), float(d00_data[f].max())) for f in DECISION_FEATURES_8}

    # 1. Production Max
    res_prod = solve_lp_production_maximization(
        model, u_base, response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert res_prod.success is True
    var_table = generate_variable_sensitivity_table(res_prod)
    assert isinstance(var_table, pd.DataFrame)
    assert len(var_table) == 8

    # 2. Cost Min
    res_cost = solve_lp_cost_minimization(
        model, u_base, target_production=22.89, response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert res_cost.success is True
    const_table = generate_constraint_sensitivity_table(res_cost)
    assert isinstance(const_table, pd.DataFrame)

    # 3. Energy Min (total)
    res_energy_tot = solve_lp_energy_minimization(
        model, u_base, target_production=22.89, energy_mode="total", response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert res_energy_tot.success is True

    # 4. Energy Min (compressor)
    res_energy_comp = solve_lp_energy_minimization(
        model, u_base, target_production=22.89, energy_mode="compressor", response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert res_energy_comp.success is True


def test_ui_goal_programming_flow(d00_data):
    """Verify Goal Programming solver execution with 8-variable model."""
    model = ProcessSurrogateModel(model_type="linear", features=DECISION_FEATURES_8).fit(d00_data)
    u_base = d00_data[DECISION_FEATURES_8].mean().to_numpy()

    goals = [
        GoalSpec(name="Product Throughput", response_tag="XMEAS_17", target=24.0, direction="max", weight=5.0, priority=1),
        GoalSpec(name="Compressor Work", response_tag="XMEAS_20", target=340.0, direction="min", weight=3.0, priority=2),
    ]

    res_w = solve_weighted_sum_gp(surrogate=model, goals=goals, u_baseline=u_base, trust_delta=10.0)
    assert res_w.success is True

    res_p = solve_preemptive_gp(surrogate=model, goals=goals, u_baseline=u_base, trust_delta=10.0)
    assert res_p.success is True


def test_ui_sensitivity_flow(d00_data):
    """Verify sensitivity computations for UI charts and tables."""
    model = ProcessSurrogateModel(model_type="ridge", alpha=10.0, features=DECISION_FEATURES_8).fit(d00_data)
    u_base = d00_data[DECISION_FEATURES_8].mean().to_numpy()

    sens_df = compute_local_what_if_sensitivity(model, u_base, d00_data, perturbation_frac=0.01)
    assert not sens_df.empty
    summary_df = summarize_largest_sensitivities(sens_df)
    assert len(summary_df) == len(RESPONSE_TARGETS)
