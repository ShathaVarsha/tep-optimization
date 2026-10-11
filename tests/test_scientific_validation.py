"""Comprehensive Scientific Validation and Verification Test Suite for TEP Optimization.

Tests verify:
  1. Component-wise purge cost calculation against hand-calculated analytical benchmarks.
  2. Full composition-based operating cost vs linear LP proxy cost and exact percentage reduction.
  3. Independent constraint feasibility checking across all three LP formulations (P1, P2, P3).
  4. Detection of deliberate constraint and bound violations.
  5. Solver failure handling and scientific feasibility distinction.
  6. Analytical known LP with verified SciPy reduced costs and shadow prices.
  7. Extrapolation diagnostics (Mahalanobis distance squared, chi-square threshold, box limits).
  8. Exact energy conversions and unit consistency.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.data.tep_loader import load_tep_dataset
from src.models.surrogate_models import (
    ProcessSurrogateModel,
    ExtrapolationDetector,
    DECISION_FEATURES_8,
    RESPONSE_TARGETS,
)
from src.optimization.cost_model import (
    calculate_purge_cost,
    calculate_compressor_cost,
    calculate_steam_cost,
    evaluate_operating_cost,
    evaluate_operating_cost_breakdown,
    calculate_cost_reduction_pct,
    KSCMH_TO_KMOL_HR,
    PRICE_PURGE_DEFAULT,
    PRICE_PURGE_BASELINE_EFFECTIVE,
    PRICE_ELECTRICITY,
    PRICE_STEAM,
    PURGE_PRICES,
    BASELINE_PURGE_MOLE_PCT,
)
from src.optimization.energy_model import (
    calculate_compressor_power,
    calculate_steam_thermal_power,
    evaluate_energy_breakdown,
    LATENT_HEAT_STEAM_KJ_KG,
    SECONDS_PER_HOUR,
)
from src.optimization.lp_optimizer import (
    solve_lp_production_maximization,
    solve_lp_cost_minimization,
    solve_lp_energy_minimization,
    verify_solution_feasibility,
    LPResult,
)


@pytest.fixture(scope="module")
def d00_dataset():
    """Load baseline dataset once for scientific validation tests."""
    return load_tep_dataset("d00")


@pytest.fixture(scope="module")
def fitted_surrogate(d00_dataset):
    """Fit 8-variable Ridge surrogate model once for validation tests."""
    sm = ProcessSurrogateModel(model_type="ridge", alpha=10.0, features=DECISION_FEATURES_8)
    sm.fit(d00_dataset, cv_splits=5)
    return sm


# ==============================================================================
# 1. HAND-CALCULATED COMPONENT-WISE PURGE COST BENCHMARK
# ==============================================================================

def test_hand_calculated_component_purge_cost():
    """Verify component-wise purge cost against a hand-calculated analytical example.

    Test parameters:
      purge_rate = 1.0 kscmh
      total_molar_flow = 1.0 * 44.619371 = 44.619371 kmol/hr
      Equimolar composition (12.5% each for A through H)
    """
    molar_flow = 1.0 * KSCMH_TO_KMOL_HR  # 44.619371 kmol/hr
    pct = 12.5  # 12.5% each

    # Analytical sum of unit prices:
    expected_avg_price = sum(PURGE_PRICES.values()) / 8.0
    # Prices: A: 2.206, B: 0.0, C: 6.016, D: 8.222, E: 10.428, F: 14.444, G: 17.653, H: 17.653
    # Sum = 76.622 $/kmol -> Average = 9.57775 $/kmol
    expected_total_cost = molar_flow * expected_avg_price

    calculated_cost = calculate_purge_cost(
        purge_rate_kscmh=1.0,
        y_A=pct, y_B=pct, y_C=pct, y_D=pct,
        y_E=pct, y_F=pct, y_G=pct, y_H=pct,
    )

    assert calculated_cost == pytest.approx(expected_total_cost, rel=1e-6)
    assert calculated_cost == pytest.approx(44.619371 * 9.57775, rel=1e-5)


def test_baseline_effective_purge_price_derivation(d00_dataset):
    """Verify that $5.80/kmol is the rounded effective price of baseline purge gas."""
    avg = d00_dataset.mean()
    p_rate = float(avg["XMEAS_10"])
    total_molar = p_rate * KSCMH_TO_KMOL_HR

    full_purge_cost = float(
        calculate_purge_cost(
            p_rate,
            avg["XMEAS_29"], avg["XMEAS_30"], avg["XMEAS_31"], avg["XMEAS_32"],
            avg["XMEAS_33"], avg["XMEAS_34"], avg["XMEAS_35"], avg["XMEAS_36"],
        )
    )

    effective_price = full_purge_cost / total_molar
    assert effective_price == pytest.approx(PRICE_PURGE_BASELINE_EFFECTIVE, rel=1e-4)
    assert round(effective_price, 2) == PRICE_PURGE_DEFAULT  # 5.80 $/kmol


# ==============================================================================
# 2. FULL COST VS LP PROXY AND COST REDUCTION PERCENTAGE
# ==============================================================================

def test_full_cost_vs_lp_proxy_consistency(fitted_surrogate, d00_dataset):
    """Verify full operating cost evaluation and consistency with LP proxy."""
    u_base = d00_dataset[DECISION_FEATURES_8].mean().to_numpy()
    resp_bounds = {t: (float(d00_dataset[t].min()), float(d00_dataset[t].max())) for t in RESPONSE_TARGETS}
    feat_bounds = {f: (float(d00_dataset[f].min()), float(d00_dataset[f].max())) for f in DECISION_FEATURES_8}
    target_prod = float(d00_dataset["XMEAS_17"].mean())

    res = solve_lp_cost_minimization(
        surrogate_model=fitted_surrogate,
        u_baseline=u_base,
        target_production=target_prod,
        response_bounds=resp_bounds,
        feature_bounds=feat_bounds,
    )

    assert res.success is True
    assert res.full_cost_baseline is not None
    assert res.full_cost_optimal is not None
    assert res.full_cost_improvement_pct is not None

    # Baseline: Full cost = $106.39/hr, LP proxy = $106.40/hr (diff < 0.02 $/hr)
    assert res.full_cost_baseline == pytest.approx(106.386, abs=0.05)
    assert res.baseline_objective_value == pytest.approx(106.401, abs=0.05)

    # Optimized: Full cost = $97.71/hr, LP proxy = $97.73/hr (diff < 0.03 $/hr)
    assert res.full_cost_optimal == pytest.approx(97.715, abs=0.1)
    assert res.objective_value == pytest.approx(97.725, abs=0.1)

    # Reductions: Both evaluate to ~8.15%
    assert res.full_cost_improvement_pct == pytest.approx(8.15, abs=0.1)
    assert res.improvement_pct == pytest.approx(8.15, abs=0.1)

    # Direct percentage formula check
    recomputed_pct = calculate_cost_reduction_pct(res.full_cost_baseline, res.full_cost_optimal)
    assert recomputed_pct == pytest.approx(res.full_cost_improvement_pct, abs=1e-5)


# ==============================================================================
# 3. INDEPENDENT FEASIBILITY VALIDATION FOR ALL THREE LP SOLUTIONS
# ==============================================================================

def test_independent_feasibility_validation_all_lps(fitted_surrogate, d00_dataset):
    """Verify that every LP solution is scientifically feasible against all mathematical constraints."""
    u_base = d00_dataset[DECISION_FEATURES_8].mean().to_numpy()
    resp_bounds = {t: (float(d00_dataset[t].min()), float(d00_dataset[t].max())) for t in RESPONSE_TARGETS}
    feat_bounds = {f: (float(d00_dataset[f].min()), float(d00_dataset[f].max())) for f in DECISION_FEATURES_8}
    target_prod = float(d00_dataset["XMEAS_17"].mean())

    # 1. Production Maximization
    r1 = solve_lp_production_maximization(
        fitted_surrogate, u_base, response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert r1.success is True
    assert r1.is_scientifically_feasible is True
    f1 = verify_solution_feasibility(
        r1, fitted_surrogate, response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert f1["is_feasible"] is True
    assert f1["num_violated"] == 0
    assert f1["max_violation"] == 0.0

    # 2. Cost Minimization
    r2 = solve_lp_cost_minimization(
        fitted_surrogate, u_base, target_production=target_prod,
        response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert r2.success is True
    assert r2.is_scientifically_feasible is True
    f2 = verify_solution_feasibility(
        r2, fitted_surrogate, target_production=target_prod,
        response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert f2["is_feasible"] is True
    assert f2["num_violated"] == 0

    # 3. Energy Minimization
    r3 = solve_lp_energy_minimization(
        fitted_surrogate, u_base, target_production=target_prod,
        response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert r3.success is True
    assert r3.is_scientifically_feasible is True
    f3 = verify_solution_feasibility(
        r3, fitted_surrogate, target_production=target_prod,
        response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert f3["is_feasible"] is True
    assert f3["num_violated"] == 0


def test_feasibility_validator_detects_deliberate_violation(fitted_surrogate, d00_dataset):
    """Verify that verify_solution_feasibility correctly identifies and flags constraint violations."""
    u_base = d00_dataset[DECISION_FEATURES_8].mean().to_numpy()
    resp_bounds = {t: (float(d00_dataset[t].min()), float(d00_dataset[t].max())) for t in RESPONSE_TARGETS}

    r1 = solve_lp_production_maximization(fitted_surrogate, u_base, response_bounds=resp_bounds)
    assert r1.success is True

    # Artificially modify optimal setpoint to violate trust bounds
    r1_tampered = LPResult(
        success=True,
        status_message=r1.status_message,
        problem_type=r1.problem_type,
        objective_name=r1.objective_name,
        objective_value=r1.objective_value,
        baseline_objective_value=r1.baseline_objective_value,
        improvement_pct=r1.improvement_pct,
        u_optimal={f: r1.u_optimal[f] + 50.0 for f in DECISION_FEATURES_8},  # Pushed 50% above baseline
        u_baseline=r1.u_baseline,
        predicted_responses=r1.predicted_responses,
        shadow_prices=r1.shadow_prices,
        slack_values=r1.slack_values,
        reduced_costs=r1.reduced_costs,
        binding_constraints=r1.binding_constraints,
        iterations=r1.iterations,
    )

    feas_tampered = verify_solution_feasibility(
        r1_tampered, fitted_surrogate, trust_delta=10.0, response_bounds=resp_bounds
    )
    assert feas_tampered["is_feasible"] is False
    assert feas_tampered["num_violated"] > 0
    assert feas_tampered["max_violation"] > 0.0
    assert len(feas_tampered["violations"]) > 0


def test_solver_failure_handling_in_validator(fitted_surrogate, d00_dataset):
    """Verify that solver failure is correctly distinguished and not marked feasible."""
    u_base = d00_dataset[DECISION_FEATURES_8].mean().to_numpy()
    resp_bounds = {t: (float(d00_dataset[t].min()), float(d00_dataset[t].max())) for t in RESPONSE_TARGETS}
    feat_bounds = {f: (float(d00_dataset[f].min()), float(d00_dataset[f].max())) for f in DECISION_FEATURES_8}

    # Request impossible quota of 40 m3/hr (nominal is ~22.9, max observed is ~25.5)
    r_infeasible = solve_lp_cost_minimization(
        fitted_surrogate, u_base, target_production=40.0,
        response_bounds=resp_bounds, feature_bounds=feat_bounds
    )
    assert r_infeasible.success is False
    assert r_infeasible.is_scientifically_feasible is False


# ==============================================================================
# 4. ANALYTICAL LP WITH KNOWN REDUCED COSTS AND SHADOW PRICES
# ==============================================================================

def test_analytical_known_lp_reduced_costs_and_shadow_prices():
    """Verify reduced-cost and shadow-price signs on an analytical textbook LP.

    Problem:
      min  z = 2*x1 + 5*x2
      s.t. x1 + x2 >= 6   (-x1 - x2 <= -6)
           x1 <= 4        (bounds: 0 <= x1 <= 4, 0 <= x2 <= 10)

    Analytical Solution:
      Since x1 is cheaper ($2 vs $5), x1 hits upper bound 4.
      x2 takes remaining requirement: x2 = 2.
      Optimal vertex: (x1*, x2*) = (4, 2), z* = 2(4) + 5(2) = 18.

    Analytical Marginals:
      - x1 is at upper bound 4: reduced cost must be negative (incentive to increase x1).
      - x2 is interior / basic (0 < 2 < 10): reduced cost must be 0.
      - Resource constraint x1 + x2 >= 6 is binding: shadow price must be positive ($5/unit).
    """
    c = [2.0, 5.0]
    A_ub = [[-1.0, -1.0]]
    b_ub = [-6.0]
    bounds = [(0.0, 4.0), (0.0, 10.0)]

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")
    assert res.success is True
    assert res.x[0] == pytest.approx(4.0)
    assert res.x[1] == pytest.approx(2.0)
    assert res.fun == pytest.approx(18.0)

    # SciPy marginals:
    # lower.marginals: [0, 0]
    # upper.marginals: [-3.0, 0.0] (for x1: increasing upper bound by 1 replaces 1 unit of x2 with x1, saving 5 - 2 = 3)
    # ineqlin.marginals: [-5.0] (relaxing RHS of -x1 - x2 <= -6 to -5 reduces cost by 5)

    assert res.upper.marginals[0] == pytest.approx(-3.0)
    assert res.upper.marginals[1] == pytest.approx(0.0)
    assert res.lower.marginals[0] == pytest.approx(0.0)
    assert res.lower.marginals[1] == pytest.approx(0.0)
    assert res.ineqlin.marginals[0] == pytest.approx(-5.0)

    # Our verified reduced-cost formula: lower + upper
    red_cost_x1 = float(res.lower.marginals[0] + res.upper.marginals[0])
    red_cost_x2 = float(res.lower.marginals[1] + res.upper.marginals[1])
    assert red_cost_x1 == pytest.approx(-3.0)  # Negative at upper bound
    assert red_cost_x2 == pytest.approx(0.0)   # Zero for basic variable

    # Our shadow price convention: -ineqlin_marginals >= 0
    shadow_price = float(-res.ineqlin.marginals[0])
    assert shadow_price == pytest.approx(5.0)  # Positive marginal savings of relaxing quota


# ==============================================================================
# 5. EXTRAPOLATION DETECTOR AND STATISTICAL DOMAIN DIAGNOSTICS
# ==============================================================================

def test_extrapolation_diagnostics_baseline_and_shifted(d00_dataset):
    """Verify Mahalanobis distance squared, chi2 threshold, and box limits."""
    detector = ExtrapolationDetector(confidence=0.99, box_expansion=0.20)
    X = d00_dataset[DECISION_FEATURES_8].to_numpy(dtype=float)
    detector.fit(X)

    u_base = np.mean(X, axis=0)
    valid_base, diag_base = detector.is_within_domain(u_base)

    # At empirical mean, D_M^2 must be 0
    assert diag_base["mahalanobis_dist_sq"] == pytest.approx(0.0, abs=1e-6)
    assert valid_base is True
    assert diag_base["mahalanobis_valid"] is True
    assert diag_base["box_valid"] is True
    # For 8 dimensions at 99% confidence: chi2.ppf(0.99, df=8) ~ 20.0902
    assert diag_base["critical_threshold"] == pytest.approx(20.09, abs=0.05)

    # Pushing a single variable far outside empirical bounds triggers box failure
    u_extreme = np.array(u_base)
    u_extreme[0] += 100.0  # Out of range
    valid_ext, diag_ext = detector.is_within_domain(u_extreme)
    assert valid_ext is False
    assert diag_ext["box_valid"] is False
    assert diag_ext["mahalanobis_dist_sq"] > diag_base["critical_threshold"]


# ==============================================================================
# 6. ENERGY FORMULATION AND CONVERSIONS
# ==============================================================================

def test_energy_conversion_factors_and_duty():
    """Verify thermal conversion factor from kg/hr steam to kW."""
    # 2257 kJ/kg / 3600 s/hr = 0.626944... kW/(kg/hr)
    conversion = LATENT_HEAT_STEAM_KJ_KG / SECONDS_PER_HOUR
    assert conversion == pytest.approx(0.6269444, rel=1e-5)

    steam_rate = 1000.0  # kg/hr
    expected_thermal_power = 1000.0 * conversion
    calc_power = calculate_steam_thermal_power(steam_rate)
    assert calc_power == pytest.approx(expected_thermal_power, rel=1e-6)

    comp_power = 300.0  # kW
    assert calculate_compressor_power(comp_power) == 300.0
