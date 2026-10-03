"""Linear Programming (LP) and Dual Simplex Benchmark for the Tennessee Eastman Process.

Formulates local linear approximations around the nominal operating baseline:
  - LP Problem 1: Production Maximization subject to Pressure and Energy Limits.
  - LP Problem 2: Operating Cost Minimization subject to Target Production Quota.

Solvers:
  - Uses the Dual Simplex algorithm via SciPy's HiGHS solver ('highs-ds').
  - Extracts basic feasible solutions, slack variables, binding constraints,
    and shadow prices (dual variables).

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.pipeline import Pipeline
import pandas as pd
from scipy.optimize import linprog

from src.models.surrogate_models import ProcessSurrogateModel, DECISION_FEATURES
from src.optimization.cost_model import evaluate_operating_cost_breakdown
from src.optimization.energy_model import evaluate_energy_breakdown




def _get_linear_coefficients(estimator):
    """Extract linear coefficients in the original input scale."""
    if isinstance(estimator, Pipeline):
        steps = estimator.named_steps
        if "poly" in steps:
            raise ValueError("LP requires a linear surrogate, not a polynomial surrogate.")
        scaler = steps.get("scaler")
        regressor = steps.get("regressor")
        if scaler is not None and regressor is not None:
            coef = np.asarray(regressor.coef_, dtype=float).ravel()
            scale = np.asarray(scaler.scale_, dtype=float)
            mean = np.asarray(scaler.mean_, dtype=float)
            return coef / scale, float(regressor.intercept_ - np.dot(coef, mean / scale))
        raise TypeError("Unsupported Pipeline structure.")
    if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
        return (
            np.asarray(estimator.coef_, dtype=float).ravel(),
            float(np.asarray(estimator.intercept_).reshape(-1)[0]),
        )
    raise TypeError("Expected a fitted linear estimator or Ridge Pipeline.")

@dataclass
class LPResult:
    """Container for Linear Programming optimization results and Simplex diagnostics."""

    success: bool
    status_message: str
    problem_type: str
    objective_name: str
    objective_value: float
    baseline_objective_value: float
    improvement_pct: float
    u_optimal: Dict[str, float]
    u_baseline: Dict[str, float]
    predicted_responses: Dict[str, float]
    shadow_prices: Dict[str, float]
    slack_values: Dict[str, float]
    reduced_costs: Dict[str, float]
    binding_constraints: List[str]
    iterations: int


def solve_lp_production_maximization(
    surrogate_model: ProcessSurrogateModel,
    u_baseline: np.ndarray,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
) -> LPResult:
    """Solve LP Problem 1: Maximize finished product throughput within energy & safety limits.

    Formulation:
      max   Product_Flow(u) = c_prod^T u + d_prod
      s.t.  Reactor_Pressure(u) <= max_pressure
            Compressor_Power(u) <= max_compressor_power
            u_base - delta <= u <= u_base + delta
            0 <= u <= 100
    """
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    # Objective: Maximize Product Flow (minimize -c^T u)
    prod_est = surrogate_model.models["XMEAS_17"]
    prod_coef, intercept_prod = _get_linear_coefficients(prod_est)
    c = -prod_coef

    # Constraints:
    # 1. Reactor Pressure <= max_pressure
    press_est = surrogate_model.models["XMEAS_07"]
    a_press, intercept_press = _get_linear_coefficients(press_est)
    b_press = max_pressure - intercept_press
    # 2. Compressor Power <= max_compressor_power
    comp_est = surrogate_model.models["XMEAS_20"]
    a_comp, intercept_comp = _get_linear_coefficients(comp_est)
    b_comp = max_compressor_power - intercept_comp
    A_ub = np.vstack([a_press, a_comp])
    b_ub = np.array([b_press, b_comp])

    # Variable Trust Bounds around baseline
    bounds = []
    for u_val in u_baseline:
        lb = max(0.0, float(u_val - trust_delta))
        ub = min(100.0, float(u_val + trust_delta))
        bounds.append((lb, ub))

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

    if not res.success:
        raise RuntimeError(f"Dual Simplex LP solver failed to converge: {res.message}")

    u_opt = res.x
    opt_prod = float(-res.fun + intercept_prod)
    base_prod = float(prod_est.predict(u_baseline.reshape(1, -1))[0])
    pct_gain = ((opt_prod - base_prod) / base_prod) * 100.0 if base_prod > 0 else 0.0

    # Predictions
    preds = surrogate_model.predict(u_opt)
    preds["XMEAS_17"] = opt_prod

    # Extract Dual Information (Shadow Prices)
    # HiGHS reports marginals on inequality constraints
    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(2)
    shadow_prices = {
        "Reactor Pressure Limit": float(ineq_marginals[0]),
        "Compressor Power Limit": float(ineq_marginals[1]),
    }

    # Slack values
    slacks = {
        "Reactor Pressure Slack (kPa)": float(max_pressure - preds["XMEAS_07"]),
        "Compressor Power Slack (kW)": float(max_compressor_power - preds["XMEAS_20"]),
    }

    # Binding constraints (slack < 1e-4)
    binding = [k for k, v in slacks.items() if abs(v) < 1e-3]

    # Reduced costs on decision variables
    lower_marginals = res.lower.marginals if hasattr(res, "lower") else np.zeros(len(u_opt))
    upper_marginals = res.upper.marginals if hasattr(res, "upper") else np.zeros(len(u_opt))
    reduced_costs = {
        feat: float(upper_marginals[i] - lower_marginals[i])
        for i, feat in enumerate(DECISION_FEATURES)
    }

    return LPResult(
        success=bool(res.success),
        status_message=str(res.message),
        problem_type="LP Formulation 1",
        objective_name="Product Flow (m3/hr)",
        objective_value=opt_prod,
        baseline_objective_value=base_prod,
        improvement_pct=pct_gain,
        u_optimal={feat: float(u_opt[i]) for i, feat in enumerate(DECISION_FEATURES)},
        u_baseline={feat: float(u_baseline[i]) for i, feat in enumerate(DECISION_FEATURES)},
        predicted_responses=preds,
        shadow_prices=shadow_prices,
        slack_values=slacks,
        reduced_costs=reduced_costs,
        binding_constraints=binding,
        iterations=int(res.nit),
    )


def solve_lp_cost_minimization(
    surrogate_model: ProcessSurrogateModel,
    u_baseline: np.ndarray,
    target_production: float = 22.89,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
) -> LPResult:
    """Solve LP Problem 2: Minimize total operating cost subject to meeting target production quota.

    Formulation:
      min   Cost(u) = c_cost^T u + d_cost
      s.t.  Product_Flow(u) >= target_production  (-Product_Flow <= -target)
            Reactor_Pressure(u) <= max_pressure
            Compressor_Power(u) <= max_compressor_power
            u_base - delta <= u <= u_base + delta
    """
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    # Formulate linear cost vector: Cost ($/hr) = Purge_cost + Comp_cost + Steam_cost
    # Purge flow: XMEAS_10 * 44.619 * ~5.8 $/kmol avg
    # Comp power: XMEAS_20 * 0.0536 $/kWh
    # Steam flow: XMEAS_19 * 0.00318 $/kg
    purge_est = surrogate_model.models["XMEAS_10"]
    steam_est = surrogate_model.models["XMEAS_19"]
    comp_est = surrogate_model.models["XMEAS_20"]

    avg_purge_price_per_kmol = 5.80  # $/kmol approximate weighted average purge price
    kscmh_to_kmol = 44.61937
    c_purge_factor = kscmh_to_kmol * avg_purge_price_per_kmol

    purge_coef, purge_intercept = _get_linear_coefficients(purge_est)
    steam_coef, steam_intercept = _get_linear_coefficients(steam_est)
    comp_coef, comp_intercept = _get_linear_coefficients(comp_est)

    c_cost = (
        purge_coef * c_purge_factor
        + steam_coef * 0.00318
        + comp_coef * 0.0536
    )
    cost_intercept = (
        purge_intercept * c_purge_factor
        + steam_intercept * 0.00318
        + comp_intercept * 0.0536
    )

    # 1. Product_Flow >= target_production -> -a_prod^T u <= -(target_production - intercept_prod)
    prod_est = surrogate_model.models["XMEAS_17"]
    prod_coef, intercept_prod = _get_linear_coefficients(prod_est)
    a_prod = -prod_coef
    b_prod = -(target_production - intercept_prod)
    # 2. Reactor Pressure <= max_pressure
    press_est = surrogate_model.models["XMEAS_07"]
    a_press, intercept_press = _get_linear_coefficients(press_est)
    b_press = max_pressure - intercept_press
    # 3. Compressor Power <= max_compressor_power
    a_comp, intercept_comp = _get_linear_coefficients(comp_est)
    b_comp = max_compressor_power - intercept_comp
    A_ub = np.vstack([a_prod, a_press, a_comp])
    b_ub = np.array([b_prod, b_press, b_comp])

    bounds = []
    for u_val in u_baseline:
        lb = max(0.0, float(u_val - trust_delta))
        ub = min(100.0, float(u_val + trust_delta))
        bounds.append((lb, ub))

    res = linprog(c=c_cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

    if not res.success:
        raise RuntimeError(f"Dual Simplex LP solver failed to converge: {res.message}")

    u_opt = res.x
    opt_cost = float(res.fun + cost_intercept)
    base_cost = float(np.dot(c_cost, u_baseline) + cost_intercept)
    pct_reduction = ((base_cost - opt_cost) / base_cost) * 100.0 if base_cost > 0 else 0.0

    preds = surrogate_model.predict(u_opt)

    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(3)
    shadow_prices = {
        "Production Quota Target (m3/hr)": float(ineq_marginals[0]),
        "Reactor Pressure Limit (kPa)": float(ineq_marginals[1]),
        "Compressor Power Limit (kW)": float(ineq_marginals[2]),
    }

    slacks = {
        "Production Surplus (m3/hr)": float(preds["XMEAS_17"] - target_production),
        "Reactor Pressure Slack (kPa)": float(max_pressure - preds["XMEAS_07"]),
        "Compressor Power Slack (kW)": float(max_compressor_power - preds["XMEAS_20"]),
    }

    binding = [k for k, v in slacks.items() if abs(v) < 1e-3]

    lower_marginals = res.lower.marginals if hasattr(res, "lower") else np.zeros(len(u_opt))
    upper_marginals = res.upper.marginals if hasattr(res, "upper") else np.zeros(len(u_opt))
    reduced_costs = {
        feat: float(upper_marginals[i] - lower_marginals[i])
        for i, feat in enumerate(DECISION_FEATURES)
    }

    return LPResult(
        success=bool(res.success),
        status_message=str(res.message),
        problem_type="LP Formulation 2",
        objective_name="Operating Cost ($/hr)",
        objective_value=opt_cost,
        baseline_objective_value=base_cost,
        improvement_pct=pct_reduction,
        u_optimal={feat: float(u_opt[i]) for i, feat in enumerate(DECISION_FEATURES)},
        u_baseline={feat: float(u_baseline[i]) for i, feat in enumerate(DECISION_FEATURES)},
        predicted_responses=preds,
        shadow_prices=shadow_prices,
        slack_values=slacks,
        reduced_costs=reduced_costs,
        binding_constraints=binding,
        iterations=int(res.nit),
    )
