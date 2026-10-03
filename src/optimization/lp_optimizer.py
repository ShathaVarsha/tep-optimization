"""Linear Programming (LP) and Dual Simplex Benchmark for the Tennessee Eastman Process.

Formulates local linear approximations around the nominal operating baseline:
  - LP Problem 1: Production Maximization subject to Pressure, Energy, and Trust Limits.
  - LP Problem 2: Operating Cost Minimization subject to Target Production Quota.
  - LP Problem 3: Energy Consumption Minimization (Total Process Power or Compressor Work).

Solvers:
  - Uses the Dual Simplex algorithm via SciPy's HiGHS solver ('highs-ds').
  - Extracts basic feasible solutions, slack variables, binding constraints,
    reduced costs, and shadow prices (dual variables).

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.pipeline import Pipeline

from src.models.surrogate_models import (
    ProcessSurrogateModel,
    DECISION_FEATURES,
    DECISION_FEATURES_7,
    DECISION_FEATURES_8,
)
from src.optimization.cost_model import (
    evaluate_operating_cost_breakdown,
    KSCMH_TO_KMOL_HR,
    PRICE_ELECTRICITY,
    PRICE_STEAM,
)
from src.optimization.energy_model import (
    evaluate_energy_breakdown,
    LATENT_HEAT_STEAM_KJ_KG,
    SECONDS_PER_HOUR,
)

STEAM_THERMAL_KW_PER_KG_HR = LATENT_HEAT_STEAM_KJ_KG / SECONDS_PER_HOUR  # ~0.62694 kW/(kg/hr)


def _get_linear_coefficients(estimator) -> Tuple[np.ndarray, float]:
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


def _resolve_features(
    surrogate_model: ProcessSurrogateModel,
    features: Optional[List[str]] = None,
) -> List[str]:
    """Resolve active decision feature list from argument or model metadata."""
    if features is not None:
        return list(features)
    if hasattr(surrogate_model, "features") and surrogate_model.features:
        return list(surrogate_model.features)
    return list(DECISION_FEATURES)


def _build_feature_bounds(
    u_baseline: np.ndarray,
    features: List[str],
    trust_delta: float = 10.0,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """Build bound tuples for each decision variable."""
    bounds = []
    for i, feat in enumerate(features):
        u_val = float(u_baseline[i])
        f_min, f_max = (0.0, 100.0)
        if feature_bounds and feat in feature_bounds:
            f_min, f_max = feature_bounds[feat]
        lb = max(f_min, u_val - trust_delta)
        ub = min(f_max, u_val + trust_delta)
        bounds.append((lb, ub))
    return bounds


def _append_response_bounds(
    surrogate_model: ProcessSurrogateModel,
    A_rows: List[np.ndarray],
    b_rows: List[float],
    constraint_names: List[str],
    response_bounds: Optional[Dict[str, Tuple[float, float]]],
) -> None:
    """Append upper and lower linear response bounds to A_ub and b_ub."""
    if not response_bounds:
        return
    for target_name, (r_min, r_max) in response_bounds.items():
        if target_name not in surrogate_model.models:
            continue
        coef, intercept = _get_linear_coefficients(surrogate_model.models[target_name])
        if r_max is not None and np.isfinite(r_max):
            A_rows.append(coef)
            b_rows.append(float(r_max - intercept))
            constraint_names.append(f"{target_name} Upper Limit")
        if r_min is not None and np.isfinite(r_min):
            A_rows.append(-coef)
            b_rows.append(float(intercept - r_min))
            constraint_names.append(f"{target_name} Lower Limit")


def solve_lp_production_maximization(
    surrogate_model: ProcessSurrogateModel,
    u_baseline: np.ndarray,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
    response_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    features: Optional[List[str]] = None,
) -> LPResult:
    """Solve LP Problem 1: Maximize finished product throughput within energy & safety limits.

    Formulation:
      max   Product_Flow(u) = c_prod^T u + d_prod
      s.t.  Reactor_Pressure(u) <= max_pressure
            Compressor_Power(u) <= max_compressor_power
            [Optional: Response Bounds]
            u_base - delta <= u <= u_base + delta
            0 <= u <= 100
    """
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    feat_names = _resolve_features(surrogate_model, features)
    u_base_arr = np.asarray(u_baseline, dtype=float).ravel()

    # Objective: Maximize Product Flow (minimize -c^T u)
    prod_est = surrogate_model.models["XMEAS_17"]
    prod_coef, intercept_prod = _get_linear_coefficients(prod_est)
    c = -prod_coef

    # Core Engineering Constraints
    press_est = surrogate_model.models["XMEAS_07"]
    a_press, intercept_press = _get_linear_coefficients(press_est)
    b_press = max_pressure - intercept_press

    comp_est = surrogate_model.models["XMEAS_20"]
    a_comp, intercept_comp = _get_linear_coefficients(comp_est)
    b_comp = max_compressor_power - intercept_comp

    A_rows = [a_press, a_comp]
    b_rows = [b_press, b_comp]
    constraint_names = ["Reactor Pressure Limit", "Compressor Power Limit"]

    # Append any additional response bounds
    _append_response_bounds(surrogate_model, A_rows, b_rows, constraint_names, response_bounds)

    A_ub = np.vstack(A_rows)
    b_ub = np.asarray(b_rows, dtype=float)

    bounds = _build_feature_bounds(u_base_arr, feat_names, trust_delta, feature_bounds)

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

    if not res.success:
        raise RuntimeError(f"Dual Simplex LP solver failed to converge: {res.message}")

    u_opt = res.x
    opt_prod = float(-res.fun + intercept_prod)
    base_prod = float(prod_est.predict(u_base_arr.reshape(1, -1))[0])
    pct_gain = ((opt_prod - base_prod) / base_prod) * 100.0 if base_prod > 0 else 0.0

    preds = surrogate_model.predict(u_opt)
    preds["XMEAS_17"] = opt_prod

    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(len(constraint_names))
    shadow_prices = {
        name: float(ineq_marginals[idx])
        for idx, name in enumerate(constraint_names)
    }

    slacks = {
        "Reactor Pressure Slack (kPa)": float(max_pressure - preds["XMEAS_07"]),
        "Compressor Power Slack (kW)": float(max_compressor_power - preds["XMEAS_20"]),
    }
    if response_bounds:
        for t_name, (r_min, r_max) in response_bounds.items():
            if t_name in preds:
                if r_max is not None:
                    slacks[f"{t_name} Upper Slack"] = float(r_max - preds[t_name])
                if r_min is not None:
                    slacks[f"{t_name} Lower Surplus"] = float(preds[t_name] - r_min)

    binding = [k for k, v in slacks.items() if abs(v) < 1e-3]

    lower_marginals = res.lower.marginals if hasattr(res, "lower") else np.zeros(len(u_opt))
    upper_marginals = res.upper.marginals if hasattr(res, "upper") else np.zeros(len(u_opt))
    reduced_costs = {
        feat: float(upper_marginals[i] - lower_marginals[i])
        for i, feat in enumerate(feat_names)
    }

    return LPResult(
        success=bool(res.success),
        status_message=str(res.message),
        problem_type="LP Formulation 1",
        objective_name="Product Flow (m3/hr)",
        objective_value=opt_prod,
        baseline_objective_value=base_prod,
        improvement_pct=pct_gain,
        u_optimal={feat: float(u_opt[i]) for i, feat in enumerate(feat_names)},
        u_baseline={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
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
    avg_purge_price_per_kmol: float = 5.80,
    response_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    features: Optional[List[str]] = None,
) -> LPResult:
    """Solve LP Problem 2: Minimize total operating cost subject to meeting target production quota.

    Formulation:
      min   Cost(u) = c_cost^T u + d_cost
      s.t.  Product_Flow(u) >= target_production
            Reactor_Pressure(u) <= max_pressure
            Compressor_Power(u) <= max_compressor_power
            [Optional: Response Bounds]
            u_base - delta <= u <= u_base + delta
    """
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    feat_names = _resolve_features(surrogate_model, features)
    u_base_arr = np.asarray(u_baseline, dtype=float).ravel()

    purge_est = surrogate_model.models["XMEAS_10"]
    steam_est = surrogate_model.models["XMEAS_19"]
    comp_est = surrogate_model.models["XMEAS_20"]

    c_purge_factor = KSCMH_TO_KMOL_HR * avg_purge_price_per_kmol

    purge_coef, purge_intercept = _get_linear_coefficients(purge_est)
    steam_coef, steam_intercept = _get_linear_coefficients(steam_est)
    comp_coef, comp_intercept = _get_linear_coefficients(comp_est)

    c_cost = (
        purge_coef * c_purge_factor
        + steam_coef * PRICE_STEAM
        + comp_coef * PRICE_ELECTRICITY
    )
    cost_intercept = (
        purge_intercept * c_purge_factor
        + steam_intercept * PRICE_STEAM
        + comp_intercept * PRICE_ELECTRICITY
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

    A_rows = [a_prod, a_press, a_comp]
    b_rows = [b_prod, b_press, b_comp]
    constraint_names = [
        "Production Quota Target (m3/hr)",
        "Reactor Pressure Limit (kPa)",
        "Compressor Power Limit (kW)",
    ]

    _append_response_bounds(surrogate_model, A_rows, b_rows, constraint_names, response_bounds)

    A_ub = np.vstack(A_rows)
    b_ub = np.asarray(b_rows, dtype=float)

    bounds = _build_feature_bounds(u_base_arr, feat_names, trust_delta, feature_bounds)

    res = linprog(c=c_cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

    if not res.success:
        raise RuntimeError(f"Dual Simplex LP solver failed to converge: {res.message}")

    u_opt = res.x
    opt_cost = float(res.fun + cost_intercept)
    base_cost = float(np.dot(c_cost, u_base_arr) + cost_intercept)
    pct_reduction = ((base_cost - opt_cost) / base_cost) * 100.0 if base_cost > 0 else 0.0

    preds = surrogate_model.predict(u_opt)

    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(len(constraint_names))
    shadow_prices = {
        name: float(ineq_marginals[idx])
        for idx, name in enumerate(constraint_names)
    }

    slacks = {
        "Production Surplus (m3/hr)": float(preds["XMEAS_17"] - target_production),
        "Reactor Pressure Slack (kPa)": float(max_pressure - preds["XMEAS_07"]),
        "Compressor Power Slack (kW)": float(max_compressor_power - preds["XMEAS_20"]),
    }
    if response_bounds:
        for t_name, (r_min, r_max) in response_bounds.items():
            if t_name in preds:
                if r_max is not None:
                    slacks[f"{t_name} Upper Slack"] = float(r_max - preds[t_name])
                if r_min is not None:
                    slacks[f"{t_name} Lower Surplus"] = float(preds[t_name] - r_min)

    binding = [k for k, v in slacks.items() if abs(v) < 1e-3]

    lower_marginals = res.lower.marginals if hasattr(res, "lower") else np.zeros(len(u_opt))
    upper_marginals = res.upper.marginals if hasattr(res, "upper") else np.zeros(len(u_opt))
    reduced_costs = {
        feat: float(upper_marginals[i] - lower_marginals[i])
        for i, feat in enumerate(feat_names)
    }

    return LPResult(
        success=bool(res.success),
        status_message=str(res.message),
        problem_type="LP Formulation 2",
        objective_name="Operating Cost ($/hr)",
        objective_value=opt_cost,
        baseline_objective_value=base_cost,
        improvement_pct=pct_reduction,
        u_optimal={feat: float(u_opt[i]) for i, feat in enumerate(feat_names)},
        u_baseline={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
        predicted_responses=preds,
        shadow_prices=shadow_prices,
        slack_values=slacks,
        reduced_costs=reduced_costs,
        binding_constraints=binding,
        iterations=int(res.nit),
    )


def solve_lp_energy_minimization(
    surrogate_model: ProcessSurrogateModel,
    u_baseline: np.ndarray,
    target_production: float = 22.89,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
    energy_mode: str = "total",
    response_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    features: Optional[List[str]] = None,
) -> LPResult:
    """Solve LP Problem 3: Minimize energy consumption subject to meeting production quota.

    Energy Modes:
      - 'total': Combined electrical work (XMEAS_20) + Reboiler steam thermal power (XMEAS_19 * 2257/3600 kW).
      - 'compressor': Compressor electrical power only (XMEAS_20 in kW).

    Formulation:
      min   Energy(u) = c_energy^T u + d_energy
      s.t.  Product_Flow(u) >= target_production
            Reactor_Pressure(u) <= max_pressure
            Compressor_Power(u) <= max_compressor_power
            [Optional: Response Bounds]
            u_base - delta <= u <= u_base + delta
    """
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    feat_names = _resolve_features(surrogate_model, features)
    u_base_arr = np.asarray(u_baseline, dtype=float).ravel()

    comp_est = surrogate_model.models["XMEAS_20"]
    comp_coef, comp_intercept = _get_linear_coefficients(comp_est)

    if energy_mode == "compressor":
        c_energy = comp_coef
        energy_intercept = comp_intercept
        obj_name = "Compressor Electrical Power (kW)"
    else:
        steam_est = surrogate_model.models["XMEAS_19"]
        steam_coef, steam_intercept = _get_linear_coefficients(steam_est)
        c_energy = comp_coef + steam_coef * STEAM_THERMAL_KW_PER_KG_HR
        energy_intercept = comp_intercept + steam_intercept * STEAM_THERMAL_KW_PER_KG_HR
        obj_name = "Total Equivalent Process Power (kW)"

    # Production quota constraint
    prod_est = surrogate_model.models["XMEAS_17"]
    prod_coef, intercept_prod = _get_linear_coefficients(prod_est)
    a_prod = -prod_coef
    b_prod = -(target_production - intercept_prod)

    # Reactor Pressure <= max_pressure
    press_est = surrogate_model.models["XMEAS_07"]
    a_press, intercept_press = _get_linear_coefficients(press_est)
    b_press = max_pressure - intercept_press

    # Compressor Power <= max_compressor_power
    a_comp, intercept_comp = comp_coef, comp_intercept
    b_comp = max_compressor_power - intercept_comp

    A_rows = [a_prod, a_press, a_comp]
    b_rows = [b_prod, b_press, b_comp]
    constraint_names = [
        "Production Quota Target (m3/hr)",
        "Reactor Pressure Limit (kPa)",
        "Compressor Power Limit (kW)",
    ]

    _append_response_bounds(surrogate_model, A_rows, b_rows, constraint_names, response_bounds)

    A_ub = np.vstack(A_rows)
    b_ub = np.asarray(b_rows, dtype=float)

    bounds = _build_feature_bounds(u_base_arr, feat_names, trust_delta, feature_bounds)

    res = linprog(c=c_energy, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

    if not res.success:
        raise RuntimeError(f"Dual Simplex LP solver failed to converge: {res.message}")

    u_opt = res.x
    opt_energy = float(res.fun + energy_intercept)
    base_energy = float(np.dot(c_energy, u_base_arr) + energy_intercept)
    pct_reduction = ((base_energy - opt_energy) / base_energy) * 100.0 if base_energy > 0 else 0.0

    preds = surrogate_model.predict(u_opt)

    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(len(constraint_names))
    shadow_prices = {
        name: float(ineq_marginals[idx])
        for idx, name in enumerate(constraint_names)
    }

    slacks = {
        "Production Surplus (m3/hr)": float(preds["XMEAS_17"] - target_production),
        "Reactor Pressure Slack (kPa)": float(max_pressure - preds["XMEAS_07"]),
        "Compressor Power Slack (kW)": float(max_compressor_power - preds["XMEAS_20"]),
    }
    if response_bounds:
        for t_name, (r_min, r_max) in response_bounds.items():
            if t_name in preds:
                if r_max is not None:
                    slacks[f"{t_name} Upper Slack"] = float(r_max - preds[t_name])
                if r_min is not None:
                    slacks[f"{t_name} Lower Surplus"] = float(preds[t_name] - r_min)

    binding = [k for k, v in slacks.items() if abs(v) < 1e-3]

    lower_marginals = res.lower.marginals if hasattr(res, "lower") else np.zeros(len(u_opt))
    upper_marginals = res.upper.marginals if hasattr(res, "upper") else np.zeros(len(u_opt))
    reduced_costs = {
        feat: float(upper_marginals[i] - lower_marginals[i])
        for i, feat in enumerate(feat_names)
    }

    return LPResult(
        success=bool(res.success),
        status_message=str(res.message),
        problem_type="LP Formulation 3",
        objective_name=obj_name,
        objective_value=opt_energy,
        baseline_objective_value=base_energy,
        improvement_pct=pct_reduction,
        u_optimal={feat: float(u_opt[i]) for i, feat in enumerate(feat_names)},
        u_baseline={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
        predicted_responses=preds,
        shadow_prices=shadow_prices,
        slack_values=slacks,
        reduced_costs=reduced_costs,
        binding_constraints=binding,
        iterations=int(res.nit),
    )
