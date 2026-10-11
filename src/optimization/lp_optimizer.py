"""Linear Programming (LP) and Dual Simplex Optimization Engine for TEP.

Formulations:
  - LP Formulation 1: Production Maximization subject to pressure, compressor, and response limits.
  - LP Formulation 2: Operating Cost Minimization subject to meeting product delivery quota.
  - LP Formulation 3: Total Energy Minimization (electrical work + stripper thermal duty).

Solver Backend:
  SciPy HiGHS Dual Simplex (`method='highs-ds'`).

Primary Reference:
  Downs, J.J. and Vogel, E.F., "A plant-wide industrial process control problem",
  Computers & Chemical Engineering, 17(3):245-255, 1993.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.pipeline import Pipeline

from src.models.surrogate_models import (
    DECISION_FEATURES,
    DECISION_FEATURES_8,
    ProcessSurrogateModel,
    RESPONSE_TARGETS,
)
from src.optimization.cost_model import (
    PRICE_ELECTRICITY,
    PRICE_PURGE_DEFAULT,
    PRICE_STEAM,
    KSCMH_TO_KMOL_HR,
    evaluate_operating_cost,
    calculate_cost_reduction_pct,
)
from src.optimization.energy_model import (
    LATENT_HEAT_STEAM_KJ_KG,
    SECONDS_PER_HOUR,
)

logger = logging.getLogger(__name__)

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
    extrapolation_status: Optional[Dict[str, Union[float, bool]]] = None
    full_cost_baseline: Optional[float] = None
    full_cost_optimal: Optional[float] = None
    full_cost_improvement_pct: Optional[float] = None
    is_scientifically_feasible: Optional[bool] = None
    feasibility_report: Optional[Dict[str, Any]] = None


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
):
    """Append empirical response bounds into LP constraint matrix A_ub @ u <= b_ub."""
    if not response_bounds:
        return

    for target_name, (r_min, r_max) in response_bounds.items():
        if target_name not in surrogate_model.models:
            continue
        est = surrogate_model.models[target_name]
        coef, intercept = _get_linear_coefficients(est)

        if r_max is not None and np.isfinite(r_max):
            A_rows.append(coef)
            b_rows.append(float(r_max - intercept))
            constraint_names.append(f"{target_name} Upper Limit")

        if r_min is not None and np.isfinite(r_min):
            A_rows.append(-coef)
            b_rows.append(float(intercept - r_min))
            constraint_names.append(f"{target_name} Lower Limit")


def verify_solution_feasibility(
    lp_result: LPResult,
    surrogate_model: ProcessSurrogateModel,
    trust_delta: float = 10.0,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    target_production: Optional[float] = None,
    response_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Independently verify all physical, engineering, and trust constraints for an LP result.

    This evaluation is performed independently of solver reported slacks by evaluating the
    fitted surrogate response models at the returned optimal setpoint u_optimal.
    """
    if not lp_result.success:
        return {
            "is_feasible": False,
            "solver_success": False,
            "num_constraints_checked": 0,
            "num_feasible": 0,
            "num_violated": 0,
            "max_violation": 0.0,
            "violations": [f"Solver reported failure: {lp_result.status_message}"],
            "details": {},
        }

    feat_names = list(lp_result.u_optimal.keys())
    u_opt_arr = np.array([lp_result.u_optimal[f] for f in feat_names], dtype=float)
    u_base_arr = np.array([lp_result.u_baseline[f] for f in feat_names], dtype=float)

    # Recompute independent surrogate predictions
    indep_preds = surrogate_model.predict(u_opt_arr)

    details: Dict[str, Dict[str, Any]] = {}
    violations: List[str] = []

    # 1. Feature trust region and physical bounds
    for i, feat in enumerate(feat_names):
        val = u_opt_arr[i]
        base_val = u_base_arr[i]
        f_min, f_max = (0.0, 100.0)
        if feature_bounds and feat in feature_bounds:
            f_min, f_max = feature_bounds[feat]

        lb = max(f_min, base_val - trust_delta)
        ub = min(f_max, base_val + trust_delta)

        lb_slack = val - lb
        lb_feasible = bool(lb_slack >= -tolerance)
        details[f"{feat} Lower Bound"] = {
            "value": float(val),
            "bound": float(lb),
            "slack": float(lb_slack),
            "is_feasible": lb_feasible,
            "type": "lower",
        }
        if not lb_feasible:
            violations.append(f"{feat} ({val:.4f}) below lower bound {lb:.4f} (viol={-lb_slack:.6f})")

        ub_slack = ub - val
        ub_feasible = bool(ub_slack >= -tolerance)
        details[f"{feat} Upper Bound"] = {
            "value": float(val),
            "bound": float(ub),
            "slack": float(ub_slack),
            "is_feasible": ub_feasible,
            "type": "upper",
        }
        if not ub_feasible:
            violations.append(f"{feat} ({val:.4f}) exceeds upper bound {ub:.4f} (viol={-ub_slack:.6f})")

    # 2. Reactor Pressure limit (XMEAS_07 <= max_pressure)
    press_val = indep_preds["XMEAS_07"]
    press_slack = max_pressure - press_val
    press_feasible = bool(press_slack >= -tolerance)
    details["Reactor Pressure Limit (kPa)"] = {
        "value": float(press_val),
        "bound": float(max_pressure),
        "slack": float(press_slack),
        "is_feasible": press_feasible,
        "type": "upper",
    }
    if not press_feasible:
        violations.append(f"Reactor Pressure ({press_val:.2f} kPa) exceeds limit {max_pressure:.2f} kPa (viol={-press_slack:.4f})")

    # 3. Compressor Power limit (XMEAS_20 <= max_compressor_power)
    comp_val = indep_preds["XMEAS_20"]
    comp_slack = max_compressor_power - comp_val
    comp_feasible = bool(comp_slack >= -tolerance)
    details["Compressor Power Limit (kW)"] = {
        "value": float(comp_val),
        "bound": float(max_compressor_power),
        "slack": float(comp_slack),
        "is_feasible": comp_feasible,
        "type": "upper",
    }
    if not comp_feasible:
        violations.append(f"Compressor Power ({comp_val:.2f} kW) exceeds limit {max_compressor_power:.2f} kW (viol={-comp_slack:.4f})")

    # 4. Production Quota Target (XMEAS_17 >= target_production)
    if target_production is not None:
        prod_val = indep_preds["XMEAS_17"]
        prod_slack = prod_val - target_production
        prod_feasible = bool(prod_slack >= -tolerance)
        details["Production Quota (m3/hr)"] = {
            "value": float(prod_val),
            "bound": float(target_production),
            "slack": float(prod_slack),
            "is_feasible": prod_feasible,
            "type": "lower",
        }
        if not prod_feasible:
            violations.append(f"Production Throughput ({prod_val:.4f} m3/hr) below target {target_production:.4f} m3/hr (viol={-prod_slack:.4f})")

    # 5. Response Empirical Bounds
    if response_bounds:
        for resp_tag, (r_min, r_max) in response_bounds.items():
            if resp_tag in indep_preds:
                r_val = indep_preds[resp_tag]
                if r_max is not None and np.isfinite(r_max):
                    u_slack = r_max - r_val
                    u_feas = bool(u_slack >= -tolerance)
                    details[f"{resp_tag} Response Max"] = {
                        "value": float(r_val),
                        "bound": float(r_max),
                        "slack": float(u_slack),
                        "is_feasible": u_feas,
                        "type": "upper",
                    }
                    if not u_feas:
                        violations.append(f"{resp_tag} ({r_val:.4f}) exceeds response max {r_max:.4f} (viol={-u_slack:.4f})")

                if r_min is not None and np.isfinite(r_min):
                    l_slack = r_val - r_min
                    l_feas = bool(l_slack >= -tolerance)
                    details[f"{resp_tag} Response Min"] = {
                        "value": float(r_val),
                        "bound": float(r_min),
                        "slack": float(l_slack),
                        "is_feasible": l_feas,
                        "type": "lower",
                    }
                    if not l_feas:
                        violations.append(f"{resp_tag} ({r_val:.4f}) below response min {r_min:.4f} (viol={-l_slack:.4f})")

    num_total = len(details)
    num_feas = sum(1 for d in details.values() if d["is_feasible"])
    num_viol = num_total - num_feas
    max_viol = max([-d["slack"] for d in details.values() if not d["is_feasible"]], default=0.0)

    return {
        "is_feasible": bool(num_viol == 0),
        "solver_success": True,
        "num_constraints_checked": num_total,
        "num_feasible": num_feas,
        "num_violated": num_viol,
        "max_violation": float(max_viol),
        "violations": violations,
        "details": details,
    }


def solve_lp_production_maximization(
    surrogate_model: Optional[ProcessSurrogateModel] = None,
    u_baseline: np.ndarray = None,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
    response_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    features: Optional[List[str]] = None,
    surrogate: Optional[ProcessSurrogateModel] = None,
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
    if surrogate_model is None and surrogate is not None:
        surrogate_model = surrogate
    if surrogate_model is None:
        raise ValueError("A fitted surrogate_model must be provided.")
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    feat_names = _resolve_features(surrogate_model, features)
    u_base_arr = np.asarray(u_baseline, dtype=float).ravel()
    if len(u_base_arr) != len(feat_names):
        raise ValueError(
            f"Length of u_baseline ({len(u_base_arr)}) does not match number of features ({len(feat_names)}: {feat_names})."
        )

    # Objective: Maximize Product Flow (solve as min -c_prod^T u)
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
        base_prod = float(prod_est.predict(u_base_arr.reshape(1, -1))[0])
        preds = surrogate_model.predict(u_base_arr)
        return LPResult(
            success=False,
            status_message=str(res.message),
            problem_type="LP Formulation 1",
            objective_name="Product Flow (m3/hr)",
            objective_value=base_prod,
            baseline_objective_value=base_prod,
            improvement_pct=0.0,
            u_optimal={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
            u_baseline={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
            predicted_responses=preds,
            shadow_prices={name: 0.0 for name in constraint_names},
            slack_values={},
            reduced_costs={feat: 0.0 for feat in feat_names},
            binding_constraints=[],
            iterations=int(getattr(res, "nit", 0)),
            is_scientifically_feasible=False,
        )

    u_opt = res.x
    opt_prod = float(-res.fun + intercept_prod)
    base_prod = float(prod_est.predict(u_base_arr.reshape(1, -1))[0])
    pct_gain = ((opt_prod - base_prod) / base_prod) * 100.0 if base_prod > 0 else 0.0

    preds = surrogate_model.predict(u_opt)
    preds["XMEAS_17"] = opt_prod

    # Standard non-negative shadow price convention (marginal improvement per RHS relaxation)
    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(len(constraint_names))
    shadow_prices = {
        name: float(-ineq_marginals[idx])
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

    # SciPy reduced costs: lower marginal >= 0 (penalty), upper marginal <= 0 (incentive)
    lower_marginals = res.lower.marginals if hasattr(res, "lower") else np.zeros(len(u_opt))
    upper_marginals = res.upper.marginals if hasattr(res, "upper") else np.zeros(len(u_opt))
    reduced_costs = {
        feat: float(lower_marginals[i] + upper_marginals[i])
        for i, feat in enumerate(feat_names)
    }

    ext_status = None
    if hasattr(surrogate_model, "check_extrapolation"):
        _, ext_status = surrogate_model.check_extrapolation(u_opt)

    result_obj = LPResult(
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
        extrapolation_status=ext_status,
    )

    feas_report = verify_solution_feasibility(
        result_obj,
        surrogate_model=surrogate_model,
        trust_delta=trust_delta,
        max_pressure=max_pressure,
        max_compressor_power=max_compressor_power,
        response_bounds=response_bounds,
        feature_bounds=feature_bounds,
    )
    result_obj.is_scientifically_feasible = feas_report["is_feasible"]
    result_obj.feasibility_report = feas_report

    return result_obj


def solve_lp_cost_minimization(
    surrogate_model: Optional[ProcessSurrogateModel] = None,
    u_baseline: np.ndarray = None,
    target_production: float = 22.89,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
    avg_purge_price_per_kmol: float = PRICE_PURGE_DEFAULT,
    response_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    features: Optional[List[str]] = None,
    surrogate: Optional[ProcessSurrogateModel] = None,
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
    if surrogate_model is None and surrogate is not None:
        surrogate_model = surrogate
    if surrogate_model is None:
        raise ValueError("A fitted surrogate_model must be provided.")
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    feat_names = _resolve_features(surrogate_model, features)
    u_base_arr = np.asarray(u_baseline, dtype=float).ravel()
    if len(u_base_arr) != len(feat_names):
        raise ValueError(
            f"Length of u_baseline ({len(u_base_arr)}) does not match number of features ({len(feat_names)}: {feat_names})."
        )

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

    res = linprog(c=c_cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")

    if not res.success:
        base_cost = float(np.dot(c_cost, u_base_arr) + cost_intercept)
        preds = surrogate_model.predict(u_base_arr)
        return LPResult(
            success=False,
            status_message=str(res.message),
            problem_type="LP Formulation 2",
            objective_name="Operating Cost ($/hr)",
            objective_value=base_cost,
            baseline_objective_value=base_cost,
            improvement_pct=0.0,
            u_optimal={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
            u_baseline={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
            predicted_responses=preds,
            shadow_prices={name: 0.0 for name in constraint_names},
            slack_values={},
            reduced_costs={feat: 0.0 for feat in feat_names},
            binding_constraints=[],
            iterations=int(getattr(res, "nit", 0)),
            is_scientifically_feasible=False,
        )

    u_opt = res.x
    opt_cost = float(res.fun + cost_intercept)
    base_cost = float(np.dot(c_cost, u_base_arr) + cost_intercept)
    pct_reduction = ((base_cost - opt_cost) / base_cost) * 100.0 if base_cost > 0 else 0.0

    preds = surrogate_model.predict(u_opt)

    # Standard non-negative shadow price convention
    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(len(constraint_names))
    shadow_prices = {
        name: float(-ineq_marginals[idx])
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
        feat: float(lower_marginals[i] + upper_marginals[i])
        for i, feat in enumerate(feat_names)
    }

    ext_status = None
    if hasattr(surrogate_model, "check_extrapolation"):
        _, ext_status = surrogate_model.check_extrapolation(u_opt)

    # Independently evaluate full component-based operating cost
    base_eval = evaluate_operating_cost(surrogate_model.predict(u_base_arr), avg_purge_price_per_kmol=avg_purge_price_per_kmol)
    opt_eval = evaluate_operating_cost(preds, avg_purge_price_per_kmol=avg_purge_price_per_kmol)
    full_cost_base = float(base_eval["full_operating_cost_usd_hr"])
    full_cost_opt = float(opt_eval["full_operating_cost_usd_hr"])
    full_cost_pct = calculate_cost_reduction_pct(full_cost_base, full_cost_opt)

    result_obj = LPResult(
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
        extrapolation_status=ext_status,
        full_cost_baseline=full_cost_base,
        full_cost_optimal=full_cost_opt,
        full_cost_improvement_pct=full_cost_pct,
    )

    feas_report = verify_solution_feasibility(
        result_obj,
        surrogate_model=surrogate_model,
        trust_delta=trust_delta,
        max_pressure=max_pressure,
        max_compressor_power=max_compressor_power,
        target_production=target_production,
        response_bounds=response_bounds,
        feature_bounds=feature_bounds,
    )
    result_obj.is_scientifically_feasible = feas_report["is_feasible"]
    result_obj.feasibility_report = feas_report

    return result_obj


def solve_lp_energy_minimization(
    surrogate_model: Optional[ProcessSurrogateModel] = None,
    u_baseline: np.ndarray = None,
    target_production: float = 22.89,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
    energy_mode: str = "total",
    response_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    features: Optional[List[str]] = None,
    surrogate: Optional[ProcessSurrogateModel] = None,
) -> LPResult:
    """Solve LP Problem 3: Minimize energy consumption subject to meeting production quota.

    Energy Modes:
      - 'total': Combined electrical work (XMEAS_20) + Reboiler steam thermal power (XMEAS_19 * 2257/3600 kW).
      - 'compressor': Compressor electrical power only (XMEAS_20 in kW).
    """
    if surrogate_model is None and surrogate is not None:
        surrogate_model = surrogate
    if surrogate_model is None:
        raise ValueError("A fitted surrogate_model must be provided.")
    if not surrogate_model.is_fitted:
        raise RuntimeError("Surrogate model must be fitted prior to LP formulation.")

    feat_names = _resolve_features(surrogate_model, features)
    u_base_arr = np.asarray(u_baseline, dtype=float).ravel()
    if len(u_base_arr) != len(feat_names):
        raise ValueError(
            f"Length of u_baseline ({len(u_base_arr)}) does not match number of features ({len(feat_names)}: {feat_names})."
        )

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
        base_energy = float(np.dot(c_energy, u_base_arr) + energy_intercept)
        preds = surrogate_model.predict(u_base_arr)
        return LPResult(
            success=False,
            status_message=str(res.message),
            problem_type="LP Formulation 3",
            objective_name=obj_name,
            objective_value=base_energy,
            baseline_objective_value=base_energy,
            improvement_pct=0.0,
            u_optimal={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
            u_baseline={feat: float(u_base_arr[i]) for i, feat in enumerate(feat_names)},
            predicted_responses=preds,
            shadow_prices={name: 0.0 for name in constraint_names},
            slack_values={},
            reduced_costs={feat: 0.0 for feat in feat_names},
            binding_constraints=[],
            iterations=int(getattr(res, "nit", 0)),
            is_scientifically_feasible=False,
        )

    u_opt = res.x
    opt_energy = float(res.fun + energy_intercept)
    base_energy = float(np.dot(c_energy, u_base_arr) + energy_intercept)
    pct_reduction = ((base_energy - opt_energy) / base_energy) * 100.0 if base_energy > 0 else 0.0

    preds = surrogate_model.predict(u_opt)

    ineq_marginals = res.ineqlin.marginals if hasattr(res, "ineqlin") else np.zeros(len(constraint_names))
    shadow_prices = {
        name: float(-ineq_marginals[idx])
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
        feat: float(lower_marginals[i] + upper_marginals[i])
        for i, feat in enumerate(feat_names)
    }

    ext_status = None
    if hasattr(surrogate_model, "check_extrapolation"):
        _, ext_status = surrogate_model.check_extrapolation(u_opt)

    # Evaluate full cost context for energy solution
    base_eval = evaluate_operating_cost(surrogate_model.predict(u_base_arr))
    opt_eval = evaluate_operating_cost(preds)
    full_cost_base = float(base_eval["full_operating_cost_usd_hr"])
    full_cost_opt = float(opt_eval["full_operating_cost_usd_hr"])
    full_cost_pct = calculate_cost_reduction_pct(full_cost_base, full_cost_opt)

    result_obj = LPResult(
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
        extrapolation_status=ext_status,
        full_cost_baseline=full_cost_base,
        full_cost_optimal=full_cost_opt,
        full_cost_improvement_pct=full_cost_pct,
    )

    feas_report = verify_solution_feasibility(
        result_obj,
        surrogate_model=surrogate_model,
        trust_delta=trust_delta,
        max_pressure=max_pressure,
        max_compressor_power=max_compressor_power,
        target_production=target_production,
        response_bounds=response_bounds,
        feature_bounds=feature_bounds,
    )
    result_obj.is_scientifically_feasible = feas_report["is_feasible"]
    result_obj.feasibility_report = feas_report

    return result_obj
