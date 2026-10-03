"""Goal Programming (GP) Optimization for the Tennessee Eastman Process.

Two formulations are provided, both solved via SciPy HiGHS Dual Simplex (linprog):

FORMULATION A - Weighted-Sum Goal Programming
  Minimize:  sum_k  w_k * (d_k_neg + d_k_pos) / |g_k|
  Subject to:
    f_k(u) + d_k_neg - d_k_pos = g_k      for all k
    d_k_neg, d_k_pos >= 0
    A_ub * u <= b_ub  (engineering safety)
    u_base - delta <= u <= u_base + delta  (trust region)

FORMULATION B - Preemptive (Lexicographic) Goal Programming
  Solve goals sequentially by priority P1 > P2 > ...
  At each level k:
    Minimize deviation penalty for goals at priority k
    Subject to all GP goal rows, safety constraints, and
    achievement constraints from higher-priority levels.

Author: Team B4 (23MNG336: Operational Research)
Primary References:
  - Charnes & Cooper (1961). Management Models and Industrial Applications.
  - Ignizio (1976). Goal Programming and Extensions.
  - Downs & Vogel (1993). Computers & Chemical Engineering, 17(3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.models.surrogate_models import ProcessSurrogateModel, DECISION_FEATURES


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GoalSpec:
    """Specification for a single goal in the GP formulation.

    Attributes:
        name:          Human-readable label.
        response_tag:  TEP variable tag (e.g. 'XMEAS_20').
        target:        Numeric goal target g_k (engineering units of the variable).
        direction:     'min' -> penalise d_k_pos only;
                       'max' -> penalise d_k_neg only;
                       'exact' -> penalise both.
        weight:        Priority weight w_k (Weighted-Sum GP) or within-level weight.
        priority:      Integer level for Preemptive GP (1 = highest priority).
    """
    name: str
    response_tag: str
    target: float
    direction: str
    weight: float = 1.0
    priority: int = 1

    def __post_init__(self):
        if self.direction not in ("min", "max", "exact"):
            raise ValueError(
                f"GoalSpec '{self.name}': direction must be 'min', 'max', or 'exact'."
            )
        if self.target == 0.0:
            raise ValueError(
                f"GoalSpec '{self.name}': target must be non-zero (used as normalisation denominator)."
            )
        if self.weight < 0.0:
            raise ValueError(f"GoalSpec '{self.name}': weight must be >= 0.")


@dataclass
class GPResult:
    """Container for Goal Programming optimisation results."""
    formulation: str
    success: bool
    status_message: str
    goals: List[GoalSpec]
    u_optimal: Dict[str, float]
    u_baseline: Dict[str, float]
    predicted_responses: Dict[str, float]
    goal_achievement: List[Dict]
    total_weighted_deviation: float
    priority_sequence: List[float]
    n_decision_vars: int
    n_deviation_vars: int
    iterations: int


# ---------------------------------------------------------------------------
# Internal LP Assembly Utilities
# ---------------------------------------------------------------------------

def _extract_linear_model(surrogate: ProcessSurrogateModel, tag: str) -> Tuple[np.ndarray, float]:
    """Extract coefficients and intercept from a fitted surrogate for a given response tag.

    For Ridge Pipeline surrogates, un-scales the coefficients to the original
    decision-variable space (before StandardScaler was applied) so they can be
    used directly in a linprog objective/constraint row.

    For Polynomial surrogates the conversion is not possible (nonlinear in u),
    and a ValueError is raised directing the caller to use a linear/ridge surrogate.

    Returns:
        coef:       np.ndarray of shape (n_features,) in original u space.
        intercept:  scalar such that prediction  coef @ u + intercept.
    """
    if tag not in surrogate.models:
        raise KeyError(f"Surrogate has no model for response tag '{tag}'.")

    estimator = surrogate.models[tag]

    if hasattr(estimator, "named_steps"):
        steps = estimator.named_steps
        if "poly" in steps:
            raise ValueError(
                f"Tag '{tag}': Polynomial surrogates cannot be directly linearised "
                "for GP. Refit surrogate with model_type='linear' or 'ridge'."
            )
        scaler = steps["scaler"]
        regressor = steps["regressor"]
        coef_scaled = np.array(regressor.coef_, dtype=float)
        coef = coef_scaled / scaler.scale_
        intercept = float(regressor.intercept_) - float(coef_scaled @ (scaler.mean_ / scaler.scale_))
    else:
        coef = np.array(estimator.coef_, dtype=float)
        intercept = float(estimator.intercept_)

    return coef, intercept


def _build_safety_constraints(
    surrogate: ProcessSurrogateModel,
    max_pressure: float,
    max_compressor_power: float,
    n_u: int,
    n_dev: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build engineering safety inequality rows in augmented [u | d_neg | d_pos] space."""
    press_coef, press_int = _extract_linear_model(surrogate, "XMEAS_07")
    comp_coef, comp_int = _extract_linear_model(surrogate, "XMEAS_20")

    zero_dev = np.zeros(n_dev)
    row_press = np.concatenate([press_coef, zero_dev])
    row_comp = np.concatenate([comp_coef, zero_dev])

    A_ub = np.vstack([row_press, row_comp])
    b_ub = np.array([max_pressure - press_int, max_compressor_power - comp_int])
    return A_ub, b_ub


def _build_trust_bounds(
    u_baseline: np.ndarray,
    trust_delta: float,
    n_dev: int,
) -> List[Tuple[float, float]]:
    """Build variable bounds for the augmented [u | d_neg | d_pos] vector."""
    bounds = []
    for u_val in u_baseline:
        lb = max(0.0, float(u_val) - trust_delta)
        ub = min(100.0, float(u_val) + trust_delta)
        bounds.append((lb, ub))
    for _ in range(n_dev):
        bounds.append((0.0, None))
    return bounds


def _build_goal_equality_rows(
    goals: List[GoalSpec],
    surrogate: ProcessSurrogateModel,
    n_u: int,
    n_goals: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build goal-row equality system: A_eq @ [u | d_neg | d_pos] = b_eq.

    Variable layout:
      [u_1 ... u_{n_u} | d_1_neg ... d_{n_goals}_neg | d_1_pos ... d_{n_goals}_pos]

    Each goal k contributes one equality row:
      c_k^T u + d_k_neg - d_k_pos = g_k - intercept_k
    """
    n_aug = n_u + 2 * n_goals
    A_eq = np.zeros((n_goals, n_aug))
    b_eq = np.zeros(n_goals)

    for k, goal in enumerate(goals):
        coef, intercept = _extract_linear_model(surrogate, goal.response_tag)
        A_eq[k, :n_u] = coef
        A_eq[k, n_u + k] = 1.0            # d_k_neg coefficient
        A_eq[k, n_u + n_goals + k] = -1.0 # d_k_pos coefficient
        b_eq[k] = goal.target - intercept

    return A_eq, b_eq


def _build_objective_vector(
    goals: List[GoalSpec],
    n_u: int,
    n_goals: int,
    active_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Build the linprog cost vector over [u | d_neg | d_pos].

    If active_indices is None, all goals are included (Weighted-Sum mode).
    If active_indices is provided, only those goal indices are penalised
    (used in Preemptive mode for a single priority level).

    Penalty terms:
      'min' direction -> penalise d_k_pos (over-achievement above target)
      'max' direction -> penalise d_k_neg (under-achievement below target)
      'exact'        -> penalise both d_k_neg and d_k_pos
    """
    n_aug = n_u + 2 * n_goals
    c = np.zeros(n_aug)
    idxs = active_indices if active_indices is not None else list(range(n_goals))

    for k in idxs:
        goal = goals[k]
        gk = abs(goal.target)
        penalty = goal.weight / gk if gk > 1e-9 else goal.weight

        if goal.direction == "min":
            c[n_u + n_goals + k] += penalty   # d_k_pos
        elif goal.direction == "max":
            c[n_u + k] += penalty              # d_k_neg
        else:  # 'exact'
            c[n_u + k] += penalty              # d_k_neg
            c[n_u + n_goals + k] += penalty   # d_k_pos

    return c


def _extract_gp_solution(
    x: np.ndarray,
    goals: List[GoalSpec],
    surrogate: ProcessSurrogateModel,
    n_u: int,
    n_goals: int,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict], float]:
    """Decode the augmented solution vector into readable results."""
    u_opt = x[:n_u]
    features = surrogate.features if hasattr(surrogate, "features") else DECISION_FEATURES
    u_optimal = {feat: float(u_opt[i]) for i, feat in enumerate(features)}
    predicted_responses = surrogate.predict(u_opt)

    goal_achievement = []
    total_dev = 0.0

    for k, goal in enumerate(goals):
        d_neg = float(x[n_u + k])
        d_pos = float(x[n_u + n_goals + k])
        achieved = predicted_responses.get(goal.response_tag, float("nan"))
        gk = abs(goal.target)
        penalty = goal.weight / gk if gk > 1e-9 else goal.weight

        if goal.direction == "min":
            total_dev += penalty * d_pos
            is_achieved = d_pos < 1e-4
        elif goal.direction == "max":
            total_dev += penalty * d_neg
            is_achieved = d_neg < 1e-4
        else:
            total_dev += penalty * (d_neg + d_pos)
            is_achieved = (d_neg + d_pos) < 1e-4

        goal_achievement.append({
            "Goal Name": goal.name,
            "Response Tag": goal.response_tag,
            "Target (g_k)": round(goal.target, 4),
            "Achieved Value": round(achieved, 4),
            "Deviation d_neg": round(d_neg, 6),
            "Deviation d_pos": round(d_pos, 6),
            "Direction": goal.direction,
            "Priority": goal.priority,
            "Weight (w_k)": goal.weight,
            "Status": "Achieved" if is_achieved else "Deviation",
        })

    return u_optimal, predicted_responses, goal_achievement, total_dev


# ---------------------------------------------------------------------------
# Public API - Weighted-Sum Goal Programming
# ---------------------------------------------------------------------------

def solve_weighted_sum_gp(
    surrogate: ProcessSurrogateModel,
    goals: List[GoalSpec],
    u_baseline: np.ndarray,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
) -> GPResult:
    """Solve the Weighted-Sum Goal Programming LP.

    Formulation:
      min  sum_k (w_k / |g_k|) * deviation_penalty_k
      s.t. f_k(u) + d_k_neg - d_k_pos = g_k    for all k
           XMEAS_07(u) <= max_pressure
           XMEAS_20(u) <= max_compressor_power
           u_base - delta <= u <= u_base + delta;  0 <= u <= 100
           d_k_neg, d_k_pos >= 0                  for all k

    Args:
        surrogate:            Fitted ProcessSurrogateModel (model_type='linear' or 'ridge').
        goals:                List of GoalSpec objects.
        u_baseline:           Nominal baseline decision vector, shape (7,).
        max_pressure:         Reactor pressure upper bound (kPa). Default: 2800 kPa.
        max_compressor_power: Compressor work upper bound (kW). Default: 355 kW.
        trust_delta:          Trust region half-width (%). Default: 10%.

    Returns:
        GPResult with formulation='Weighted-Sum'.
    """
    if not surrogate.is_fitted:
        raise RuntimeError("Surrogate must be fitted before calling GP solver.")

    n_goals = len(goals)
    n_u = len(surrogate.features) if hasattr(surrogate, "features") else len(DECISION_FEATURES)
    n_dev = 2 * n_goals
    n_aug = n_u + n_dev

    c = _build_objective_vector(goals, n_u, n_goals)
    A_eq, b_eq = _build_goal_equality_rows(goals, surrogate, n_u, n_goals)
    A_ub, b_ub = _build_safety_constraints(surrogate, max_pressure, max_compressor_power, n_u, n_dev)
    bounds = _build_trust_bounds(u_baseline, trust_delta, n_dev)

    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ds")

    if res.success:
        u_optimal, preds, achievement, total_dev = _extract_gp_solution(
            res.x, goals, surrogate, n_u, n_goals
        )
    else:
        u_optimal = {feat: float(u_baseline[i]) for i, feat in enumerate(DECISION_FEATURES)}
        preds = surrogate.predict(u_baseline)
        achievement = []
        total_dev = float("inf")

    return GPResult(
        formulation="Weighted-Sum",
        success=bool(res.success),
        status_message=str(res.message),
        goals=goals,
        u_optimal=u_optimal,
        u_baseline={feat: float(u_baseline[i]) for i, feat in enumerate(DECISION_FEATURES)},
        predicted_responses=preds,
        goal_achievement=achievement,
        total_weighted_deviation=total_dev,
        priority_sequence=[],
        n_decision_vars=n_u,
        n_deviation_vars=n_dev,
        iterations=int(res.nit),
    )


# ---------------------------------------------------------------------------
# Public API - Preemptive (Lexicographic) Goal Programming
# ---------------------------------------------------------------------------

def solve_preemptive_gp(
    surrogate: ProcessSurrogateModel,
    goals: List[GoalSpec],
    u_baseline: np.ndarray,
    max_pressure: float = 2800.0,
    max_compressor_power: float = 355.0,
    trust_delta: float = 10.0,
) -> GPResult:
    """Solve the Preemptive (Lexicographic) Goal Programming LP.

    Goals are ordered by their priority attribute (1 = highest).
    At each priority level, an LP is solved minimising only that level's deviation.
    The achieved deviation is then frozen as an additional inequality constraint
    before moving to the next lower priority level.

    This guarantees that no higher-priority goal is ever degraded to improve a
    lower-priority goal (Ignizio 1976 lexicographic dominance property).

    Args:
        surrogate:            Fitted ProcessSurrogateModel (model_type='linear' or 'ridge').
        goals:                List of GoalSpec objects with priority attributes set.
        u_baseline:           Nominal baseline decision vector, shape (7,).
        max_pressure:         Reactor pressure upper bound (kPa). Default: 2800 kPa.
        max_compressor_power: Compressor work upper bound (kW). Default: 355 kW.
        trust_delta:          Trust region half-width (%). Default: 10%.

    Returns:
        GPResult with formulation='Preemptive'.
    """
    if not surrogate.is_fitted:
        raise RuntimeError("Surrogate must be fitted before calling GP solver.")

    n_goals = len(goals)
    n_u = len(surrogate.features) if hasattr(surrogate, "features") else len(DECISION_FEATURES)
    n_dev = 2 * n_goals

    sorted_priorities = sorted(set(g.priority for g in goals))
    priority_groups: Dict[int, List[int]] = {
        p: [i for i, g in enumerate(goals) if g.priority == p]
        for p in sorted_priorities
    }

    A_ub_base, b_ub_base = _build_safety_constraints(
        surrogate, max_pressure, max_compressor_power, n_u, n_dev
    )
    A_eq, b_eq = _build_goal_equality_rows(goals, surrogate, n_u, n_goals)
    base_bounds = _build_trust_bounds(u_baseline, trust_delta, n_dev)

    extra_ineq_rows: List[np.ndarray] = []
    extra_ineq_rhs: List[float] = []
    priority_sequence: List[float] = []
    status_messages: List[str] = []
    total_iterations = 0
    last_x: Optional[np.ndarray] = None
    all_success = True

    for p in sorted_priorities:
        idxs = priority_groups[p]
        c = _build_objective_vector(goals, n_u, n_goals, active_indices=idxs)

        if extra_ineq_rows:
            A_ub = np.vstack([A_ub_base] + extra_ineq_rows)
            b_ub = np.concatenate([b_ub_base, extra_ineq_rhs])
        else:
            A_ub = A_ub_base
            b_ub = b_ub_base.copy()

        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=base_bounds, method="highs-ds")

        total_iterations += int(res.nit)
        status_messages.append(f"P{p}: {res.message}")

        if not res.success:
            all_success = False
            priority_sequence.append(float("inf"))
            break

        z_star = float(res.fun)
        priority_sequence.append(z_star)
        last_x = res.x.copy()

        # Freeze this priority level: its deviation cannot worsen in subsequent solves
        extra_ineq_rows.append(c.reshape(1, -1))
        extra_ineq_rhs.append(z_star + 1e-6)

    if all_success and last_x is not None:
        u_optimal, preds, achievement, total_dev = _extract_gp_solution(
            last_x, goals, surrogate, n_u, n_goals
        )
    else:
        u_optimal = {feat: float(u_baseline[i]) for i, feat in enumerate(DECISION_FEATURES)}
        preds = surrogate.predict(u_baseline)
        achievement = []
        total_dev = float("inf")

    return GPResult(
        formulation="Preemptive",
        success=all_success,
        status_message=" | ".join(status_messages),
        goals=goals,
        u_optimal=u_optimal,
        u_baseline={feat: float(u_baseline[i]) for i, feat in enumerate(DECISION_FEATURES)},
        predicted_responses=preds,
        goal_achievement=achievement,
        total_weighted_deviation=total_dev,
        priority_sequence=priority_sequence,
        n_decision_vars=n_u,
        n_deviation_vars=n_dev,
        iterations=total_iterations,
    )


# ---------------------------------------------------------------------------
# Reporting Utility
# ---------------------------------------------------------------------------

def format_gp_report(result: GPResult) -> str:
    """Format a comprehensive text Goal Programming result report."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"OPERATIONS RESEARCH GOAL PROGRAMMING REPORT: {result.formulation}")
    lines.append("=" * 80)
    lines.append(f"Solver Engine    : SciPy HiGHS Dual Simplex ('highs-ds')")
    lines.append(f"Status           : {'SUCCESS' if result.success else 'FAILED'}")
    lines.append(f"Solver Messages  : {result.status_message}")
    lines.append(f"Total Iterations : {result.iterations}")
    lines.append(f"Decision Vars    : {result.n_decision_vars} | Deviation Vars: {result.n_deviation_vars}")

    if result.formulation == "Preemptive" and result.priority_sequence:
        lines.append("\nPriority Sequence Objective Values (Z_k*):")
        for i, z in enumerate(result.priority_sequence, start=1):
            lines.append(f"  P{i}: {z:.6f}  (dimensionless weighted deviation)")

    lines.append(f"\nTotal Weighted Normalised Deviation: {result.total_weighted_deviation:.6f}")

    lines.append("\n" + "-" * 80)
    lines.append("GOAL ACHIEVEMENT SUMMARY")
    lines.append("-" * 80)
    if result.goal_achievement:
        df = pd.DataFrame(result.goal_achievement)
        lines.append(df.to_string(index=False))
    else:
        lines.append("  (No goals achieved - solver infeasible)")

    lines.append("\n" + "-" * 80)
    lines.append("OPTIMAL DECISION VARIABLE SETPOINTS")
    lines.append("-" * 80)
    for tag, val in result.u_optimal.items():
        base = result.u_baseline.get(tag, 0.0)
        delta = val - base
        lines.append(f"  {tag:12s}: {val:8.3f}%  (baseline: {base:8.3f}%  delta={delta:+.3f}%)")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default TEP Goal Set (canonical three-goal set for Mode 1)
# ---------------------------------------------------------------------------

def build_default_tep_goals() -> List[GoalSpec]:
    """Return the canonical three-goal set for TEP Mode 1 economic optimisation.

    Goal targets are derived from Downs & Vogel (1993) Table 1/Table 4 Mode 1 values:
      - Product flow nominal: 22.89 m3/hr -> target: match or exceed (maximise)
      - Compressor power nominal: 341.4 kW -> target: 5% reduction = 324.3 kW
      - Purge rate nominal: 0.3371 kscmh -> target: 10% reduction = 0.303 kscmh

    Priority order for Preemptive GP:
      P1: Production continuity (most critical to maintain)
      P2: Compressor power reduction and purge rate reduction (energy/cost goals)

    Weighted-Sum weights reflect course emphasis and relative importance.
    """
    return [
        GoalSpec(
            name="Meet Production Quota (>=22.89 m3/hr)",
            response_tag="XMEAS_17",
            target=22.89,
            direction="max",
            weight=2.0,
            priority=1,
        ),
        GoalSpec(
            name="Reduce Compressor Power (target 324.3 kW)",
            response_tag="XMEAS_20",
            target=324.3,
            direction="min",
            weight=1.5,
            priority=2,
        ),
        GoalSpec(
            name="Reduce Purge Rate (target 0.303 kscmh)",
            response_tag="XMEAS_10",
            target=0.303,
            direction="min",
            weight=1.0,
            priority=2,
        ),
    ]
