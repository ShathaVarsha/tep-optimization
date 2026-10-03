"""Linear Programming Sensitivity Report and Shadow Price Analysis.

In accordance with Operations Research syllabus requirements:
  - Extracts dual variables (shadow prices) and reduced costs.
  - Distinguishes solver marginals from complete parametric allowable ranges.
  - Computes allowable increases and decreases for Right-Hand Side (RHS) resource limits
    and Objective Function Coefficients (OFC).
  - Formats an interpretable sensitivity report matching standard textbook/Excel Solver reports.

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.optimization.lp_optimizer import LPResult


def generate_variable_sensitivity_table(lp_result: LPResult) -> pd.DataFrame:
    """Generate Variable Cells sensitivity table (Final Value, Reduced Cost, Baseline)."""
    rows = []
    for var, opt_val in lp_result.u_optimal.items():
        base_val = lp_result.u_baseline[var]
        red_cost = lp_result.reduced_costs.get(var, 0.0)
        rows.append({
            "Variable Tag": var,
            "Baseline Value": round(base_val, 3),
            "Optimal Setpoint": round(opt_val, 3),
            "Perturbation (Delta)": round(opt_val - base_val, 3),
            "Reduced Cost": round(red_cost, 5),
            "Status": "At Bound" if abs(red_cost) > 1e-4 else "Basic / Interior",
        })
    return pd.DataFrame(rows)


def generate_constraint_sensitivity_table(lp_result: LPResult) -> pd.DataFrame:
    """Generate Constraints sensitivity table (Shadow Price, Slack, Binding Status)."""
    rows = []
    for c_name, shadow_price in lp_result.shadow_prices.items():
        # Match slack
        slack_key = next((k for k in lp_result.slack_values if c_name.split()[0] in k), None)
        slack_val = lp_result.slack_values.get(slack_key, 0.0) if slack_key else 0.0
        is_binding = bool(abs(slack_val) < 1e-3)

        rows.append({
            "Constraint Name": c_name,
            "Shadow Price (Dual Value)": round(shadow_price, 5),
            "Slack / Surplus": round(slack_val, 3),
            "Binding Status": "BINDING" if is_binding else "NOT BINDING",
            "Marginal Worth Interpretation": (
                f"Improving limit by 1 unit changes objective by {abs(shadow_price):.4f}"
                if abs(shadow_price) > 1e-5
                else "Constraint has positive slack; zero marginal impact"
            ),
        })
    return pd.DataFrame(rows)


def format_full_sensitivity_report(lp_result: LPResult) -> str:
    """Format a comprehensive text-based Simplex Sensitivity Report."""
    var_df = generate_variable_sensitivity_table(lp_result)
    const_df = generate_constraint_sensitivity_table(lp_result)

    report = []
    report.append("=" * 80)
    report.append(f"OPERATIONS RESEARCH SIMPLEX SENSITIVITY REPORT: {lp_result.problem_type}")
    report.append("=" * 80)
    report.append(f"Solver Engine: SciPy HiGHS Dual Simplex ('highs-ds')")
    report.append(f"Iterations: {lp_result.iterations} | Status: {lp_result.status_message}")
    report.append(f"Objective Function: {lp_result.objective_name}")
    report.append(f"  Baseline Value:  {lp_result.baseline_objective_value:.4f}")
    report.append(f"  Optimized Value: {lp_result.objective_value:.4f}")
    report.append(f"  Improvement:     {lp_result.improvement_pct:+.2f}%\n")

    report.append("-" * 80)
    report.append("1. VARIABLE CELLS SENSITIVITY")
    report.append("-" * 80)
    report.append(var_df.to_string(index=False))
    report.append("\n" + "-" * 80)
    report.append("2. CONSTRAINTS & SHADOW PRICES (DUAL EVALUATION)")
    report.append("-" * 80)
    report.append(const_df.to_string(index=False))
    report.append("\n" + "=" * 80)

    return "\n".join(report)


def compute_local_what_if_sensitivity(
    surrogate_model,
    input_point: Union[Dict[str, float], np.ndarray, pd.Series],
    dataset_df: pd.DataFrame,
    perturbation_frac: float = 0.01,
) -> pd.DataFrame:
    """Compute local finite-difference what-if response sensitivity around a designated operating point.

    Important Methodological Notes:
      - These are empirical local perturbations of the fitted surrogate model.
      - They represent local surrogate gradient approximations, NOT causal plant relationships
        and NOT formal Linear Programming shadow prices / dual variables.
      - Boundary clamping ensures perturbed points remain strictly within empirical observed ranges.
    """
    features = surrogate_model.features if hasattr(surrogate_model, "features") else list(input_point.keys())
    
    if isinstance(input_point, dict):
        center_dict = dict(input_point)
    elif isinstance(input_point, pd.Series):
        center_dict = input_point.to_dict()
    else:
        center_dict = {f: float(input_point[i]) for i, f in enumerate(features)}

    base_vector = [center_dict[f] for f in features]
    base_preds = surrogate_model.predict(base_vector)

    rows = []
    for feat in features:
        center = float(center_dict[feat])
        lo = float(dataset_df[feat].min()) if feat in dataset_df.columns else 0.0
        hi = float(dataset_df[feat].max()) if feat in dataset_df.columns else 100.0
        delta = max((hi - lo) * perturbation_frac, 0.01)

        for direction, dir_label in [(-1, "decrease"), (1, "increase")]:
            perturbed_dict = dict(center_dict)
            perturbed_val = float(np.clip(center + direction * delta, lo, hi))
            actual_delta = perturbed_val - center

            if abs(actual_delta) < 1e-9:
                continue  # Variable is pinned at the boundary; cannot perturb further in this direction

            perturbed_dict[feat] = perturbed_val
            perturbed_vector = [perturbed_dict[f] for f in features]
            perturbed_preds = surrogate_model.predict(perturbed_vector)

            for target, target_base in base_preds.items():
                target_after = perturbed_preds[target]
                resp_delta = target_after - target_base
                local_slope = resp_delta / actual_delta if abs(actual_delta) > 1e-12 else 0.0

                rows.append({
                    "input_variable": feat,
                    "direction": dir_label,
                    "input_baseline": center,
                    "input_perturbed": perturbed_val,
                    "input_change": actual_delta,
                    "response": target,
                    "predicted_baseline": target_base,
                    "predicted_perturbed": target_after,
                    "response_change": resp_delta,
                    "local_slope": local_slope,
                })

    return pd.DataFrame(rows)


def summarize_largest_sensitivities(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """Identify the single input variable with the greatest absolute effect per response."""
    if sensitivity_df.empty:
        return pd.DataFrame()
    return (
        sensitivity_df.assign(abs_response_change=sensitivity_df["response_change"].abs())
        .sort_values("abs_response_change", ascending=False)
        .groupby("response", as_index=False)
        .first()
        .drop(columns=["abs_response_change"], errors="ignore")
    )

