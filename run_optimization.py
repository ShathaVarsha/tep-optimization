"""Master Reproducible Operations Research Optimization Pipeline for TEP.

Executes:
  1. Data Loading and Integrity Verification (d00.dat).
  2. 8-Variable Surrogate Model Training & Chronological Cross-Validation.
  3. Linear Programming Solvers via SciPy HiGHS Dual Simplex:
     - Scenario A: Production Maximization (+7.73%)
     - Scenario B: Operating Cost Minimization (-8.15%)
     - Scenario C: Total Process Energy Minimization (-3.46%)
  4. Local Finite-Difference What-If Sensitivity Analysis.
  5. Export of Structured Optimization JSON, CSV summaries, and Sensitivity Reports.

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.models.surrogate_models import (
    ProcessSurrogateModel,
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
    compute_local_what_if_sensitivity,
    summarize_largest_sensitivities,
    generate_variable_sensitivity_table,
    generate_constraint_sensitivity_table,
)

OUT_DIR = Path("results/optimization")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = OUT_DIR / "xmv11_experiment"
EXP_DIR.mkdir(parents=True, exist_ok=True)


def lp_result_to_dict(result: LPResult) -> dict:
    """Serialize LPResult dataclass to JSON-compatible dictionary."""
    return {
        "success": result.success,
        "status_message": result.status_message,
        "problem_type": result.problem_type,
        "objective_name": result.objective_name,
        "objective_value": result.objective_value,
        "baseline_objective_value": result.baseline_objective_value,
        "improvement_pct": result.improvement_pct,
        "u_optimal": result.u_optimal,
        "u_baseline": result.u_baseline,
        "predicted_responses": result.predicted_responses,
        "shadow_prices": result.shadow_prices,
        "slack_values": result.slack_values,
        "reduced_costs": result.reduced_costs,
        "binding_constraints": result.binding_constraints,
        "iterations": result.iterations,
    }


def main():
    print("=" * 80)
    print("TENNESSEE EASTMAN PROCESS: REPRODUCIBLE OPTIMIZATION PIPELINE")
    print("=" * 80)

    # 1. Load Dataset
    print("\n[Step 1/5] Loading normal-operation benchmark dataset (d00.dat)...")
    df = load_tep_dataset("d00")
    print(f"  • Shape: {df.shape[0]} samples × {df.shape[1]} process variables")
    print(f"  • Integrity: NaN count = {df.isna().sum().sum()}, Inf count = {np.isinf(df.to_numpy()).sum()}")

    # 2. Train Surrogate Models
    print("\n[Step 2/5] Training 8-Variable Ridge Surrogates (+XMV_11) with TimeSeriesSplit CV...")
    model = ProcessSurrogateModel(
        model_type="ridge",
        alpha=10.0,
        features=DECISION_FEATURES_8,
    ).fit(df, cv_splits=5)

    metrics_df = pd.DataFrame(model.metrics).T
    metrics_df.to_csv(OUT_DIR / "surrogate_metrics_8vars.csv")
    metrics_df.to_csv(EXP_DIR / "surrogate_metrics_8vars.csv")
    print("  • Cross-Validation Metrics Table:")
    print(metrics_df[["train_r2", "cv_r2", "cv_mae", "cv_rmse", "cv_rel_error_pct"]].round(4).to_string())

    # 3. Setup Bounds & Initial Point
    u_base = df[DECISION_FEATURES_8].mean().to_numpy()
    resp_bounds = {t: (float(df[t].min()), float(df[t].max())) for t in RESPONSE_TARGETS}
    feat_bounds = {f: (float(df[f].min()), float(df[f].max())) for f in DECISION_FEATURES_8}

    # 4. Execute Optimization Scenarios
    print("\n[Step 3/5] Solving Linear Programming Formulations via SciPy HiGHS Dual Simplex...")

    # Problem 1: Production Maximization
    res_prod = solve_lp_production_maximization(
        surrogate_model=model,
        u_baseline=u_base,
        max_pressure=2800.0,
        max_compressor_power=355.0,
        trust_delta=10.0,
        response_bounds=resp_bounds,
        feature_bounds=feat_bounds,
    )
    print(f"  • Production Maximization: {res_prod.baseline_objective_value:.4f} -> {res_prod.objective_value:.4f} m³/hr ({res_prod.improvement_pct:+.2f}%) [Status: {res_prod.status_message}]")

    # Problem 2: Operating Cost Minimization
    target_production = float(df["XMEAS_17"].mean())
    res_cost = solve_lp_cost_minimization(
        surrogate_model=model,
        u_baseline=u_base,
        target_production=target_production,
        max_pressure=2800.0,
        max_compressor_power=355.0,
        trust_delta=10.0,
        response_bounds=resp_bounds,
        feature_bounds=feat_bounds,
    )
    print(f"  • Cost Minimization:       ${res_cost.baseline_objective_value:.2f} -> ${res_cost.objective_value:.2f}/hr ({res_cost.improvement_pct:+.2f}%) [Status: {res_cost.status_message}]")

    # Problem 3: Total Energy Minimization
    res_energy = solve_lp_energy_minimization(
        surrogate_model=model,
        u_baseline=u_base,
        target_production=target_production,
        max_pressure=2800.0,
        max_compressor_power=355.0,
        trust_delta=10.0,
        energy_mode="total",
        response_bounds=resp_bounds,
        feature_bounds=feat_bounds,
    )
    print(f"  • Energy Minimization:     {res_energy.baseline_objective_value:.2f} -> {res_energy.objective_value:.2f} kW ({res_energy.improvement_pct:+.2f}%) [Status: {res_energy.status_message}]")

    # 5. Local What-If Sensitivity Analysis
    print("\n[Step 4/5] Computing Local Finite-Difference What-If Sensitivity around Baseline...")
    sens_df = compute_local_what_if_sensitivity(model, u_base, df, perturbation_frac=0.01)
    sens_df.to_csv(OUT_DIR / "local_what_if_sensitivity.csv", index=False)
    sens_df.to_csv(EXP_DIR / "local_what_if_sensitivity.csv", index=False)

    summary_sens = summarize_largest_sensitivities(sens_df)
    summary_sens.to_csv(OUT_DIR / "largest_local_sensitivity_by_response.csv", index=False)
    summary_sens.to_csv(EXP_DIR / "local_what_if_sensitivity_summary.csv", index=False)
    print("  • Dominant Local Sensitivity per Process Response:")
    print(summary_sens[["response", "input_variable", "direction", "response_change", "local_slope"]].round(4).to_string(index=False))

    # 6. Save JSON & CSV Deliverables
    print("\n[Step 5/5] Exporting Full Optimization Results & Comparison Tables...")
    full_results = {
        "dataset": "d00",
        "dataset_shape": list(df.shape),
        "features": DECISION_FEATURES_8,
        "note": "Surrogate-based Linear Programming optimization results with empirical response bounding.",
        "production_maximization": lp_result_to_dict(res_prod),
        "cost_minimization": lp_result_to_dict(res_cost),
        "energy_minimization": lp_result_to_dict(res_energy),
    }

    json_path = OUT_DIR / "optimization_results_8vars.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    with open(EXP_DIR / "optimization_results_8vars.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    summary_table = pd.DataFrame([
        {
            "Problem": "Production Maximization",
            "Objective": res_prod.objective_name,
            "Baseline Value": round(res_prod.baseline_objective_value, 4),
            "Optimized Value": round(res_prod.objective_value, 4),
            "Improvement (%)": round(res_prod.improvement_pct, 2),
            "Iterations": res_prod.iterations,
        },
        {
            "Problem": "Operating Cost Minimization",
            "Objective": res_cost.objective_name,
            "Baseline Value": round(res_cost.baseline_objective_value, 4),
            "Optimized Value": round(res_cost.objective_value, 4),
            "Improvement (%)": round(res_cost.improvement_pct, 2),
            "Iterations": res_cost.iterations,
        },
        {
            "Problem": "Total Energy Minimization",
            "Objective": res_energy.objective_name,
            "Baseline Value": round(res_energy.baseline_objective_value, 4),
            "Optimized Value": round(res_energy.objective_value, 4),
            "Improvement (%)": round(res_energy.improvement_pct, 2),
            "Iterations": res_energy.iterations,
        },
    ])
    summary_table.to_csv(OUT_DIR / "optimization_summary_8vars.csv", index=False)
    summary_table.to_csv(EXP_DIR / "optimization_summary_8vars.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL OPTIMIZATION SUMMARY (8-VARIABLE FORMULATION)")
    print("=" * 80)
    print(summary_table.to_string(index=False))
    print(f"\nAll artifacts generated and persisted in {OUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
