"""End-to-end numerical validation — TEP Optimization Project.
Mirrors run_optimization.py exactly to reproduce the published results.
Run:  python validate_results.py
"""
import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.models.surrogate_models import (
    ProcessSurrogateModel, DECISION_FEATURES_8, RESPONSE_TARGETS
)
from src.optimization.lp_optimizer import (
    solve_lp_production_maximization,
    solve_lp_cost_minimization,
    solve_lp_energy_minimization,
)
from src.optimization.cost_model import evaluate_operating_cost_breakdown
from src.optimization.energy_model import evaluate_energy_breakdown
from src.optimization.goal_programming import GoalSpec, solve_weighted_sum_gp, solve_preemptive_gp

OK  = "[PASS]"
ERR = "[FAIL]"
results = []

def check(label, value, lo, hi, unit=""):
    ok = lo <= value <= hi
    tag = OK if ok else ERR
    print("  %s %-42s %9.4f %s  [%.2f - %.2f]" % (tag, label + ":", value, unit, lo, hi))
    results.append(ok)
    return ok

sep = "=" * 70
print(sep)
print("  TEP OPTIMIZATION - FULL NUMERICAL VALIDATION")
print(sep)

# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------
print("\n[1/5] Data Loading & Integrity")
df = load_tep_dataset('d00')
check("Dataset rows",    float(len(df)),    499, 501)
check("Dataset columns", float(df.shape[1]), 51,  53)
nan_ct = int(df.isna().sum().sum())
inf_ct = int(np.isinf(df.to_numpy()).sum())
print("  %s NaN values: %d  [expected 0]" % (OK if nan_ct == 0 else ERR, nan_ct))
print("  %s Inf values: %d  [expected 0]" % (OK if inf_ct == 0 else ERR, inf_ct))
results += [nan_ct == 0, inf_ct == 0]

# ------------------------------------------------------------------
# 2. BASELINE ENERGY & COST
# ------------------------------------------------------------------
print("\n[2/5] Baseline Energy & Cost Models (Downs & Vogel 1993)")
cost   = evaluate_operating_cost_breakdown(df)
energy = evaluate_energy_breakdown(df)
check("Baseline total cost",    cost["total_operating_cost_usd_hr"],  104.0, 108.0, "$/hr")
check("Purge loss cost",        cost["purge_cost_usd_hr"],             82.0,  92.0, "$/hr")
check("Compressor elec. cost",  cost["compressor_cost_usd_hr"],        17.0,  20.0, "$/hr")
check("Steam cost",             cost["steam_cost_usd_hr"],              0.5,   1.5, "$/hr")
check("Compressor power",       energy["compressor_power_kw"],         338.0, 345.0, "kW")
check("Steam thermal power",    energy["steam_thermal_power_kw"],      142.0, 147.0, "kW")
check("Total energy footprint", energy["total_energy_power_kw"],       483.0, 490.0, "kW")

# ------------------------------------------------------------------
# 3. SURROGATE MODELS  (using sm.metrics dict after fit())
# ------------------------------------------------------------------
print("\n[3/5] 8-Variable Ridge Surrogate Models (TimeSeriesSplit CV k=5)")
sm = ProcessSurrogateModel(model_type='ridge', alpha=10.0, features=DECISION_FEATURES_8)
sm.fit(df, cv_splits=5)
mdict = sm.metrics   # {response_tag: {cv_r2, cv_mae, ...}}
for tag, label, min_r2 in [
    ("XMEAS_17", "Product Flow     (XMEAS_17)", 0.99),
    ("XMEAS_10", "Purge Rate       (XMEAS_10)", 0.85),
    ("XMEAS_19", "Stripper Steam   (XMEAS_19)", 0.70),
    ("XMEAS_07", "Reactor Pressure (XMEAS_07)", 0.20),
    ("XMEAS_20", "Compressor Power (XMEAS_20)", 0.25),
]:
    r2 = float(mdict[tag]["cv_r2"]) if tag in mdict else -999.0
    ok = r2 >= min_r2
    print("  %s %-38s CV R2: %.4f  [min: %.2f]" % (OK if ok else ERR, label, r2, min_r2))
    results.append(ok)

# ------------------------------------------------------------------
# 4. LP OPTIMIZATION  (replicate run_optimization.py exactly)
# ------------------------------------------------------------------
print("\n[4/5] Linear Programming Results (HiGHS Dual Simplex)")
u_base      = df[DECISION_FEATURES_8].mean().to_numpy()
resp_bounds = {t: (float(df[t].min()), float(df[t].max())) for t in RESPONSE_TARGETS}
feat_bounds = {f: (float(df[f].min()), float(df[f].max())) for f in DECISION_FEATURES_8}
target_prod = float(df["XMEAS_17"].mean())

r1 = solve_lp_production_maximization(
    surrogate_model=sm, u_baseline=u_base,
    max_pressure=2800.0, max_compressor_power=355.0,
    trust_delta=10.0, response_bounds=resp_bounds, feature_bounds=feat_bounds,
)
r2 = solve_lp_cost_minimization(
    surrogate_model=sm, u_baseline=u_base,
    target_production=target_prod,
    max_pressure=2800.0, max_compressor_power=355.0,
    trust_delta=10.0, response_bounds=resp_bounds, feature_bounds=feat_bounds,
)
r3 = solve_lp_energy_minimization(
    surrogate_model=sm, u_baseline=u_base,
    target_production=target_prod,
    max_pressure=2800.0, max_compressor_power=355.0,
    trust_delta=10.0, response_bounds=resp_bounds, feature_bounds=feat_bounds,
)

# Production maximization  (published: baseline=22.91 -> opt=24.68, +7.73%)
check("Prod Max  baseline",   r1.baseline_objective_value,  22.5, 23.3, "m3/hr")
check("Prod Max  optimized",  r1.objective_value,           24.0, 25.5, "m3/hr")
check("Prod Max  improvement",r1.improvement_pct,            5.0, 10.0, "%")
print("  %s Prod Max solver: %s" % (OK if r1.success else ERR, r1.status_message[:60]))
results.append(r1.success)

# Cost minimization  (published: $106.40 -> $97.73, -8.15%)
check("Cost Min  baseline",   r2.baseline_objective_value, 104.0, 108.0, "$/hr")
check("Cost Min  optimized",  r2.objective_value,           94.0, 101.0, "$/hr")
check("Cost Min  improvement",r2.improvement_pct,            5.0,  12.0, "%")
print("  %s Cost Min solver: %s" % (OK if r2.success else ERR, r2.status_message[:60]))
results.append(r2.success)

# Energy minimization  (published: 485.84 -> 469.03, -3.46%)
check("Energy Min baseline",  r3.baseline_objective_value, 482.0, 490.0, "kW")
check("Energy Min optimized", r3.objective_value,          463.0, 478.0, "kW")
check("Energy Min improve",   r3.improvement_pct,            2.0,   7.0, "%")
print("  %s Energy Min solver: %s" % (OK if r3.success else ERR, r3.status_message[:60]))
results.append(r3.success)

# Safety constraints
if r1.predicted_responses:
    press = r1.predicted_responses.get("XMEAS_07", 9999.0)
    comp  = r1.predicted_responses.get("XMEAS_20", 9999.0)
    ok_p = press <= 2800.0
    ok_c = comp  <= 355.0
    print("  %s Safety: Reactor Pressure %.1f kPa  (<= 2800 kPa limit)" % (OK if ok_p else ERR, press))
    print("  %s Safety: Compressor Power  %.1f kW   (<= 355 kW rating)" % (OK if ok_c else ERR, comp))
    results += [ok_p, ok_c]

# ------------------------------------------------------------------
# 5. GOAL PROGRAMMING
# ------------------------------------------------------------------
print("\n[5/5] Goal Programming (Weighted-Sum & Preemptive Lexicographic)")
goals = [
    GoalSpec(name="Production Quota", response_tag="XMEAS_17",
             target=24.0, direction="max", weight=2.0, priority=1),
    GoalSpec(name="Cost Reduction",   response_tag="XMEAS_20",
             target=338.0, direction="min", weight=1.0, priority=2),
    GoalSpec(name="Pressure Target",  response_tag="XMEAS_07",
             target=2705.0, direction="exact", weight=0.5, priority=3),
]
ws_r  = solve_weighted_sum_gp(sm, goals, u_base)
pre_r = solve_preemptive_gp(sm, goals, u_base)
ok_ws  = ws_r.success
ok_pre = pre_r.success
dev_ws = ws_r.total_weighted_deviation if ok_ws else float('nan')
print("  %s Weighted-Sum GP: success=%-5s | Weighted deviation: %.4f" % (
    OK if ok_ws else ERR, str(ok_ws), dev_ws))
print("  %s Preemptive GP:   success=%-5s | Priority levels: %d" % (
    OK if ok_pre else ERR, str(ok_pre), len(pre_r.priority_sequence)))
results += [ok_ws, ok_pre]

# ------------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------------
print("\n" + sep)
passed  = sum(1 for r in results if r)
total   = len(results)
pct     = 100.0 * passed / total if total else 0
verdict = "ALL CHECKS PASSED" if passed == total else ("ISSUES: %d check(s) FAILED" % (total - passed))
print("  RESULT: %d/%d checks passed (%.0f%%)  --  %s" % (passed, total, pct, verdict))
print(sep)
