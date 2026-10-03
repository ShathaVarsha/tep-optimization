import json
from pathlib import Path

p = Path("results/optimization/optimization_results.json")
data = json.loads(p.read_text(encoding="utf-8"))

for key in ("production_maximization", "cost_minimization"):
    r = data[key]
    print("\n" + "=" * 65)
    print(r["problem_type"])
    print("Solver success:", r["success"])
    print("Iterations:", r["iterations"])
    print("Baseline objective:", r["baseline_objective_value"])
    print("Optimized objective:", r["objective_value"])
    print("Reported improvement (%):", r["improvement_pct"])

    print("\nOptimized decision variables:")
    for name, value in r["u_optimal"].items():
        print(f"  {name}: {value:.6f}")

    print("\nPredicted responses:")
    for name, value in r["predicted_responses"].items():
        print(f"  {name}: {value:.6f}")

    print("\nConstraint slacks:")
    for name, value in r["slack_values"].items():
        print(f"  {name}: {value:.6f}")

    print("\nBinding constraints:", r["binding_constraints"])
