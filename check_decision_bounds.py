import json
from pathlib import Path

data = json.loads(
    Path("results/optimization/optimization_results.json").read_text(encoding="utf-8")
)

for key in ("production_maximization", "cost_minimization"):
    r = data[key]
    print("\n" + "=" * 75)
    print(r["problem_type"])

    optimal = r["u_optimal"]
    baseline = r.get("u_baseline")

    if baseline is None:
        print("Baseline decision variables are not stored in this result.")
        print("Available result fields:", list(r.keys()))
        continue

    for name, opt in optimal.items():
        base = baseline[name]
        lower = max(0.0, base - 10.0)
        upper = min(100.0, base + 10.0)

        if abs(opt - lower) < 1e-5:
            status = "LOWER TRUST BOUND ACTIVE"
        elif abs(opt - upper) < 1e-5:
            status = "UPPER TRUST BOUND ACTIVE"
        else:
            status = "Interior"

        print(
            f"{name}: baseline={base:.4f}, optimized={opt:.4f}, "
            f"allowed=[{lower:.4f}, {upper:.4f}] -> {status}"
        )
