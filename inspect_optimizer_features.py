from pathlib import Path

for filename in [
    "src/models/surrogate_models.py",
    "src/optimization/lp_optimizer.py",
    "run_optimization.py",
]:
    p = Path(filename)
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    print(f"\n{'=' * 65}\n{filename}\n{'=' * 65}")

    terms = (
        "DECISION_FEATURES", "XMV_09", "decision_bounds",
        "bounds", "extrapolation", "features", "baseline"
    )
    shown = set()
    for i, line in enumerate(lines):
        if any(term.lower() in line.lower() for term in terms):
            start, end = max(0, i - 2), min(len(lines), i + 3)
            for j in range(start, end):
                if j not in shown:
                    print(f"{j+1}: {lines[j]}")
                    shown.add(j)
            print()
