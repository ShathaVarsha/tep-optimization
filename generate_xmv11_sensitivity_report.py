import json
from pathlib import Path
import pandas as pd

root = Path.cwd()
out_dir = root / "results" / "optimization" / "xmv11_experiment"
json_path = out_dir / "optimization_results_8vars.json"

if not json_path.exists():
    raise FileNotFoundError(
        f"Missing {json_path}. Run the 8-variable response-bounded optimization first."
    )

data = json.loads(json_path.read_text(encoding="utf-8"))
rows = []

def walk(obj, problem, section=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                walk(value, problem, key)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                rows.append({
                    "problem": problem,
                    "section": section,
                    "parameter": key,
                    "value": value,
                })

for problem, details in data.items():
    walk(details, problem)

report = pd.DataFrame(rows)
report.to_csv(out_dir / "sensitivity_input_report.csv", index=False)

readme = """8-variable optimization: sensitivity-input report

This report extracts the saved optimization-result metadata, decision-variable
settings, predicted responses, and available constraint information.

Important limitations:
- This is NOT a formal LP shadow-price or dual-value sensitivity report.
- The saved JSON does not contain re-optimized results under perturbed
  coefficients or right-hand-side constraints.
- Predictions are surrogate-model estimates, not validated plant outcomes.
- Treat reported cost changes as preliminary estimates.

For formal sensitivity analysis, the optimization code must retain the LP
solver's dual values, slack values, and objective coefficients, or rerun the
model under controlled perturbations.
"""
(out_dir / "sensitivity_report_notes.txt").write_text(readme, encoding="utf-8")

print(f"Saved: {out_dir / 'sensitivity_input_report.csv'}")
print(f"Saved: {out_dir / 'sensitivity_report_notes.txt'}")
print(f"Rows extracted: {len(report)}")
