from pathlib import Path
import re

path = Path(r"src/optimization/lp_optimizer.py")
source = path.read_text(encoding="utf-8")

if "_get_linear_coefficients" in source:
    raise SystemExit("Helper already exists. No changes made; backup is available.")

if "from sklearn.pipeline import Pipeline" not in source:
    source = re.sub(
        r"(?m)^import numpy as np\s*$",
        "import numpy as np\nfrom sklearn.pipeline import Pipeline",
        source,
        count=1,
    )

helper = '''

def _get_linear_coefficients(estimator):
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

'''

match = re.search(r"(?m)^@dataclass\b", source)
if not match:
    raise SystemExit("Could not locate @dataclass. No changes written.")
source = source[:match.start()] + helper + source[match.start():]

# Update coefficient access in both optimization functions.
replacements = {
    r"(?m)^(\s*)c = -prod_est\.coef_\s*$":
        r"\1prod_coef, intercept_prod = _get_linear_coefficients(prod_est)\n\1c = -prod_coef",
    r"(?m)^(\s*)intercept_prod = float\(prod_est\.intercept_\)\s*$": "",
    r"(?m)^(\s*)a_press = press_est\.coef_\s*$":
        r"\1a_press, intercept_press = _get_linear_coefficients(press_est)",
    r"(?m)^(\s*)b_press = max_pressure - float\(press_est\.intercept_\)\s*$":
        r"\1b_press = max_pressure - intercept_press",
    r"(?m)^(\s*)a_comp = comp_est\.coef_\s*$":
        r"\1a_comp, intercept_comp = _get_linear_coefficients(comp_est)",
    r"(?m)^(\s*)b_comp = max_compressor_power - float\(comp_est\.intercept_\)\s*$":
        r"\1b_comp = max_compressor_power - intercept_comp",
    r"(?m)^(\s*)a_prod = -prod_est\.coef_\s*$":
        r"\1prod_coef, intercept_prod = _get_linear_coefficients(prod_est)\n\1a_prod = -prod_coef",
    r"(?m)^(\s*)b_prod = -\(target_production - float\(prod_est\.intercept_\)\)\s*$":
        r"\1b_prod = -(target_production - intercept_prod)",
}
for pattern, replacement in replacements.items():
    source = re.sub(pattern, replacement, source)

# Convert the cost objective to original-scale coefficients.
start = source.find("    c_cost = (")
if start == -1:
    raise SystemExit("Could not locate c_cost block. Original file remains unchanged.")
end_match = re.search(r"(?m)^    # 1\. Product_Flow", source[start:])
if not end_match:
    raise SystemExit("Could not locate end of cost coefficient block. Original file remains unchanged.")

cost_block = '''    purge_coef, purge_intercept = _get_linear_coefficients(purge_est)
    steam_coef, steam_intercept = _get_linear_coefficients(steam_est)
    comp_coef, comp_intercept = _get_linear_coefficients(comp_est)

    c_cost = (
        purge_coef * c_purge_factor
        + steam_coef * 0.00318
        + comp_coef * 0.0536
    )
    cost_intercept = (
        purge_intercept * c_purge_factor
        + steam_intercept * 0.00318
        + comp_intercept * 0.0536
    )

'''
source = source[:start] + cost_block + source[start + end_match.start():]

# Stop if any direct model coefficient access remains in the optimization functions.
func_start = source.find("def solve_lp_production_maximization")
if func_start == -1:
    raise SystemExit("Production function not found.")
if re.search(r"\b(?:prod_est|press_est|comp_est|purge_est|steam_est)\.(?:coef_|intercept_)", source[func_start:]):
    raise SystemExit("Some direct coefficient accesses remain. Original file remains unchanged.")

path.write_text(source, encoding="utf-8")
print("Updated lp_optimizer.py. Backup saved as src/optimization/lp_optimizer.py.backup")
