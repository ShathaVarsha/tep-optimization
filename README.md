# TEP Process Optimization Suite

## Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process

**Course:** 23MNG336: Operational Research | **Team:** B4 | **Level:** Final-Year B.Tech AI & Data Science

---

## Key Results

| Optimization Scenario | Baseline | Optimized | Improvement |
|---|---|---|---|
| **Production Maximization** | 22.91 m³/hr | **24.68 m³/hr** | **+7.73%** |
| **Operating Cost Minimization** | $106.40/hr | **$97.73/hr** | **-8.15%** |
| **Total Energy Minimization** | 485.84 kW | **469.03 kW** | **-3.46%** |
| Product Flow Surrogate R² | — | **0.9951** | — |
| Automated Tests | — | **70/70 passed** | 100% |

---

## What This Project Does

This end-to-end Operations Research project applies data-driven optimization to the industry-standard **Tennessee Eastman Process (TEP)** benchmark — a realistic chemical plant simulation used worldwide in process control research.

### Core Pipeline

1. **Data Ingestion** — Loads all 44 official `.dat` benchmark files (Downs & Vogel, 1993) with automated orientation detection and integrity auditing (0 NaN, 0 Inf values)

2. **Baseline Validation** — Augmented Dickey-Fuller (ADF) unit-root tests confirm true statistical steady-state operation across all 7 key process variables

3. **Surrogate Modeling** — Trains L₂-regularized Ridge Regression (α=10) models via 5-fold TimeSeriesSplit cross-validation, mapping 8 manipulated valve setpoints → 5 process responses

4. **The XMV_11 Discovery** — Adding the Condenser Cooling Water Valve (XMV_11) to the decision space elevated product flow prediction from R²=−0.063 to **R²=0.9951**

5. **Linear Programming** — SciPy HiGHS Dual Simplex solver with trust regions and empirical response bounding achieves all four optimization objectives

6. **Goal Programming** — Weighted-Sum and Preemptive (Lexicographic) formulations handle competing trade-offs between throughput, cost, and pressure stability

7. **Sensitivity Analysis** — LP shadow prices (dual variables) and local finite-difference what-if gradients quantify constraint value and variable influence

8. **Interactive Dashboard** — Full-stack Streamlit web app with live solver execution, Plotly charts, and CSV exports

---

## Running Locally

### Prerequisites
- Python 3.12+
- All packages in `requirements.txt`

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run streamlit_app.py

# 4. Run all tests
python -m pytest tests/ -v

# 5. Full numerical validation (32 checks)
python validate_results.py
```

---

## Project Structure

```
OR/
├── streamlit_app.py              # Main dashboard (7 interactive views)
├── requirements.txt              # Python dependencies
├── validate_results.py           # 32-check numerical validation script
├── run_optimization.py           # Reproducible CLI optimization pipeline
├── generate_final_plots.py       # Publication-quality figure generator
│
├── src/
│   ├── data/
│   │   ├── tep_loader.py         # TEP benchmark data loader
│   │   └── statistics_generator.py
│   ├── process/
│   │   ├── baseline_analyzer.py  # ADF stationarity + drift analysis
│   │   └── variable_dictionary.py
│   ├── models/
│   │   └── surrogate_models.py   # Ridge / OLS / Polynomial surrogates
│   ├── optimization/
│   │   ├── lp_optimizer.py       # HiGHS Dual Simplex LP
│   │   ├── goal_programming.py   # Weighted-Sum & Preemptive GP
│   │   ├── cost_model.py         # Downs & Vogel economic cost model
│   │   └── energy_model.py       # Thermodynamic energy model
│   └── analysis/
│       └── lp_sensitivity.py     # Shadow prices & what-if sensitivity
│
├── tests/                        # 70 pytest unit tests (all passing)
│   ├── test_data_loader.py
│   ├── test_cost_energy.py
│   ├── test_surrogate.py
│   ├── test_lp.py
│   ├── test_goal_programming.py
│   ├── test_ui_workflow.py
│   └── test_variable_dictionary.py
│
├── results/optimization/         # JSON results, CSV summaries, plots
├── OR_final_visuals/             # 14 publication-quality figures
├── docs/                         # Parameter traceability tables
├── FINAL_VALIDATION_REPORT.md    # Complete engineering validation report
└── PRESENTATION_OUTLINE.md       # 15-slide defense presentation outline
```

---

## Data Source

The Tennessee Eastman Process benchmark dataset is from:
> Downs, J.J. and Vogel, E.F. (1993). "A plant-wide industrial process control problem."
> *Computers & Chemical Engineering*, 17(3), 245–255.

The dataset contains 44 `.dat` files (training + testing, normal + 21 fault modes) with 52 process variables.

---

## Tech Stack

| Component | Technology |
|---|---|
| LP Solver | SciPy HiGHS Dual Simplex |
| Surrogate Models | scikit-learn Ridge Regression |
| Statistical Testing | statsmodels ADF |
| Dashboard | Streamlit 1.65 + Plotly |
| Testing | pytest (70 tests) |
| Language | Python 3.12 |
