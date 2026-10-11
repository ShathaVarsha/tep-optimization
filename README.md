# Tennessee Eastman Process Optimization Suite

Data-driven energy, cost, and throughput optimization for the Tennessee Eastman Process (TEP) benchmark.

---

## Key Results

| Optimization Scenario | Baseline | Optimized | Change |
|---|---|---|---|
| **Production Maximization** | 22.91 m³/hr | **24.68 m³/hr** | **+7.73%** |
| **Operating Cost Minimization** | $106.40/hr (proxy) / $106.39/hr (full) | **$97.73/hr (proxy) / $97.71/hr (full)** | **-8.15%** |
| **Total Energy Minimization** | 485.84 kW | **469.03 kW** | **-3.46%** |
| Product Flow Surrogate R² | — | **0.9951** | — |
| Automated Tests | — | **88/88 passed** | 100% |

---

## Overview

This project applies data-driven optimization to the Tennessee Eastman Process (TEP), an established industrial benchmark developed by Downs & Vogel (1993). The plant model simulates a two-phase chemical process comprising a reactor, condenser, recycle compressor, vapor-liquid separator, and stripper column.

### Core Workflow

1. **Data Ingestion**  
   Loads all 44 benchmark `.dat` files with automated array transposition detection and integrity auditing (0 missing values, 0 infinities).

2. **Stationarity Checks**  
   Augmented Dickey-Fuller (ADF) unit-root tests confirm steady-state operation across key process variables in baseline dataset `d00.dat`.

3. **Surrogate Modeling**  
   Trains L₂-regularized Ridge Regression models (α = 10) using 5-fold TimeSeriesSplit cross-validation, mapping manipulated valve positions to critical plant responses.

4. **8-Variable Decision Space (`XMV_11`)**  
   Adding the condenser cooling water valve (`XMV_11`) resolves product flow prediction, increasing cross-validated R² on finished product flow from -0.063 to **0.9951** due to vapor-liquid condensation equilibrium.

5. **Linear Programming**  
   Formulates local linear approximations solved via the SciPy HiGHS Dual Simplex solver (`highs-ds`) with trust regions and empirical response bounding.

6. **Goal Programming**  
   Solves multi-objective trade-offs between throughput quotas, compressor electrical load, and reactor pressure stabilization using both Weighted-Sum and Preemptive (Lexicographic) formulations.

7. **Sensitivity Analysis**  
   Calculates Simplex shadow prices (dual variables) and local finite-difference gradients to identify binding bottlenecks.

8. **Interactive Dashboard**  
   Full-featured Streamlit application with live solver runs, scenario comparison, and interactive parameter adjustment.

---

## Getting Started

### Prerequisites
- Python 3.12+
- Packages listed in `requirements.txt`

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run streamlit_app.py
# or: streamlit run app.py

# 4. Run automated tests
python -m pytest tests/ -v

# 5. Run end-to-end numerical validation
python validate_results.py
```

---

## Repository Structure

```
OR/
├── streamlit_app.py              # Main dashboard application
├── app.py                        # Lightweight entrypoint launcher
├── requirements.txt              # Project dependencies
├── validate_results.py           # 50-check numerical validation script
├── run_optimization.py           # Reproducible CLI optimization pipeline
├── generate_final_plots.py       # Plot and figure generator
│
├── src/
│   ├── data/
│   │   ├── tep_loader.py         # Benchmark data ingestion
│   │   └── statistics_generator.py
│   ├── process/
│   │   ├── baseline_analyzer.py  # Stationarity and drift checks
│   │   └── variable_dictionary.py# 52-variable definitions and bounds
│   ├── models/
│   │   └── surrogate_models.py   # Ridge, OLS, and polynomial surrogates
│   ├── optimization/
│   │   ├── lp_optimizer.py       # Dual Simplex LP formulations
│   │   ├── goal_programming.py   # Weighted-Sum and Preemptive GP
│   │   ├── cost_model.py         # Downs & Vogel operating cost model
│   │   └── energy_model.py       # Thermodynamic electrical and thermal power
│   └── analysis/
│       └── lp_sensitivity.py     # Dual shadow prices and local sensitivity
│
├── tests/                        # 88 automated pytest cases
│   ├── test_data_loader.py
│   ├── test_cost_energy.py
│   ├── test_surrogate.py
│   ├── test_lp.py
│   ├── test_goal_programming.py
│   ├── test_ui_workflow.py
│   ├── test_variable_dictionary.py
│   └── test_scientific_validation.py
│
├── results/optimization/         # Optimization JSON and CSV records
├── OR_final_visuals/             # Exported publication figures
├── docs/                         # Parameter traceability tables
├── SCIENTIFIC_VALIDATION_REPORT.md # Mathematical and physical validation details
├── FINAL_VALIDATION_REPORT.md    # Complete numerical validation results
└── PRESENTATION_OUTLINE.md       # Slide deck outline and speaking notes
```

---

## Benchmark Source

The dataset and process model are from:
> Downs, J.J. and Vogel, E.F. (1993). "A plant-wide industrial process control problem."  
> *Computers & Chemical Engineering*, 17(3), 245–255.

---

## Tech Stack

| Component | Technology |
|---|---|
| LP Solver | SciPy HiGHS Dual Simplex (`highs-ds`) |
| Surrogate Models | scikit-learn Ridge Regression |
| Statistical Testing | statsmodels Augmented Dickey-Fuller (ADF) |
| Dashboard | Streamlit + Plotly |
| Test Suite | pytest (88 unit and integration tests) |
| Language | Python 3.12 |
