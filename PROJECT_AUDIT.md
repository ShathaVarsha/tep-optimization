# Tennessee Eastman Process (TEP) Optimization — Project Audit Report

---

## 1. Executive Summary

This audit documents the baseline architecture, surrogate models, and optimization components for the Tennessee Eastman Process (TEP) project. The workflow brings together:
1. Benchmark TEP industrial dataset ingestion and validation (`d00.dat`–`d21_te.dat`).
2. Stationarity testing, descriptive statistics, and baseline operational extraction.
3. Surrogate modeling (OLS, Ridge, Polynomial Ridge) mapping manipulated variables to critical plant responses.
4. Linear Programming (LP) via Dual Simplex (SciPy HiGHS) and Goal Programming (Weighted-Sum and Preemptive).
5. Chemical plant operating cost and thermal/electrical energy modeling.
6. Local finite-difference sensitivity and LP duality analysis (shadow prices, reduced costs, slacks).
7. Interactive User Interface and automated test verification.

The audit identifies foundational components across the codebase, reconciles the 7-variable baseline with the enhanced 8-variable formulation (`+ XMV_11`), integrates an explicit energy minimization objective, and tracks the resolution of known gaps.

---

## 2. Directory Structure and Component Mapping

```text
OR/
├── The Tennessee Eastman process (TEP)/  # Authoritative benchmark .dat files
│   └── The Tennessee Eastman process (TEP)/TEdata/TEdata/ (d00.dat - d21_te.dat: 44 files)
├── src/
│   ├── data/
│   │   ├── tep_loader.py               # Ingestion, transposition correction, integrity audit
│   │   └── statistics_generator.py     # Descriptive stats, IQR outliers, correlations
│   ├── process/
│   │   ├── variable_dictionary.py      # 52-variable catalog, engineering bounds, nominal points
│   │   └── baseline_analyzer.py        # ADF stationarity, rolling drift, domain checks
│   ├── models/
│   │   └── surrogate_models.py         # OLS, Ridge, Poly surrogates + Mahalanobis extrapolation
│   ├── optimization/
│   │   ├── cost_model.py               # Downs & Vogel (1993) economic purge, comp, steam model
│   │   ├── energy_model.py             # Electrical work (kW) + Reboiler thermal duty (kW)
│   │   ├── lp_optimizer.py             # Dual Simplex (HiGHS) LP: Max Production, Min Cost
│   │   ├── lp_optimizer_xmv11_experiment.py # 8-variable experimental copy
│   │   └── goal_programming.py        # Weighted-Sum & Preemptive Goal Programming
│   └── analysis/
│       └── lp_sensitivity.py           # Variable reduced costs, constraint shadow prices
├── tests/
│   ├── test_data_loader.py             # 7 tests (shapes, 44-file audit, statistics)
│   ├── test_variable_dictionary.py     # 6 tests (52 variables, bounds, units)
│   ├── test_surrogate.py               # 5 tests (ADF, Ridge, Poly, extrapolation)
│   ├── test_cost_energy.py             # 4 tests (pricing, thermal duty, baseline d00)
│   ├── test_lp.py                      # 3 tests (Dual Simplex, constraints, sensitivity)
│   └── test_goal_programming.py        # 35 tests (Weighted-Sum, Preemptive lexicographic)
├── results/
│   └── optimization/                   # Summary CSVs, JSON records, diagnostic outputs
│       └── xmv11_experiment/          # 8-variable surrogate & optimization artifacts
├── OR_final_visuals/                   # Pre-rendered PNG plots & summary CSVs
└── [Diagnostic & Experiment Scripts]   # Root scripts to be unified and cleaned
```

---

## 3. Component Status and Findings

### 3.1 Data Pipeline (`src/data/`)
* **Status:** Working and Validated.
* **Key Observations:**
  - `tep_loader.py` properly identifies that `d00.dat` is formatted as (52, 500) and automatically transposes it to (500, 52) samples × variables.
  - Confirms zero NaN and zero infinite values across all 44 `.dat` files in the benchmark directory.
  - `statistics_generator.py` provides skewness, kurtosis, IQR outlier detection, and correlation matrices.
  - Robust default path resolution handles nested workspace paths reliably.

### 3.2 Process Knowledge and Baseline Analysis (`src/process/`)
* **Status:** Working.
* **Key Observations:**
  - `variable_dictionary.py` provides an authoritative catalog of all 41 measured (`XMEAS_01` to `XMEAS_41`) and 11 manipulated (`XMV_01` to `XMV_11`) variables, complete with nominal setpoints, engineering bounds, and shutdown limits from Downs & Vogel (1993).
  - `baseline_analyzer.py` implements Augmented Dickey-Fuller (ADF) testing and rolling drift metrics to confirm steady-state operation of `d00.dat`.
* **Resolved Typing Defect:**
  - `Union` import in `src/process/baseline_analyzer.py` corrected to ensure strict type compliance.

### 3.3 Surrogate Modeling (`src/models/surrogate_models.py`)
* **Status:** Working, with dual-architecture support.
* **Key Finding (The `XMV_11` Discrepancy):**
  - The initial 7-decision variable set `[XMV_01, XMV_02, XMV_03, XMV_04, XMV_05, XMV_06, XMV_09]` yielded a negative cross-validated R² score (R² = -0.0627, MAE = 0.4976 m³/hr) on `XMEAS_17` (Product Flow), indicating negligible predictive capacity on `d00.dat`.
  - Adding `XMV_11` (Condenser Cooling Water Valve) as the 8th decision variable improved TimeSeriesSplit cross-validation R² to **0.9951** (MAE = 0.0308 m³/hr).
  - Cross-dataset audits across `d00` through `d05` show strong negative correlation (r = -0.9996 in `d00`, r ≤ -0.938 across normal and fault datasets). In the TEP control architecture, condenser cooling water governs vapor condensation and the liquid inventory feeding the product stream.
* **Architecture Enhancement:**
  - `surrogate_models.py` supports both the canonical 7-feature set and the enhanced 8-feature set configurably.

### 3.4 Energy Modeling (`src/optimization/energy_model.py`)
* **Status:** Complete.
* **Key Observations:**
  - Distinguishes direct electrical work measured by `XMEAS_20` (Compressor Power in kW) from reboiler steam thermal duty derived from `XMEAS_19` (Stripper Steam mass flow in kg/hr × latent heat λ = 2257 kJ/kg / 3600 s/hr = kW).
  - Baseline total equivalent power: 341.47 kW (electrical) + 144.37 kW (thermal) = 485.84 kW.
* **Optimization Integration:**
  - Explicit energy minimization LP objective added alongside the compressor power inequality constraint (P_comp ≤ 355 kW).

### 3.5 Operating Cost Modeling (`src/optimization/cost_model.py`)
* **Status:** Working and mathematically validated.
* **Key Observations:**
  - Implements the Downs & Vogel (1993) economic objective:
    Cost ($/hr) = Purge Gas Loss + Compressor Electricity + Steam Cost
  - Uses the molar flow conversion (44.61937 kmol/hr per kscmh) derived from the original Fortran simulation driver `teprob.f`.
  - Baseline cost on `d00.dat` evaluates to **$106.39/hr** (Purge: $87.35/hr, Compressor: $18.30/hr, Steam: $0.73/hr).
* **Linear Cost Model Proxy:**
  - The linear optimizer uses an average purge price of $5.80/kmol, evaluating to $106.40/hr at baseline (matching the $106.39/hr composition-weighted benchmark within 0.01%).
  - This proxy permits direct solution via Linear Programming without requiring bilinear composition NLP terms.

### 3.6 Linear Programming and Sensitivity (`src/optimization/lp_optimizer.py`, `src/analysis/lp_sensitivity.py`)
* **Status:** Working with empirical response bounding.
* **Key Observations:**
  - Uses SciPy HiGHS Dual Simplex solver (`method='highs-ds'`).
  - Solves Production Maximization, Cost Minimization, and Energy Minimization.
  - Extracts dual variables (shadow prices), reduced costs, and binding constraint slacks.
* **Response Bounding Integration:**
  - In unconstrained 7-variable runs, cost minimization pushed decision variables to ±10% bounds based on an unreliable product surrogate.
  - In the 8-variable formulation with empirical process response bounding, optimization yields verified results:
    - Production Maximization: 22.91 → 24.68 m³/hr (+7.73%).
    - Cost Minimization: $106.40 → $97.73/hr (-8.15%).
    - Energy Minimization: 485.84 → 469.03 kW (-3.46%).

### 3.7 Goal Programming (`src/optimization/goal_programming.py`)
* **Status:** Fully functional and tested.
* **Key Observations:** Implements both Weighted-Sum GP and Preemptive (Lexicographic) GP, penalizing normalized deviations from user targets across production, energy, and reactor pressure.

### 3.8 User Interface & Backend Integration
* **Status:** Completed.
* **Delivered Application:** The interactive Streamlit dashboard (`streamlit_app.py`) provides:
  - Dataset overview and variable dictionary explorer.
  - Baseline stationarity & energy/cost diagnostics.
  - Surrogate model selector (7-variable vs 8-variable; Ridge, Linear, Poly) with validation metrics (R², MAE, RMSE).
  - Interactive LP solver supporting all three objectives (Production, Cost, Energy) and Goal Programming.
  - Side-by-side comparison tables, interactive Plotly charts, and CSV exports.
  - Local sensitivity and Simplex dual shadow price explorers.

---

## 4. Summary of Identified Items and Resolutions

1. **Typing Import:** Added `Union` import in `src/process/baseline_analyzer.py`.
2. **Surrogate Architecture:** Updated `src/models/surrogate_models.py` to support both 7-feature and 8-feature (`+ XMV_11`) sets.
3. **Energy Objective:** Formulated and verified the Energy Minimization LP objective in `lp_optimizer.py`.
4. **Interactive Dashboard:** Built and verified `streamlit_app.py` with full backend integration.
5. **Test Coverage:** Suite expanded to 88 unit tests and 50 independent validation checks.

---

## 5. Implementation Roadmap Summary

| Phase | Module / Action | Outcome |
| :---: | :--- | :--- |
| **Phase 1** | Baseline Analysis & Typing | Fixed typing imports; verified ADF stationarity across all 44 datasets. |
| **Phase 2** | Surrogate Enhancements | Supported configurable feature sets; documented R² = 0.9951 Condenser CW breakthrough. |
| **Phase 3** | Energy Optimization | Formulated explicit Energy Minimization LP objective; verified against steam and electrical models. |
| **Phase 4** | Interactive Dashboard | Deployed `streamlit_app.py` with multi-scenario comparison and sensitivity views. |
| **Phase 5** | Test & Validation Suite | Validated 88 tests and 50 numerical checks with zero regressions. |
| **Phase 6** | Documentation & Reporting | Generated comprehensive validation reports, audit documentation, and presentation materials. |
