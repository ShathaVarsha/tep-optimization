# Tennessee Eastman Process (TEP) Optimization — Project Audit Report

**Course:** 23MNG336: Operational Research | Final Year B.Tech AI & Data Science  
**Project Title:** Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process (TEP)  
**Date of Audit:** October 4, 2026  
**Auditor:** Senior Operations Research & Machine Learning Engineer (AI Pair Programmer)

---

## 1. Executive Summary

This comprehensive audit evaluates the existing codebase for the Tennessee Eastman Process (TEP) optimization project. The project aims to develop a data-driven optimization framework combining:
1. Benchmark TEP industrial dataset ingestion and validation (`d00.dat`–`d21.dat`).
2. Stationarity testing, descriptive statistics, and baseline operational extraction.
3. Surrogate modeling (OLS, Ridge, Polynomial Ridge) mapping manipulated variables to critical plant responses.
4. Linear Programming (LP) via Dual Simplex (SciPy HiGHS) and Goal Programming (Weighted-Sum and Preemptive).
5. Chemical plant operating cost and thermal/electrical energy modeling.
6. Local finite-difference sensitivity and LP duality analysis (shadow prices, reduced costs, slacks).
7. Interactive User Interface and automated test verification.

The codebase currently has strong foundational components (60 passing pytest cases), but also contains critical discrepancies between the initial 7-variable LP implementation and an experimental 8-variable formulation (`+ XMV_11`), an incomplete energy LP objective, a missing interactive UI, and ad-hoc scripts in the root directory that need modular consolidation.

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
* **Status:** **Working and Validated.**
* **Strengths:** 
  - `tep_loader.py` properly identifies that `d00.dat` is formatted as (52, 500) and automatically transposes it to (500, 52) samples × variables.
  - Confirms zero NaN and zero infinite values across all 44 `.dat` files in the benchmark directory.
  - `statistics_generator.py` provides skewness, kurtosis, IQR outlier detection, and correlation matrices.
* **Findings/Gaps:**
  - Robust default path resolution handles nested workspace paths reliably.

### 3.2 Process Knowledge and Baseline Analysis (`src/process/`)
* **Status:** **Working with minor typing defect.**
* **Strengths:**
  - `variable_dictionary.py` provides complete, authoritative cataloging of all 41 measured (`XMEAS_01` to `XMEAS_41`) and 11 manipulated (`XMV_01` to `XMV_11`) variables, complete with nominal setpoints, engineering bounds, and shutdown limits from Downs & Vogel (1993).
  - `baseline_analyzer.py` implements Augmented Dickey-Fuller (ADF) testing and rolling drift metrics to prove steady-state operation of `d00.dat`.
* **Defect Identified:**
  - In `src/process/baseline_analyzer.py` line 40, `Union` is referenced in a type annotation (`Dict[str, Union[float, bool, int]]`) but was omitted from `from typing import Dict, List, Optional`.

### 3.3 Surrogate Modeling (`src/models/surrogate_models.py`)
* **Status:** **Working, but critical model specification disparity identified.**
* **Key Finding (The `XMV_11` Discrepancy):**
  - The base module `surrogate_models.py` uses 7 decision variables: `[XMV_01, XMV_02, XMV_03, XMV_04, XMV_05, XMV_06, XMV_09]`.
  - Under this 7-feature set, predicting `XMEAS_17` (Product Flow) yields a negative cross-validated $R^2$ score ($R^2 = -0.0627$, MAE = 0.4976 m³/hr), indicating that the 7 features have virtually zero explanatory power for product delivery flow on `d00.dat`.
  - When `XMV_11` (Condenser Cooling Water Valve) is added as the 8th decision variable, the TimeSeriesSplit cross-validation $R^2$ jumps to **$0.9951$** (MAE = 0.0308 m³/hr).
  - Cross-dataset audits across `d00` through `d05` prove that `XMV_11` and `XMEAS_17` exhibit an extreme negative correlation ($r = -0.9996$ in `d00`, $r \le -0.938$ across all normal/fault datasets). This is because in the TEP plant control scheme, the condenser cooling water directly governs vapor condensation and liquid inventory feeding the product stream.
* **Architecture Enhancement Needed:**
  - `surrogate_models.py` must support configurable decision feature sets (both the canonical 7-feature set and the enhanced 8-feature set) without code duplication.

### 3.4 Energy Modeling (`src/optimization/energy_model.py`)
* **Status:** **Partially complete; LP optimization objective missing.**
* **Strengths:**
  - Distinguishes direct electrical work measured by `XMEAS_20` (Compressor Power in kW) from reboiler steam thermal duty derived from `XMEAS_19` (Stripper Steam mass flow in kg/hr $\times$ latent heat $\lambda = 2257\text{ kJ/kg} / 3600\text{ s/hr} = \text{kW}$).
  - Evaluates baseline total equivalent power: $341.47\text{ kW}$ (electrical) $+ 144.37\text{ kW}$ (thermal) $= 485.84\text{ kW}$.
* **Gaps:**
  - No energy-minimization LP formulation exists in `lp_optimizer.py`. Energy was only enforced as an upper-bound inequality constraint ($P_{\text{comp}} \le 355\text{ kW}$), rather than an explicit optimization objective.

### 3.5 Operating Cost Modeling (`src/optimization/cost_model.py`)
* **Status:** **Working and mathematically validated.**
* **Strengths:**
  - Implements the exact Downs & Vogel (1993) economic objective:
    $$\text{Cost } (\$/\text{hr}) = \text{Purge Gas Loss} + \text{Compressor Electricity} + \text{Steam Cost}$$
  - Correctly applies the molar flow conversion ($44.61937\text{ kmol/hr per kscmh}$) derived from the original Fortran simulation driver `teprob.f`.
  - Baseline cost on `d00.dat` evaluates to **$\$106.39/\text{hr}$** (Purge: $\$87.35/\text{hr}$, Compressor: $\$18.30/\text{hr}$, Steam: $\$0.73/\text{hr}$).
* **Clarification of Approximate Cost Model:**
  - The linear optimizer uses an approximate average purge price of $\$5.80/\text{kmol}$, which evaluates to $\$106.40/\text{hr}$ at baseline—in virtually exact agreement with the $\$106.39/\text{hr}$ composition-weighted benchmark.
  - This approximation must be explicitly documented so users understand why linear programming can solve it directly without requiring bilinear composition NLP terms.

### 3.6 Linear Programming and Sensitivity (`src/optimization/lp_optimizer.py`, `src/analysis/lp_sensitivity.py`)
* **Status:** **Working, but needs unification and response bounding.**
* **Strengths:**
  - Uses SciPy's HiGHS Dual Simplex solver (`method='highs-ds'`).
  - Solves both Problem 1 (Production Maximization) and Problem 2 (Cost Minimization).
  - Extracts dual variables (shadow prices), reduced costs, and binding constraint slacks.
* **Findings & Rectifications Needed:**
  - In the 7-variable formulation, cost minimization drove decision variables to the $\pm 10\%$ trust bounds, claiming a 27.35% cost reduction, but on an unreliable product flow surrogate.
  - In the 8-variable formulation (`compare_xmv11_optimization_response_bounded.py`), with all responses constrained to empirical operating bounds, optimization yields the well-validated preliminary results:
    - Production Maximization: $22.91 \to 24.68\text{ m}^3/\text{hr}$ ($+7.73\%$).
    - Cost Minimization: $\$106.40 \to \$97.73/\text{hr}$ ($-8.15\%$).
  - We must unify these into a single, clean optimizer interface supporting 7-var or 8-var modes, configurable trust bounds, response bounds, and the 3rd objective (Energy Minimization).

### 3.7 Goal Programming (`src/optimization/goal_programming.py`)
* **Status:** **Fully functional and extensively tested.**
* **Strengths:** 35 unit tests passing. Implements both Weighted-Sum GP and Preemptive (Lexicographic) GP, penalizing positive/negative deviation variables normalized by target values.

### 3.8 User Interface & Backend Integration
* **Status:** **Missing.**
* **Gap:** `requirements.txt` specifies `streamlit>=1.28.0` and `plotly>=5.17.0`, but no Streamlit application file exists yet.
* **Action Required:** Build a complete, state-of-the-art interactive Streamlit application (`src/ui/app.py` and `streamlit_app.py`) providing:
  - Dataset overview and variable dictionary explorer.
  - Baseline stationarity & energy/cost diagnostics.
  - Surrogate model selector (7-variable vs 8-variable; Ridge, Linear, Poly) with validation metrics ($R^2$, MAE, RMSE).
  - Interactive LP solver supporting all three objectives (Production, Cost, Energy) and Goal Programming.
  - Real-time side-by-side comparison tables, interactive Plotly charts, and CSV exports.
  - Local sensitivity and Simplex dual shadow price explorers.

---

## 4. Confirmed Defects and Inconsistencies

1. **Missing Typing Import:** `src/process/baseline_analyzer.py` misses `Union` import from `typing`.
2. **Hardcoded Feature List:** `src/models/surrogate_models.py` hardcodes 7 features, despite the project's discovery that 8 features (`+ XMV_11`) are required for accurate product flow prediction.
3. **Missing Energy Minimization LP Formulation:** Energy is only modeled as a post-hoc calculation or passive constraint, not an active LP objective function.
4. **Duplicate / Scratch Scripts in Root:** Over 15 loose scripts (`_patch_lp_optimizer.py`, `compare_xmv11_optimization_response_bounded.py`, etc.) contain critical logic that belongs in `src/`.
5. **No Frontend / Interactive Dashboard:** No Streamlit app exists to interact with the backend modules.

---

## 5. Prioritized Implementation Plan

| Priority | Phase | Module / Action | Target Outcome |
| :---: | :---: | :--- | :--- |
| **P1** | Phase 2 & 3 | Fix `baseline_analyzer.py` typing; update `surrogate_models.py` to support configurable feature sets (7-var and 8-var) | Zero typing errors; surrogate models seamlessly train with both feature sets; transparent metrics recorded. |
| **P2** | Phase 4 & 6 | Implement Energy Minimization LP objective in `lp_optimizer.py`; add response bounding support | Support all 3 LP objectives (Production, Cost, Energy) in unified solver with Dual Simplex. |
| **P3** | Phase 6 & 7 | Consolidate 8-variable optimization & local what-if sensitivity into `src/optimization/` and `src/analysis/` | Eliminate reliance on scratch root scripts; generate clean CSV and JSON outputs. |
| **P4** | Phase 8 | Build a premium interactive Streamlit Web Application (`streamlit_app.py`) with Plotly visualizations | Complete UI-backend integration allowing live scenario runs, parameter tuning, and CSV exports. |
| **P5** | Phase 9 | Expand automated test suite in `tests/` to cover new energy LP, 8-variable optimizer, and sensitivity | Ensure 100% test pass rate with zero regressions across all modules. |
| **P6** | Phase 10 | Generate `FINAL_VALIDATION_REPORT.md`, academic report materials, presentation outline, and clean documentation | Submission-ready academic deliverables with defensible engineering interpretations. |
