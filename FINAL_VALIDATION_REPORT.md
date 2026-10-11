# Final Validation and Verification Report

## Tennessee Eastman Process Optimization

---

## 1. Executive Summary

This report documents the numerical, mathematical, and software validation of the data-driven optimization framework for the Tennessee Eastman industrial chemical manufacturing process (Downs & Vogel, 1993).

Key findings and deliverables:

1. **Benchmark Ingestion and Integrity:** Automated detection of orientation layout across all 44 benchmark `.dat` files with zero missing values and zero infinities.
2. **Stationarity Validation:** Augmented Dickey-Fuller (ADF) unit-root testing (p < 0.05 across all monitored variables) and rolling drift analysis (< 5.0%) confirm steady-state baseline operation in `d00.dat`.
3. **Surrogate Model Formulation:** Resolving the product flow prediction bottleneck by including the condenser cooling water valve (`XMV_11`) in the decision space, raising cross-validated R² on finished product flow from -0.063 to **0.9951** (MAE = 0.0308 m³/hr).
4. **Energy and Cost Modeling:** Separating direct electrical work (`XMEAS_20`, 341.47 kW) from stripper reboiler thermal duty (`XMEAS_19`, 144.37 kW equivalent), defining a baseline power footprint of 485.84 kW. The Downs & Vogel operating cost model evaluates baseline operating cost at **$106.39/hr** (detailed composition-weighted) versus **$106.40/hr** (linear LP proxy).
5. **Linear Programming (Dual Simplex):** Formulated and solved using SciPy HiGHS Dual Simplex (`highs-ds`) with trust regions and empirical response bounding:
   - **Production Maximization:** Throughput increases from 22.9093 m³/hr to **24.6813 m³/hr** (+7.73%) while respecting reactor pressure (≤ 2800 kPa) and compressor power limits (≤ 355 kW).
   - **Operating Cost Minimization:** Cost decreases from $106.40/hr to **$97.73/hr** (-8.15%) while satisfying the production quota of 22.89 m³/hr.
   - **Total Energy Minimization:** Equivalent process power decreases from 485.84 kW to **469.03 kW** (-3.46%).
   - **Compressor Power Minimization:** Direct motor electrical draw decreases from 341.47 kW to **337.77 kW** (-1.08%).
6. **Multi-Objective Goal Programming:** Formulated and solved via both Weighted-Sum and Preemptive (Lexicographic) GP, resolving trade-offs between throughput, pressure stability, and compressor work.
7. **Sensitivity Analysis:** Clear distinction between Simplex shadow prices (dual variables) and local finite-difference what-if response sensitivities.
8. **Interactive Dashboard:** Complete Streamlit web app (`streamlit_app.py`) providing interactive scenario configuration, parameter tuning, live solver runs, and Plotly visualization.
9. **Automated Verification:** 88 automated pytest test cases and 50 independent numerical validation checks pass cleanly with zero failures.

---

## 2. Dataset Auditing and Preprocessing Validation

### 2.1 Benchmark Directory Structure
The workspace contains the benchmark files distributed with the original simulation:
- 44 benchmark `.dat` files located in `The Tennessee Eastman process (TEP)/The Tennessee Eastman process (TEP)/TEdata/TEdata`.
- Files include training sets (`d00.dat` to `d21.dat`) and testing sets (`d00_te.dat` to `d21_te.dat`).

### 2.2 Transposition Detection and Orientation
The ingestion module `src/data/tep_loader.py` dynamically checks array dimensions:
- `d00.dat` is stored as an array of shape (52, 500).
- The loader automatically detects that the first dimension equals the 52 process variables, transposing the array to (500, 52) (500 time observations × 52 process variables).
- Testing datasets (`d00_te.dat` through `d21_te.dat`) have native dimensions of (960, 52), requiring no transposition.

### 2.3 Numerical Integrity Audit
Automated audit across all 44 benchmark files (`test_data_loader.py`) established:
- **Missing Values:** 0 NaN values across all 44 datasets.
- **Infinite Values:** 0 Inf values across all 44 datasets.
- **Variable Mapping:** Exactly 52 standardized column tags assigned: 41 measured variables (`XMEAS_01` to `XMEAS_41`) and 11 manipulated control valve variables (`XMV_01` to `XMV_11`).

---

## 3. Baseline Stationarity & Operating Conditions

### 3.1 Steady-State Verification (`d00.dat`)
In chemical process optimization, surrogate models and linear programs must be anchored around a validated stationary baseline. `src/process/baseline_analyzer.py` verifies this via:
1. **Augmented Dickey-Fuller (ADF) Unit-Root Test:** Null hypothesis of a unit root is rejected (p < 0.05) for all key state variables.
2. **Rolling Mean Drift:** Evaluates drift over rolling windows of 50 samples. Normalized drift across the entire 500-sample run is < 5.0% for all critical variables.

| Variable Tag | Variable Description | Baseline Mean | ADF Statistic | p-value | Critical Value (5%) | Rolling Drift (%) | Steady-State Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `XMEAS_07` | Reactor Pressure (kPa gauge) | 2705.40 | -4.612 | 0.00012 | -2.867 | 0.21% | STATIONARY |
| `XMEAS_08` | Reactor Level (%) | 65.03 | -5.124 | 0.00001 | -2.867 | 0.45% | STATIONARY |
| `XMEAS_09` | Reactor Temperature (°C) | 120.41 | -4.891 | 0.00004 | -2.867 | 0.18% | STATIONARY |
| `XMEAS_10` | Purge Gas Flow Rate (kscmh) | 0.3376 | -3.987 | 0.00148 | -2.867 | 1.12% | STATIONARY |
| `XMEAS_17` | Product Delivery Rate (m³/hr) | 22.9093 | -4.231 | 0.00057 | -2.867 | 0.68% | STATIONARY |
| `XMEAS_19` | Stripper Steam Flow (kg/hr) | 230.28 | -4.550 | 0.00016 | -2.867 | 0.94% | STATIONARY |
| `XMEAS_20` | Compressor Work (kW) | 341.47 | -4.318 | 0.00041 | -2.867 | 0.32% | STATIONARY |

---

## 4. Surrogate Modeling & The XMV_11 Discovery

### 4.1 Feature Selection and Model Architecture
Surrogate models approximate the complex dynamics of the TEP simulation around the nominal operating baseline:
- **Model Formulations Evaluated:** Ordinary Least Squares (OLS), Ridge Regression (L₂ regularized, α = 10.0), and Polynomial Ridge (degree = 2, α = 50.0).
- **Validation Scheme:** TimeSeriesSplit chronological cross-validation (k = 5 splits) to prevent temporal data leakage.

### 4.2 Product Flow Prediction Bottleneck & Empirical Resolution
In the initial project implementation, 7 manipulated variables were used:
`u₇ = [XMV_01, XMV_02, XMV_03, XMV_04, XMV_05, XMV_06, XMV_09]`

While u₇ predicted purge flow (R² = 0.878) and steam flow (R² = 0.774), it failed on Product Flow (`XMEAS_17`):
- **7-Variable Model on `XMEAS_17`:** Cross-validated R² = -0.0627, MAE = 0.4976 m³/hr. A negative R² indicates the model performs worse than simply predicting the mean.

**Root Cause:** In the Tennessee Eastman control architecture (Downs & Vogel 1993, Section 3), reactor condenser duty controls the condensation rate of recycle gas and product vapors. The condenser cooling water valve is `XMV_11`. When cooling water flow increases, condensation increases, directly governing the liquid level entering the stripper and product stream.

**The 8-Variable Formulation (+ XMV_11):**
`u₈ = [XMV_01, XMV_02, XMV_03, XMV_04, XMV_05, XMV_06, XMV_09, XMV_11]`

Adding `XMV_11` resolves the model:
- Cross-validated R² for `XMEAS_17` jumps from -0.0627 to **0.9951**.
- MAE drops from 0.4976 m³/hr to **0.0308 m³/hr** (relative error 0.13%).

### 4.3 Generalization Across Benchmark Datasets
Cross-dataset correlation between `XMV_11` and `XMEAS_17` was evaluated across six independent operating datasets:

| Dataset ID | Operating Condition | Observations | Correlation r(XMV_11, XMEAS_17) |
| :---: | :--- | :---: | :---: |
| `d00` | Normal Operation (Training) | 500 | -0.9996 |
| `d01` | Fault 1 (A/C feed ratio step) | 480 | -0.9746 |
| `d02` | Fault 2 (B composition step) | 480 | -0.9510 |
| `d03` | Fault 3 (D feed temp step) | 480 | -0.9996 |
| `d04` | Fault 4 (Reactor cooling water step) | 480 | -0.9994 |
| `d05` | Fault 5 (Condenser cooling water step)| 480 | -0.9380 |

The strong inverse correlation is consistent across both normal and fault modes.

### 4.4 Final 8-Variable Ridge Model Metrics

| Response Variable Tag | Response Name | Baseline Mean | Train R² | CV R² (k=5) | CV MAE | CV RMSE | CV Relative Error (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `XMEAS_17` | Product Delivery Rate (m³/hr) | 22.9093 | 0.9989 | **0.9951** | 0.0308 | 0.0390 | 0.13% |
| `XMEAS_07` | Reactor Pressure (kPa gauge) | 2705.40 | 0.6623 | **0.3005** | 2.9304 | 3.6014 | 0.11% |
| `XMEAS_10` | Purge Flow Rate (kscmh) | 0.3376 | 0.8979 | **0.8775** | 0.0032 | 0.0040 | 0.95% |
| `XMEAS_19` | Stripper Steam Flow (kg/hr) | 230.28 | 0.9591 | **0.7733** | 1.8042 | 2.2163 | 0.79% |
| `XMEAS_20` | Compressor Power (kW) | 341.47 | 0.7747 | **0.3602** | 0.6185 | 0.7360 | 0.18% |

---

## 5. Thermodynamic Energy & Operating Cost Models

### 5.1 Energy Model Formulation
Physical sensor measurements are distinguished from thermodynamic proxies:

1. **Compressor Electrical Power (kW):** Directly measured by sensor `XMEAS_20`. This represents electrical power consumed by the recycle gas compressor motor:  
   `P_comp = XMEAS_20 (kW)`
2. **Stripper Reboiler Thermal Power (kW):** Derived from steam mass flow (`XMEAS_19`, kg/hr) using the latent heat of saturated low-pressure steam (λ = 2257 kJ/kg):  
   `P_steam = (XMEAS_19 × 2257 kJ/kg) / 3600 s/hr = 0.62694 × XMEAS_19 (kW)`
3. **Total Equivalent Process Power Footprint (kW):**  
   `P_total = P_comp + P_steam = XMEAS_20 + 0.62694 × XMEAS_19 (kW)`

**Baseline Energy Values on `d00`:**
- Compressor Electrical Power: 341.47 kW (70.28% of total)
- Stripper Reboiler Thermal Duty: 144.37 kW (29.72% of total)
- Total Process Power: **485.84 kW**

**Physical Limitations:**  
Sensible heat removed by reactor cooling water (`XMV_10`) and condenser cooling water (`XMV_11`) is omitted from power calculations because benchmark datasets record only valve opening percentages and exit temperatures, omitting cooling water mass flow rates.

### 5.2 Operating Cost Model Formulation
Downs & Vogel (1993, Section 4, Table 4) define the operating cost function as:  
`Cost ($/hr) = Purge Gas Loss Cost + Compressor Electricity Cost + Steam Cost`

Economic parameters:
- Electricity Price: $0.0536/kWh
- Steam Price: $0.00318/kg
- Purge Component Molar Prices: A = $2.206, C = $6.016, D = $8.222, E = $10.428, F = $14.444, G = $17.653, H = $17.653 ($/kmol).
- Molar Flow Conversion Factor: Derived from Fortran simulation code `teprob.f` (lines 679–688):  
  `1 kscmh = (35.3145 / 0.359) × 0.45359237 = 44.61937 kmol/hr`

**Baseline Cost Comparison:**
- **Detailed Composition-Weighted Model:** Evaluates instantaneous purge compositions across Stream 9 (`XMEAS_29` through `XMEAS_36`):  
  `Detailed Baseline Cost = $87.35 (purge) + $18.30 (comp) + $0.73 (steam) = $106.39/hr`
- **Linear LP Proxy Model:** Uses empirical weighted average purge price of $5.80/kmol:  
  `Linear Proxy Baseline Cost = (0.3376 × 44.61937 × 5.80) + (341.47 × 0.0536) + (230.28 × 0.00318) = $106.40/hr`

The linear proxy matches the composition-weighted benchmark within $0.01/hr (0.01%), allowing standard Linear Programming without bilinear composition terms.

---

## 6. Linear Programming Optimization & Results

### 6.1 Mathematical Formulations (Dual Simplex)

All linear programs are solved using SciPy HiGHS Dual Simplex (`highs-ds`) with trust region constraints `u_base − δ ≤ u ≤ u_base + δ` (δ = 10.0 percentage points) and empirical response bounding `y_min ≤ ŷ(u) ≤ y_max`.

#### Formulation 1: Production Maximization
```text
Maximize:   Product Flow(u) = c_prodᵀ u + d_prod
Subject to: Reactor Pressure(u) ≤ 2800 kPa
            Compressor Power(u) ≤ 355 kW
            y_min ≤ ŷ_j(u) ≤ y_max  for all j in {17, 7, 10, 19, 20}
            u_base − 10 ≤ u ≤ u_base + 10
            0 ≤ u ≤ 100
```

#### Formulation 2: Operating Cost Minimization
```text
Minimize:   Operating Cost(u) = c_costᵀ u + d_cost
Subject to: Product Flow(u) ≥ 22.89 m³/hr  (Production Quota)
            Reactor Pressure(u) ≤ 2800 kPa
            Compressor Power(u) ≤ 355 kW
            y_min ≤ ŷ_j(u) ≤ y_max
            u_base − 10 ≤ u ≤ u_base + 10
```

#### Formulation 3: Total Energy Minimization
```text
Minimize:   P_total(u) = c_energyᵀ u + d_energy
Subject to: Product Flow(u) ≥ 22.89 m³/hr
            Reactor Pressure(u) ≤ 2800 kPa
            Trust region bounds: u_base − 10 ≤ u ≤ u_base + 10
```

---

### 6.2 Summary of Optimization Scenarios

| Optimization Scenario | Objective Function | Baseline Value | Optimized Value | Change | Solver Iterations | Solution Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Problem 1: Production Max** | Product Flow (XMEAS_17) | 22.9093 m³/hr | **24.6813 m³/hr** | **+7.73%** | 2 | Optimal (HiGHS) |
| **Problem 2: Cost Min** | Operating Cost ($/hr) | $106.4015/hr | **$97.7254/hr** | **-8.15%** | 4 | Optimal (HiGHS) |
| **Problem 3: Energy Min (Total)**| Total Power (kW) | 485.8436 kW | **469.0334 kW** | **-3.46%** | 4 | Optimal (HiGHS) |
| **Problem 4: Energy Min (Comp)** | Compressor Work (kW) | 341.4695 kW | **337.7700 kW** | **-1.08%** | 2 | Optimal (HiGHS) |

---

### 6.3 Optimal Manipulated Variable Setpoints (Valves %)

| Variable Tag | Variable Description | Baseline Setpoint | Production Max Setpoint | Cost Min Setpoint | Energy Min Setpoint | Units |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `XMV_01` | D Feed Flow Valve | 63.031% | 64.436% (+1.41%) | 61.296% (-1.74%) | 61.296% (-1.74%) | % |
| `XMV_02` | E Feed Flow Valve | 54.006% | 55.111% (+1.11%) | 54.990% (+0.98%) | 55.050% (+1.04%) | % |
| `XMV_03` | A Feed Flow Valve | 24.721% | 33.682% (+8.96%) | 17.781% (-6.94%) | 17.781% (-6.94%) | % |
| `XMV_04` | A+C Feed Flow Valve | 61.322% | 57.903% (-3.42%) | 57.903% (-3.42%) | 57.903% (-3.42%) | % |
| `XMV_05` | Compressor Recycle Valve | 22.252% | 20.811% (-1.44%) | 21.147% (-1.11%) | 20.811% (-1.44%) | % |
| `XMV_06` | Purge Valve | 40.082% | 36.385% (-3.70%) | 36.128% (-3.95%) | 36.385% (-3.70%) | % |
| `XMV_09` | Stripper Steam Valve | 47.461% | 42.162% (-5.30%) | 42.162% (-5.30%) | 42.162% (-5.30%) | % |
| `XMV_11` | Condenser Cooling Water Valve | 18.216% | 14.009% (-4.21%) | 18.259% (+0.04%) | 18.259% (+0.04%) | % |

---

### 6.4 Predicted Process State Responses under Optimal Policies

| Process State Response | Baseline | Production Max | Cost Min | Energy Min | Safety Limit / Alarm | Units |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `XMEAS_17` (Product Delivery) | 22.909 | **24.681** | 22.909 | 22.909 | Lower Quota: 22.89 | m³/hr |
| `XMEAS_07` (Reactor Pressure) | 2705.4 | 2702.7 | 2698.7 | 2698.7 | Max Safety: 2800 | kPa gauge |
| `XMEAS_10` (Purge Gas Flow) | 0.3376 | 0.3051 | 0.3051 | 0.3051 | Upper Bound: 0.40 | kscmh |
| `XMEAS_19` (Stripper Steam Flow)| 230.28 | 210.55 | 209.37 | 209.37 | Max Capacity: 1500 | kg/hr |
| `XMEAS_20` (Compressor Power) | 341.47 | 337.77 | 337.77 | 337.77 | Max Rating: 355 | kW |
| Operating Cost | $106.40 | $98.67 | **$97.73** | $97.73 | — | $/hr |
| Total Process Power | 485.84 | 469.77 | 469.03 | **469.03** | — | kW |

---

## 7. Sensitivity Analysis: Duality vs Local What-If

### 7.1 Linear Programming Duality (Shadow Prices)
In Linear Programming, shadow prices represent the marginal change in optimal objective value per unit relaxation of a binding constraint (∂z*/∂b_i):
- In **Production Maximization**, the binding constraints are the empirical upper limit on product flow and the empirical lower limit on compressor work.
- In **Cost Minimization**, the production quota constraint (`XMEAS_17 ≥ 22.89 m³/hr`) is strictly binding with zero surplus slack.

### 7.2 Local What-If Finite-Difference Sensitivity
To evaluate local operational sensitivity around the baseline, 1% perturbations (Δu_i = ±0.01 × span(u_i)) were evaluated against the surrogate model:

| Response Variable | Most Sensitive Manipulated Input | Perturbation Direction | Response Change (Δy) | Local Sensitivity Slope (Δy / Δu) | Engineering Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `XMEAS_07` (Pressure) | `XMV_05` (Compressor Recycle) | Increase | +0.2145 kPa | +8.4188 | Opening recycle valve raises suction pressure and reactor backpressure. |
| `XMEAS_10` (Purge Rate) | `XMV_06` (Purge Valve) | Decrease | -0.0007 kscmh | +0.0075 | Purge valve directly throttles stream 9 vent flow. |
| `XMEAS_17` (Product Flow) | `XMV_11` (Condenser CW Valve)| Increase | -0.0377 m³/hr | -0.4128 | Condenser cooling rate directly influences product condensation equilibrium. |
| `XMEAS_19` (Steam Flow) | `XMV_09` (Steam Valve) | Decrease | -0.3674 kg/hr | +3.6876 | Steam valve directly throttles stripper reboiler heating utility. |
| `XMEAS_20` (Compressor Work)| `XMV_05` (Compressor Recycle) | Increase | +0.0631 kW | +2.4757 | Higher recycle flow increases volumetric flow through the centrifugal impeller. |

**Methodological Distinction:**  
These finite-difference sensitivities are localized numerical approximations of surrogate partial derivatives. They must not be confused with:
1. True physical plant causal dynamics (which include closed-loop PID controller feedback reactions).
2. Global LP shadow prices (which evaluate changes to the active basic feasible solution basis).

---

## 8. User Interface & Software Architecture

### 8.1 Streamlit Interactive Dashboard (`streamlit_app.py`)
A full-stack application built using Streamlit and Plotly provides 8 interactive views:
1. **Executive Overview & Process Flowsheet:** High-level KPI cards, 5-unit TEP flowsheet summary, and literature parameter tables.
2. **Benchmark Dataset & Variable Catalog Explorer:** 44-file integrity audit, 52-variable searchable dictionary, time series trajectories, empirical distributions, and cross-variable correlation heatmaps.
3. **Baseline Stationarity & Equilibrium Analysis:** Augmented Dickey-Fuller stationarity tests (p < 0.05), rolling drift metrics (< 5.0%), and Downs & Vogel baseline cost/energy breakdowns.
4. **Machine Learning Surrogate Modeling:** Interactive selector for Ridge, Linear, and Polynomial models; 7-var vs 8-var mode toggle; TimeSeriesSplit cross-validation tables (R², MAE, RMSE); live parity scatter plots and extrapolation boundary indicators.
5. **Linear Programming: Production Maximization:** Interactive solver panel with sliders for trust region delta (±δ%), pressure limits, and compressor power ratings; live HiGHS Dual Simplex solver execution; side-by-side comparison tables, bar charts, and CSV exports.
6. **Linear Programming: Operating Cost & Energy Minimization:** Solves cost minimization, total equivalent power minimization, or compressor electrical power minimization subject to target production quota.
7. **Multi-Objective Goal Programming:** Interactive weighted-sum and preemptive lexicographic solver with user-defined target values and priority rankings.
8. **Sensitivity Analysis, Dual Shadow Prices & Report Exporter:** Dual Simplex LP shadow prices, local finite-difference what-if perturbation calculator with dominant sensitivity bar charts, master scenario comparison table, and downloadable publication artifacts.

---

## 9. Automated Testing & Verification Suite

An 88-test automated test suite was constructed using `pytest`, covering all modules and integrations:
- `tests/test_data_loader.py` (7 tests): Array shapes, transposition, zero NaN/Inf across all 44 datasets.
- `tests/test_variable_dictionary.py` (6 tests): Catalog completeness, tag indices, engineering bounds.
- `tests/test_cost_energy.py` (4 tests): Pricing calculations, thermal duty conversions, baseline d00 costs.
- `tests/test_surrogate.py` (7 tests): ADF stationarity, Ridge/OLS/Poly fitting, Mahalanobis extrapolation detection, fold-level metric preservation, holdout testing, 8-variable model performance (R² > 0.95).
- `tests/test_lp.py` (9 tests): Production maximization, cost minimization, energy minimization, response-bounded optimization, LP sensitivity reports, local what-if sensitivity, dimension mismatch validation.
- `tests/test_goal_programming.py` (38 tests): Weighted-sum and preemptive lexicographic goal formulations, deviation variables, priority sequencing, canonical scenarios, dimension validation.
- `tests/test_ui_workflow.py` (8 tests): End-to-end data pipeline, baseline calculations, multi-objective LP execution, GP execution, sensitivity computation.
- `tests/test_scientific_validation.py` (9 tests): Hand-calculated component purge costs, exact baseline effective price derivation ($5.798963/kmol), full component vs linear proxy cost evaluations, independent mathematical feasibility of all 3 LP solutions, deliberate violation detection, solver-failure handling, analytical LP with known reduced costs/shadow prices, extrapolation diagnostics, and thermodynamic power conversions.

In addition, `validate_results.py` executes an independent 50-check numerical verification protocol (50/50 passing, 100%) verifying all mathematical definitions, constraints, sensitivities, and solver outputs.

---

## 10. Engineering Limitations and Conclusions

### 10.1 Limitations & Disclaimers
1. **Surrogate Approximation:** The optimization solutions are derived from linear and regularized Ridge surrogates fitted to nominal steady-state data (`d00.dat`). Cross-validation confirms high statistical accuracy (R² = 0.995 on product flow), but these are local mathematical models rather than dynamic plant simulations.
2. **Approximate Purge Cost:** The optimizer uses a weighted-average purge price ($5.80/kmol), which closely models baseline conditions ($106.40/hr vs $106.39/hr) but assumes constant stream composition.
3. **Safety and Dynamic Stability:** Respecting reactor pressure (≤ 2800 kPa) and compressor power (≤ 355 kW) constraints does not guarantee dynamic plant controllability or safety under major process disturbances. Dynamic closed-loop validation in the Fortran simulation driver is recommended for plant deployment.

### 10.2 Summary
The project applies Operations Research methodologies—regularized surrogate modeling, Dual Simplex linear programming, Goal Programming, and duality/sensitivity analysis—to the Tennessee Eastman benchmark process. Incorporating the condenser cooling water valve `XMV_11` enables accurate throughput prediction, yielding an optimization model achieving:
- **+7.73%** product throughput gain,
- **-8.15%** operating cost reduction, and
- **-3.46%** primary energy footprint reduction.
