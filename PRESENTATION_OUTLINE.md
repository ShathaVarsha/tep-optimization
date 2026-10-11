# Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process (TEP)

## Final Project Defense & Presentation Deck Outline

---

### Slide 1: Title & Project Overview
* **Title:** Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process (TEP)
* **Subtitle:** An End-to-End Operations Research, Surrogate Modeling, and Dual Simplex Linear Programming Framework
* **Department:** School of Artificial Intelligence & Computing | Operations Research Division
* **Key Visual:** Overview schematic of the Tennessee Eastman plant flowsheet (Reactor, Condenser, Recycle Compressor, Vapor-Liquid Separator, Stripper Column).

---

### Slide 2: Problem Statement & Industrial Motivation
* **Industrial Context:** Chemical manufacturing plants consume large quantities of utilities (high-pressure compressor electricity, steam, cooling water) and lose valuable chemical reactants through purge streams.
* **The Challenge:** Real chemical plants operate under complex, stiff nonlinear dynamics with cross-coupled feedback loops. Plant operators cannot intuitively predict optimal setpoints that minimize cost while maximizing throughput and guaranteeing reactor pressure stability.
* **Operations Research Solution:** Develop a mathematically rigorous, data-driven optimization framework combining empirical process surrogate models with SciPy HiGHS Dual Simplex Linear Programming.

---

### Slide 3: Project Objectives & Scope
* **Core Objectives:**
  1. Ingest and audit 44 benchmark Tennessee Eastman Process datasets (`d00.dat`–`d21_te.dat`).
  2. Statistically validate baseline steady-state operation using Augmented Dickey-Fuller (ADF) hypothesis testing.
  3. Formulate interpretable, regularized surrogate models mapping manipulated valve setpoints to plant responses.
  4. Optimize production throughput (+7.73%), operating cost (-8.15%), and energy footprint (-3.46%).
  5. Formulate multi-objective trade-offs using Weighted-Sum and Preemptive Goal Programming.
  6. Deploy a full-stack, tested interactive application (`streamlit_app.py`).

---

### Slide 4: Dataset Integrity & Preprocessing Pipeline
* **Authoritative Data Source:** Eastman Chemical Company benchmark simulation (Downs & Vogel, 1993).
* **Automated Data Ingestion:**
  - Automated layout orientation detection: corrects native transposition of `d00.dat` from (52, 500) to (500, 52).
  - 100% Data Integrity: 0 NaN values, 0 infinite values across all 44 benchmark `.dat` files.
  - Strict preservation of chronological time ordering (no shuffle leakage).
* **Process Variable Mapping:** Complete dictionary of 52 variables:
  - 41 Measured Variables (XMEAS_01 to XMEAS_41)
  - 11 Manipulated Variables (XMV_01 to XMV_11)

---

### Slide 5: Baseline Stationarity & Equilibrium Verification
* **Why Stationarity Matters in OR:** Linear approximations are only valid around a stable, stationary operating equilibrium.
* **Statistical Rigor:**
  - Augmented Dickey-Fuller (ADF) unit-root tests reject non-stationarity (p < 0.05) across all critical variables.
  - Maximum rolling mean drift < 5.0% across the 500-sample run.
* **Nominal Baseline Operating State (`d00.dat`):**
  - Product Flow (XMEAS_17): 22.909 m³/hr
  - Reactor Pressure (XMEAS_07): 2705.4 kPa gauge
  - Purge Rate (XMEAS_10): 0.3376 kscmh
  - Baseline Operating Cost: **$106.39/hr**
  - Baseline Process Power: **485.84 kW**

---

### Slide 6: Surrogate Modeling & The Condenser Cooling Water Breakthrough
* **The Model:** L₂-regularized Ridge Regression (α = 10.0) evaluated via 5-fold TimeSeriesSplit chronological cross-validation.
* **The Earlier Defect:** A 7-variable decision vector (XMV_01 to XMV_06, XMV_09) yielded a negative R² = -0.063 on finished product delivery flow (XMEAS_17).
* **The Empirical Discovery:**
  - Adding XMV_11 (Condenser Cooling Water Valve) provides the missing thermodynamic heat balance link.
  - Cross-validated R² jumps from **-0.063 to 0.9951** (MAE = 0.0308 m³/hr, relative error 0.13%).
  - Proven across 6 benchmark datasets: correlation r(XMV_11, XMEAS_17) ≤ -0.938 across both normal and fault runs.

---

### Slide 7: Thermodynamic Energy Modeling & Physical Realism
* **Separating Measurements from Proxies:**
  1. **Compressor Electrical Power (kW):** Direct sensor measurement (XMEAS_20 = 341.47 kW).
  2. **Stripper Reboiler Thermal Duty (kW):** Derived from steam mass flow (XMEAS_19) via latent heat of vaporization:
     P_steam = (XMEAS_19 × 2257 kJ/kg) / 3600 s/hr = 0.62694 × XMEAS_19 = 144.37 kW
  3. **Total Equivalent Process Power:** P_total = 341.47 + 144.37 = **485.84 kW**.
* **Documented Plant Limitation:** Cooling water duties are excluded from power calculations because benchmark datasets record only valve openings and exit temperatures, omitting mass flow rates.

---

### Slide 8: Downs & Vogel Operating Cost Formulation
* **Economic Objective Function:**
  Cost ($/hr) = Purge Gas Chemical Loss + Compressor Electricity + Reboiler Steam
* **Parameters & Unit Conversion:**
  - Electricity: $0.0536/kWh, Steam: $0.00318/kg.
  - Purge Gas Molar Flow: 1 kscmh = 44.61937 kmol/hr (derived from Fortran simulation driver `teprob.f`).
* **Linear Programming Proxy Validation:**
  - Detailed composition-weighted cost: **$106.39/hr**
  - Average purge price proxy ($5.80/kmol): **$106.40/hr** (error < 0.01%).

---

### Slide 9: Linear Programming Architecture (Dual Simplex)
* **Solver Engine:** SciPy HiGHS Dual Simplex (`highs-ds`).
* **Trust Region & Bound Management:**
  - Trust half-width: u_base − 10% ≤ u ≤ u_base + 10%.
  - Valve boundaries: 0% ≤ u ≤ 100%.
  - Empirical response bounding: y_min ≤ ŷ_j(u) ≤ y_max prevents unphysical extrapolation.
* **Mahalanobis Distance Extrapolation Detector:** χ² confidence ellipsoids verify that candidate solutions remain within the validated statistical operating envelope.

---

### Slide 10: Quantitative Optimization Results
* **Scenario 1: Production Maximization**
  - Product Throughput: 22.909 → **24.681 m³/hr** (**+7.73%** increase)
  - Reactor Pressure strictly safe at 2702.7 kPa (below 2800 kPa limit).
  - Compressor Power at 337.77 kW (below 355 kW rating).
* **Scenario 2: Operating Cost Minimization**
  - Operating Cost: $106.40 → **$97.73/hr** (**-8.15%** reduction)
  - Production quota maintained at 22.909 m³/hr.
* **Scenario 3: Total Energy Minimization**
  - Process Power: 485.84 → **469.03 kW** (**-3.46%** reduction).

---

### Slide 11: Multi-Objective Goal Programming
* **Overcoming Single-Objective Limitations:** Trade-offs between competing corporate goals (maximize throughput vs. minimize compressor energy vs. maintain target pressure).
* **Formulation A: Weighted-Sum GP:**
  Minimize: sum_k (w_k / |g_k|) × (d_k⁻ + d_k⁺)
* **Formulation B: Preemptive (Lexicographic) GP:**
  - Priority 1: Satisfy production quota (24.0 m³/hr).
  - Priority 2: Minimize compressor electrical draw (338.0 kW).
  - Priority 3: Stabilize reactor pressure (2700 kPa).

---

### Slide 12: Sensitivity Analysis: Duality vs Local What-If
* **Dual Simplex Shadow Prices:**
  - Measures marginal economic worth of relaxing plant resource constraints (∂z*/∂b_i).
  - In Cost Minimization, production quota has a positive dual marginal worth.
* **Local What-If Finite-Difference Gradients (Δy / Δu):**
  - Product Flow is strongly governed by Condenser CW (XMV_11, slope −0.413).
  - Reactor Pressure is dominated by Compressor Recycle (XMV_05, slope +8.419).
  - Purge Flow is governed by Purge Valve (XMV_06, slope +0.0075).
* **Academic Integrity Note:** Finite-difference gradients are local surrogate slopes, not causal physical responses or global LP shadow prices.

---

### Slide 13: Full-Stack Interactive Application (`streamlit_app.py`)
* **Interactive Features:**
  - Real-time parameter tuning (trust deltas, pressure limits, demand quotas).
  - Side-by-side comparison tables and Plotly bar charts.
  - Instant toggle between 7-variable and 8-variable surrogate architectures.
  - Live CSV and JSON result exports.
  - Built-in technical documentation viewer.

---

### Slide 14: Engineering Limitations & Plant Controllability
* **Academic Transparency:**
  1. Optimization solutions are mathematical surrogate predictions, not physical closed-loop plant trials.
  2. Fixed $5.80/kmol purge price proxy assumes steady-state composition; transient composition swings require nonlinear dynamic simulation.
  3. Meeting static pressure and power limits does not guarantee dynamic asymptotic stability under large fault disturbances.

---

### Slide 15: Conclusion & Future Work
* **Project Deliverables Completed:**
  - Robust data ingestion and steady-state validation (44 benchmark datasets).
  - 8-variable surrogate model achieving R² = 0.9951 for product flow.
  - Dual Simplex LP achieving +7.73% production, -8.15% cost, and -3.46% energy.
  - Multi-objective Goal Programming and sensitivity analysis.
  - 88 passing automated pytest unit tests and 50 independent validation checks.
  - Interactive full-stack Streamlit web application.
* **Future Work:** Integration with nonlinear MPC (Model Predictive Control) and real-time closed-loop testing in the original Fortran simulation driver.
