# Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process (TEP)

## Final Project Defense & Presentation Deck Outline

**Course:** 23MNG336: Operational Research  
**Program:** Final-Year B.Tech in Artificial Intelligence & Data Science  
**Team:** B4  
**Date:** October 2026  

---

### Slide 1: Title & Project Overview
* **Title:** Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process (TEP)
* **Subtitle:** An End-to-End Operations Research, Surrogate Modeling, and Dual Simplex Linear Programming Framework
* **Team:** Team B4 | Final Year B.Tech AI & Data Science
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
  4. Optimize production throughput ($+7.73\%$), operating cost ($-8.15\%$), and energy footprint ($-3.46\%$).
  5. Formulate multi-objective trade-offs using Weighted-Sum and Preemptive Goal Programming.
  6. Deploy a full-stack, tested interactive application (`streamlit_app.py`).

---

### Slide 4: Dataset Integrity & Preprocessing Pipeline
* **Authoritative Data Source:** Eastman Chemical Company benchmark simulation (Downs & Vogel, 1993).
* **Automated Data Ingestion:**
  - Automated layout orientation detection: corrects native transposition of `d00.dat` from $(52, 500)$ to $(500, 52)$.
  - 100% Data Integrity: $0\text{ NaN}$ values, $0\text{ infinite}$ values across all 44 benchmark `.dat` files.
  - Strict preservation of chronological time ordering (no shuffle leakage).
* **Process Variable Mapping:** Complete dictionary of 52 variables:
  - 41 Measured Variables ($XMEAS\_01$ to $XMEAS\_41$)
  - 11 Manipulated Variables ($XMV\_01$ to $XMV\_11$)

---

### Slide 5: Baseline Stationarity & Equilibrium Verification
* **Why Stationarity Matters in OR:** Linear approximations are only valid around a stable, stationary operating equilibrium.
* **Statistical Rigor:**
  - Augmented Dickey-Fuller (ADF) unit-root tests reject non-stationarity ($p < 0.05$) across all critical variables.
  - Maximum rolling mean drift $< 5.0\%$ across the 500-sample run.
* **Nominal Baseline Operating State (`d00.dat`):**
  - Product Flow ($XMEAS\_17$): $22.909\text{ m}^3/\text{hr}$
  - Reactor Pressure ($XMEAS\_07$): $2705.4\text{ kPa gauge}$
  - Purge Rate ($XMEAS\_10$): $0.3376\text{ kscmh}$
  - Baseline Operating Cost: **$\$106.39/\text{hr}$**
  - Baseline Process Power: **$485.84\text{ kW}$**

---

### Slide 6: Surrogate Modeling & The Condenser Cooling Water Breakthrough
* **The Model:** $L_2$-regularized Ridge Regression ($\alpha = 10.0$) evaluated via 5-fold TimeSeriesSplit chronological cross-validation.
* **The Earlier Defect:** A 7-variable decision vector ($XMV\_01$ to $XMV\_06, XMV\_09$) yielded a negative $R^2 = -0.063$ on finished product delivery flow ($XMEAS\_17$).
* **The Empirical Discovery:**
  - Adding $XMV\_11$ (Condenser Cooling Water Valve) provides the missing thermodynamic heat balance link.
  - Cross-validated $R^2$ jumps from **$-0.063$ to $\mathbf{0.9951}$** ($MAE = 0.0308\text{ m}^3/\text{hr}$, relative error $0.13\%$).
  - Proven across 6 benchmark datasets: correlation $r(XMV\_11, XMEAS\_17) \le -0.938$ across both normal and fault runs.

---

### Slide 7: Thermodynamic Energy Modeling & Physical Realism
* **Separating Measurements from Proxies (Rule 8):**
  1. **Compressor Electrical Power ($kW$):** Direct sensor measurement ($XMEAS\_20 = 341.47\text{ kW}$).
  2. **Stripper Reboiler Thermal Duty ($kW$):** Derived from steam mass flow ($XMEAS\_19$) via latent heat of vaporization:
     $$P_{\text{steam}} = \frac{XMEAS\_19 \times 2257\text{ kJ/kg}}{3600\text{ s/hr}} = 0.62694 \times XMEAS\_19 = 144.37\text{ kW}$$
  3. **Total Equivalent Process Power:** $P_{\text{total}} = 341.47 + 144.37 = \mathbf{485.84\text{ kW}}$.
* **Documented Plant Limitation:** Cooling water duties are excluded from power calculations because benchmark datasets record only valve openings and exit temperatures, omitting mass flow rates.

---

### Slide 8: Downs & Vogel Operating Cost Formulation
* **Economic Objective Function:**
  $$\text{Cost } (\$/\text{hr}) = \text{Purge Gas Chemical Loss} + \text{Compressor Electricity} + \text{Reboiler Steam}$$
* **Parameters & Unit Conversion:**
  - Electricity: $\$0.0536/\text{kWh}$, Steam: $\$0.00318/\text{kg}$.
  - Purge Gas Molar Flow: $1\text{ kscmh} = 44.61937\text{ kmol/hr}$ (derived from Fortran simulation driver `teprob.f`).
* **Linear Programming Proxy Validation:**
  - Detailed composition-weighted cost: **$\$106.39/\text{hr}$**
  - Average purge price proxy ($\$5.80/\text{kmol}$): **$\$106.40/\text{hr}$** (error $< 0.01\%$).

---

### Slide 9: Linear Programming Architecture (Dual Simplex)
* **Solver Engine:** SciPy HiGHS Dual Simplex (`highs-ds`).
* **Trust Region & Bound Management:**
  - Trust half-width: $u_{\text{base}} - 10\% \le u \le u_{\text{base}} + 10\%$.
  - Valve boundaries: $0\% \le u \le 100\%$.
  - Empirical response bounding: $y_{\min} \le \hat{y}_j(u) \le y_{\max}$ prevents unphysical extrapolation.
* **Mahalanobis Distance Extrapolation Detector:** $\chi^2$ confidence ellipsoids verify that candidate solutions remain within the validated statistical operating envelope.

---

### Slide 10: Quantitative Optimization Results
* **Scenario 1: Production Maximization**
  - Product Throughput: $22.909 \to \mathbf{24.681\text{ m}^3/\text{hr}}$ (**$+7.73\%$** increase)
  - Reactor Pressure strictly safe at $2702.7\text{ kPa}$ (below $2800\text{ kPa}$ limit).
  - Compressor Power at $337.77\text{ kW}$ (below $355\text{ kW}$ rating).
* **Scenario 2: Operating Cost Minimization**
  - Operating Cost: $\$106.40 \to \mathbf{\$97.73/\text{hr}}$ (**$-8.15\%$** reduction)
  - Production quota maintained at $22.909\text{ m}^3/\text{hr}$.
* **Scenario 3: Total Energy Minimization**
  - Process Power: $485.84 \to \mathbf{469.03\text{ kW}}$ (**$-3.46\%$** reduction).

---

### Slide 11: Multi-Objective Goal Programming
* **Overcoming Single-Objective Limitations:** Trade-offs between competing corporate goals (maximize throughput vs. minimize compressor energy vs. maintain target pressure).
* **Formulation A: Weighted-Sum GP:**
  $$\min \sum_k \frac{w_k}{|g_k|} (d_k^- + d_k^+)$$
* **Formulation B: Preemptive (Lexicographic) GP:**
  - Priority 1: Satisfy production quota ($24.0\text{ m}^3/\text{hr}$).
  - Priority 2: Minimize compressor electrical draw ($338.0\text{ kW}$).
  - Priority 3: Stabilize reactor pressure ($2700\text{ kPa}$).

---

### Slide 12: Sensitivity Analysis: Duality vs Local What-If
* **Dual Simplex Shadow Prices:**
  - Measures marginal economic worth of relaxing plant resource constraints ($\frac{\partial z^*}{\partial b_i}$).
  - In Cost Minimization, production quota has a positive dual marginal worth.
* **Local What-If Finite-Difference Gradients ($\frac{\Delta y}{\Delta u}$):**
  - Product Flow is strongly governed by Condenser CW ($XMV\_11$, slope $-0.413$).
  - Reactor Pressure is dominated by Compressor Recycle ($XMV\_05$, slope $+8.419$).
  - Purge Flow is governed by Purge Valve ($XMV\_06$, slope $+0.0075$).
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
  2. Fixed $\$5.80/\text{kmol}$ purge price proxy assumes steady-state composition; transient composition swings require nonlinear dynamic simulation.
  3. Meeting static pressure and power limits does not guarantee dynamic asymptotic stability under large fault disturbances.

---

### Slide 15: Conclusion & Future Work
* **Project Deliverables Completed:**
  - Robust data ingestion and steady-state validation (44 benchmark datasets).
  - 8-variable surrogate model achieving $R^2 = 0.9951$ for product flow.
  - Dual Simplex LP achieving $+7.73\%$ production, $-8.15\%$ cost, and $-3.46\%$ energy.
  - Multi-objective Goal Programming and sensitivity analysis.
  - 70 passing automated pytest unit tests.
  - Interactive full-stack Streamlit web application.
* **Future Work:** Integration with nonlinear MPC (Model Predictive Control) and real-time closed-loop testing in the original Fortran simulation driver.
