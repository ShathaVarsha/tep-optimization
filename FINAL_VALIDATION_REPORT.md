# Final Validation and Verification Report: Tennessee Eastman Process Optimization

**Course:** 23MNG336: Operational Research  
**Project Title:** Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process (TEP)  
**Academic Level:** Final-Year B.Tech in Artificial Intelligence & Data Science  
**Author:** Team B4  
**Date of Completion:** October 4, 2026  
**Status:** **Fully Validated, Tested, and Submission-Ready**

---

## 1. Executive Summary

This report presents the complete mathematical, engineering, and software validation of the data-driven optimization framework developed for the Tennessee Eastman industrial chemical manufacturing process. Utilizing the standard benchmark dataset generated from the Eastman Chemical Company simulation (Downs & Vogel, 1993), this project combines:

1. **Benchmark Ingestion and Integrity Auditing:** Automated detection of orientation layout (transposition from $52 \times 500$ to $500 \times 52$) across all 44 benchmark `.dat` files with 100% data integrity ($0$ missing values, $0$ infinities).
2. **Stationarity and Baseline Validation:** Multi-criteria Augmented Dickey-Fuller (ADF) unit-root testing ($p < 0.05$ across all state variables) and rolling drift verification confirming true statistical steady-state operation of nominal baseline dataset `d00.dat`.
3. **Surrogate Model Formulation and Breakthrough:** Discovery and empirical resolution of the product flow prediction bottleneck. While standard 7-variable models exhibited near-zero predictive power for finished product delivery rate ($XMEAS\_17$, TimeSeriesSplit $R^2 = -0.063$), augmenting the decision space with the condenser cooling water valve ($XMV\_11$) elevated cross-validated $R^2$ to **$0.9951$** ($MAE = 0.0308\text{ m}^3/\text{hr}$).
4. **Thermodynamic Energy and Operating Cost Modeling:** Explicit segregation of direct sensor electrical work ($XMEAS\_20$, $341.47\text{ kW}$) from stripper reboiler thermal duty ($XMEAS\_19$, $144.37\text{ kW}$ equivalent), establishing a rigorous baseline power footprint of $485.84\text{ kW}$. Rigorous implementation of the Downs & Vogel (1993) economic cost model evaluates the baseline operating cost at **$\$106.39/\text{hr}$** (detailed composition-weighted) versus **$\$106.40/\text{hr}$** (linear LP proxy).
5. **Linear Programming via Dual Simplex:** Formulated and solved using SciPy HiGHS Dual Simplex (`highs-ds`) with trust regions and empirical response bounding:
   - **Production Maximization:** Predicted finished product delivery rate increases from **$22.9093\text{ m}^3/\text{hr}$** to **$24.6813\text{ m}^3/\text{hr}$** (**$+7.73\%$** increase) while strictly respecting reactor pressure ($\le 2800\text{ kPa}$) and compressor power limits ($\le 355\text{ kW}$).
   - **Operating Cost Minimization:** Approximate operating cost decreases from **$\$106.40/\text{hr}$** to **$\$97.73/\text{hr}$** (**$-8.15\%$** reduction) while satisfying the baseline production quota of $22.89\text{ m}^3/\text{hr}$.
   - **Total Energy Minimization:** Equivalent process power decreases from **$485.84\text{ kW}$** to **$469.03\text{ kW}$** (**$-3.46\%$** reduction).
   - **Compressor Electrical Minimization:** Direct motor electrical draw decreases from **$341.47\text{ kW}$** to **$337.77\text{ kW}$** (**$-1.08\%$**).
6. **Multi-Objective Goal Programming:** Formulated and solved via both Weighted-Sum and Preemptive (Lexicographic) GP, resolving competing trade-offs between throughput, pressure stability, and compressor work.
7. **Sensitivity Analysis:** Clear distinction between formal Simplex shadow prices (dual variables) and local finite-difference what-if response sensitivities.
8. **Interactive User Interface:** Built and verified a full-stack interactive Streamlit dashboard (`streamlit_app.py`) allowing interactive scenario configuration, parameter tuning, live solver runs, and Plotly visualization.
9. **Automated Verification:** 70 automated pytest test cases pass cleanly with zero failures.

---

## 2. Dataset Auditing and Preprocessing Validation

### 2.1 Benchmark Directory Structure
The workspace contains the authoritative benchmark files distributed with the original simulation:
- 44 benchmark `.dat` files located in `The Tennessee Eastman process (TEP)/The Tennessee Eastman process (TEP)/TEdata/TEdata`.
- Files include training sets (`d00.dat` to `d21.dat`) and testing sets (`d00_te.dat` to `d21_te.dat`).

### 2.2 Transposition Detection and Orientation
The ingestion module `src/data/tep_loader.py` dynamically checks array dimensions:
- `d00.dat` is stored as an array of shape $(52, 500)$.
- The loader automatically detects that the first dimension equals the 52 process variables, transposing the array to $(500, 52)$ (500 chronological time observations $\times$ 52 process variables).
- Testing datasets (`d00_te.dat` through `d21_te.dat`) have native dimensions of $(960, 52)$, requiring no transposition.

### 2.3 Numerical Integrity Audit
Automated audit across all 44 benchmark files (`test_data_loader.py`) established:
- **Missing Values:** $0\text{ NaN}$ values across all 44 datasets.
- **Infinite Values:** $0\text{ Inf}$ values across all 44 datasets.
- **Variable Mapping:** Exactly 52 standardized column tags assigned: 41 continuous measured variables (`XMEAS_01` to `XMEAS_41`) and 11 manipulated control valve variables (`XMV_01` to `XMV_11`).

---

## 3. Baseline Stationarity & Operating Conditions

### 3.1 Steady-State Verification (`d00.dat`)
In chemical process optimization, surrogate models and linear programs must be anchored around a validated stationary baseline. `src/process/baseline_analyzer.py` verifies this via:
1. **Augmented Dickey-Fuller (ADF) Unit-Root Test:** Null hypothesis of a unit root is rejected ($p < 0.05$) for all key state variables.
2. **Rolling Mean Drift:** Evaluates drift over rolling windows of 50 samples. Normalized drift across the entire 500-sample run is $< 5.0\%$ for all critical variables.

| Variable Tag | Variable Description | Nominal Baseline Mean | ADF Statistic | p-value | Critical Value (5%) | Rolling Drift (%) | Steady-State Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `XMEAS_07` | Reactor Pressure (kPa gauge) | $2705.40$ | $-4.612$ | $0.00012$ | $-2.867$ | $0.21\%$ | **STATIONARY** |
| `XMEAS_08` | Reactor Level (%) | $65.03$ | $-5.124$ | $0.00001$ | $-2.867$ | $0.45\%$ | **STATIONARY** |
| `XMEAS_09` | Reactor Temperature (°C) | $120.41$ | $-4.891$ | $0.00004$ | $-2.867$ | $0.18\%$ | **STATIONARY** |
| `XMEAS_10` | Purge Gas Flow Rate (kscmh) | $0.3376$ | $-3.987$ | $0.00148$ | $-2.867$ | $1.12\%$ | **STATIONARY** |
| `XMEAS_17` | Product Delivery Rate (m³/hr) | $22.9093$ | $-4.231$ | $0.00057$ | $-2.867$ | $0.68\%$ | **STATIONARY** |
| `XMEAS_19` | Stripper Steam Flow (kg/hr) | $230.28$ | $-4.550$ | $0.00016$ | $-2.867$ | $0.94\%$ | **STATIONARY** |
| `XMEAS_20` | Compressor Work (kW) | $341.47$ | $-4.318$ | $0.00041$ | $-2.867$ | $0.32\%$ | **STATIONARY** |

---

## 4. Surrogate Modeling & The $XMV\_11$ Discovery

### 4.1 Feature Selection and Model Architecture
Surrogate models approximate the complex, stiff nonlinear differential-algebraic equations (DAE) of the TEP simulation around the nominal operating baseline.
- **Model Formulations Evaluated:** Ordinary Least Squares (OLS), Ridge Regression ($L_2$ regularized, $\alpha = 10.0$), and Polynomial Ridge ($\text{degree} = 2, \alpha = 50.0$).
- **Validation Scheme:** TimeSeriesSplit chronological cross-validation ($k = 5$ splits) to prevent temporal data leakage.

### 4.2 The Product Flow Prediction Bottleneck & Empirical Breakthrough
In the initial project implementation, 7 manipulated variables were used:
$$u_7 = [XMV\_01, XMV\_02, XMV\_03, XMV\_04, XMV\_05, XMV\_06, XMV\_09]$$

While $u_7$ adequately predicted purge flow ($R^2 = 0.878$) and steam flow ($R^2 = 0.774$), it failed completely on Product Flow ($XMEAS\_17$):
- **7-Variable Model on $XMEAS\_17$:** Cross-validated $R^2 = \mathbf{-0.0627}$, $MAE = 0.4976\text{ m}^3/\text{hr}$. A negative $R^2$ demonstrates that the model performs worse than simply predicting the mean.

**Root Cause Analysis:** In the Tennessee Eastman process control architecture (Downs & Vogel 1993, Section 3), the reactor condenser duty controls the condensation rate of unreacted recycle gas and product vapors. The condenser cooling water valve is **$XMV\_11$**. When cooling water flow increases, condensation increases, which directly regulates the liquid level entering the stripper and product stream.

**The 8-Variable Formulation ($+ XMV\_11$):**
$$u_8 = [XMV\_01, XMV\_02, XMV\_03, XMV\_04, XMV\_05, XMV\_06, XMV\_09, \mathbf{XMV\_11}]$$

Adding $XMV\_11$ produced a dramatic empirical breakthrough:
- Cross-validated $R^2$ for $XMEAS\_17$ jumped from **$-0.0627$ to $\mathbf{0.9951}$**!
- $MAE$ dropped from $0.4976\text{ m}^3/\text{hr}$ to **$0.0308\text{ m}^3/\text{hr}$** (relative error $0.13\%$).

### 4.3 Generalization Across Benchmark Datasets
To verify that this strong correlation is not an artifact of overfitting on `d00.dat`, cross-dataset correlation between $XMV\_11$ and $XMEAS\_17$ was evaluated across six independent operating datasets:

| Dataset ID | Operating Condition | Observations | Correlation $r(XMV\_11, XMEAS\_17)$ |
| :---: | :--- | :---: | :---: |
| `d00` | Normal Operation (Training) | 500 | **$-0.9996$** |
| `d01` | Fault 1 (A/C feed ratio step) | 480 | **$-0.9746$** |
| `d02` | Fault 2 (B composition step) | 480 | **$-0.9510$** |
| `d03` | Fault 3 (D feed temp step) | 480 | **$-0.9996$** |
| `d04` | Fault 4 (Reactor cooling water step) | 480 | **$-0.9994$** |
| `d05` | Fault 5 (Condenser cooling water step)| 480 | **$-0.9380$** |

The relationship is robust, physically grounded, and consistent across normal and fault modes.

### 4.4 Final 8-Variable Ridge Model Metrics

| Response Variable Tag | Response Name | Baseline Mean | Train $R^2$ | CV $R^2$ ($k=5$) | CV MAE | CV RMSE | CV Relative Error (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `XMEAS_17` | Product Delivery Rate (m³/hr) | $22.9093$ | $0.9989$ | **$0.9951$** | $0.0308$ | $0.0390$ | $0.13\%$ |
| `XMEAS_07` | Reactor Pressure (kPa gauge) | $2705.40$ | $0.6623$ | **$0.3005$** | $2.9304$ | $3.6014$ | $0.11\%$ |
| `XMEAS_10` | Purge Flow Rate (kscmh) | $0.3376$ | $0.8979$ | **$0.8775$** | $0.0032$ | $0.0040$ | $0.95\%$ |
| `XMEAS_19` | Stripper Steam Flow (kg/hr) | $230.28$ | $0.9591$ | **$0.7733$** | $1.8042$ | $2.2163$ | $0.79\%$ |
| `XMEAS_20` | Compressor Power (kW) | $341.47$ | $0.7747$ | **$0.3602$** | $0.6185$ | $0.7360$ | $0.18\%$ |

---

## 5. Thermodynamic Energy & Operating Cost Models

### 5.1 Energy Model Formulation
In accordance with Rule 8 and Phase 4 requirements, physical measurements are strictly separated from thermodynamic proxies:

1. **Compressor Electrical Power ($kW$):** Directly measured by sensor $XMEAS\_20$. This represents true electrical power consumed by the recycle gas centrifugal compressor drive motor.
   $$P_{\text{comp}} = XMEAS\_20 \quad [\text{kW}]$$
2. **Stripper Reboiler Thermal Power ($kW$):** Derived from steam mass flow ($XMEAS\_19$, kg/hr) using the latent heat of saturated low-pressure steam ($\lambda = 2257\text{ kJ/kg}$):
   $$P_{\text{steam}} = \frac{XMEAS\_19 \times 2257\text{ kJ/kg}}{3600\text{ s/hr}} = 0.62694 \times XMEAS\_19 \quad [\text{kW}]$$
3. **Total Equivalent Process Power Footprint ($kW$):**
   $$P_{\text{total}} = P_{\text{comp}} + P_{\text{steam}} = XMEAS\_20 + 0.62694 \times XMEAS\_19 \quad [\text{kW}]$$

**Baseline Energy Values on `d00`:**
- Compressor Electrical Power: $341.47\text{ kW}$ ($70.28\%$ of total)
- Stripper Reboiler Thermal Duty: $144.37\text{ kW}$ ($29.72\%$ of total)
- Total Primary Process Power: **$485.84\text{ kW}$**

**Documented Physical Limitations:**
Sensible heat removed by reactor cooling water ($XMV\_10$) and condenser cooling water ($XMV\_11$) is not converted into power because the benchmark dataset records only valve opening percentage (%) and exit temperature ($XMEAS\_21, XMEAS\_22$), without recording the cooling water mass flow rate. Converting cooling water to energy without mass flow data would require unverified assumptions regarding supply water header pressure. Thus, $P_{\text{total}}$ represents the true electrical and steam utility power footprint supported by the dataset.

### 5.2 Operating Cost Model Formulation
Downs & Vogel (1993, Section 4, Table 4) define the operating cost function as:
$$\text{Cost } (\$/\text{hr}) = \text{Purge Gas Loss Cost} + \text{Compressor Electricity Cost} + \text{Steam Cost}$$

Economic parameters:
- Electricity Price: $\$0.0536/\text{kWh}$
- Steam Price: $\$0.00318/\text{kg}$
- Purge Component Molar Prices: $A=\$2.206$, $C=\$6.016$, $D=\$8.222$, $E=\$10.428$, $F=\$14.444$, $G=\$17.653$, $H=\$17.653$ (\$/kmol).
- Molar Flow Conversion Factor: Derived from the Fortran simulation source code `teprob.f` (lines 679–688):
  $$1\text{ kscmh} = \frac{35.3145}{0.359} \times 0.45359237 = 44.61937\text{ kmol/hr}$$

**Baseline Cost Comparison:**
- **Detailed Composition-Weighted Model:** Evaluates instantaneous purge compositions across Stream 9 ($XMEAS\_29$ through $XMEAS\_36$).
  $$\text{Detailed Baseline Cost} = \$87.35\text{ (purge)} + \$18.30\text{ (comp)} + \$0.73\text{ (steam)} = \mathbf{\$106.39/\text{hr}}$$
- **Linear LP Proxy Model:** Uses the empirical weighted average purge price of $\$5.80/\text{kmol}$:
  $$\text{Linear Proxy Baseline Cost} = (0.3376 \times 44.61937 \times 5.80) + (341.47 \times 0.0536) + (230.28 \times 0.00318) = \mathbf{\$106.40/\text{hr}}$$
The linear proxy matches the rigorous composition-weighted benchmark within $\$0.01/\text{hr}$ ($0.01\%$), enabling exact Linear Programming formulation without nonlinear bilinear composition terms.

---

## 6. Linear Programming Optimization & Results

### 6.1 Mathematical Formulations (Dual Simplex)

All linear programs are solved using SciPy HiGHS Dual Simplex (`highs-ds`) with trust region constraints $u_{\text{base}} - \delta \le u \le u_{\text{base}} + \delta$ ($\delta = 10.0$ percentage points) and empirical response bounding $y_{\min} \le \hat{y}(u) \le y_{\max}$.

#### Formulation 1: Production Maximization
$$\max_{u} \quad \text{Product Flow}(u) = c_{\text{prod}}^T u + d_{\text{prod}}$$
$$\text{s.t.} \quad \text{Reactor Pressure}(u) \le 2800\text{ kPa}$$
$$\text{Compressor Power}(u) \le 355\text{ kW}$$
$$y_{\min} \le \hat{y}_j(u) \le y_{\max} \quad \forall j \in \{17, 7, 10, 19, 20\}$$
$$u_{\text{base}} - 10 \le u \le u_{\text{base}} + 10, \quad 0 \le u \le 100$$

#### Formulation 2: Operating Cost Minimization
$$\min_{u} \quad \text{Operating Cost}(u) = c_{\text{cost}}^T u + d_{\text{cost}}$$
$$\text{s.t.} \quad \text{Product Flow}(u) \ge 22.89\text{ m}^3/\text{hr} \quad (\text{Production Quota})$$
$$\text{Reactor Pressure}(u) \le 2800\text{ kPa}$$
$$\text{Compressor Power}(u) \le 355\text{ kW}$$
$$y_{\min} \le \hat{y}_j(u) \le y_{\max}, \quad u_{\text{base}} - 10 \le u \le u_{\text{base}} + 10$$

#### Formulation 3: Total Energy Minimization
$$\min_{u} \quad P_{\text{total}}(u) = c_{\text{energy}}^T u + d_{\text{energy}}$$
$$\text{s.t.} \quad \text{Product Flow}(u) \ge 22.89\text{ m}^3/\text{hr}, \quad \text{Reactor Pressure}(u) \le 2800\text{ kPa}, \quad \text{Bounds}$$

---

### 6.2 Summary of Optimization Scenarios

| Optimization Scenario | Objective Function | Baseline Value | Optimized Value | Improvement | Solver Iterations | Solution Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Problem 1: Production Max** | Product Flow ($XMEAS\_17$) | $22.9093\text{ m}^3/\text{hr}$ | **$24.6813\text{ m}^3/\text{hr}$** | **$+7.73\%$** | 2 | Optimal (HiGHS) |
| **Problem 2: Cost Min** | Operating Cost ($\$/\text{hr}$) | $\$106.4015/\text{hr}$ | **$\$97.7254/\text{hr}$** | **$-8.15\%$** | 4 | Optimal (HiGHS) |
| **Problem 3: Energy Min (Total)**| Total Power ($kW$) | $485.8436\text{ kW}$ | **$469.0334\text{ kW}$** | **$-3.46\%$** | 4 | Optimal (HiGHS) |
| **Problem 4: Energy Min (Comp)** | Compressor Work ($kW$) | $341.4695\text{ kW}$ | **$337.7700\text{ kW}$** | **$-1.08\%$** | 2 | Optimal (HiGHS) |

---

### 6.3 Optimal Manipulated Variable Setpoints (Valves %)

| Variable Tag | Variable Description | Baseline Setpoint | Production Max Setpoint | Cost Min Setpoint | Energy Min Setpoint | Units |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `XMV_01` | D Feed Flow Valve | $63.031\%$ | $64.436\%$ ($+1.41\%$) | $61.296\%$ ($-1.74\%$) | $61.296\%$ ($-1.74\%$) | % |
| `XMV_02` | E Feed Flow Valve | $54.006\%$ | $55.111\%$ ($+1.11\%$) | $54.990\%$ ($+0.98\%$) | $55.050\%$ ($+1.04\%$) | % |
| `XMV_03` | A Feed Flow Valve | $24.721\%$ | $33.682\%$ ($+8.96\%$) | $17.781\%$ ($-6.94\%$) | $17.781\%$ ($-6.94\%$) | % |
| `XMV_04` | A+C Feed Flow Valve | $61.322\%$ | $57.903\%$ ($-3.42\%$) | $57.903\%$ ($-3.42\%$) | $57.903\%$ ($-3.42\%$) | % |
| `XMV_05` | Compressor Recycle Valve | $22.252\%$ | $20.811\%$ ($-1.44\%$) | $21.147\%$ ($-1.11\%$) | $20.811\%$ ($-1.44\%$) | % |
| `XMV_06` | Purge Valve | $40.082\%$ | $36.385\%$ ($-3.70\%$) | $36.128\%$ ($-3.95\%$) | $36.385\%$ ($-3.70\%$) | % |
| `XMV_09` | Stripper Steam Valve | $47.461\%$ | $42.162\%$ ($-5.30\%$) | $42.162\%$ ($-5.30\%$) | $42.162\%$ ($-5.30\%$) | % |
| `XMV_11` | Condenser Cooling Water Valve | $18.216\%$ | $14.009\%$ ($-4.21\%$) | $18.259\%$ ($+0.04\%$) | $18.259\%$ ($+0.04\%$) | % |

---

### 6.4 Predicted Process State Responses under Optimal Policies

| Process State Response | Baseline | Production Max | Cost Min | Energy Min | Safety Limit / Alarm | Units |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `XMEAS_17` (Product Delivery) | $22.909$ | **$24.681$** | $22.909$ | $22.909$ | Lower Quota: $22.89$ | m³/hr |
| `XMEAS_07` (Reactor Pressure) | $2705.4$ | $2702.7$ | $2698.7$ | $2698.7$ | Max Safety: $2800$ | kPa gauge |
| `XMEAS_10` (Purge Gas Flow) | $0.3376$ | $0.3051$ | $0.3051$ | $0.3051$ | Upper Bound: $0.40$ | kscmh |
| `XMEAS_19` (Stripper Steam Flow)| $230.28$ | $210.55$ | $209.37$ | $209.37$ | Max Capacity: $1500$| kg/hr |
| `XMEAS_20` (Compressor Power) | $341.47$ | $337.77$ | $337.77$ | $337.77$ | Max Rating: $355$ | kW |
| **Operating Cost** | $\$106.40$ | $\$98.67$ | **$\$97.73$** | $\$97.73$ | — | \$/hr |
| **Total Process Power** | $485.84$ | $469.77$ | $469.03$ | **$469.03$** | — | kW |

---

## 7. Sensitivity Analysis: Duality vs Local What-If

### 7.1 Linear Programming Duality (Shadow Prices)
In Linear Programming, shadow prices represent the marginal change in optimal objective value per unit relaxation of a binding constraint ($\frac{\partial z^*}{\partial b_i}$).
- In **Production Maximization**, the binding constraints are the empirical upper limit on product flow and the empirical lower limit on compressor work.
- In **Cost Minimization**, the production quota constraint ($XMEAS\_17 \ge 22.89\text{ m}^3/\text{hr}$) is strictly binding with zero surplus slack.

### 7.2 Local What-If Finite-Difference Sensitivity
To evaluate local operational sensitivity around the nominal baseline, 1% perturbations ($\Delta u_i = \pm 0.01 \times \text{span}(u_i)$) were evaluated against the surrogate model with boundary clamping.

| Response Variable | Most Sensitive Manipulated Input | Perturbation Direction | Response Change ($\Delta y$) | Local Sensitivity Slope ($\Delta y / \Delta u$) | Engineering Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `XMEAS_07` (Pressure) | `XMV_05` (Compressor Recycle) | Increase | $+0.2145\text{ kPa}$ | $+8.4188$ | Opening recycle valve raises suction pressure and reactor backpressure. |
| `XMEAS_10` (Purge Rate) | `XMV_06` (Purge Valve) | Decrease | $-0.0007\text{ kscmh}$| $+0.0075$ | Purge valve directly throttles stream 9 vent flow. |
| `XMEAS_17` (Product Flow) | `XMV_11` (Condenser CW Valve)| Increase | $-0.0377\text{ m}^3/\text{hr}$ | $-0.4128$ | Condenser cooling rate directly influences product condensation equilibrium. |
| `XMEAS_19` (Steam Flow) | `XMV_09` (Steam Valve) | Decrease | $-0.3674\text{ kg/hr}$| $+3.6876$ | Steam valve directly throttles stripper reboiler heating utility. |
| `XMEAS_20` (Compressor Work)| `XMV_05` (Compressor Recycle) | Increase | $+0.0631\text{ kW}$ | $+2.4757$ | Higher recycle flow increases volumetric flow through the centrifugal impeller. |

**Critical Methodological Distinction:**
These finite-difference sensitivities are localized numerical approximations of surrogate partial derivatives. They must **not** be confused with:
1. True physical plant causal dynamics (which include closed-loop PID controller feedback reactions).
2. Global LP shadow prices (which evaluate changes to the active basic feasible solution basis).

---

## 8. User Interface & Software Architecture

### 8.1 Streamlit Interactive Dashboard (`streamlit_app.py`)
A modern, dark-mode full-stack application was built using Streamlit and Plotly, providing 7 functional views:
1. **Dataset & Process Catalog:** 44-file integrity audit, 52-variable searchable dictionary, descriptive statistics, and correlation heatmaps.
2. **Baseline & Stationarity:** Augmented Dickey-Fuller stationarity tests, rolling drift metrics, and Downs & Vogel baseline cost/energy breakdowns.
3. **Surrogate Modeling:** Interactive selector for Ridge, Linear, and Polynomial models; 7-var vs 8-var mode toggle; TimeSeriesSplit cross-validation tables ($R^2$, MAE, RMSE); live response curve overlays.
4. **Linear Programming Engine:** Multi-objective solver panel (Production Max, Cost Min, Energy Min) with sliders for trust region delta ($\pm \delta\%$), pressure limits, and compressor power ratings; live HiGHS Dual Simplex solver execution; side-by-side comparison tables, bar charts, and CSV exports.
5. **Goal Programming Studio:** Interactive weighted-sum and preemptive lexicographic solver with user-defined target values and priority rankings.
6. **Sensitivity Explorer:** Interactive what-if perturbation calculator with adjustable step sizes and dominant sensitivity bar charts.
7. **Reports & Audit:** In-app markdown viewer for project audit and validation documentation.

---

## 9. Automated Testing & Verification Suite

A 70-test automated test suite was constructed using `pytest`, covering all modules and integrations:
- `tests/test_data_loader.py` (7 tests): Array shapes, transposition, zero NaN/Inf across all 44 datasets.
- `tests/test_variable_dictionary.py` (6 tests): Catalog completeness, tag indices, engineering bounds.
- `tests/test_cost_energy.py` (4 tests): Pricing calculations, thermal duty conversions, baseline d00 costs.
- `tests/test_surrogate.py` (6 tests): ADF stationarity, Ridge/OLS/Poly fitting, Mahalanobis extrapolation detection, 8-variable model performance ($R^2 > 0.95$).
- `tests/test_lp.py` (7 tests): Production maximization, cost minimization, energy minimization (total and compressor modes), 8-variable response-bounded optimization, LP sensitivity reports, local what-if sensitivity.
- `tests/test_goal_programming.py` (35 tests): Weighted-sum and preemptive lexicographic goal formulations, deviation variables, priority sequencing, safety constraints.
- `tests/test_ui_workflow.py` (5 tests): End-to-end data pipeline, baseline calculations, multi-objective LP execution, GP execution, sensitivity computation.

**Test Run Results (October 4, 2026):**
```text
============================= 70 passed in 4.18s ==============================
```

---

## 10. Engineering Limitations and Academic Conclusion

### 10.1 Limitations & Disclaimers
1. **Surrogate Approximation:** The optimization solutions are derived from linear and regularized Ridge surrogates fitted to nominal steady-state data (`d00.dat`). While cross-validation confirms high accuracy ($R^2 = 0.995$ on product flow), these are mathematical surrogate predictions rather than physical closed-loop plant trials.
2. **Approximate Purge Cost:** The optimizer uses a weighted-average purge price ($\$5.80/\text{kmol}$), which accurately models baseline conditions ($\$106.40/\text{hr}$ vs $\$106.39/\text{hr}$) but neglects nonlinear composition fluctuations during large transients.
3. **Safety and Dynamic Stability:** Respecting reactor pressure ($\le 2800\text{ kPa}$) and compressor power ($\le 355\text{ kW}$) constraints does not inherently guarantee dynamic plant controllability or safety under major process disturbances. Dynamic closed-loop validation in the Fortran simulation driver is recommended for plant deployment.

### 10.2 Academic Conclusion
This project successfully applies Operations Research methodologies—specifically regularized surrogate modeling, Dual Simplex linear programming, Goal Programming, and duality/sensitivity analysis—to the Tennessee Eastman benchmark process. By discovering the critical role of condenser cooling water valve $XMV\_11$, the framework provides a statistically validated, academically defensible optimization model achieving:
- **$+7.73\%$** predicted product throughput gain,
- **$-8.15\%$** predicted operating cost reduction, and
- **$-3.46\%$** predicted primary energy footprint reduction,
fully integrated into a tested, interactive full-stack application.
