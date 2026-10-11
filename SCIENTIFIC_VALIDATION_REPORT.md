# Scientific and Mathematical Validation Report

## Tennessee Eastman Process Optimization

---

## 1. Executive Summary & Purpose

This report provides the mathematical verification and engineering assessment of the Operations Research optimization framework developed for the Tennessee Eastman Process (TEP).

The project implements:
1. **Empirical Process Surrogates:** Regularized Ridge and Linear models mapping manipulated valve setpoints (decision vector u) to critical plant state and economic responses (response vector y).
2. **Linear Programming Formulations:** Solved with the Dual Simplex algorithm (`highs-ds` in SciPy) for Production Maximization, Operating Cost Minimization, and Energy Footprint Minimization.
3. **Multi-Objective Goal Programming:** Weighted-Sum and Preemptive (Lexicographic) models handling trade-offs between product throughput, compressor power, purge gas loss, and reactor pressure safety.
4. **Independent Numerical Verification:** Automated end-to-end testing (`validate_results.py`, 50 checks) and unit test suite (`pytest`, 94 tests).

### Software Verification vs. Closed-Loop Plant Validation
A key distinction in process systems engineering is between algorithmic software verification and physical plant validation:
- **Software Verification:** Confirms that the code correctly implements the mathematical formulations, simplex pivots, shadow prices, and priority freeze constraints without numerical errors.
- **Physical Plant Validation:** The surrogate models are fitted on steady-state baseline simulation data (`d00.dat` from Downs & Vogel, 1993). In a physical plant, shifting setpoints by ±10% engages secondary feedback loops and dynamic controller actions. As a result, the optimization candidates represent optimal setpoints within the empirical surrogate trust boundaries, rather than guaranteed dynamic equilibria.

---

## 2. Surrogate Model Validation & Investigation of XMV_11

### 2.1 The 7-Variable vs. 8-Variable Decision Space
Initial formulations used a 7-variable decision vector:
`u₇ = [XMV_01, XMV_02, XMV_03, XMV_04, XMV_05, XMV_06, XMV_09]`
representing feed flows A, D, E, A+C, compressor recycle, purge valve, and stripper steam.

On undisturbed baseline data (`d00.dat`), this 7-variable model had no predictive power for finished product flow (`XMEAS_17`):
- Cross-Validation R² = -0.0592
- Chronological Holdout R² = -0.0278

### 2.2 Role of Condenser Cooling Water (XMV_11)
In the TEP flowsheet (Downs & Vogel, 1993), reactor effluent vapors pass through the condenser before entering the vapor-liquid separator and stripper column. Valve `XMV_11` regulates condenser cooling water:
1. **Condensation Mechanism:** Higher cooling water flow reduces condenser exit temperature, condensing volatile product components (G and H) from the recycle gas into liquid.
2. **Flow Coupling:** Condensed liquid flows directly to the stripper column base as stripper underflow, measured by flow sensor `XMEAS_17`.
3. **Statistical Correlation:** In `d00.dat`, Pearson correlation between `XMV_11` and `XMEAS_17` is r = -0.9996.
4. **8-Variable Model Performance:** Expanding the decision space to `u₈ = [u₇, XMV_11]` provides strong predictive accuracy across cross-validation and chronological holdout splits:

| Response Variable | Tag | Units | 5-Fold CV R² | 80/20 Holdout R² | Train R² | Mean Relative Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Product Flow | XMEAS_17 | m³/hr | 0.9951 | 0.9987 | 0.9975 | 0.06% |
| Purge Rate | XMEAS_10 | kscmh | 0.8775 | 0.8812 | 0.8841 | 0.78% |
| Stripper Steam | XMEAS_19 | kg/hr | 0.7733 | 0.7804 | 0.7791 | 0.83% |
| Reactor Pressure | XMEAS_07 | kPa gauge | 0.3005 | 0.3120 | 0.3214 | 0.05% |
| Compressor Power | XMEAS_20 | kW | 0.3602 | 0.3715 | 0.3750 | 0.28% |

### 2.3 Chronological Holdout Evaluation
To prevent data leakage in time-series evaluation:
- **TimeSeriesSplit (k = 5):** Evaluates each fold strictly on subsequent chronological data.
- **80/20 Chronological Holdout:** The first 400 samples (t = 0 to 399) form the training set, and the final 100 samples (t = 400 to 499) form the holdout test set. Product flow holdout test R² is 0.9987, confirming model stability on unseen future data.

---

## 3. Extrapolation Detection & Domain Boundaries

### 3.1 Extrapolation Detector Formulation
To detect when candidate setpoints move outside the empirical training distribution, the model uses a dual statistical and hyper-box check:
1. **Regularized Mahalanobis Distance:**  
   `D_M²(u) = (u − ū)ᵀ · (Σ + λI)⁻¹ · (u − ū)`  
   where `ū` is the baseline mean, `Σ` is the sample covariance matrix, and `λ = 10⁻⁴` provides numerical stability.
2. **Chi-Squared Cutoff:**  
   `D_M²(u) ≤ χ²(p, 1 − α)`  
   For p = 8 variables at 99% confidence (α = 0.01), χ²(8, 0.99) = 20.09.
3. **Hyper-Box Bounds:**  
   `u_min − 0.20·Δu ≤ u ≤ u_max + 0.20·Δu`

### 3.2 Domain Results
- **Nominal Baseline Point (u_base):**  
  D_M²(u_base) = 0.00 ≤ 20.09 (inside empirical domain).
- **LP Optimal Points (u_opt at ±10% trust bounds):**  
  D_M²(u_opt) ≈ 61.0 > 20.09 (flagged as statistical extrapolation).
- **Physical Interpretation:**  
  The unperturbed baseline data (`d00.dat`) represents steady-state operation with low variance. Moving manipulated valves across a ±10% trust region pushes variables outside the historical steady-state covariance ellipsoid. Enforcing empirical response bounds (historical minimum and maximum values on state variables) prevents unphysical predictions while permitting local exploration.

---

## 4. Goal Programming Semantics and Structure

### 4.1 Direction Penalty Logic
Goal Programming converts multi-objective criteria into deviation variables:  
`f_k(u) + d_k⁻ − d_k⁺ = g_k,  with d_k⁻, d_k⁺ ≥ 0`

Penalties in the objective vector c follow the target direction:
1. **Maximization (`direction='max'`):** Target is f_k(u) ≥ g_k. Under-achievement incurs d_k⁻ > 0. Penalty applies to d_k⁻ only; d_k⁺ is free.
2. **Minimization (`direction='min'`):** Target is f_k(u) ≤ g_k. Over-achievement incurs d_k⁺ > 0. Penalty applies to d_k⁺ only; d_k⁻ is free.
3. **Set-point Tracking (`direction='exact'`):** Target is f_k(u) = g_k. Any deviation is penalized (both d_k⁻ and d_k⁺).

Deviations are normalized by `|g_k|` to maintain dimensionless comparability across different units (m³/hr, kW, kPa).

### 4.2 Preemptive Priority Preservation
In Preemptive Goal Programming, goals are grouped into ordered priorities P₁ > P₂ > ... > P_L.
- At priority level ℓ, the solver minimizes deviation: `z_ℓ = sum(w_k · d_k / |g_k|)`.
- Once optimal deviation z_ℓ* is found, a freeze constraint is added for subsequent levels:  
  `sum(w_k · d_k / |g_k|) ≤ z_ℓ* + ε  (ε = 10⁻⁶)`
- **Numerical Verification:** Solving subsequent levels P₂ and P₃ preserves level 1 achievement: `z₁ = 0.000000 ≤ z₁* + 10⁻⁴`, satisfying lexicographic dominance.

### 4.3 Canonical Goal Sets

#### Scenario A: Operational Stability and Safety (`build_operational_safety_goals()`)
- P1: Product Throughput (XMEAS_17) ≥ 24.0 m³/hr (direction: max, weight: 2.0)
- P2: Compressor Electrical Power (XMEAS_20) ≤ 338.0 kW (direction: min, weight: 1.0)
- P3: Reactor Pressure (XMEAS_07) = 2705.0 kPa (direction: exact, weight: 0.5)

*Correction Note:* In earlier drafts, Goal 2 was mislabeled as "Cost Reduction" while targeting compressor electrical work (`XMEAS_20` in kW). It is now correctly labeled "Compressor Power Reduction".

#### Scenario B: Economic Conservation (`build_default_tep_goals()`)
Derived from Downs & Vogel (1993) Mode 1 values:
- P1: Product Quota (XMEAS_17) ≥ 22.89 m³/hr (direction: max, weight: 2.0)
- P2: Compressor Electrical Power (XMEAS_20) ≤ 324.3 kW (5% reduction, direction: min, weight: 1.5)
- P2: Purge Vent Rate (XMEAS_10) ≤ 0.303 kscmh (10% reduction, direction: min, weight: 1.0)

Both scenarios solve to feasibility with zero deviation within the allowed trust region.

---

## 5. Cost and Energy Balance Verification

### 5.1 Operating Cost Balance
From Downs & Vogel (1993, Section 4):  
`Cost_total ($/hr) = Cost_purge + Cost_electricity + Cost_steam`

- **Purge Loss Cost:** Evaluated from component mole fractions y_A through y_H (`XMEAS_29`–`XMEAS_36`) and component prices ($2.206 to $17.653/kmol) with molar conversion factor 44.61937 kmol/hr per kscmh. Baseline = $87.3508/hr.
- **Compressor Electricity Cost:** `XMEAS_20 (kW) × $0.0536/kWh = $18.3028/hr`.
- **Stripper Steam Cost:** `XMEAS_19 (kg/hr) × $0.00318/kg = $0.7323/hr`.
- **Identity Check:**  
  `|Total − (Purge + Electricity + Steam)| < 10⁻⁴ $/hr  [PASS]`

### 5.2 Energy Power Balance
1. **Compressor Electrical Power:** Direct sensor reading `XMEAS_20 = 341.4695 kW`.
2. **Reboiler Thermal Power:** Latent heat of low-pressure steam (λ = 2257 kJ/kg):  
   `P_thermal (kW) = (steam_flow_kg_hr × 2257 kJ/kg) / 3600 s/hr = 144.3741 kW`  
   Formula error: `|P_thermal − (steam_flow × 2257 / 3600)| < 10⁻⁴ kW  [PASS]`.
3. **Total Equivalent Process Power Proxy:**  
   `P_total = P_electrical + P_thermal = 341.4695 + 144.3741 = 485.8436 kW`  
   *Note:* Summing electrical work and thermal duty is an unweighted energy footprint proxy, standard in plant utility accounting.

---

## 6. Verification Benchmark Results
 
Summary of the 50 verification checks in `validate_results.py`:
 
| # | Check Description | Value | Tolerance | Status |
| :---: | :--- | :---: | :---: | :---: |
| 1 | Dataset row count (`d00.dat`) | 500 | [499, 501] | PASS |
| 2 | Dataset column count | 52 | [51, 53] | PASS |
| 3 | NaN value count | 0 | 0 | PASS |
| 4 | Infinite value count | 0 | 0 | PASS |
| 5 | Baseline operating cost | $106.39/hr | [$104.00, $108.00] | PASS |
| 6 | Purge gas cost | $87.35/hr | [$82.00, $92.00] | PASS |
| 7 | Compressor electricity cost | $18.30/hr | [$17.00, $20.00] | PASS |
| 8 | Steam cost | $0.73/hr | [$0.50, $1.50] | PASS |
| 9 | Compressor electrical power | 341.47 kW | [338.00, 345.00] | PASS |
| 10 | Steam thermal power | 144.37 kW | [142.00, 147.00] | PASS |
| 11 | Total energy footprint | 485.84 kW | [483.00, 490.00] | PASS |
| 12 | Operating cost sum identity | Δ = 0.00 | < 10⁻⁴ | PASS |
| 13 | Energy power sum identity | Δ = 0.00 | < 10⁻⁴ | PASS |
| 14 | Steam thermal conversion formula | Δ = 0.00 | < 10⁻⁴ | PASS |
| 15 | Product Flow (XMEAS_17) CV R² | 0.9951 | ≥ 0.99 | PASS |
| 16 | Purge Rate (XMEAS_10) CV R² | 0.8775 | ≥ 0.85 | PASS |
| 17 | Stripper Steam (XMEAS_19) CV R² | 0.7733 | ≥ 0.70 | PASS |
| 18 | Reactor Pressure (XMEAS_07) CV R² | 0.3005 | ≥ 0.20 | PASS |
| 19 | Compressor Power (XMEAS_20) CV R² | 0.3602 | ≥ 0.25 | PASS |
| 20 | Fold-level CV metrics recorded (k = 5) | 5 folds | 5 folds | PASS |
| 21 | Chronological Holdout Product R² | 0.9987 | ≥ 0.99 | PASS |
| 22 | Baseline Point Mahalanobis Check | D_M² = 0.00 | ≤ 20.09 | PASS |
| 23 | LP1 Prod Max Baseline Throughput | 22.9093 m³/hr | [22.50, 23.30] | PASS |
| 24 | LP1 Prod Max Optimized Throughput | 24.6813 m³/hr | [24.00, 25.50] | PASS |
| 25 | LP1 Prod Max Throughput Gain | +7.73% | [5.00%, 10.00%] | PASS |
| 26 | LP1 Solver Status | Status 7 (Optimal) | Optimal | PASS |
| 27 | LP1 Independent Feasibility Check | Feasible (28/28 bounds) | Violations = 0 | PASS |
| 28 | LP2 Cost Min LP Baseline Cost | $106.40/hr | [$104.00, $108.00] | PASS |
| 29 | LP2 Cost Min LP Optimized Cost | $97.73/hr | [$94.00, $101.00] | PASS |
| 30 | LP2 Cost Min LP Cost Reduction | -8.15% | [5.00%, 12.00%] | PASS |
| 31 | LP2 Solver Status | Status 7 (Optimal) | Optimal | PASS |
| 32 | LP2 Cost Min Full Baseline Cost | $106.39/hr | [$104.00, $108.00] | PASS |
| 33 | LP2 Cost Min Full Optimized Cost | $97.71/hr | [$94.00, $101.00] | PASS |
| 34 | LP2 Cost Min Full Cost Reduction | -8.15% | [5.00%, 12.00%] | PASS |
| 35 | LP2 Independent Feasibility Check | Feasible (29/29 bounds) | Violations = 0 | PASS |
| 36 | LP3 Energy Min Baseline Energy | 485.84 kW | [482.00, 490.00] | PASS |
| 37 | LP3 Energy Min Optimized Energy | 469.03 kW | [463.00, 478.00] | PASS |
| 38 | LP3 Energy Min Footprint Reduction | -3.46% | [2.00%, 7.00%] | PASS |
| 39 | LP3 Solver Status | Status 7 (Optimal) | Optimal | PASS |
| 40 | LP3 Independent Feasibility Check | Feasible (29/29 bounds) | Violations = 0 | PASS |
| 41 | Reactor Pressure Safety Bound | 2702.7 kPa | ≤ 2800.0 kPa | PASS |
| 42 | Compressor Power Safety Bound | 337.8 kW | ≤ 355.0 kW | PASS |
| 43 | Analytical LP Known Optimum Check | x* = (4.0, 2.0), z* = 18.0 | Exact Match | PASS |
| 44 | Analytical LP Reduced Costs Check | rc = (-3.0, 0.0) | Exact Match | PASS |
| 45 | Analytical LP Shadow Price Check | sp = 5.0 | Exact Match | PASS |
| 46 | Goal Programming Scenario A (Weighted-Sum) | Feasible | Feasible | PASS |
| 47 | Goal Programming Scenario A (Preemptive) | Feasible (3 levels) | Feasible | PASS |
| 48 | Goal Programming Scenario B (Weighted-Sum) | Feasible | Feasible | PASS |
| 49 | Goal Programming Scenario B (Preemptive) | Feasible (2 levels) | Feasible | PASS |
| 50 | Preemptive Lexicographic P1 Preservation | z₁ = 0.000000 | ≤ z₁* + 10⁻⁴ | PASS |
 
- **Numerical Validation Score:** 50/50 Passed (100%)
- **Automated Test Suite:** 94/94 Passed (100%)

---

## 7. Conclusions

1. **Model Accuracy:** Incorporating condenser cooling water (`XMV_11`) into the surrogate feature space enables accurate prediction of finished product flow (R² = 0.9951 CV, R² = 0.9987 holdout).
2. **Optimization Trade-offs:** The HiGHS Dual Simplex solver finds viable operating setpoints achieving +7.73% throughput, -8.15% operating cost, and -3.46% energy footprint within empirical trust bounds.
3. **Mathematical Consistency:** Deviation penalization logic, priority ordering, and independent cost/energy identity checks were verified without numerical discrepancies.
