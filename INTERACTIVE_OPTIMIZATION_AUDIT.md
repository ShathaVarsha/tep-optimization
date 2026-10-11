# Interactive Optimization Controls Audit & Verification Report

## Tennessee Eastman Process Optimization Dashboard

---

## 1. Executive Summary

This audit resolved the issue where modifying sliders, targets, or weights in the dashboard previously appeared to leave optimization results unchanged. A systematic inspection of the mathematical formulations, data contracts, and Streamlit execution model revealed four root causes:

1. **Stateless UI Reset on Widget Interaction:** In Streamlit, adjusting a widget triggers a full script rerun where `st.button` evaluates to `False`. Previously, solver results were not stored in `st.session_state`, so moving a slider reset previous outputs or forced reruns without scenario continuity.
2. **Mathematical Invariance Under Binding Constraints:** In several operating regimes, the mathematical optimum legitimately remained unchanged:
   - In **Production Maximization**, expanding trust bounds beyond ±10% encounters active empirical lower bounds on purge venting (`XMEAS_10`) and compressor work (`XMEAS_20`). Throughput plateaus at 24.6813 m³/hr because these response bounds act as active bottlenecks.
   - In **Cost Minimization**, varying production quotas between 20.0 and 24.6 m³/hr produced the same minimum cost ($97.73/hr) because condenser cooling (`XMV_11`) satisfies throughput demands without increasing purge loss or electricity penalties (Quota Shadow Price = $0.00/m³).
   - In **Goal Programming**, when targets are mutually attainable within trust bounds, all deviation variables (d⁻, d⁺) equal zero (0.0000). Multiplying zero deviation by different weights produces zero penalty.
3. **Priority Sliders in Preemptive GP:** In View 7, priority levels in `GoalSpec` were hardcoded to `priority = 1, 2, 3`. The sliders adjusted weights rather than lexicographic levels, rendering priority re-ordering non-functional.
4. **Unhandled Solver Infeasibility:** When quotas exceeding capacity (e.g., > 24.68 m³/hr) were entered, `lp_optimizer.py` previously raised an unhandled `RuntimeError`, crashing the app.

All four issues were addressed. The controls now have verified mathematical connections, session-state persistence, graceful infeasibility handling, and dynamic recalculation of operational and economic metrics.

---

## 2. Interactive Control Audit Matrix

| View | Control | Type | Parameter / Constraint Modified | Formulation Effect | Verified Behavior |
|---|---|---|---|---|---|
| **View 5: Production Max** | Trust Region Half-Width | Slider (±2.0% to ±20.0%) | `u_base − δ ≤ u ≤ u_base + δ` | Modifies decision variable bounds in HiGHS solver | Below 10%, throughput scales with δ; above 10%, it plateaus at 24.6813 m³/hr due to active response bounds. |
| **View 5: Production Max** | Empirical Response Bounding | Checkbox (Toggle) | `A_resp · u ≤ b_resp` | Toggles historical min/max constraints on state variables | When disabled, throughput expands slightly to 24.6843 m³/hr (+0.003 m³/hr) while core safety limits (P ≤ 2800 kPa, W ≤ 355 kW) remain strictly enforced. |
| **View 5: Production Max** | Solve Button | Button | Solver execution trigger | Executes HiGHS Dual Simplex and updates session state | Increments scenario ID, recalculates dependent metrics, and generates a change comparison note. |
| **View 6: Cost & Energy** | Minimization Objective | Radio (3 options) | Objective vector cᵀ | Swaps between Operating Cost ($/hr), Total Power (kW), and Compressor Work (kW) | Switches cost vector in `linprog`. Cost minimizes to $97.73/hr, Total Power to 469.03 kW, Compressor Work to 337.77 kW. |
| **View 6: Cost & Energy** | Production Quota Target | Slider (20.0 to 25.0 m³/hr) | `-c_prodᵀ u ≤ -(g_target − d_prod)` | Sets lower bound on finished product flow `XMEAS_17 ≥ g_target` | Adjusts `XMV_11` setpoint dynamically. Quotas > 24.68 m³/hr return an infeasible status gracefully without crashing. |
| **View 6: Cost & Energy** | Trust Region Half-Width | Slider (±2.0% to ±15.0%) | Variable bounds l_i, u_i | Sets allowable deviation around nominal baseline | Constrains decision vector search radius. |
| **View 6: Cost & Energy** | Solve Button | Button | Solver execution trigger | Solves LP Problem 2 or 3 and updates session state | Updates scenario badge, logs whether cost is unchanged due to `XMV_11` slack, and recomputes metrics. |
| **View 7: Goal Programming** | Algorithm Mode | Radio (2 options) | Single LP vs. Lexicographic Priority Sequence | Swaps between Weighted-Sum GP and Preemptive Lexicographic GP | Weighted-sum penalizes `sum(w_k / g_k · d_k)`; Preemptive solves successive LPs freezing higher priority deviations. |
| **View 7: Goal Programming** | Targets (Prod, Comp, Press) | Number Inputs (3 fields) | Goal Equality Constraints `A_eq · x = b_eq` | Modifies target values g_k for `XMEAS_17, 20, 07` | Alters right-hand side `b_eq` vector. When competing targets are set, dictates Pareto trade-off point. |
| **View 7: Goal Programming** | Goal Weights (w_k) | Sliders (1.0 to 10.0) | Objective vector coefficients c | Scales penalty multipliers `(w_k / |g_k|)` in Weighted-Sum mode | Modifies relative penalty weights. When trade-offs occur, shifts optimal balance toward higher-weighted goals. |
| **View 7: Goal Programming** | Lexicographic Priorities (P_k) | Selectboxes (P1, P2, P3) | `priority` attribute in `GoalSpec` | Groups goals into priority sets solved in sequential stages | Changes priority order in Preemptive mode. Solving P1=Product first vs P1=Compressor first yields distinct operating points. |
| **View 7: Goal Programming** | Trust Region Bounds | Slider (±2.0% to ±20.0%) | Decision variable bounds | Sets trust region half-width around baseline | Expands or contracts GP trust region. |
| **View 7: Goal Programming** | Solve Button | Button | Solver execution trigger | Solves GP and updates session state | Recalculates operational metrics and updates summary table and charts. |

---

## 3. Code Modifications

### 3.1 `src/optimization/lp_optimizer.py`
- Removed unhandled `raise RuntimeError(...)` when `not res.success`.
- Implemented graceful failure return: returns an `LPResult(success=False, status_message=str(res.message), ...)` containing baseline vectors, empty slacks, zero reduced costs, and the HiGHS solver message (e.g., `HiGHS Status 8: Infeasible`).
- Allows the caller and dashboard to detect infeasibility and explain remediation steps without terminating execution.

### 3.2 `streamlit_app.py`
- Added `compute_all_operational_metrics(u_vector, surrogate_model)`: Derives all 11 dependent operational metrics for any operating vector u (throughput, pressure, purge flow, steam flow, compressor power, steam thermal duty, total power, purge cost, electricity cost, steam cost, total cost).
- Added `render_scenario_badge(...)`: Displays a clean metadata badge with scenario index, execution time, and solver iterations.
- Added `render_change_explanation(...)`: Explains whether and why the objective or variables changed.
- View 5 (Production Maximization): Integrated `st.session_state["v5_scenario"]`. Initializes with default benchmark solve.
- View 6 (Cost & Energy Minimization): Integrated `st.session_state["v6_scenario"]`. Displays actionable feedback for infeasible quota requests (> 24.68 m³/hr) and explains constant cost regimes ($97.73/hr) due to non-cost-bearing `XMV_11` cooling water capacity.
- View 7 (Goal Programming): Integrated `st.session_state["v7_scenario"]`. Renders weight sliders in Weighted-Sum mode and priority rank selectboxes (P1, P2, P3) in Preemptive mode.

### 3.3 `tests/test_ui_workflow.py`
- `test_ui_interactive_trust_region_sensitivity`: Asserts that trust regions of ±2% vs ±10% and empirical bounding alter the problem formulation and objective value.
- `test_ui_interactive_cost_quota_sensitivity`: Asserts that quotas of 21.0 vs 24.0 m³/hr produce different valve settings (`XMV_11`), and that an unreachable quota (35.0 m³/hr) returns `LPResult(success=False)` without raising an exception.
- `test_ui_interactive_goal_programming_priorities`: Asserts that re-ordering lexicographic priorities (P1=Product vs P1=Compressor) under competing targets (28.0 m³/hr vs 310.0 kW) produces distinct trade-off vectors.

---

## 4. Controlled Scenario Test Results

### 4.1 Production Maximization Scenarios (View 5)

| Scenario | Trust Delta (δ) | Empirical Bounds | Solver Status | Product Flow (XMEAS_17) | Total Cost ($/hr) | Compressor Power (kW) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | ±2.0% | Enforced | Optimal (2 iter) | 23.7572 m³/hr | $97.73 | 337.8 kW | Tightly constrained by ±2% trust region. |
| 2 | ±5.0% | Enforced | Optimal (2 iter) | 24.6771 m³/hr | $97.73 | 337.8 kW | Search space expanded; throughput increases +3.87%. |
| 3 | ±10.0% | Enforced | Optimal (2 iter) | 24.6813 m³/hr | $97.73 | 337.8 kW | Benchmark optimum. Purge and compressor limits become active. |
| 4 | ±15.0% | Enforced | Optimal (2 iter) | 24.6813 m³/hr | $97.73 | 337.8 kW | Throughput plateaus because active response bounds constrain further expansion. |
| 5 | ±10.0% | Disabled | Optimal (2 iter) | 24.6843 m³/hr | $97.73 | 337.8 kW | Disabling response bounds permits +0.003 m³/hr gain until valve limits bind. |

### 4.2 Cost and Energy Minimization Scenarios (View 6)

| Scenario | Objective | Quota Target | Trust Delta | Solver Status | Objective Value | Throughput | Valve XMV_11 | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | Operating Cost | 21.00 m³/hr | ±10.0% | Optimal (2 iter) | $97.73/hr | 21.00 m³/hr | 20.46% | Lower quota met by throttling XMV_11. |
| 2 | Operating Cost | 22.89 m³/hr | ±10.0% | Optimal (2 iter) | $97.73/hr | 22.89 m³/hr | 18.31% | Baseline quota met exactly. Cost unchanged because XMV_11 has zero cost penalty. |
| 3 | Operating Cost | 24.00 m³/hr | ±10.0% | Optimal (2 iter) | $97.73/hr | 24.00 m³/hr | 15.61% | Quota met. Cost remains $97.73/hr; quota shadow price is $0.00/m³. |
| 4 | Operating Cost | 24.68 m³/hr | ±10.0% | Optimal (2 iter) | $97.73/hr | 24.68 m³/hr | 14.01% | Upper feasibility limit of throughput. |
| 5 | Operating Cost | 25.00 m³/hr | ±10.0% | Infeasible | N/A | N/A | N/A | Quota exceeds maximum throughput capacity (24.68 m³/hr). |
| 6 | Total Power | 22.89 m³/hr | ±10.0% | Optimal (2 iter) | 469.03 kW | 22.89 m³/hr | 18.31% | Total power minimized (Compressor 337.8 kW + Reboiler 131.2 kW). |
| 7 | Compressor Work | 22.89 m³/hr | ±10.0% | Optimal (2 iter) | 337.77 kW | 22.89 m³/hr | 18.31% | Compressor electrical power minimized directly. |

### 4.3 Multi-Objective Goal Programming Scenarios (View 7)

| Scenario | Formulation | Targets (Prod, Comp, Press) | Weights / Priorities | Total Deviation | Achieved Prod | Achieved Comp | Notes |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| 1 | Weighted-Sum | 24.0, 340.0, 2705.4 | w = (5.0, 3.0, 1.0) | 0.0000 | 24.00 m³/hr | 340.0 kW | All targets feasible within trust region; deviations equal zero. |
| 2 | Weighted-Sum | 24.0, 340.0, 2705.4 | w = (1.0, 10.0, 1.0) | 0.0000 | 24.00 m³/hr | 340.0 kW | When all deviations are zero, changing weights scales zeros: total deviation remains zero. |
| 3 | Preemptive | 28.0, 310.0, 2705.4 | P1=Prod > P2=Comp > P3=Press | 0.0381 | 27.26 m³/hr | 317.8 kW | Prioritizes throughput; compressor power rises to 317.8 kW. |
| 4 | Preemptive | 28.0, 310.0, 2705.4 | P1=Comp > P2=Prod > P3=Press | 0.3018 | 19.55 m³/hr | 310.0 kW | Prioritizes compressor curtailment; throughput falls to 19.55 m³/hr. |

---

## 5. Verification Summary

- `pytest tests/`: 94/94 passed in ~4 seconds.
- `pytest tests/test_ui_workflow.py`: 8/8 passed.
- `python validate_results.py`: 50/50 checks passed (100%).
- All 11 operational and economic metrics recalculate dynamically on each scenario solve.
