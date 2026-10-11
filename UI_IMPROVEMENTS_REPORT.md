# Tennessee Eastman Process Optimization Dashboard — UI Improvements & Verification Report

**Course:** 23MNG336 — Operations Research  
**Team:** B4  
**Date:** October 11, 2026  
**Status:** Fully Resolved, Validated, and Submission-Ready

---

## 1. Summary of UI Improvements

| Area | Previous State | Improved State | Benefit |
|---|---|---|---|
| **Theme & Aesthetic** | Dark mode (`#0d1117`) with low contrast and neon accents | Clean light scientific theme (`#ffffff` main, `#f8fafc` cards, `#0f172a` text, `#0284c7` blue accent) | Meets academic submission requirements; high contrast and legibility |
| **Sidebar Branding** | Broken external image (`img.icons8.com/...`) | Clean inline SVG/CSS brand badge ("TE" monogram with bold header) | Zero external network dependency; reliable offline rendering |
| **Navigation IA** | Lengthy technical titles | Concise 8-view structure: Overview, Process Explorer, Dataset & Diagnostics, Predictive Models, Production Optimization, Cost & Energy, Goal Programming, Sensitivity & Reports | Intuitive user journey from data to models to optimization |
| **KPI Cards** | Hardcoded values and inconsistent styling | Standardized 4 primary KPI cards with value, unit, baseline, delta %, tag (Model-Predicted, Calculated, etc.), and plain-English description | Immediate clarity for first-time viewers |
| **Data Visualizations** | Dark Plotly templates (`plotly_dark`) | Clean scientific light templates (`plotly_white`) with high-contrast palette and readable axes | Publication-ready figures matching academic standards |
| **Process Flowsheet** | Simple text dataframe | Structured process flow schematic diagram + detailed unit operations table + glossary | Visually clarifies feeds, reactor, condenser, separator, compressor, stripper, and recycle loops |
| **LaTeX & Units** | Raw unrendered LaTeX strings (`\text{hr}`, `\text{kWh}`, `\text{kg}`, `\approx`) | Clean readable unicode text (`$/hr`, `$/kWh`, `$/kg`, `m³/hr`, `kPa`, `kW`, `kJ/kg`) with `st.latex()` reserved for equations | No broken or awkward markup in tables and UI text |
| **Methodology Disclaimers** | Scattered or missing | Prominent callout banners clarifying benchmark simulation nature, statistical stationarity vs physical safety, and trust regions | Transparent and academically defensible presentation |

---

## 2. Bug Fixes

### 2.1 Resolution of `KeyError: 'steam_thermal_duty_kw'`
- **Root Cause:** In `src/optimization/energy_model.py`, the return dictionary from `evaluate_energy_breakdown()` defined `"steam_thermal_power_kw"`. However, View 3 in `streamlit_app.py` previously attempted to access `energy_base['steam_thermal_duty_kw']`, triggering an unhandled `KeyError` whenever View 3 was loaded.
- **Resolution:**
  1. Updated `src/optimization/energy_model.py` to provide both `"steam_thermal_power_kw"` and `"steam_thermal_duty_kw"` (as an exact alias) in the dictionary return schema.
  2. Updated `streamlit_app.py` to safely access `energy_base.get("steam_thermal_duty_kw", energy_base.get("steam_thermal_power_kw", 0.0))`.
  3. Added an automated regression test in `tests/test_cost_energy.py` asserting that `"steam_thermal_duty_kw"` is present and strictly equals `"steam_thermal_power_kw"`.

### 2.2 Plotly Trace Types in Optimization Profiles
- **Root Cause:** In View 5, decision variable profiles previously attempted to pass dictionaries directly to `go.Bar(y=...)`.
- **Resolution:** Extracted explicit float lists matching feature ordering: `y=[lp_res.u_baseline.get(f, 0.0) for f in DECISION_FEATURES_8]`.

### 2.3 Goal Programming Subtraction Bug
- **Root Cause:** In View 7, calculating valve deltas attempted `gp_res.u_optimal - u_base` where `u_optimal` was a dictionary, raising `TypeError`.
- **Resolution:** Extracted a NumPy array `u_opt_arr = np.array([gp_res.u_optimal.get(f, 0.0) for f in DECISION_FEATURES_8])` to compute delta percentages cleanly.

---

## 3. List of Modified Files

1. **`streamlit_app.py`**: Redesigned with light scientific theme, 8 clear views, Plotly white templates, no broken images, and eliminated raw LaTeX.
2. **`.streamlit/config.toml`**: Configured default Streamlit theme to light (`base = "light"`, `primaryColor = "#0284c7"`, `backgroundColor = "#ffffff"`).
3. **`src/optimization/energy_model.py`**: Added `"steam_thermal_duty_kw"` alias to prevent KeyError.
4. **`tests/test_cost_energy.py`**: Added regression test asserting presence and equivalence of `"steam_thermal_duty_kw"`.
5. **`UI_IMPROVEMENTS_REPORT.md`**: Created comprehensive documentation of design overhaul and test outcomes.

---

## 4. Test Results and Numerical Evidence

| Test Command | Scope | Actual Outcome | Status |
|---|---|---|:---:|
| `python -m py_compile app.py streamlit_app.py` | Syntax & compilation check | Exit code 0, clean compilation | **PASS** |
| `python -m pytest tests/ -v` | Complete 70-test suite | **70 passed in 2.73s** (100%) | **PASS** |
| `python -m pytest tests/test_cost_energy.py -v` | Energy & cost regression tests | **4 passed in 0.64s** | **PASS** |
| `python validate_results.py` | 32-check numerical benchmark | **32/32 checks passed (100%)** | **PASS** |
| `python run_optimization.py` | Optimization pipeline & artifacts | Exit code 0, all JSON/CSV outputs regenerated | **PASS** |
| `python generate_final_plots.py` | Publication visual figures | Exit code 0, 5 figures rendered in high resolution | **PASS** |
| Programmatic 8-View Simulation | End-to-end simulated view execution | All 8 views executed with zero exceptions | **PASS** |
| Streamlit Server Launch | Background daemon & HTTP probe | Server started, `HTTP 200` received | **PASS** |

---

## 5. Verification of All 8 Dashboard Views

1. **View 1: Overview**
   - Renders 4 primary KPI cards with baseline, optimized values, and percentage gains.
   - Normalized comparison bar chart rendered using `plotly_white`.
   - "What do these results mean?" plain-English summary cards verified.
2. **View 2: Process Explorer**
   - 5 major unit operations table with measured (XMEAS) and manipulated (XMV) tags.
   - Text/ASCII material and recycle stream diagram verified.
   - Variable catalog with category filters and CSV export verified.
3. **View 3: Dataset & Diagnostics**
   - Dataset statistics for 500 samples across 52 variables verified.
   - Time series trajectory and histogram distribution rendered with light theme.
   - Augmented Dickey-Fuller stationarity table ($p < 0.05$) rendered with statistical vs physical explanation.
   - Baseline economics and energy footprint breakdown verified without KeyError.
4. **View 4: Predictive Models**
   - Detailed explanation of the XMV_11 discovery (correlation vs causation, condensation equilibrium).
   - TimeSeriesSplit CV table ($k=5$) displaying $R^2 = 0.9951$ for product flow.
   - Actual vs predicted parity plot rendered with light theme and 45-degree reference line.
5. **View 5: Production Optimization**
   - 10-point transparent engineering formulation.
   - Live HiGHS Dual Simplex solver button executes and finds optimal solution ($24.68\text{ m}^3/\text{hr}$, $+7.73\%$).
   - Optimal valve setpoints, binding constraints, and shadow prices displayed.
6. **View 6: Cost & Energy**
   - Minimizes Operating Cost ($-8.15\%$), Total Process Power ($-3.46\%$), or Compressor Electrical Work ($-1.08\%$).
   - Explicit segregation between electrical work (kW) and thermal duty (kWth).
7. **View 7: Goal Programming**
   - Weighted-Sum and Preemptive Lexicographic modes execute to feasibility ($0.0000$ deviation).
   - Goal achievement table and recommended valve positions displayed.
8. **View 8: Sensitivity & Reports**
   - Clear distinction between LP shadow prices and local finite-difference what-if gradients.
   - Dominant gradient bar chart rendered with light theme.
   - Working CSV download buttons for master summary and local sensitivity gradients.

---

## 6. Stability of Numerical Optimization Results

No optimization mathematics, cost formulas, or solver parameters were altered. All reported numerical values are identically reproduced:
- **Product Throughput:** Baseline $22.9093\text{ m}^3/\text{hr}$ $\rightarrow$ Optimized **$24.6813\text{ m}^3/\text{hr}$** (**$+7.73\%$**)
- **Hourly Operating Cost:** Baseline $\$106.4015/\text{hr}$ $\rightarrow$ Optimized **$\$97.7254/\text{hr}$** (**$-8.15\%$**)
- **Process Energy Footprint:** Baseline $485.8436\text{ kW}$ $\rightarrow$ Optimized **$469.0334\text{ kW}$** (**$-3.46\%$**)
- **Compressor Electrical Power:** Baseline $341.4695\text{ kW}$ $\rightarrow$ Optimized **$337.7700\text{ kW}$** (**$-1.08\%$**)

---

## 7. How to Launch the Application

From the project root directory, launch the dashboard using either command:

```bash
python -m streamlit run streamlit_app.py
```
or:
```bash
streamlit run app.py
```

The application will launch on `http://localhost:8501` in clean, professional light theme mode.
