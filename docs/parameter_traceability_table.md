# Parameter and Engineering Traceability Table

**Project:** Chemical Manufacturing Energy and Cost Optimization Using the Tennessee Eastman Process  
**Course & Team:** 23MNG336: Operational Research | Team B4  
**Primary References:**
1. Downs, J.J. and Vogel, E.F., *"A plant-wide industrial process control problem"*, Computers & Chemical Engineering, 17(3):245-255, 1993.
2. Russell, E.L., Chiang, L.H., and Braatz, R.D., *"Data-driven Techniques for Fault Detection and Diagnosis in Chemical Processes"*, Springer, 2000.
3. Tennessee Eastman Simulation Code (`teprob.f` and `temain_mod.f`).

---

## 1. Process Variable Catalog & Bounds (52 Variables)

| Tag | Index | Variable Name | Type | Units | Nominal (Mode 1) | Operating Bounds | Safety Alarms / Shutdown | Primary Literature Source | Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `XMEAS_01` | 1 | A Feed Flow (Stream 1) | Measured | kscmh | 0.2505 | [0.0, 1.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_02` | 2 | D Feed Flow (Stream 2) | Measured | kg/hr | 3657.0 | [0.0, 6000.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_03` | 3 | E Feed Flow (Stream 3) | Measured | kg/hr | 4440.0 | [0.0, 6000.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_04` | 4 | A+C Feed Flow (Stream 4) | Measured | kscmh | 9.347 | [0.0, 15.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_05` | 5 | Recycle Flow (Stream 8) | Measured | kscmh | 26.902 | [0.0, 50.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_06` | 6 | Reactor Feed Rate (Stream 6) | Measured | kscmh | 32.253 | [0.0, 60.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_07` | 7 | Reactor Pressure | Measured | kPa gauge | 2705.0 | [0.0, 3200.0] | High Alarm: 2895 kPa; Shutdown: 3000 kPa | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_08` | 8 | Reactor Level | Measured | % | 65.0 | [30.0, 80.0] | Low: 30%, High: 80%; Shutdown: [20%, 90%] | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_09` | 9 | Reactor Temperature | Measured | °C | 120.40 | [0.0, 200.0] | High Alarm: 175.0 °C | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_10` | 10 | Purge Rate (Stream 9) | Measured | kscmh | 0.3371 | [0.0, 2.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_11` | 11 | Separator Temperature | Measured | °C | 80.11 | [0.0, 150.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_12` | 12 | Separator Level | Measured | % | 50.0 | [30.0, 80.0] | Low: 30%, High: 80% | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_13` | 13 | Separator Pressure | Measured | kPa gauge | 2633.7 | [0.0, 3200.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_14` | 14 | Separator Underflow (Stream 10)| Measured | m³/hr | 25.16 | [0.0, 50.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_15` | 15 | Stripper Level | Measured | % | 50.0 | [30.0, 80.0] | Low: 30%, High: 80% | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_16` | 16 | Stripper Pressure | Measured | kPa gauge | 2604.8 | [0.0, 3200.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_17` | 17 | Product Delivery Rate (Stream 11)| Measured | m³/hr | 22.89 | [0.0, 50.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_18` | 18 | Stripper Temperature | Measured | °C | 65.73 | [0.0, 150.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_19` | 19 | Stripper Steam Flow | Measured | kg/hr | 517.9 | [0.0, 1500.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_20` | 20 | Compressor Electrical Work | Measured | kW | 341.4 | [0.0, 600.0] | Max motor power: 500 kW | Downs & Vogel (1993), Table 1; `teprob.f` | **Verified Literature** |
| `XMEAS_21` | 21 | Reactor Cooling Water Out Temp | Measured | °C | 94.60 | [0.0, 150.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_22` | 22 | Separator Cooling Water Out Temp | Measured | °C | 45.69 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_23`..`28` | 23–28 | Reactor Feed Composition (A–F) | Measured | mol % | Table 1 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_29`..`36` | 29–36 | Purge Gas Composition (A–H) | Measured | mol % | Table 1 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMEAS_37`..`41` | 37–41 | Product Composition (D, E, F, G, H)| Measured | mol % | Table 1 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 1 | **Verified Literature** |
| `XMV_01` | 42 | D Feed Valve (Stream 2) | Manipulated | % | 63.05 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_02` | 43 | E Feed Valve (Stream 3) | Manipulated | % | 53.98 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_03` | 44 | A Feed Valve (Stream 1) | Manipulated | % | 24.64 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_04` | 45 | A+C Feed Valve (Stream 4) | Manipulated | % | 61.30 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_05` | 46 | Compressor Recycle Valve | Manipulated | % | 22.21 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_06` | 47 | Purge Valve (Stream 9) | Manipulated | % | 40.06 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_07` | 48 | Separator Liquid Flow Valve | Manipulated | % | 38.10 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_08` | 49 | Stripper Product Flow Valve | Manipulated | % | 46.53 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_09` | 50 | Stripper Steam Valve | Manipulated | % | 47.45 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_10` | 51 | Reactor Cooling Water Valve | Manipulated | % | 41.11 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |
| `XMV_11` | 52 | Condenser Cooling Water Valve | Manipulated | % | 18.11 | [0.0, 100.0] | — | Downs & Vogel (1993), Table 2 | **Verified Literature** |

---

## 2. Economic & Operating Cost Coefficients (Downs & Vogel 1993)

| Cost Parameter | Engineering Value | Units | Source Reference | Project Classification |
| :--- | :---: | :---: | :--- | :---: |
| **Purge Cost (Component A)** | \$2.206 | \$/kmol | Downs & Vogel (1993), Table 4 | **Verified Literature** |
| **Purge Cost (Component C)** | \$6.016 | \$/kmol | Downs & Vogel (1993), Table 4 | **Verified Literature** |
| **Purge Cost (Component D)** | \$8.222 | \$/kmol | Downs & Vogel (1993), Table 4 | **Verified Literature** |
| **Purge Cost (Component E)** | \$10.428 | \$/kmol | Downs & Vogel (1993), Table 4 | **Verified Literature** |
| **Purge Cost (Component F)** | \$14.444 | \$/kmol | Downs & Vogel (1993), Table 4 | **Verified Literature** |
| **Stripper Steam Price** | \$0.00318 | \$/kg | Downs & Vogel (1993), Table 4 | **Verified Literature** |
| **Compressor Electrical Power Price** | \$0.0536 | \$/kWh | Downs & Vogel (1993), Table 4 | **Verified Literature** |
| **Product Loss Cost (Unreacted A, C, D, E)** | Same component molar prices | \$/kmol | Downs & Vogel (1993), Section 4 | **Verified Literature** |
| **Literature Mode 1 Base Operating Cost** | $\approx \$310 - \$340$ | \$/hr | Downs & Vogel (1993), Bathelt et al. (2015) | **Literature Benchmark Hypothesis** |
