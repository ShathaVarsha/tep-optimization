"""Downs & Vogel (1993) Operating Cost Model for the Tennessee Eastman Process.

Primary Source:
  Downs, J.J. and Vogel, E.F., "A plant-wide industrial process control problem",
  Computers & Chemical Engineering, 17(3):245-255, 1993 (Section 4, Table 4).

Mathematical Formulation:
  Total Operating Cost ($/hr) = Purge Gas Loss Cost + Compressor Electricity Cost + Stripper Reboiler Steam Cost

Note on Product Losses:
  In Downs & Vogel (1993) Table 4, purge gas losses explicitly price product components
  G and H at $17.653/kmol (the finished product valuation). Therefore, product losses vented
  through the purge stream are already fully captured within the purge loss cost, avoiding double-counting.
"""

from __future__ import annotations

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd

# Standard economic parameters from Downs & Vogel (1993) Table 4
PRICE_PURGE_A = 2.206   # $/kmol (reactant A)
PRICE_PURGE_B = 0.000   # $/kmol (inert nitrogen/methane proxy, zero economic penalty)
PRICE_PURGE_C = 6.016   # $/kmol (reactant C)
PRICE_PURGE_D = 8.222   # $/kmol (feed reactant D)
PRICE_PURGE_E = 10.428  # $/kmol (feed reactant E)
PRICE_PURGE_F = 14.444  # $/kmol (byproduct F)
PRICE_PURGE_G = 17.653  # $/kmol (product G lost in purge stream)
PRICE_PURGE_H = 17.653  # $/kmol (product H lost in purge stream)

PURGE_PRICES = {
    "A": PRICE_PURGE_A,
    "B": PRICE_PURGE_B,
    "C": PRICE_PURGE_C,
    "D": PRICE_PURGE_D,
    "E": PRICE_PURGE_E,
    "F": PRICE_PURGE_F,
    "G": PRICE_PURGE_G,
    "H": PRICE_PURGE_H,
}

# Empirical baseline steady-state purge stream mole percentages from d00.dat:
BASELINE_PURGE_MOLE_PCT = {
    "A": 33.0015,
    "B": 13.8229,
    "C": 24.0195,
    "D": 1.2516,
    "E": 18.5725,
    "F": 2.2591,
    "G": 4.8412,
    "H": 2.2963,
}

# Exact effective composition-weighted unit price derived from d00 baseline:
# sum(y_i / 100 * P_i) = $5.798963 / kmol
PRICE_PURGE_BASELINE_EFFECTIVE = 5.798963

# Default rounded unit price proxy used in the linear programming objective ($/kmol)
PRICE_PURGE_DEFAULT = 5.80

PRICE_STEAM = 0.00318        # $/kg of stripper reboiler steam
PRICE_ELECTRICITY = 0.0536   # $/kWh of compressor electrical power

# Physical molar flow conversion factor derived from Fortran teprob.f (lines 679-688):
# XMEAS(10) in kscmh = FTM(10) * 0.359 / 35.3145 (where FTM is lbmol/hr)
# 1 lbmol = 0.45359237 kmol
# kmol/hr per kscmh = (35.3145 / 0.359) * 0.45359237 = 44.619371 kmol/hr per kscmh
KSCMH_TO_KMOL_HR = (35.3145 / 0.359) * 0.45359237


def calculate_purge_cost(
    purge_rate_kscmh: Union[float, np.ndarray, pd.Series],
    y_A: Union[float, np.ndarray, pd.Series],
    y_B: Union[float, np.ndarray, pd.Series],
    y_C: Union[float, np.ndarray, pd.Series],
    y_D: Union[float, np.ndarray, pd.Series],
    y_E: Union[float, np.ndarray, pd.Series],
    y_F: Union[float, np.ndarray, pd.Series],
    y_G: Union[float, np.ndarray, pd.Series],
    y_H: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate the economic loss from vented purge gas ($/hr) using component-specific prices.
    
    y_i values are mole percentages (0-100).
    """
    total_molar_flow = purge_rate_kscmh * KSCMH_TO_KMOL_HR

    cost_A = (total_molar_flow * (y_A / 100.0)) * PRICE_PURGE_A
    cost_B = (total_molar_flow * (y_B / 100.0)) * PRICE_PURGE_B
    cost_C = (total_molar_flow * (y_C / 100.0)) * PRICE_PURGE_C
    cost_D = (total_molar_flow * (y_D / 100.0)) * PRICE_PURGE_D
    cost_E = (total_molar_flow * (y_E / 100.0)) * PRICE_PURGE_E
    cost_F = (total_molar_flow * (y_F / 100.0)) * PRICE_PURGE_F
    cost_G = (total_molar_flow * (y_G / 100.0)) * PRICE_PURGE_G
    cost_H = (total_molar_flow * (y_H / 100.0)) * PRICE_PURGE_H

    return cost_A + cost_B + cost_C + cost_D + cost_E + cost_F + cost_G + cost_H


def calculate_compressor_cost(
    compressor_work_kw: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate compressor electrical operating cost ($/hr)."""
    return compressor_work_kw * PRICE_ELECTRICITY


def calculate_steam_cost(
    steam_flow_kg_hr: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate stripper reboiler steam operating cost ($/hr)."""
    return steam_flow_kg_hr * PRICE_STEAM


def evaluate_operating_cost_breakdown(
    data: Union[pd.DataFrame, pd.Series, Dict[str, float]],
) -> Dict[str, float]:
    """Evaluate full operating cost breakdown on a given state observation or dataset mean.

    Returns dict containing:
      - purge_cost_usd_hr
      - compressor_cost_usd_hr
      - steam_cost_usd_hr
      - total_operating_cost_usd_hr
    """
    if isinstance(data, pd.DataFrame):
        avg = data.mean()
    else:
        avg = data

    purge_rate = float(avg["XMEAS_10"])
    y_A = float(avg["XMEAS_29"])
    y_B = float(avg["XMEAS_30"])
    y_C = float(avg["XMEAS_31"])
    y_D = float(avg["XMEAS_32"])
    y_E = float(avg["XMEAS_33"])
    y_F = float(avg["XMEAS_34"])
    y_G = float(avg["XMEAS_35"])
    y_H = float(avg["XMEAS_36"])

    p_cost = float(calculate_purge_cost(purge_rate, y_A, y_B, y_C, y_D, y_E, y_F, y_G, y_H))
    c_cost = float(calculate_compressor_cost(float(avg["XMEAS_20"])))
    s_cost = float(calculate_steam_cost(float(avg["XMEAS_19"])))
    total = p_cost + c_cost + s_cost

    return {
        "purge_cost_usd_hr": p_cost,
        "compressor_cost_usd_hr": c_cost,
        "steam_cost_usd_hr": s_cost,
        "total_operating_cost_usd_hr": total,
    }


def evaluate_operating_cost(
    predicted_responses: Dict[str, float],
    compositions: Optional[Dict[str, float]] = None,
    avg_purge_price_per_kmol: float = PRICE_PURGE_DEFAULT,
) -> Dict[str, float]:
    """Independently evaluate both full component-based cost and LP linear proxy cost.

    Args:
        predicted_responses: Dict with keys 'XMEAS_10' (kscmh), 'XMEAS_19' (kg/hr), 'XMEAS_20' (kW).
        compositions: Optional dict with keys 'A'..'H' (or 'XMEAS_29'..'XMEAS_36').
                      If omitted, uses empirical baseline steady-state compositions.
        avg_purge_price_per_kmol: Unit purge gas valuation for LP linear objective ($/kmol).

    Returns:
        Dict containing full operating cost, individual cost components, LP proxy cost,
        and the approximation difference.
    """
    purge_rate = float(predicted_responses["XMEAS_10"])
    comp_power = float(predicted_responses["XMEAS_20"])
    steam_rate = float(predicted_responses["XMEAS_19"])

    if compositions is not None:
        comp_dict = {}
        for key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            if key in compositions:
                comp_dict[key] = float(compositions[key])
            else:
                idx = 29 + ["A", "B", "C", "D", "E", "F", "G", "H"].index(key)
                comp_dict[key] = float(compositions.get(f"XMEAS_{idx}", BASELINE_PURGE_MOLE_PCT[key]))
    else:
        comp_dict = dict(BASELINE_PURGE_MOLE_PCT)

    full_purge_cost = float(
        calculate_purge_cost(
            purge_rate,
            comp_dict["A"],
            comp_dict["B"],
            comp_dict["C"],
            comp_dict["D"],
            comp_dict["E"],
            comp_dict["F"],
            comp_dict["G"],
            comp_dict["H"],
        )
    )
    compressor_cost = float(calculate_compressor_cost(comp_power))
    steam_cost = float(calculate_steam_cost(steam_rate))
    full_total_cost = full_purge_cost + compressor_cost + steam_cost

    # Simplified linear LP objective proxy
    lp_purge_cost = float((purge_rate * KSCMH_TO_KMOL_HR) * avg_purge_price_per_kmol)
    lp_proxy_cost = float(lp_purge_cost + compressor_cost + steam_cost)

    return {
        "full_operating_cost_usd_hr": full_total_cost,
        "purge_cost_full_usd_hr": full_purge_cost,
        "compressor_cost_usd_hr": compressor_cost,
        "steam_cost_usd_hr": steam_cost,
        "lp_proxy_cost_usd_hr": lp_proxy_cost,
        "lp_purge_cost_usd_hr": lp_purge_cost,
        "proxy_difference_usd_hr": lp_proxy_cost - full_total_cost,
    }


def calculate_cost_reduction_pct(baseline_cost: float, optimized_cost: float) -> float:
    """Calculate percentage cost reduction: 100 * (baseline - optimized) / baseline."""
    if baseline_cost <= 0.0:
        return 0.0
    return float(100.0 * (baseline_cost - optimized_cost) / baseline_cost)
