"""Downs & Vogel (1993) Operating Cost Model for the Tennessee Eastman Process.

Primary Source:
  Downs, J.J. and Vogel, E.F., "A plant-wide industrial process control problem",
  Computers & Chemical Engineering, 17(3):245-255, 1993 (Section 4, Table 4).

Mathematical Formulation:
  Total Operating Cost ($/hr) = Purge Loss Cost + Compressor Work Cost + Steam Cost + Product Loss Cost

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
import pandas as pd

# Standard economic parameters from Downs & Vogel (1993) Table 4
PRICE_PURGE_A = 2.206   # $/kmol
PRICE_PURGE_B = 0.000   # $/kmol (inert nitrogen/methane proxy, zero economic penalty)
PRICE_PURGE_C = 6.016   # $/kmol
PRICE_PURGE_D = 8.222   # $/kmol
PRICE_PURGE_E = 10.428  # $/kmol
PRICE_PURGE_F = 14.444  # $/kmol (byproduct)
PRICE_PURGE_G = 17.653  # $/kmol (product G lost in purge)
PRICE_PURGE_H = 17.653  # $/kmol (product H lost in purge)

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
    """Calculate the economic loss from vented purge gas ($/hr)."""
    total_molar_flow = purge_rate_kscmh * KSCMH_TO_KMOL_HR
    
    # y_i values are mole percentages (0-100)
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


def evaluate_operating_cost_breakdown(data: Union[pd.DataFrame, pd.Series, Dict[str, float]]) -> Dict[str, float]:
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
