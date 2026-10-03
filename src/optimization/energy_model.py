"""Energy Consumption and Thermal Duty Modeling for the Tennessee Eastman Process.

Primary Sources:
  1. Downs, J.J. and Vogel, E.F. (1993), Table 1 & Table 4.
  2. Simulation driver 'teprob.f', lines 698-701.

Energy Breakdown:
  1. Compressor Electrical Power (kW): Measured directly via XMEAS_20.
  2. Stripper Reboiler Thermal Power (kW): Derived from steam flow (XMEAS_19) using
     the latent heat of saturated low-pressure steam (lambda = 2257 kJ/kg).
  3. Total Primary Equivalent Power (kW): Total energy footprint proxy.

Author: Team B4 (23MNG336: Operational Research)
"""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
import pandas as pd

# Latent heat of vaporization of saturated steam at typical stripper pressure (~2257 kJ/kg)
LATENT_HEAT_STEAM_KJ_KG = 2257.0
SECONDS_PER_HOUR = 3600.0


def calculate_compressor_power(
    compressor_work_kw: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Return the electrical compressor power in kW [Direct Sensor Measurement XMEAS_20]."""
    return compressor_work_kw


def calculate_steam_thermal_power(
    steam_flow_kg_hr: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """Convert steam mass flow rate (kg/hr) into equivalent thermal power (kW).

    P_thermal (kW) = (steam_flow_kg_hr * 2257 kJ/kg) / 3600 s/hr
    """
    return (steam_flow_kg_hr * LATENT_HEAT_STEAM_KJ_KG) / SECONDS_PER_HOUR


def evaluate_energy_breakdown(data: Union[pd.DataFrame, pd.Series, Dict[str, float]]) -> Dict[str, float]:
    """Evaluate full electrical, thermal, and combined energy consumption metrics.

    Returns dict containing:
      - compressor_power_kw: Direct electrical work (kW)
      - steam_thermal_power_kw: Reboiler thermal power (kW)
      - total_energy_power_kw: Combined equivalent energy rate (kW)
    """
    if isinstance(data, pd.DataFrame):
        avg = data.mean()
    else:
        avg = data

    comp_kw = float(avg["XMEAS_20"])
    steam_flow = float(avg["XMEAS_19"])
    thermal_kw = float(calculate_steam_thermal_power(steam_flow))
    total_kw = comp_kw + thermal_kw

    return {
        "compressor_power_kw": comp_kw,
        "steam_thermal_power_kw": thermal_kw,
        "total_energy_power_kw": total_kw,
    }
