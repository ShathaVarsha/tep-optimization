"""Automated test suite for TEP Operating Cost and Energy Models.

Author: Team B4 (23MNG336: Operational Research)
"""

import pytest
import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.optimization.cost_model import (
    calculate_purge_cost,
    calculate_compressor_cost,
    calculate_steam_cost,
    evaluate_operating_cost_breakdown,
    PRICE_ELECTRICITY,
    PRICE_STEAM,
)
from src.optimization.energy_model import (
    calculate_compressor_power,
    calculate_steam_thermal_power,
    evaluate_energy_breakdown,
)


def test_zero_flow_zero_cost():
    """Verify that zero flows yield zero costs."""
    assert calculate_purge_cost(0.0, 30.0, 10.0, 20.0, 1.0, 20.0, 2.0, 5.0, 2.0) == 0.0
    assert calculate_compressor_cost(0.0) == 0.0
    assert calculate_steam_cost(0.0) == 0.0
    assert calculate_steam_thermal_power(0.0) == 0.0


def test_unit_pricing_calculations():
    """Verify linear proportionality against official unit costs."""
    kw = 100.0
    assert calculate_compressor_cost(kw) == pytest.approx(100.0 * PRICE_ELECTRICITY)
    assert calculate_compressor_power(kw) == 100.0

    steam_kg = 500.0
    assert calculate_steam_cost(steam_kg) == pytest.approx(500.0 * PRICE_STEAM)
    expected_thermal_kw = (500.0 * 2257.0) / 3600.0
    assert calculate_steam_thermal_power(steam_kg) == pytest.approx(expected_thermal_kw)


def test_baseline_cost_evaluation_on_d00():
    """Evaluate empirical operating cost on normal operation baseline (d00.dat)."""
    df = load_tep_dataset("d00")
    cost_breakdown = evaluate_operating_cost_breakdown(df)

    assert "purge_cost_usd_hr" in cost_breakdown
    assert "compressor_cost_usd_hr" in cost_breakdown
    assert "steam_cost_usd_hr" in cost_breakdown
    assert "total_operating_cost_usd_hr" in cost_breakdown

    # Ensure all components are strictly positive
    assert cost_breakdown["purge_cost_usd_hr"] > 50.0
    assert cost_breakdown["compressor_cost_usd_hr"] > 10.0
    assert cost_breakdown["steam_cost_usd_hr"] > 0.1
    assert cost_breakdown["total_operating_cost_usd_hr"] > 80.0

    # Verify empirical alignment: compressor cost is ~18.30 $/hr (341.5 kW * 0.0536)
    assert cost_breakdown["compressor_cost_usd_hr"] == pytest.approx(341.47 * 0.0536, rel=1e-2)


def test_baseline_energy_evaluation_on_d00():
    """Evaluate empirical energy metrics on normal operation baseline (d00.dat)."""
    df = load_tep_dataset("d00")
    energy_breakdown = evaluate_energy_breakdown(df)

    assert "compressor_power_kw" in energy_breakdown
    assert "steam_thermal_power_kw" in energy_breakdown
    assert "total_energy_power_kw" in energy_breakdown

    # Compressor power should match nominal setpoint within +- 2%
    assert energy_breakdown["compressor_power_kw"] == pytest.approx(341.4, abs=5.0)
    assert energy_breakdown["steam_thermal_power_kw"] > 0.0
    assert energy_breakdown["total_energy_power_kw"] > energy_breakdown["compressor_power_kw"]
