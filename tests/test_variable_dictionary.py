"""Automated test suite for TEP Variable Dictionary, Engineering Units, and Bounds.

Author: Team B4 (23MNG336: Operational Research)
"""

import pytest
import pandas as pd

from src.process.variable_dictionary import (
    VARIABLE_CATALOG,
    ProcessVariable,
    get_variable_metadata,
    get_catalog_dataframe,
)
from src.data.tep_loader import TEP_COLUMN_NAMES, MEASURED_VARS, MANIPULATED_VARS


def test_catalog_completeness():
    """Verify that all 52 variables are mapped in the catalog without missing entries."""
    assert len(VARIABLE_CATALOG) == 52
    for col in TEP_COLUMN_NAMES:
        assert col in VARIABLE_CATALOG, f"Column '{col}' missing from variable catalog"


def test_bounds_validity():
    """Verify that every variable has strictly lower < upper bounds."""
    for tag, var in VARIABLE_CATALOG.items():
        assert var.min_bound < var.max_bound, f"Variable {tag} has invalid bounds: [{var.min_bound}, {var.max_bound}]"


def test_nominal_within_bounds():
    """Verify that every documented nominal setpoint lies strictly within operating bounds."""
    for tag, var in VARIABLE_CATALOG.items():
        assert (
            var.min_bound <= var.nominal_setpoint <= var.max_bound
        ), f"Variable {tag} nominal {var.nominal_setpoint} is outside [{var.min_bound}, {var.max_bound}]"


def test_manipulated_variable_bounds():
    """Verify that all 11 manipulated actuator valves have span [0, 100]%."""
    for tag in MANIPULATED_VARS:
        var = VARIABLE_CATALOG[tag]
        assert var.is_manipulated is True
        assert var.units == "%"
        assert var.min_bound == 0.0
        assert var.max_bound == 100.0


def test_critical_safety_limits_downs_vogel():
    """Verify that critical reactor safety limits match Downs & Vogel (1993) Table 1."""
    p_reactor = VARIABLE_CATALOG["XMEAS_07"]
    assert p_reactor.units == "kPa gauge"
    assert p_reactor.high_alarm == 2895.0
    assert p_reactor.high_shutdown == 3000.0

    t_reactor = VARIABLE_CATALOG["XMEAS_09"]
    assert t_reactor.units == "Deg C"
    assert t_reactor.high_alarm == 175.0

    l_reactor = VARIABLE_CATALOG["XMEAS_08"]
    assert l_reactor.low_alarm == 30.0
    assert l_reactor.high_alarm == 80.0
    assert l_reactor.low_shutdown == 20.0
    assert l_reactor.high_shutdown == 90.0

    # Compressor Work variable verification
    p_comp = VARIABLE_CATALOG["XMEAS_20"]
    assert p_comp.name == "Compressor Work"
    assert p_comp.units == "kW"
    assert p_comp.nominal_setpoint == 341.4


def test_dataframe_export():
    """Verify that the variable catalog converts cleanly to a DataFrame for UI display."""
    df = get_catalog_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 52
    assert "Tag" in df.columns
    assert "Units" in df.columns
    assert "Nominal" in df.columns
    assert "High Alarm" in df.columns
