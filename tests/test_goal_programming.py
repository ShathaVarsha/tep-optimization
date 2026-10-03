"""Automated test suite for Phase 6: Goal Programming (Weighted-Sum and Preemptive).

Tests verify:
  1. GoalSpec dataclass validation.
  2. Weighted-Sum GP: feasibility, deviation variables, goal achievement structure.
  3. Preemptive GP: lexicographic ordering, priority sequence, frozen level constraints.
  4. Engineering safety constraints are respected in both formulations.
  5. Sensitivity of total deviation to weight changes.
  6. Format report generation.

Author: Team B4 (23MNG336: Operational Research)
"""

import pytest
import numpy as np
import pandas as pd

from src.data.tep_loader import load_tep_dataset
from src.models.surrogate_models import ProcessSurrogateModel, DECISION_FEATURES
from src.optimization.goal_programming import (
    GoalSpec,
    GPResult,
    solve_weighted_sum_gp,
    solve_preemptive_gp,
    format_gp_report,
    build_default_tep_goals,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gp_setup():
    """Load baseline data and fit linear surrogate once for all GP tests."""
    df = load_tep_dataset("d00")
    model = ProcessSurrogateModel(model_type="linear").fit(df)
    u_base = df[DECISION_FEATURES].mean().to_numpy()
    return model, u_base


@pytest.fixture(scope="module")
def default_goals():
    """Return canonical TEP Mode 1 three-goal set."""
    return build_default_tep_goals()


# ---------------------------------------------------------------------------
# 1. GoalSpec Validation
# ---------------------------------------------------------------------------

class TestGoalSpec:
    def test_valid_goal_creation(self):
        g = GoalSpec(name="Test", response_tag="XMEAS_17", target=22.89, direction="max", weight=1.0, priority=1)
        assert g.name == "Test"
        assert g.direction == "max"
        assert g.priority == 1

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            GoalSpec(name="Bad", response_tag="XMEAS_17", target=22.89, direction="above")

    def test_zero_target_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            GoalSpec(name="Bad", response_tag="XMEAS_17", target=0.0, direction="max")

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="weight must be"):
            GoalSpec(name="Bad", response_tag="XMEAS_17", target=22.89, direction="max", weight=-0.1)

    def test_all_directions_accepted(self):
        for d in ("min", "max", "exact"):
            g = GoalSpec(name="G", response_tag="XMEAS_20", target=341.4, direction=d)
            assert g.direction == d


# ---------------------------------------------------------------------------
# 2. Weighted-Sum Goal Programming
# ---------------------------------------------------------------------------

class TestWeightedSumGP:
    def test_ws_gp_returns_gpresult(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert isinstance(result, GPResult)

    def test_ws_gp_formulation_label(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert result.formulation == "Weighted-Sum"

    def test_ws_gp_solver_succeeds(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert result.success is True, f"Solver failed: {result.status_message}"

    def test_ws_gp_goal_achievement_count(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert len(result.goal_achievement) == len(default_goals)

    def test_ws_gp_goal_achievement_schema(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        for row in result.goal_achievement:
            assert "Goal Name" in row
            assert "Target (g_k)" in row
            assert "Achieved Value" in row
            assert "Deviation d_neg" in row
            assert "Deviation d_pos" in row
            assert "Status" in row
            # Deviation variables must be non-negative
            assert row["Deviation d_neg"] >= -1e-6
            assert row["Deviation d_pos"] >= -1e-6

    def test_ws_gp_decision_variables_within_trust_region(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base, trust_delta=10.0)
        for feat, opt_val in result.u_optimal.items():
            base_val = result.u_baseline[feat]
            assert abs(opt_val - base_val) <= 10.0 + 1e-3, (
                f"{feat}: optimal={opt_val:.3f} deviates from baseline={base_val:.3f} by "
                f"{abs(opt_val - base_val):.3f} (exceeds trust_delta=10.0)"
            )

    def test_ws_gp_safety_constraints_respected(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base,
                                       max_pressure=2800.0, max_compressor_power=355.0)
        preds = result.predicted_responses
        assert preds["XMEAS_07"] <= 2800.0 + 1e-2, (
            f"Reactor pressure {preds['XMEAS_07']:.1f} kPa exceeds safety limit 2800 kPa"
        )
        assert preds["XMEAS_20"] <= 355.0 + 1e-2, (
            f"Compressor power {preds['XMEAS_20']:.1f} kW exceeds safety limit 355 kW"
        )

    def test_ws_gp_total_deviation_non_negative(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert result.total_weighted_deviation >= 0.0

    def test_ws_gp_no_priority_sequence(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert result.priority_sequence == []

    def test_ws_gp_iterations_recorded(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert result.iterations >= 0

    def test_ws_gp_n_deviation_vars(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        assert result.n_deviation_vars == 2 * len(default_goals)

    def test_ws_gp_higher_weight_reduces_deviation(self, gp_setup):
        """Increasing the weight on a goal should reduce (not increase) its deviation."""
        model, u_base = gp_setup

        goals_low = [
            GoalSpec(name="Production", response_tag="XMEAS_17", target=22.89, direction="max", weight=0.1, priority=1),
            GoalSpec(name="Comp Power", response_tag="XMEAS_20", target=324.3, direction="min", weight=1.0, priority=2),
        ]
        goals_high = [
            GoalSpec(name="Production", response_tag="XMEAS_17", target=22.89, direction="max", weight=5.0, priority=1),
            GoalSpec(name="Comp Power", response_tag="XMEAS_20", target=324.3, direction="min", weight=1.0, priority=2),
        ]

        r_low = solve_weighted_sum_gp(model, goals_low, u_base)
        r_high = solve_weighted_sum_gp(model, goals_high, u_base)

        if r_low.success and r_high.success:
            dev_low = r_low.goal_achievement[0]["Deviation d_neg"]  # XMEAS_17 under-achievement
            dev_high = r_high.goal_achievement[0]["Deviation d_neg"]
            # Higher weight should achieve equal or smaller deviation for that goal
            assert dev_high <= dev_low + 1e-3, (
                f"Higher weight did not reduce deviation: low={dev_low:.4f}, high={dev_high:.4f}"
            )


# ---------------------------------------------------------------------------
# 3. Preemptive Goal Programming
# ---------------------------------------------------------------------------

class TestPreemptiveGP:
    def test_preemptive_gp_returns_gpresult(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        assert isinstance(result, GPResult)

    def test_preemptive_gp_formulation_label(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        assert result.formulation == "Preemptive"

    def test_preemptive_gp_solver_succeeds(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        assert result.success is True, f"Preemptive GP failed: {result.status_message}"

    def test_preemptive_gp_priority_sequence_length(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        n_levels = len(set(g.priority for g in default_goals))
        assert len(result.priority_sequence) == n_levels

    def test_preemptive_gp_priority_sequence_non_negative(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        for z in result.priority_sequence:
            assert z >= 0.0, f"Priority objective value {z} is negative"

    def test_preemptive_gp_goal_achievement_count(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        assert len(result.goal_achievement) == len(default_goals)

    def test_preemptive_gp_safety_constraints_respected(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base,
                                     max_pressure=2800.0, max_compressor_power=355.0)
        preds = result.predicted_responses
        assert preds["XMEAS_07"] <= 2800.0 + 1e-2
        assert preds["XMEAS_20"] <= 355.0 + 1e-2

    def test_preemptive_gp_decision_variables_within_trust_region(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base, trust_delta=10.0)
        for feat, opt_val in result.u_optimal.items():
            base_val = result.u_baseline[feat]
            assert abs(opt_val - base_val) <= 10.0 + 1e-3

    def test_preemptive_gp_status_contains_priority_levels(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        # Status message should mention each priority level
        n_levels = len(set(g.priority for g in default_goals))
        for p in range(1, n_levels + 1):
            assert f"P{p}:" in result.status_message

    def test_preemptive_gp_p1_deviation_not_worsened_by_p2(self, gp_setup):
        """Core preemptive property: P2 solve must not worsen P1 achievement."""
        model, u_base = gp_setup
        goals = [
            GoalSpec("Production", "XMEAS_17", 22.89, "max", weight=2.0, priority=1),
            GoalSpec("Comp Power", "XMEAS_20", 324.3, "min", weight=1.5, priority=2),
        ]
        result = solve_preemptive_gp(model, goals, u_base)
        assert result.success

        # P1 level objective value
        z1 = result.priority_sequence[0]

        # Manually solve only P1 to get its standalone optimum
        goals_p1_only = [goals[0]]
        r_p1 = solve_weighted_sum_gp(model, goals_p1_only, u_base)

        if r_p1.success:
            z1_standalone = r_p1.total_weighted_deviation
            # The preemptive P1 objective must be within tolerance of the standalone optimum
            assert z1 <= z1_standalone + 1e-3, (
                f"Preemptive P1 z={z1:.6f} worse than standalone P1 z={z1_standalone:.6f}"
            )

    def test_preemptive_vs_weighted_sum_both_succeed(self, gp_setup, default_goals):
        """Both formulations should produce successful solves for the canonical goal set."""
        model, u_base = gp_setup
        r_ws = solve_weighted_sum_gp(model, default_goals, u_base)
        r_pre = solve_preemptive_gp(model, default_goals, u_base)
        assert r_ws.success
        assert r_pre.success


# ---------------------------------------------------------------------------
# 4. Default TEP Goal Set
# ---------------------------------------------------------------------------

class TestDefaultGoalSet:
    def test_default_goals_count(self, default_goals):
        assert len(default_goals) == 3

    def test_default_goals_tags(self, default_goals):
        tags = {g.response_tag for g in default_goals}
        assert "XMEAS_17" in tags
        assert "XMEAS_20" in tags
        assert "XMEAS_10" in tags

    def test_default_goals_priorities(self, default_goals):
        priorities = [g.priority for g in default_goals]
        assert 1 in priorities
        assert 2 in priorities

    def test_default_goals_targets_from_downs_vogel(self, default_goals):
        prod_goal = next(g for g in default_goals if g.response_tag == "XMEAS_17")
        assert prod_goal.target == pytest.approx(22.89, abs=0.01)
        comp_goal = next(g for g in default_goals if g.response_tag == "XMEAS_20")
        assert comp_goal.target == pytest.approx(324.3, abs=0.5)


# ---------------------------------------------------------------------------
# 5. Report Formatting
# ---------------------------------------------------------------------------

class TestGPReporting:
    def test_ws_gp_report_generation(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        report = format_gp_report(result)
        assert isinstance(report, str)
        assert "GOAL PROGRAMMING REPORT" in report
        assert "Weighted-Sum" in report
        assert "GOAL ACHIEVEMENT SUMMARY" in report
        assert "OPTIMAL DECISION VARIABLE SETPOINTS" in report

    def test_preemptive_gp_report_generation(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_preemptive_gp(model, default_goals, u_base)
        report = format_gp_report(result)
        assert "Preemptive" in report
        assert "Priority Sequence Objective Values" in report

    def test_report_contains_all_decision_variables(self, gp_setup, default_goals):
        model, u_base = gp_setup
        result = solve_weighted_sum_gp(model, default_goals, u_base)
        report = format_gp_report(result)
        for feat in DECISION_FEATURES:
            assert feat in report, f"{feat} missing from GP report"
