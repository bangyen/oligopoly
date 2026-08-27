"""Test policy subsidy shocks with per-unit validation.

This module tests the subsidy policy shock implementation, ensuring that
per-unit subsidies increase profits by σ*qty for each firm.
"""

import math

import pytest

from sim.policy.policy_shocks import apply_subsidy_shock


def test_subsidy_shock_basic() -> None:
    """Test basic subsidy shock application."""
    base_profits = [100.0, 150.0, 200.0]
    quantities = [10.0, 15.0, 20.0]
    subsidy_per_unit = 5.0

    subsidized_profits = apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)

    # Each firm gets subsidy_per_unit * quantity added to profit
    expected_profits = [150.0, 225.0, 300.0]  # 100+5*10, 150+5*15, 200+5*20

    for i, (actual, expected) in enumerate(zip(subsidized_profits, expected_profits)):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"


def test_subsidy_shock_zero_subsidy() -> None:
    """Test subsidy shock with zero subsidy (no change)."""
    base_profits = [100.0, 150.0, 200.0]
    quantities = [10.0, 15.0, 20.0]
    subsidy_per_unit = 0.0

    subsidized_profits = apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)

    # With σ=0, profits should be unchanged
    assert subsidized_profits == base_profits


def test_subsidy_shock_fractional_quantities() -> None:
    """Test subsidy shock with fractional quantities."""
    base_profits = [100.0, 150.0]
    quantities = [10.5, 15.25]
    subsidy_per_unit = 2.0

    subsidized_profits = apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)

    # 100 + 2*10.5 = 121.0, 150 + 2*15.25 = 180.5
    expected_profits = [121.0, 180.5]

    for i, (actual, expected) in enumerate(zip(subsidized_profits, expected_profits)):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"


def test_subsidy_shock_zero_quantities() -> None:
    """Test subsidy shock with zero quantities (no subsidy)."""
    base_profits = [100.0, 150.0]
    quantities = [0.0, 0.0]
    subsidy_per_unit = 5.0

    subsidized_profits = apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)

    # With zero quantities, profits should be unchanged
    assert subsidized_profits == base_profits


def test_subsidy_shock_negative_subsidy() -> None:
    """Test subsidy shock with negative subsidy (should raise error)."""
    base_profits = [100.0, 150.0]
    quantities = [10.0, 15.0]
    subsidy_per_unit = -5.0

    with pytest.raises(ValueError, match="Subsidy per unit must be non-negative"):
        apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)


def test_subsidy_shock_mismatched_lengths() -> None:
    """Test subsidy shock with mismatched profit and quantity lists."""
    base_profits = [100.0, 150.0]
    quantities = [10.0, 15.0, 20.0]  # Different length
    subsidy_per_unit = 5.0

    with pytest.raises(
        ValueError, match="Profits list length.*must match quantities list length"
    ):
        apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)


def test_subsidy_shock_empty_lists() -> None:
    """Test subsidy shock with empty lists."""
    base_profits: list[float] = []
    quantities: list[float] = []
    subsidy_per_unit = 5.0

    subsidized_profits = apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)

    assert subsidized_profits == []


def test_subsidy_shock_single_firm() -> None:
    """Test subsidy shock with single firm."""
    base_profits = [100.0]
    quantities = [10.0]
    subsidy_per_unit = 3.0

    subsidized_profits = apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)

    expected_profits = [130.0]  # 100 + 3*10
    assert math.isclose(subsidized_profits[0], expected_profits[0], abs_tol=1e-6)


def test_subsidy_shock_large_values() -> None:
    """Test subsidy shock with large values."""
    base_profits = [1000000.0, 2000000.0]
    quantities = [1000.0, 2000.0]
    subsidy_per_unit = 100.0

    subsidized_profits = apply_subsidy_shock(base_profits, quantities, subsidy_per_unit)

    # 1000000 + 100*1000 = 1100000, 2000000 + 100*2000 = 2200000
    expected_profits = [1100000.0, 2200000.0]

    for i, (actual, expected) in enumerate(zip(subsidized_profits, expected_profits)):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"
