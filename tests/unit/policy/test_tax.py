"""Test policy tax shocks with τ=0.2 validation.

This module tests the tax policy shock implementation, ensuring that
with a tax rate of 0.2, profits are reduced to 80% of their base value.
"""

import math

import pytest

from sim.policy.policy_shocks import apply_tax_shock


def test_tax_shock_basic() -> None:
    """Test basic tax shock application with τ=0.2."""
    base_profits = [100.0, 150.0, 200.0]
    tax_rate = 0.2

    taxed_profits = apply_tax_shock(base_profits, tax_rate)

    # With τ=0.2, profits should be 80% of base
    expected_profits = [80.0, 120.0, 160.0]

    for i, (actual, expected) in enumerate(zip(taxed_profits, expected_profits)):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"


def test_tax_shock_zero_rate() -> None:
    """Test tax shock with zero tax rate (no change)."""
    base_profits = [100.0, 150.0, 200.0]
    tax_rate = 0.0

    taxed_profits = apply_tax_shock(base_profits, tax_rate)

    # With τ=0, profits should be unchanged
    assert taxed_profits == base_profits


def test_tax_shock_high_rate() -> None:
    """Test tax shock with high tax rate."""
    base_profits = [100.0, 150.0, 200.0]
    tax_rate = 0.5

    taxed_profits = apply_tax_shock(base_profits, tax_rate)

    # With τ=0.5, profits should be 50% of base
    expected_profits = [50.0, 75.0, 100.0]

    for i, (actual, expected) in enumerate(zip(taxed_profits, expected_profits)):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"


def test_tax_shock_invalid_rate() -> None:
    """Test tax shock with invalid tax rates."""
    base_profits = [100.0, 150.0, 200.0]

    # Negative tax rate should raise ValueError
    with pytest.raises(ValueError, match="Tax rate must be in"):
        apply_tax_shock(base_profits, -0.1)

    # Tax rate >= 1.0 should raise ValueError
    with pytest.raises(ValueError, match="Tax rate must be in"):
        apply_tax_shock(base_profits, 1.0)

    with pytest.raises(ValueError, match="Tax rate must be in"):
        apply_tax_shock(base_profits, 1.5)


def test_tax_shock_empty_profits() -> None:
    """Test tax shock with empty profits list."""
    base_profits: list[float] = []
    tax_rate = 0.2

    taxed_profits = apply_tax_shock(base_profits, tax_rate)

    assert taxed_profits == []


def test_tax_shock_single_firm() -> None:
    """Test tax shock with single firm."""
    base_profits = [100.0]
    tax_rate = 0.2

    taxed_profits = apply_tax_shock(base_profits, tax_rate)

    expected_profits = [80.0]
    assert math.isclose(taxed_profits[0], expected_profits[0], abs_tol=1e-6)


def test_tax_shock_precision() -> None:
    """Test tax shock with floating point precision."""
    base_profits = [33.333333, 66.666667]
    tax_rate = 0.2

    taxed_profits = apply_tax_shock(base_profits, tax_rate)

    # Should handle floating point arithmetic correctly
    expected_profits = [26.666666, 53.333334]

    for i, (actual, expected) in enumerate(zip(taxed_profits, expected_profits)):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"
