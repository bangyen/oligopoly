"""Test cases for demand non-negativity constraint.

This module tests scenarios where the demand function Q(p) = max(0, α - β*p)
returns zero due to high prices.
"""

import pytest

from src.sim.bertrand import bertrand_simulation, calculate_demand


class TestDemandNonNegativity:
    """Test cases for demand non-negativity constraint."""

    def test_demand_zero_when_price_too_high(self) -> None:
        """Test that demand is zero when price exceeds alpha/beta."""
        alpha, beta = 50.0, 2.0
        costs = [10.0, 15.0]
        prices = [
            40.0,
            35.0,
        ]  # Price 40 > alpha/beta = 25, so Q(40) = max(0, 50-2*40) = 0

        result = bertrand_simulation(alpha, beta, costs, prices)

        # All firms should get zero demand
        assert result.quantities[0] == pytest.approx(0.0, abs=1e-6)
        assert result.quantities[1] == pytest.approx(0.0, abs=1e-6)

        # Total demand should be zero
        assert result.total_demand == pytest.approx(0.0, abs=1e-6)

        # All profits should be zero
        assert result.profits[0] == pytest.approx(0.0, abs=1e-6)
        assert result.profits[1] == pytest.approx(0.0, abs=1e-6)

    def test_demand_boundary_case(self) -> None:
        """Test demand at the boundary where Q = 0."""
        alpha, beta = 100.0, 1.0
        costs = [20.0, 25.0]
        prices = [100.0, 105.0]  # At boundary: Q(100) = max(0, 100-1*100) = 0

        result = bertrand_simulation(alpha, beta, costs, prices)

        assert result.total_demand == pytest.approx(0.0, abs=1e-6)
        assert all(q == pytest.approx(0.0, abs=1e-6) for q in result.quantities)
        assert all(pi == pytest.approx(0.0, abs=1e-6) for pi in result.profits)

    def test_demand_just_above_zero(self) -> None:
        """Test demand just above zero boundary."""
        alpha, beta = 100.0, 1.0
        costs = [20.0, 25.0]
        prices = [
            99.9,
            105.0,
        ]  # Just below boundary: Q(99.9) = max(0, 100-1*99.9) = 0.1

        result = bertrand_simulation(alpha, beta, costs, prices)

        expected_demand = 100.0 - 1.0 * 99.9  # 0.1
        assert result.total_demand == pytest.approx(expected_demand, abs=1e-6)
        assert result.quantities[0] == pytest.approx(expected_demand, abs=1e-6)
        assert result.quantities[1] == pytest.approx(0.0, abs=1e-6)

    def test_calculate_demand_function(self) -> None:
        """Test the calculate_demand helper function directly."""
        # Normal case
        assert calculate_demand(100.0, 1.0, 20.0) == pytest.approx(80.0, abs=1e-6)

        # Boundary case
        assert calculate_demand(100.0, 1.0, 100.0) == pytest.approx(0.0, abs=1e-6)

        # Negative demand case (should return 0)
        assert calculate_demand(100.0, 1.0, 150.0) == pytest.approx(0.0, abs=1e-6)

        # Edge case with decimal precision
        assert calculate_demand(50.0, 2.0, 25.0) == pytest.approx(0.0, abs=1e-6)
        assert calculate_demand(50.0, 2.0, 24.9) == pytest.approx(0.2, abs=1e-6)
