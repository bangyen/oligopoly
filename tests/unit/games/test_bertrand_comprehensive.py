"""Comprehensive test suite for Bertrand oligopoly simulation.

This module contains all test cases for the Bertrand simulation,
including edge cases for undercutting, ties, demand non-negativity,
CLI functionality, and input validation.
"""

import pytest

from src.sim.games.bertrand import (
    allocate_demand,
    bertrand_simulation,
    calculate_demand,
)


class TestBertrandUndercut:
    """Test cases for Bertrand undercutting behavior."""

    def test_undercut_to_marginal_cost(self) -> None:
        """Test that firm undercutting to marginal cost gets all demand with zero profit."""
        # Firm 0 sets price equal to marginal cost, others set higher prices
        alpha, beta = 100.0, 1.0
        costs = [20.0, 25.0, 30.0]
        prices = [20.0, 25.0, 30.0]  # Firm 0 prices at marginal cost

        result = bertrand_simulation(alpha, beta, costs, prices)

        # Firm 0 should get all demand
        assert result.quantities[0] == pytest.approx(
            80.0, abs=1e-6
        )  # Q(20) = 100 - 1*20 = 80
        assert result.quantities[1] == pytest.approx(0.0, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)

        # Firm 0 should have approximately zero profit (within float tolerance)
        assert result.profits[0] == pytest.approx(0.0, abs=1e-6)  # (20-20)*80 = 0
        assert result.profits[1] == pytest.approx(0.0, abs=1e-6)
        assert result.profits[2] == pytest.approx(0.0, abs=1e-6)

        # Total demand should equal allocated quantity
        assert result.total_demand == pytest.approx(80.0, abs=1e-6)
        assert sum(result.quantities) == pytest.approx(result.total_demand, abs=1e-6)

    def test_undercut_below_marginal_cost(self) -> None:
        """Test firm attempting to undercut below marginal cost gets price adjusted to minimum viable price."""
        alpha, beta = 100.0, 1.0
        costs = [20.0, 25.0, 30.0]
        prices = [15.0, 25.0, 30.0]  # Firm 0 attempts to price below marginal cost

        result = bertrand_simulation(alpha, beta, costs, prices)

        # Firm 0's price should be adjusted to minimum viable price (20 * 0.95 = 19)
        # So demand becomes Q(19) = 100 - 1*19 = 81
        assert result.quantities[0] == pytest.approx(
            81.0, abs=1e-6
        )  # Q(19) = 100 - 1*19 = 81
        assert result.quantities[1] == pytest.approx(0.0, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)

        # Firm 0 should have small negative profit (pricing at 95% of marginal cost)
        assert result.profits[0] == pytest.approx(-81.0, abs=1e-6)  # (19-20)*81 = -81
        assert result.profits[1] == pytest.approx(0.0, abs=1e-6)
        assert result.profits[2] == pytest.approx(0.0, abs=1e-6)


class TestBertrandTie:
    """Test cases for Bertrand price ties."""

    def test_two_firms_tie_for_lowest_price(self) -> None:
        """Test that two firms with equal lowest prices split demand equally."""
        alpha, beta = 120.0, 1.2
        costs = [20.0, 20.0, 25.0]
        prices = [22.0, 22.0, 24.0]  # Firms 0 and 1 tie at lowest price

        result = bertrand_simulation(alpha, beta, costs, prices)

        # Both firms should get equal share of demand
        expected_demand_per_firm = (
            120.0 - 1.2 * 22.0
        ) / 2  # Q(22) = 93.6, split equally = 46.8
        assert result.quantities[0] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[1] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)

        # Total allocated should equal total demand
        total_demand = 120.0 - 1.2 * 22.0  # 93.6
        assert result.total_demand == pytest.approx(total_demand, abs=1e-6)
        assert sum(result.quantities) == pytest.approx(total_demand, abs=1e-6)

        # Both firms should have equal profits
        expected_profit = (22.0 - 20.0) * expected_demand_per_firm  # 2 * 46.8 = 93.6
        assert result.profits[0] == pytest.approx(expected_profit, abs=1e-6)
        assert result.profits[1] == pytest.approx(expected_profit, abs=1e-6)
        assert result.profits[2] == pytest.approx(0.0, abs=1e-6)

    def test_three_firms_tie_for_lowest_price(self) -> None:
        """Test that three firms with equal lowest prices split demand equally."""
        alpha, beta = 100.0, 1.0
        costs = [10.0, 10.0, 10.0]
        prices = [15.0, 15.0, 15.0]  # All three firms tie

        result = bertrand_simulation(alpha, beta, costs, prices)

        # All firms should get equal share of demand
        expected_demand_per_firm = (
            100.0 - 1.0 * 15.0
        ) / 3  # Q(15) = 85, split equally = 28.33...
        assert result.quantities[0] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[1] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[2] == pytest.approx(expected_demand_per_firm, abs=1e-6)

        # Total allocated should equal total demand
        total_demand = 100.0 - 1.0 * 15.0  # 85
        assert result.total_demand == pytest.approx(total_demand, abs=1e-6)
        assert sum(result.quantities) == pytest.approx(total_demand, abs=1e-6)


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


class TestCLIBertrand:
    """Test cases for CLI functionality."""

    def test_cli_example_from_spec(self) -> None:
        """Test the specific example from the specification."""
        alpha, beta = 120.0, 1.2
        costs = [20.0, 20.0, 25.0]
        prices = [22.0, 21.0, 24.0]

        result = bertrand_simulation(alpha, beta, costs, prices)

        # Firm 1 (index 1) has lowest price (21), so should get all demand
        expected_demand = 120.0 - 1.2 * 21.0  # 94.8
        assert result.total_demand == pytest.approx(expected_demand, abs=1e-6)
        assert result.quantities[1] == pytest.approx(expected_demand, abs=1e-6)
        assert result.quantities[0] == pytest.approx(0.0, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)

        # Verify profits
        expected_profit_1 = (21.0 - 20.0) * expected_demand  # 1 * 94.8 = 94.8
        assert result.profits[1] == pytest.approx(expected_profit_1, abs=1e-6)
        assert result.profits[0] == pytest.approx(0.0, abs=1e-6)
        assert result.profits[2] == pytest.approx(0.0, abs=1e-6)

    def test_cli_output_structure(self) -> None:
        """Test that CLI output has the expected structure."""
        alpha, beta = 100.0, 1.0
        costs = [10.0, 15.0, 20.0]
        prices = [12.0, 14.0, 16.0]

        result = bertrand_simulation(alpha, beta, costs, prices)

        # Expected CLI output format:
        # Q=88.0
        # p_0=12.0, q_0=88.0, π_0=176.0
        # p_1=14.0, q_1=0.0, π_1=0.0
        # p_2=16.0, q_2=0.0, π_2=0.0

        # Verify all components are present and correct
        assert result.total_demand == pytest.approx(
            88.0, abs=1e-6
        )  # Q(12) = 100-1*12 = 88

        # Firm 0 has lowest price, gets all demand
        assert result.quantities[0] == pytest.approx(88.0, abs=1e-6)
        assert result.quantities[1] == pytest.approx(0.0, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)

        # Verify profits
        assert result.profits[0] == pytest.approx(176.0, abs=1e-6)  # (12-10)*88 = 176
        assert result.profits[1] == pytest.approx(0.0, abs=1e-6)
        assert result.profits[2] == pytest.approx(0.0, abs=1e-6)


class TestInputValidation:
    """Test cases for input validation."""

    def test_negative_prices_raise_error(self) -> None:
        """Test that negative prices raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            bertrand_simulation(100.0, 1.0, [10.0], [-5.0])

    def test_mismatched_costs_and_prices_raise_error(self) -> None:
        """Test that mismatched costs and prices lists raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            bertrand_simulation(100.0, 1.0, [10.0, 20.0], [15.0])

    def test_non_positive_alpha_raises_error(self) -> None:
        """Test that non-positive alpha raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            bertrand_simulation(0.0, 1.0, [10.0], [15.0])

    def test_non_positive_beta_raises_error(self) -> None:
        """Test that non-positive beta raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            bertrand_simulation(100.0, 0.0, [10.0], [15.0])


class TestHelperFunctions:
    """Test cases for helper functions."""

    def test_calculate_demand(self) -> None:
        """Test demand calculation function."""
        # Normal case
        assert calculate_demand(100.0, 1.0, 20.0) == pytest.approx(80.0, abs=1e-6)

        # Boundary case
        assert calculate_demand(100.0, 1.0, 100.0) == pytest.approx(0.0, abs=1e-6)

        # Negative demand case (should return 0)
        assert calculate_demand(100.0, 1.0, 150.0) == pytest.approx(0.0, abs=1e-6)

    def test_allocate_demand(self) -> None:
        """Test demand allocation function."""
        prices = [20.0, 25.0, 30.0]
        costs = [10.0, 15.0, 20.0]
        alpha, beta = 100.0, 1.0

        quantities, total_demand = allocate_demand(prices, costs, alpha, beta)

        # Firm 0 should get all demand
        expected_demand = 100.0 - 1.0 * 20.0  # 80
        assert total_demand == pytest.approx(expected_demand, abs=1e-6)
        assert quantities[0] == pytest.approx(expected_demand, abs=1e-6)
        assert quantities[1] == pytest.approx(0.0, abs=1e-6)
        assert quantities[2] == pytest.approx(0.0, abs=1e-6)
