"""Test cases for Bertrand price ties.

This module tests scenarios where multiple firms set the same lowest price
and must split the market demand equally.
"""

import pytest

from src.sim.bertrand import bertrand_simulation


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

    def test_tie_with_different_costs(self) -> None:
        """Test tie scenario where firms have different costs but same prices."""
        alpha, beta = 80.0, 0.8
        costs = [15.0, 20.0, 25.0]
        prices = [18.0, 18.0, 22.0]  # Firms 0 and 1 tie at lowest price

        result = bertrand_simulation(alpha, beta, costs, prices)

        # Both firms should get equal share of demand
        expected_demand_per_firm = (
            80.0 - 0.8 * 18.0
        ) / 2  # Q(18) = 65.6, split equally = 32.8
        assert result.quantities[0] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[1] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)

        # Profits should differ based on costs
        expected_profit_0 = (18.0 - 15.0) * expected_demand_per_firm  # 3 * 32.8 = 98.4
        expected_profit_1 = (
            18.0 - 20.0
        ) * expected_demand_per_firm  # -2 * 32.8 = -65.6
        assert result.profits[0] == pytest.approx(expected_profit_0, abs=1e-6)
        assert result.profits[1] == pytest.approx(expected_profit_1, abs=1e-6)
        assert result.profits[2] == pytest.approx(0.0, abs=1e-6)
