"""Test cases for CLI functionality.

This module tests the command-line interface for Bertrand simulations,
including output format and the specific example from the specification.
"""

import pytest

from src.sim.games.bertrand import bertrand_simulation
from tests.utils import (
    assert_bertrand_cli_format,
    assert_bertrand_output_format,
    create_sample_bertrand_cli_config,
    create_sample_bertrand_config,
)


class TestCLIBertrand:
    """Test cases for CLI functionality."""

    def test_cli_output_format(self) -> None:
        """Test that CLI produces correct output format."""
        config = create_sample_bertrand_config()

        result = bertrand_simulation(
            config["alpha"], config["beta"], config["costs"], config["prices"]
        )

        assert_bertrand_output_format(result, len(config["costs"]))

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
        config = create_sample_bertrand_cli_config()

        result = bertrand_simulation(
            config["alpha"], config["beta"], config["costs"], config["prices"]
        )

        assert_bertrand_cli_format(
            result,
            config["expected_demand"],
            config["expected_quantities"],
            config["expected_profits"],
        )

    def test_cli_with_ties(self) -> None:
        """Test CLI output format when firms tie for lowest price."""
        alpha, beta = 80.0, 0.8
        costs = [15.0, 15.0, 20.0]
        prices = [18.0, 18.0, 22.0]  # Firms 0 and 1 tie

        result = bertrand_simulation(alpha, beta, costs, prices)

        # Both firms should split demand equally
        expected_demand_per_firm = (80.0 - 0.8 * 18.0) / 2  # 65.6 / 2 = 32.8
        assert result.quantities[0] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[1] == pytest.approx(expected_demand_per_firm, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)

        # Total demand should equal sum of allocated quantities
        total_demand = 80.0 - 0.8 * 18.0  # 65.6
        assert result.total_demand == pytest.approx(total_demand, abs=1e-6)
        assert sum(result.quantities) == pytest.approx(total_demand, abs=1e-6)

    def test_cli_with_zero_demand(self) -> None:
        """Test CLI output when demand is zero."""
        alpha, beta = 50.0, 2.0
        costs = [10.0, 15.0]
        prices = [30.0, 35.0]  # Both prices > alpha/beta = 25

        result = bertrand_simulation(alpha, beta, costs, prices)

        # All quantities and profits should be zero
        assert result.total_demand == pytest.approx(0.0, abs=1e-6)
        assert all(q == pytest.approx(0.0, abs=1e-6) for q in result.quantities)
        assert all(pi == pytest.approx(0.0, abs=1e-6) for pi in result.profits)
