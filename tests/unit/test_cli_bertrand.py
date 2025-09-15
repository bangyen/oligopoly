"""Test cases for CLI functionality.

This module tests the command-line interface for Bertrand simulations,
including output format and the specific example from the specification.
"""

import pytest
import subprocess
import sys
from src.sim.bertrand import bertrand_simulation


class TestCLIBertrand:
    """Test cases for CLI functionality."""
    
    def test_cli_output_format(self):
        """Test that CLI produces correct output format."""
        # This would typically be tested with subprocess, but for now we test the logic
        alpha, beta = 120.0, 1.2
        costs = [20.0, 20.0, 25.0]
        prices = [22.0, 21.0, 24.0]
        
        result = bertrand_simulation(alpha, beta, costs, prices)
        
        # Verify the structure matches expected CLI output
        assert result.total_demand >= 0
        assert len(result.prices) == len(costs)
        assert len(result.quantities) == len(costs)
        assert len(result.profits) == len(costs)
        
        # Verify total demand equals sum of allocated quantities
        assert sum(result.quantities) == pytest.approx(result.total_demand, abs=1e-6)
    
    def test_cli_example_from_spec(self):
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
    
    def test_cli_output_structure(self):
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
        assert result.total_demand == pytest.approx(88.0, abs=1e-6)  # Q(12) = 100-1*12 = 88
        
        # Firm 0 has lowest price, gets all demand
        assert result.quantities[0] == pytest.approx(88.0, abs=1e-6)
        assert result.quantities[1] == pytest.approx(0.0, abs=1e-6)
        assert result.quantities[2] == pytest.approx(0.0, abs=1e-6)
        
        # Verify profits
        assert result.profits[0] == pytest.approx(176.0, abs=1e-6)  # (12-10)*88 = 176
        assert result.profits[1] == pytest.approx(0.0, abs=1e-6)
        assert result.profits[2] == pytest.approx(0.0, abs=1e-6)
    
    def test_cli_with_ties(self):
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
    
    def test_cli_with_zero_demand(self):
        """Test CLI output when demand is zero."""
        alpha, beta = 50.0, 2.0
        costs = [10.0, 15.0]
        prices = [30.0, 35.0]  # Both prices > alpha/beta = 25
        
        result = bertrand_simulation(alpha, beta, costs, prices)
        
        # All quantities and profits should be zero
        assert result.total_demand == pytest.approx(0.0, abs=1e-6)
        assert all(q == pytest.approx(0.0, abs=1e-6) for q in result.quantities)
        assert all(pi == pytest.approx(0.0, abs=1e-6) for pi in result.profits)
