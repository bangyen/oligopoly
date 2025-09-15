"""Tests for Cournot input validation functionality.

This module tests the input validation logic in the Cournot simulation,
ensuring that invalid inputs are properly rejected with meaningful error messages.
"""

import pytest
from src.sim.cournot import (
    cournot_simulation, 
    validate_quantities, 
    parse_costs, 
    parse_quantities
)


class TestCournotInputValidation:
    """Test cases for input validation in Cournot simulation."""
    
    def test_negative_quantities_raise_error(self) -> None:
        """Test that negative quantities raise ValueError."""
        with pytest.raises(ValueError, match="Quantity q_0 = -5.0 must be non-negative"):
            cournot_simulation(a=100, b=1, costs=[10], quantities=[-5])
            
    def test_multiple_negative_quantities(self) -> None:
        """Test validation with multiple negative quantities."""
        with pytest.raises(ValueError, match="Quantity q_1 = -10.0 must be non-negative"):
            cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[5, -10])
            
    def test_zero_quantities_allowed(self) -> None:
        """Test that zero quantities are allowed."""
        result = cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[0, 0])
        assert result.price == 100.0
        assert result.profits[0] == 0.0  # (100 - 10) * 0 = 0
        assert result.profits[1] == 0.0  # (100 - 20) * 0 = 0
        
    def test_mismatched_costs_quantities_lengths(self) -> None:
        """Test error when costs and quantities lists have different lengths."""
        with pytest.raises(ValueError, match="Costs list length \\(2\\) must match quantities list length \\(3\\)"):
            cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[10, 20, 30])
            
    def test_parse_costs_valid_format(self) -> None:
        """Test parsing valid costs string."""
        costs = parse_costs("10,20,30")
        assert costs == [10.0, 20.0, 30.0]
        
    def test_parse_costs_with_spaces(self) -> None:
        """Test parsing costs string with spaces."""
        costs = parse_costs(" 10 , 20 , 30 ")
        assert costs == [10.0, 20.0, 30.0]
        
    def test_parse_costs_invalid_format(self) -> None:
        """Test parsing invalid costs string."""
        with pytest.raises(ValueError, match="Invalid costs format '10,abc,30'"):
            parse_costs("10,abc,30")
            
    def test_parse_costs_empty_string(self) -> None:
        """Test parsing empty costs string."""
        with pytest.raises(ValueError, match="Costs list cannot be empty"):
            parse_costs("")
            
    def test_parse_costs_single_value(self) -> None:
        """Test parsing single cost value."""
        costs = parse_costs("15.5")
        assert costs == [15.5]
        
    def test_parse_quantities_valid_format(self) -> None:
        """Test parsing valid quantities string."""
        quantities = parse_quantities("10,20,30")
        assert quantities == [10.0, 20.0, 30.0]
        
    def test_parse_quantities_with_spaces(self) -> None:
        """Test parsing quantities string with spaces."""
        quantities = parse_quantities(" 10 , 20 , 30 ")
        assert quantities == [10.0, 20.0, 30.0]
        
    def test_parse_quantities_invalid_format(self) -> None:
        """Test parsing invalid quantities string."""
        with pytest.raises(ValueError, match="Invalid quantities format '10,xyz,30'"):
            parse_quantities("10,xyz,30")
            
    def test_parse_quantities_empty_string(self) -> None:
        """Test parsing empty quantities string."""
        with pytest.raises(ValueError, match="Quantities list cannot be empty"):
            parse_quantities("")
            
    def test_parse_quantities_single_value(self) -> None:
        """Test parsing single quantity value."""
        quantities = parse_quantities("25.5")
        assert quantities == [25.5]
        
    def test_validate_quantities_function(self) -> None:
        """Test the validate_quantities function directly."""
        # Valid quantities should not raise
        validate_quantities([0, 10, 20.5])
        
        # Negative quantities should raise
        with pytest.raises(ValueError, match="Quantity q_1 = -5.0 must be non-negative"):
            validate_quantities([10, -5, 20])
            
    def test_fractional_quantities_allowed(self) -> None:
        """Test that fractional quantities are allowed."""
        result = cournot_simulation(a=100, b=1, costs=[10], quantities=[10.5])
        assert result.price == 89.5
        assert result.profits[0] == (89.5 - 10) * 10.5
