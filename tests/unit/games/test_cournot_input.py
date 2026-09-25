"""Tests for Cournot input validation functionality.

This module tests the input validation logic in the Cournot simulation,
ensuring that invalid inputs are properly rejected with meaningful error messages.
"""

import pytest

from sim.games.cournot import cournot_simulation, validate_quantities


class TestCournotInputValidation:
    """Test cases for input validation in Cournot simulation."""

    def test_negative_quantities_raise_error(self) -> None:
        """Test that negative quantities raise ValueError."""
        with pytest.raises(
            ValueError, match="Quantity q_0 = -5.0 must be non-negative"
        ):
            cournot_simulation(a=100, b=1, costs=[10], quantities=[-5])

    def test_multiple_negative_quantities(self) -> None:
        """Test validation with multiple negative quantities."""
        with pytest.raises(
            ValueError, match="Quantity q_1 = -10.0 must be non-negative"
        ):
            cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[5, -10])

    def test_zero_quantities_allowed(self) -> None:
        """Test that zero quantities are allowed."""
        result = cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[0, 0])
        assert result.price == 100.0
        assert result.profits[0] == 0.0  # (100 - 10) * 0 = 0
        assert result.profits[1] == 0.0  # (100 - 20) * 0 = 0

    def test_mismatched_costs_quantities_lengths(self) -> None:
        """Test error when costs and quantities lists have different lengths."""
        with pytest.raises(
            ValueError,
            match="Costs list length \\(2\\) must match quantities list length \\(3\\)",
        ):
            cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[10, 20, 30])

    def test_validate_quantities_function(self) -> None:
        """Test the validate_quantities function directly."""
        # Valid quantities should not raise
        validate_quantities([0, 10, 20.5])

        # Negative quantities should raise
        with pytest.raises(
            ValueError, match="Quantity q_1 = -5.0 must be non-negative"
        ):
            validate_quantities([10, -5, 20])

    def test_fractional_quantities_allowed(self) -> None:
        """Test that fractional quantities are allowed."""
        result = cournot_simulation(a=100, b=1, costs=[10], quantities=[10.5])
        assert result.price == 89.5
        assert result.profits[0] == (89.5 - 10) * 10.5
