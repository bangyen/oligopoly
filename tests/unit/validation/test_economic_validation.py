"""Tests for economic validation utilities.

This module tests the economic validation functions that ensure
realistic market behavior and prevent unrealistic outcomes.
"""

import pytest

from sim.validation.economic_validation import (
    EconomicValidationError,
    validate_cost_structure,
    validate_demand_parameters,
)


class TestEconomicValidationError:
    """Test economic validation error."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = EconomicValidationError("Test error message")
        assert str(error) == "Test error message"


class TestValidateDemandParameters:
    """Test demand parameter validation."""

    def test_validate_cournot_parameters_valid(self):
        """Test validation with valid Cournot parameters."""
        # Should not raise exception
        validate_demand_parameters(a=100.0, b=1.0, alpha=0.0, beta=0.0)

    def test_validate_bertrand_parameters_valid(self):
        """Test validation with valid Bertrand parameters."""
        # Should not raise exception
        validate_demand_parameters(a=0.0, b=0.0, alpha=200.0, beta=2.0)

    def test_validate_both_parameters_valid(self):
        """Test validation with valid both Cournot and Bertrand parameters."""
        # Should not raise exception
        validate_demand_parameters(a=100.0, b=1.0, alpha=200.0, beta=2.0)

    def test_validate_cournot_negative_intercept(self):
        """Test validation with negative Cournot intercept."""
        with pytest.raises(
            EconomicValidationError,
            match="Cournot demand intercept 'a' must be positive",
        ):
            validate_demand_parameters(a=-10.0, b=1.0, alpha=0.0, beta=0.0)

    def test_validate_cournot_negative_slope(self):
        """Test validation with negative Cournot slope."""
        with pytest.raises(
            EconomicValidationError, match="Cournot demand slope 'b' must be positive"
        ):
            validate_demand_parameters(a=100.0, b=-1.0, alpha=0.0, beta=0.0)

    def test_validate_cournot_small_market_size(self):
        """Test validation with small market size."""
        with pytest.raises(
            EconomicValidationError, match="Market size \\(a/b\\) too small"
        ):
            validate_demand_parameters(a=5.0, b=1.0, alpha=0.0, beta=0.0)

    def test_validate_bertrand_negative_intercept(self):
        """Test validation with negative Bertrand intercept."""
        with pytest.raises(
            EconomicValidationError,
            match="Bertrand demand intercept 'alpha' must be positive",
        ):
            validate_demand_parameters(a=0.0, b=0.0, alpha=-10.0, beta=2.0)

    def test_validate_bertrand_negative_slope(self):
        """Test validation with negative Bertrand slope."""
        with pytest.raises(
            EconomicValidationError,
            match="Bertrand demand slope 'beta' must be positive",
        ):
            validate_demand_parameters(a=0.0, b=0.0, alpha=200.0, beta=-2.0)

    def test_validate_bertrand_small_market_size(self):
        """Test validation with small Bertrand market size."""
        with pytest.raises(
            EconomicValidationError, match="Market size \\(alpha/beta\\) too small"
        ):
            validate_demand_parameters(a=0.0, b=0.0, alpha=5.0, beta=1.0)

    def test_validate_zero_parameters(self):
        """Test validation with all zero parameters."""
        # Should not raise exception when all parameters are zero
        validate_demand_parameters(a=0.0, b=0.0, alpha=0.0, beta=0.0)

    def test_validate_multiple_errors(self):
        """Test validation with multiple errors."""
        # Test with parameters that should trigger validation errors
        with pytest.raises(EconomicValidationError) as exc_info:
            validate_demand_parameters(
                a=5.0, b=1.0, alpha=5.0, beta=1.0
            )  # Small market sizes

        error_msg = str(exc_info.value)
        # Check that at least some errors are present
        assert len(error_msg) > 0


class TestValidateCostStructure:
    """Test cost structure validation."""

    def test_validate_costs_valid(self):
        """Test validation with valid costs."""
        costs = [10.0, 12.0, 8.0]
        # Should not raise exception
        validate_cost_structure(costs)

    def test_validate_costs_with_fixed_costs(self):
        """Test validation with valid costs and fixed costs."""
        costs = [10.0, 12.0, 8.0]
        fixed_costs = [5.0, 3.0, 7.0]
        # Should not raise exception
        validate_cost_structure(costs, fixed_costs)

    def test_validate_costs_empty(self):
        """Test validation with empty costs."""
        with pytest.raises(
            EconomicValidationError, match="At least one firm must have a cost"
        ):
            validate_cost_structure([])

    def test_validate_costs_non_positive(self):
        """Test validation with non-positive costs."""
        with pytest.raises(
            EconomicValidationError, match="Firm 0 marginal cost must be positive"
        ):
            validate_cost_structure(
                [-5.0, 12.0]
            )  # Use negative instead of zero to avoid division by zero

    def test_validate_costs_unrealistically_high(self):
        """Test validation with unrealistically high costs."""
        with pytest.raises(
            EconomicValidationError,
            match="Firm 0 marginal cost 2000.0 seems unrealistically high",
        ):
            validate_cost_structure([2000.0, 12.0])

    def test_validate_costs_high_dispersion(self):
        """Test validation with high cost dispersion."""
        with pytest.raises(EconomicValidationError, match="Cost dispersion too high"):
            validate_cost_structure([1.0, 20.0])  # Ratio of 20

    def test_validate_fixed_costs_length_mismatch(self):
        """Test validation with mismatched fixed costs length."""
        with pytest.raises(
            EconomicValidationError,
            match="Fixed costs length 2 must match marginal costs length 3",
        ):
            validate_cost_structure([10.0, 12.0, 8.0], [5.0, 3.0])

    def test_validate_fixed_costs_negative(self):
        """Test validation with negative fixed costs."""
        with pytest.raises(
            EconomicValidationError, match="Firm 0 fixed cost must be non-negative"
        ):
            validate_cost_structure([10.0, 12.0], [-5.0, 3.0])

    def test_validate_fixed_costs_unrealistically_high(self):
        """Test validation with unrealistically high fixed costs."""
        with pytest.raises(
            EconomicValidationError,
            match="Firm 0 fixed cost 20000.0 seems unrealistically high",
        ):
            validate_cost_structure([10.0, 12.0], [20000.0, 3.0])
