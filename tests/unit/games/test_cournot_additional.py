"""Additional tests for cournot.py to improve coverage.

This module tests additional edge cases, error handling, and validation
scenarios in the Cournot game implementation.
"""

import math
from unittest.mock import patch

import pytest

from sim.games.cournot import (
    CournotResult,
    cournot_segmented_simulation,
    cournot_simulation,
    validate_quantities,
)
from sim.models.models import DemandSegment, SegmentedDemand
from sim.validation.economic_validation import EconomicValidationError


class TestValidateQuantities:
    """Test the validate_quantities function."""

    def test_validate_quantities_valid(self):
        """Test validate_quantities with valid quantities."""
        quantities = [10.0, 20.0, 30.0]
        # Should not raise any exception
        validate_quantities(quantities)

    def test_validate_quantities_negative(self):
        """Test validate_quantities with negative quantities."""
        quantities = [10.0, -5.0, 30.0]
        with pytest.raises(ValueError) as exc_info:
            validate_quantities(quantities)
        assert "Quantity q_1 = -5.0 must be non-negative" in str(exc_info.value)

    def test_validate_quantities_zero(self):
        """Test validate_quantities with zero quantities."""
        quantities = [10.0, 0.0, 30.0]
        # Should not raise any exception (zero is valid)
        validate_quantities(quantities)

    def test_validate_quantities_empty(self):
        """Test validate_quantities with empty list."""
        quantities = []
        # Should not raise any exception (empty list is valid)
        validate_quantities(quantities)


class TestCournotResult:
    """Test the CournotResult class."""

    def test_cournot_result_creation(self):
        """Test creating a CournotResult instance."""
        result = CournotResult(
            price=50.0, quantities=[20.0, 15.0], profits=[800.0, 600.0]
        )
        assert result.price == 50.0
        assert result.quantities == [20.0, 15.0]
        assert result.profits == [800.0, 600.0]

    def test_cournot_result_repr(self):
        """Test CournotResult string representation."""
        result = CournotResult(
            price=50.0, quantities=[20.0, 15.0], profits=[800.0, 600.0]
        )
        repr_str = repr(result)
        assert "CournotResult" in repr_str
        assert "price=50.0" in repr_str
        assert "quantities=[20.0, 15.0]" in repr_str
        assert "profits=[800.0, 600.0]" in repr_str


class TestCournotSimulationAdditional:
    """Test additional scenarios for cournot_simulation function."""

    def test_cournot_simulation_mismatched_lengths(self):
        """Test cournot_simulation with mismatched cost and quantity lengths."""
        with pytest.raises(ValueError) as exc_info:
            cournot_simulation(
                a=100.0,
                b=1.0,
                costs=[10.0, 12.0],
                quantities=[20.0],  # Mismatched lengths
            )
        assert "Costs list length (2) must match quantities list length (1)" in str(
            exc_info.value
        )

    def test_cournot_simulation_with_capacity_limits(self):
        """Test cournot_simulation with capacity limits."""
        result = cournot_simulation(
            a=100.0,
            b=1.0,
            costs=[10.0, 12.0],
            quantities=[50.0, 30.0],  # High quantities
            capacity_limits=[25.0, 20.0],  # Capacity constraints
        )
        assert result.price > 0
        assert result.quantities[0] <= 25.0  # Should be constrained
        assert result.quantities[1] <= 20.0  # Should be constrained

    def test_cournot_simulation_capacity_limits_mismatch(self):
        """Test cournot_simulation with mismatched capacity limits length."""
        with pytest.raises(ValueError) as exc_info:
            cournot_simulation(
                a=100.0,
                b=1.0,
                costs=[10.0, 12.0],
                quantities=[20.0, 15.0],
                capacity_limits=[25.0],  # Mismatched length
            )
        assert "Capacity limits length (1) must match quantities length (2)" in str(
            exc_info.value
        )

    def test_cournot_simulation_with_fixed_costs(self):
        """Test cournot_simulation with fixed costs."""
        result = cournot_simulation(
            a=100.0,
            b=1.0,
            costs=[10.0, 12.0],
            quantities=[20.0, 15.0],
            fixed_costs=[100.0, 80.0],
        )
        assert result.price > 0
        assert len(result.profits) == 2
        # Profits should account for fixed costs
        assert (
            result.profits[0] < (result.price - 10.0) * 20.0
        )  # Should be less due to fixed cost
        assert (
            result.profits[1] < (result.price - 12.0) * 15.0
        )  # Should be less due to fixed cost

    def test_cournot_simulation_fixed_costs_mismatch(self):
        """Test cournot_simulation with mismatched fixed costs length."""
        with pytest.raises(ValueError) as exc_info:
            cournot_simulation(
                a=100.0,
                b=1.0,
                costs=[10.0, 12.0],
                quantities=[20.0, 15.0],
                fixed_costs=[100.0],  # Mismatched length
            )
        assert "Fixed costs length 1 must match marginal costs length 2" in str(
            exc_info.value
        )

    def test_cournot_simulation_economic_validation_error(self):
        """Test cournot_simulation with economic validation error."""
        with patch("sim.games.cournot.validate_demand_parameters") as mock_validate:
            mock_validate.side_effect = EconomicValidationError(
                "Invalid demand parameters"
            )

            with pytest.raises(ValueError) as exc_info:
                cournot_simulation(
                    a=100.0, b=1.0, costs=[10.0, 12.0], quantities=[20.0, 15.0]
                )
            assert "Invalid demand parameters" in str(exc_info.value)

    def test_cournot_simulation_cost_validation_error(self):
        """Test cournot_simulation with cost validation error."""
        with patch("sim.games.cournot.validate_cost_structure") as mock_validate:
            mock_validate.side_effect = EconomicValidationError(
                "Invalid cost structure"
            )

            with pytest.raises(ValueError) as exc_info:
                cournot_simulation(
                    a=100.0, b=1.0, costs=[10.0, 12.0], quantities=[20.0, 15.0]
                )
            assert "Invalid cost structure" in str(exc_info.value)

    def test_cournot_simulation_zero_total_quantity(self):
        """Test cournot_simulation with zero total quantity."""
        result = cournot_simulation(
            a=100.0, b=1.0, costs=[10.0, 12.0], quantities=[0.0, 0.0]
        )
        assert result.price == 100.0  # Should be the intercept
        assert result.quantities == [0.0, 0.0]
        assert result.profits == [0.0, 0.0]

    def test_cournot_simulation_high_total_quantity(self):
        """Test cournot_simulation with high total quantity (price = 0)."""
        result = cournot_simulation(
            a=100.0,
            b=1.0,
            costs=[10.0, 12.0],
            quantities=[60.0, 50.0],  # Total = 110 > 100
        )
        assert result.price == 0.0
        assert result.quantities == [60.0, 50.0]
        assert result.profits == [-600.0, -600.0]


class TestCournotSegmentedSimulationAdditional:
    """Test additional scenarios for cournot_segmented_simulation function."""

    def test_cournot_segmented_simulation_mismatched_lengths(self):
        """Test cournot_segmented_simulation with mismatched cost and quantity lengths."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.2, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        with pytest.raises(ValueError) as exc_info:
            cournot_segmented_simulation(
                segmented_demand=segmented_demand,
                costs=[10.0, 12.0],
                quantities=[20.0],  # Mismatched lengths
            )
        assert "Costs list length (2) must match quantities list length (1)" in str(
            exc_info.value
        )

    def test_cournot_segmented_simulation_cost_validation_error(self):
        """Test cournot_segmented_simulation with cost validation error."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.2, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        with patch("sim.games.cournot.validate_cost_structure") as mock_validate:
            mock_validate.side_effect = EconomicValidationError(
                "Invalid cost structure"
            )

            with pytest.raises(ValueError) as exc_info:
                cournot_segmented_simulation(
                    segmented_demand=segmented_demand,
                    costs=[10.0, 12.0],
                    quantities=[20.0, 15.0],
                )
            assert "Invalid cost structure" in str(exc_info.value)

    def test_cournot_segmented_simulation_negative_weighted_beta(self):
        """Test cournot_segmented_simulation with negative weighted beta."""
        segments = [
            DemandSegment(alpha=100.0, beta=-1.0, weight=0.6),  # Negative beta
            DemandSegment(alpha=80.0, beta=1.2, weight=0.4),
        ]
        # This should raise a ValueError during SegmentedDemand creation
        with pytest.raises(ValueError, match="beta parameter must be positive"):
            SegmentedDemand(segments=segments)

    def test_cournot_segmented_simulation_with_fixed_costs(self):
        """Test cournot_segmented_simulation with fixed costs."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.2, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        result = cournot_segmented_simulation(
            segmented_demand=segmented_demand,
            costs=[10.0, 12.0],
            quantities=[20.0, 15.0],
            fixed_costs=[100.0, 80.0],
        )
        assert result.price > 0
        assert len(result.profits) == 2
        # Profits should account for fixed costs
        assert result.profits[0] < (result.price - 10.0) * 20.0
        assert result.profits[1] < (result.price - 12.0) * 15.0

    def test_cournot_segmented_simulation_fixed_costs_mismatch(self):
        """Test cournot_segmented_simulation with mismatched fixed costs length."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.2, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        with pytest.raises(ValueError) as exc_info:
            cournot_segmented_simulation(
                segmented_demand=segmented_demand,
                costs=[10.0, 12.0],
                quantities=[20.0, 15.0],
                fixed_costs=[100.0],  # Mismatched length
            )
        assert "Fixed costs length 1 must match marginal costs length 2" in str(
            exc_info.value
        )

    def test_cournot_segmented_simulation_zero_total_quantity(self):
        """Test cournot_segmented_simulation with zero total quantity."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.2, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        result = cournot_segmented_simulation(
            segmented_demand=segmented_demand,
            costs=[10.0, 12.0],
            quantities=[0.0, 0.0],
        )
        # Price should be weighted_alpha / weighted_beta
        expected_price = (100.0 * 0.6 + 80.0 * 0.4) / (1.0 * 0.6 + 1.2 * 0.4)
        assert math.isclose(result.price, expected_price, rel_tol=1e-6)
        assert result.quantities == [0.0, 0.0]
        assert result.profits == [0.0, 0.0]

    def test_cournot_segmented_simulation_high_total_quantity(self):
        """Test cournot_segmented_simulation with high total quantity."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.2, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        result = cournot_segmented_simulation(
            segmented_demand=segmented_demand,
            costs=[10.0, 12.0],
            quantities=[100.0, 50.0],  # Very high quantities
        )
        # Price is max(0, (weighted_alpha - total_qty) / weighted_beta) = 0
        assert result.price == 0.0
        assert result.quantities == [100.0, 50.0]

    def test_cournot_segmented_simulation_no_price_floor(self):
        """Price comes from demand alone; there is no floor at marginal cost."""
        segments = [
            DemandSegment(alpha=10.0, beta=1.0, weight=0.6),  # Low alpha
            DemandSegment(alpha=8.0, beta=1.2, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        result = cournot_segmented_simulation(
            segmented_demand=segmented_demand,
            costs=[5.0, 6.0],
            quantities=[20.0, 15.0],
        )
        # weighted_alpha = 9.2, total_qty = 35 -> demand price clamps to 0
        assert result.price == 0.0
