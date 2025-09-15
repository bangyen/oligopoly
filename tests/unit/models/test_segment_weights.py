"""Test cases for segmented demand weight validation and total demand calculation.

This module tests the validation of segment weights (must sum to 1) and
verifies that total demand equals the sum of individual segment demands.
"""

import math

import pytest

from src.sim.models.models import DemandSegment, SegmentedDemand


class TestSegmentWeightValidation:
    """Test cases for segment weight validation."""

    def test_valid_weights_sum_to_one(self) -> None:
        """Test that segments with weights summing to 1 are valid."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.4),
        ]

        # Should not raise an error
        segmented_demand = SegmentedDemand(segments=segments)
        assert len(segmented_demand.segments) == 2

    def test_weights_sum_to_one_with_floating_point_precision(self) -> None:
        """Test that weights summing to approximately 1 are valid."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.333333),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.333333),
            DemandSegment(alpha=90.0, beta=1.2, weight=0.333334),  # Sums to ~1.0
        ]

        # Should not raise an error
        segmented_demand = SegmentedDemand(segments=segments)
        assert len(segmented_demand.segments) == 3

    def test_invalid_weights_sum_not_one_raises_error(self) -> None:
        """Test that segments with weights not summing to 1 raise ValueError."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.3),  # Sums to 0.9, not 1.0
        ]

        with pytest.raises(ValueError, match="Segment weights must sum to 1.0"):
            SegmentedDemand(segments=segments)

    def test_invalid_weights_sum_to_zero_raises_error(self) -> None:
        """Test that segments with weights summing to 0 raise ValueError."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.0),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.0),
        ]

        with pytest.raises(ValueError, match="Segment weights must sum to 1.0"):
            SegmentedDemand(segments=segments)

    def test_invalid_weights_sum_to_two_raises_error(self) -> None:
        """Test that segments with weights summing to 2 raise ValueError."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=1.0),
            DemandSegment(alpha=80.0, beta=1.5, weight=1.0),  # Sums to 2.0
        ]

        with pytest.raises(ValueError, match="Segment weights must sum to 1.0"):
            SegmentedDemand(segments=segments)

    def test_single_segment_weight_one_is_valid(self) -> None:
        """Test that a single segment with weight 1.0 is valid."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=1.0),
        ]

        segmented_demand = SegmentedDemand(segments=segments)
        assert len(segmented_demand.segments) == 1


class TestTotalDemandCalculation:
    """Test cases for total demand calculation."""

    def test_total_demand_equals_sum_of_segment_demands(self) -> None:
        """Test that total demand equals weighted sum of segment demands."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        price = 20.0

        # Calculate total demand using the method
        total_demand = segmented_demand.total_demand(price)

        # Calculate manually
        segment_demands = segmented_demand.segment_demands(price)
        manual_total = sum(
            segment.weight * demand
            for segment, demand in zip(segments, segment_demands)
        )

        assert math.isclose(total_demand, manual_total, abs_tol=1e-10)

    def test_total_demand_with_zero_price(self) -> None:
        """Test total demand calculation at zero price."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.5),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        price = 0.0
        total_demand = segmented_demand.total_demand(price)

        # At price 0, demand = alpha for each segment
        expected_total = 0.5 * 100.0 + 0.5 * 80.0  # 90.0
        assert math.isclose(total_demand, expected_total, abs_tol=1e-10)

    def test_total_demand_with_high_price(self) -> None:
        """Test total demand calculation at high price (some segments may have zero demand)."""
        segments = [
            DemandSegment(
                alpha=100.0, beta=1.0, weight=0.5
            ),  # Q(80) = max(0, 100-80) = 20
            DemandSegment(
                alpha=60.0, beta=1.0, weight=0.5
            ),  # Q(80) = max(0, 60-80) = 0
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        price = 80.0
        total_demand = segmented_demand.total_demand(price)

        # Only first segment contributes
        expected_total = 0.5 * 20.0 + 0.5 * 0.0  # 10.0
        assert math.isclose(total_demand, expected_total, abs_tol=1e-10)

    def test_total_demand_with_very_high_price(self) -> None:
        """Test total demand calculation at very high price (all segments have zero demand)."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.5),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        price = 200.0  # Higher than any segment's alpha/beta
        total_demand = segmented_demand.total_demand(price)

        # All segments should have zero demand
        assert math.isclose(total_demand, 0.0, abs_tol=1e-10)

    def test_segment_demands_method(self) -> None:
        """Test the segment_demands method returns correct individual demands."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        price = 30.0
        segment_demands = segmented_demand.segment_demands(price)

        # Expected demands: Q1 = max(0, 100-30) = 70, Q2 = max(0, 80-1.5*30) = 35
        expected_demands = [70.0, 35.0]

        assert len(segment_demands) == 2
        for actual, expected in zip(segment_demands, expected_demands):
            assert math.isclose(actual, expected, abs_tol=1e-10)

    def test_three_segments_total_demand(self) -> None:
        """Test total demand calculation with three segments."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.4),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.3),
            DemandSegment(alpha=90.0, beta=1.2, weight=0.3),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        price = 25.0
        total_demand = segmented_demand.total_demand(price)

        # Manual calculation:
        # Q1 = max(0, 100-25) = 75, weighted: 0.4 * 75 = 30
        # Q2 = max(0, 80-1.5*25) = 42.5, weighted: 0.3 * 42.5 = 12.75
        # Q3 = max(0, 90-1.2*25) = 60, weighted: 0.3 * 60 = 18
        # Total: 30 + 12.75 + 18 = 60.75
        expected_total = 60.75

        assert math.isclose(total_demand, expected_total, abs_tol=1e-10)


class TestDemandSegmentIndividual:
    """Test cases for individual DemandSegment behavior."""

    def test_demand_segment_calculation(self) -> None:
        """Test individual segment demand calculation."""
        segment = DemandSegment(alpha=100.0, beta=1.0, weight=0.5)

        # Test normal case
        assert math.isclose(segment.demand(20.0), 80.0, abs_tol=1e-10)  # 100 - 20

        # Test boundary case
        assert math.isclose(segment.demand(100.0), 0.0, abs_tol=1e-10)  # 100 - 100

        # Test negative demand case
        assert math.isclose(
            segment.demand(150.0), 0.0, abs_tol=1e-10
        )  # max(0, 100-150)

    def test_demand_segment_with_different_beta(self) -> None:
        """Test segment with different beta parameter."""
        segment = DemandSegment(alpha=100.0, beta=2.0, weight=0.5)

        # Q(20) = max(0, 100 - 2*20) = 60
        assert math.isclose(segment.demand(20.0), 60.0, abs_tol=1e-10)

        # Q(50) = max(0, 100 - 2*50) = 0
        assert math.isclose(segment.demand(50.0), 0.0, abs_tol=1e-10)

    def test_demand_segment_repr(self) -> None:
        """Test string representation of DemandSegment."""
        segment = DemandSegment(alpha=100.0, beta=1.0, weight=0.5)
        repr_str = repr(segment)

        assert "DemandSegment" in repr_str
        assert "alpha=100.0" in repr_str
        assert "beta=1.0" in repr_str
        assert "weight=0.5" in repr_str

    def test_segmented_demand_repr(self) -> None:
        """Test string representation of SegmentedDemand."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)
        repr_str = repr(segmented_demand)

        assert "SegmentedDemand" in repr_str
        assert "segments=2" in repr_str
