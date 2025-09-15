"""Test cases for segmented demand allocation in Bertrand competition.

This module tests the allocation logic where each segment chooses the firm
with the lowest price, with ties splitting equally within each segment.
"""

import math

import pytest

from src.sim.games.bertrand import (
    allocate_segmented_demand,
    bertrand_segmented_simulation,
)
from src.sim.models.models import DemandSegment, SegmentedDemand


class TestSegmentAllocation:
    """Test cases for segment-based demand allocation."""

    def test_single_firm_cheapest_gets_all_segments(self) -> None:
        """Test that if Firm A is cheapest, it gets all segment demand."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        prices = [20.0, 25.0, 30.0]  # Firm 0 has lowest price
        costs = [10.0, 15.0, 20.0]

        quantities, total_demand = allocate_segmented_demand(
            prices, costs, segmented_demand
        )

        # Firm 0 should get all demand from both segments
        # Segment 1: Q(20) = 80, weighted: 0.6 * 80 = 48
        # Segment 2: Q(20) = 50, weighted: 0.4 * 50 = 20
        # Total for firm 0: 48 + 20 = 68
        expected_firm_0_demand = 0.6 * 80.0 + 0.4 * 50.0  # 68.0

        assert math.isclose(quantities[0], expected_firm_0_demand, abs_tol=1e-10)
        assert math.isclose(quantities[1], 0.0, abs_tol=1e-10)
        assert math.isclose(quantities[2], 0.0, abs_tol=1e-10)

        # Total demand should equal sum of weighted segment demands
        assert math.isclose(total_demand, expected_firm_0_demand, abs_tol=1e-10)

    def test_two_firms_tie_split_each_segment(self) -> None:
        """Test that with ties, each firm gets half within each segment."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.5),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        prices = [20.0, 20.0, 25.0]  # Firms 0 and 1 tie at lowest price
        costs = [10.0, 10.0, 15.0]

        quantities, total_demand = allocate_segmented_demand(
            prices, costs, segmented_demand
        )

        # Each firm should get half of each segment's demand
        # Segment 1: Q(20) = 80, each firm gets 0.5 * 0.5 * 80 = 20
        # Segment 2: Q(20) = 50, each firm gets 0.5 * 0.5 * 50 = 12.5
        # Each firm total: 20 + 12.5 = 32.5
        expected_per_firm = 0.5 * 0.5 * 80.0 + 0.5 * 0.5 * 50.0  # 32.5

        assert math.isclose(quantities[0], expected_per_firm, abs_tol=1e-10)
        assert math.isclose(quantities[1], expected_per_firm, abs_tol=1e-10)
        assert math.isclose(quantities[2], 0.0, abs_tol=1e-10)

        # Total demand should equal sum of weighted segment demands
        expected_total = 0.5 * 80.0 + 0.5 * 50.0  # 65.0
        assert math.isclose(total_demand, expected_total, abs_tol=1e-10)

    def test_three_firms_tie_split_each_segment(self) -> None:
        """Test that three firms tying split each segment equally."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        prices = [20.0, 20.0, 20.0]  # All three firms tie
        costs = [10.0, 10.0, 10.0]

        quantities, total_demand = allocate_segmented_demand(
            prices, costs, segmented_demand
        )

        # Each firm should get 1/3 of each segment's demand
        # Segment 1: Q(20) = 80, each firm gets 0.6 * (1/3) * 80 = 16
        # Segment 2: Q(20) = 50, each firm gets 0.4 * (1/3) * 50 = 6.67
        # Each firm total: 16 + 6.67 = 22.67
        expected_per_firm = 0.6 * (1 / 3) * 80.0 + 0.4 * (1 / 3) * 50.0  # 22.67

        for i in range(3):
            assert math.isclose(quantities[i], expected_per_firm, abs_tol=1e-10)

        # Total demand should equal sum of weighted segment demands
        expected_total = 0.6 * 80.0 + 0.4 * 50.0  # 68.0
        assert math.isclose(total_demand, expected_total, abs_tol=1e-10)

    def test_segment_with_zero_demand_at_price(self) -> None:
        """Test allocation when some segments have zero demand at the price."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),  # Q(80) = 20
            DemandSegment(alpha=60.0, beta=1.0, weight=0.5),  # Q(80) = 0 (negative)
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        prices = [80.0, 85.0, 90.0]  # Firm 0 has lowest price
        costs = [40.0, 45.0, 50.0]

        quantities, total_demand = allocate_segmented_demand(
            prices, costs, segmented_demand
        )

        # Only first segment contributes demand
        # Segment 1: Q(80) = 20, weighted: 0.5 * 20 = 10
        # Segment 2: Q(80) = 0, weighted: 0.5 * 0 = 0
        expected_firm_0_demand = 0.5 * 20.0 + 0.5 * 0.0  # 10.0

        assert math.isclose(quantities[0], expected_firm_0_demand, abs_tol=1e-10)
        assert math.isclose(quantities[1], 0.0, abs_tol=1e-10)
        assert math.isclose(quantities[2], 0.0, abs_tol=1e-10)

        assert math.isclose(total_demand, expected_firm_0_demand, abs_tol=1e-10)

    def test_all_segments_zero_demand(self) -> None:
        """Test allocation when all segments have zero demand."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),  # Q(150) = 0
            DemandSegment(alpha=80.0, beta=1.0, weight=0.5),  # Q(150) = 0
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        prices = [150.0, 160.0, 170.0]  # All prices too high
        costs = [40.0, 45.0, 50.0]

        quantities, total_demand = allocate_segmented_demand(
            prices, costs, segmented_demand
        )

        # All firms should get zero demand
        for i in range(3):
            assert math.isclose(quantities[i], 0.0, abs_tol=1e-10)

        assert math.isclose(total_demand, 0.0, abs_tol=1e-10)


class TestBertrandSegmentedSimulation:
    """Test cases for the full Bertrand segmented simulation."""

    def test_bertrand_segmented_simulation_basic(self) -> None:
        """Test basic Bertrand segmented simulation."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.6),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.4),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [10.0, 15.0, 20.0]
        prices = [20.0, 25.0, 30.0]  # Firm 0 has lowest price

        result = bertrand_segmented_simulation(segmented_demand, costs, prices)

        # Verify result structure
        assert hasattr(result, "total_demand")
        assert hasattr(result, "prices")
        assert hasattr(result, "quantities")
        assert hasattr(result, "profits")

        # Verify quantities match allocation
        expected_firm_0_demand = 0.6 * 80.0 + 0.4 * 50.0  # 68.0
        assert math.isclose(result.quantities[0], expected_firm_0_demand, abs_tol=1e-10)
        assert math.isclose(result.quantities[1], 0.0, abs_tol=1e-10)
        assert math.isclose(result.quantities[2], 0.0, abs_tol=1e-10)

        # Verify profits calculation
        expected_profit_0 = (20.0 - 10.0) * expected_firm_0_demand  # 10 * 68 = 680
        assert math.isclose(result.profits[0], expected_profit_0, abs_tol=1e-10)
        assert math.isclose(result.profits[1], 0.0, abs_tol=1e-10)
        assert math.isclose(result.profits[2], 0.0, abs_tol=1e-10)

    def test_bertrand_segmented_simulation_with_ties(self) -> None:
        """Test Bertrand segmented simulation with price ties."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.5),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [10.0, 10.0, 15.0]
        prices = [20.0, 20.0, 25.0]  # Firms 0 and 1 tie

        result = bertrand_segmented_simulation(segmented_demand, costs, prices)

        # Each firm should get half the demand
        expected_per_firm = 0.5 * 0.5 * 80.0 + 0.5 * 0.5 * 50.0  # 32.5

        assert math.isclose(result.quantities[0], expected_per_firm, abs_tol=1e-10)
        assert math.isclose(result.quantities[1], expected_per_firm, abs_tol=1e-10)
        assert math.isclose(result.quantities[2], 0.0, abs_tol=1e-10)

        # Both firms should have equal profits
        expected_profit = (20.0 - 10.0) * expected_per_firm  # 10 * 32.5 = 325
        assert math.isclose(result.profits[0], expected_profit, abs_tol=1e-10)
        assert math.isclose(result.profits[1], expected_profit, abs_tol=1e-10)
        assert math.isclose(result.profits[2], 0.0, abs_tol=1e-10)

    def test_bertrand_segmented_simulation_validation(self) -> None:
        """Test input validation in Bertrand segmented simulation."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=1.0),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        # Test negative prices
        with pytest.raises(ValueError, match="must be non-negative"):
            bertrand_segmented_simulation(segmented_demand, [10.0], [-5.0])

        # Test mismatched costs and prices
        with pytest.raises(ValueError, match="must match"):
            bertrand_segmented_simulation(segmented_demand, [10.0, 20.0], [15.0])

    def test_bertrand_segmented_simulation_empty_prices(self) -> None:
        """Test Bertrand segmented simulation with empty prices list."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=1.0),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        result = bertrand_segmented_simulation(segmented_demand, [], [])

        assert result.total_demand == 0.0
        assert result.quantities == []
        assert result.profits == []
        assert result.prices == []


class TestSegmentAllocationEdgeCases:
    """Test edge cases for segment allocation."""

    def test_single_segment_allocation(self) -> None:
        """Test allocation with single segment."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=1.0),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        prices = [20.0, 25.0, 30.0]
        costs = [10.0, 15.0, 20.0]

        quantities, total_demand = allocate_segmented_demand(
            prices, costs, segmented_demand
        )

        # Should behave like single-segment case
        expected_demand = 80.0  # 100 - 20
        assert math.isclose(quantities[0], expected_demand, abs_tol=1e-10)
        assert math.isclose(quantities[1], 0.0, abs_tol=1e-10)
        assert math.isclose(quantities[2], 0.0, abs_tol=1e-10)
        assert math.isclose(total_demand, expected_demand, abs_tol=1e-10)

    def test_three_segments_allocation(self) -> None:
        """Test allocation with three segments."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.4),
            DemandSegment(alpha=80.0, beta=1.5, weight=0.3),
            DemandSegment(alpha=90.0, beta=1.2, weight=0.3),
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        prices = [20.0, 25.0, 30.0]  # Firm 0 has lowest price
        costs = [10.0, 15.0, 20.0]

        quantities, total_demand = allocate_segmented_demand(
            prices, costs, segmented_demand
        )

        # Calculate expected demand for firm 0
        # Segment 1: Q(20) = 80, weighted: 0.4 * 80 = 32
        # Segment 2: Q(20) = 50, weighted: 0.3 * 50 = 15
        # Segment 3: Q(20) = 66, weighted: 0.3 * 66 = 19.8
        # Total: 32 + 15 + 19.8 = 66.8
        expected_firm_0_demand = 0.4 * 80.0 + 0.3 * 50.0 + 0.3 * 66.0  # 66.8

        assert math.isclose(quantities[0], expected_firm_0_demand, abs_tol=1e-10)
        assert math.isclose(quantities[1], 0.0, abs_tol=1e-10)
        assert math.isclose(quantities[2], 0.0, abs_tol=1e-10)
        assert math.isclose(total_demand, expected_firm_0_demand, abs_tol=1e-10)
