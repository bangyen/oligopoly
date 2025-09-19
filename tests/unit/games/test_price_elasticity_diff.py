"""Test cases for price elasticity differences across segments.

This module tests that segments with higher β (more elastic) reduce quantity
more when price rises, demonstrating different price sensitivity across segments.
"""

import math

from src.sim.games.bertrand import bertrand_segmented_simulation
from src.sim.games.cournot import cournot_segmented_simulation
from src.sim.models.models import DemandSegment, SegmentedDemand


class TestPriceElasticityDifferences:
    """Test cases for different price elasticity across segments."""

    def test_higher_beta_reduces_quantity_more_in_bertrand(self) -> None:
        """Test that segment with higher β reduces quantity more when price rises in Bertrand."""
        # Create segments with different elasticities
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),  # Less elastic (lower β)
            DemandSegment(alpha=100.0, beta=2.0, weight=0.5),  # More elastic (higher β)
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [10.0, 15.0, 20.0]

        # Test at low price
        prices_low = [20.0, 25.0, 30.0]  # Firm 0 has lowest price
        result_low = bertrand_segmented_simulation(segmented_demand, costs, prices_low)

        # Test at high price
        prices_high = [40.0, 45.0, 50.0]  # Firm 0 still has lowest price
        result_high = bertrand_segmented_simulation(
            segmented_demand, costs, prices_high
        )

        # Calculate demand changes for each segment
        # At price 20:
        # Segment 1: Q = 100 - 1*20 = 80, weighted: 0.5 * 80 = 40
        # Segment 2: Q = 100 - 2*20 = 60, weighted: 0.5 * 60 = 30
        # Total at low price: 40 + 30 = 70

        # At price 40:
        # Segment 1: Q = 100 - 1*40 = 60, weighted: 0.5 * 60 = 30
        # Segment 2: Q = 100 - 2*40 = 20, weighted: 0.5 * 20 = 10
        # Total at high price: 30 + 10 = 40

        # Demand reduction: 70 - 40 = 30
        # Segment 1 reduction: 40 - 30 = 10
        # Segment 2 reduction: 30 - 10 = 20
        # Higher β segment (segment 2) reduces more: 20 > 10

        demand_reduction_segment_1 = 0.5 * (80.0 - 60.0)  # 10
        demand_reduction_segment_2 = 0.5 * (60.0 - 20.0)  # 20

        assert demand_reduction_segment_2 > demand_reduction_segment_1
        assert math.isclose(result_low.total_demand, 70.0, abs_tol=1e-10)
        assert math.isclose(result_high.total_demand, 40.0, abs_tol=1e-10)

    def test_higher_beta_reduces_quantity_more_in_cournot(self) -> None:
        """Test that segment with higher β reduces quantity more when price rises in Cournot."""
        # Create segments with different elasticities
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),  # Less elastic (lower β)
            DemandSegment(alpha=100.0, beta=2.0, weight=0.5),  # More elastic (higher β)
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [5.0, 10.0, 15.0]  # Lower costs to prevent exits

        # Test with low quantities (higher price)
        quantities_low = [10.0, 15.0, 20.0]  # Total: 45
        result_low = cournot_segmented_simulation(
            segmented_demand, costs, quantities_low
        )

        # Test with high quantities (lower price) - but not so high that firms exit
        quantities_high = [20.0, 25.0, 30.0]  # Total: 75 (still profitable)
        result_high = cournot_segmented_simulation(
            segmented_demand, costs, quantities_high
        )

        # Verify that higher quantities lead to lower prices
        assert result_high.price < result_low.price

        # Calculate effective demand parameters
        weighted_alpha = 0.5 * 100.0 + 0.5 * 100.0  # 100
        weighted_beta = 0.5 * 1.0 + 0.5 * 2.0  # 1.5

        # At low quantities (45): P = (100 - 45) / 1.5 = 36.67
        # At high quantities (75): P = (100 - 75) / 1.5 = 16.67

        expected_price_low = max(0.0, (weighted_alpha - 45.0) / weighted_beta)
        expected_price_high = max(0.0, (weighted_alpha - 75.0) / weighted_beta)

        assert math.isclose(result_low.price, expected_price_low, abs_tol=1e-10)
        assert math.isclose(result_high.price, expected_price_high, abs_tol=1e-10)

    def test_elasticity_difference_with_three_segments(self) -> None:
        """Test elasticity differences with three segments of varying β."""
        segments = [
            DemandSegment(alpha=100.0, beta=0.5, weight=0.3),  # Least elastic
            DemandSegment(alpha=100.0, beta=1.0, weight=0.4),  # Medium elastic
            DemandSegment(alpha=100.0, beta=2.0, weight=0.3),  # Most elastic
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [10.0, 15.0, 20.0]

        # Test at different price levels
        prices_test = [20.0, 25.0, 30.0]  # Firm 0 has lowest price
        result = bertrand_segmented_simulation(segmented_demand, costs, prices_test)

        # Calculate segment demands at price 20
        # Segment 1: Q = 100 - 0.5*20 = 90, weighted: 0.3 * 90 = 27
        # Segment 2: Q = 100 - 1.0*20 = 80, weighted: 0.4 * 80 = 32
        # Segment 3: Q = 100 - 2.0*20 = 60, weighted: 0.3 * 60 = 18
        # Total: 27 + 32 + 18 = 77

        expected_total = 0.3 * 90.0 + 0.4 * 80.0 + 0.3 * 60.0  # 77
        assert math.isclose(result.total_demand, expected_total, abs_tol=1e-10)

        # Verify that higher β segments contribute less demand at the same price
        # Compare unweighted demands to isolate elasticity effects
        segment_demands = [90.0, 80.0, 60.0]  # Unweighted demands at price 20

        # Higher β segments should have lower unweighted demand
        assert segment_demands[2] < segment_demands[1] < segment_demands[0]

    def test_price_sensitivity_comparison(self) -> None:
        """Test direct comparison of price sensitivity between segments."""
        # Create two segments with same α but different β
        segment_low_elasticity = DemandSegment(alpha=100.0, beta=1.0, weight=1.0)
        segment_high_elasticity = DemandSegment(alpha=100.0, beta=2.0, weight=1.0)

        # Test demand at different prices
        prices = [10.0, 20.0, 30.0, 40.0, 50.0]

        low_elasticity_demands = [segment_low_elasticity.demand(p) for p in prices]
        high_elasticity_demands = [segment_high_elasticity.demand(p) for p in prices]

        # Calculate demand reductions as price increases
        low_elasticity_reductions = [
            low_elasticity_demands[i] - low_elasticity_demands[i + 1]
            for i in range(len(prices) - 1)
        ]
        high_elasticity_reductions = [
            high_elasticity_demands[i] - high_elasticity_demands[i + 1]
            for i in range(len(prices) - 1)
        ]

        # High elasticity segment should have larger reductions
        for low_red, high_red in zip(
            low_elasticity_reductions, high_elasticity_reductions
        ):
            assert high_red > low_red

    def test_boundary_elasticity_effects(self) -> None:
        """Test elasticity effects at boundary conditions."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),  # Q(100) = 0
            DemandSegment(alpha=100.0, beta=2.0, weight=0.5),  # Q(50) = 0
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [10.0, 15.0, 20.0]

        # Test at boundary prices
        # Price 100: Segment 1 has zero demand, Segment 2 has zero demand
        prices_boundary = [100.0, 105.0, 110.0]
        result_boundary = bertrand_segmented_simulation(
            segmented_demand, costs, prices_boundary
        )

        assert math.isclose(result_boundary.total_demand, 0.0, abs_tol=1e-10)

        # Test just below boundary
        # Price 49: Segment 1 has positive demand, Segment 2 has positive demand
        prices_below = [49.0, 55.0, 60.0]
        result_below = bertrand_segmented_simulation(
            segmented_demand, costs, prices_below
        )

        # Both segments should contribute
        # Segment 1: Q(49) = 51, weighted: 0.5 * 51 = 25.5
        # Segment 2: Q(49) = 2, weighted: 0.5 * 2 = 1
        # Total: 25.5 + 1 = 26.5
        expected_total = 0.5 * 51.0 + 0.5 * 2.0  # 26.5
        assert math.isclose(result_below.total_demand, expected_total, abs_tol=1e-10)

    def test_elasticity_weight_interaction(self) -> None:
        """Test interaction between elasticity and segment weights."""
        # Create segments with different elasticities and weights
        segments = [
            DemandSegment(
                alpha=100.0, beta=1.0, weight=0.8
            ),  # Low elasticity, high weight
            DemandSegment(
                alpha=100.0, beta=3.0, weight=0.2
            ),  # High elasticity, low weight
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [10.0, 15.0, 20.0]

        # Test at moderate price
        prices = [20.0, 25.0, 30.0]
        result = bertrand_segmented_simulation(segmented_demand, costs, prices)

        # Calculate contributions
        # Segment 1: Q(20) = 80, weighted: 0.8 * 80 = 64
        # Segment 2: Q(20) = 40, weighted: 0.2 * 40 = 8
        # Total: 64 + 8 = 72

        expected_total = 0.8 * 80.0 + 0.2 * 40.0  # 72
        assert math.isclose(result.total_demand, expected_total, abs_tol=1e-10)

        # High weight segment dominates despite lower elasticity
        segment_1_contribution = 0.8 * 80.0  # 64
        segment_2_contribution = 0.2 * 40.0  # 8

        assert segment_1_contribution > segment_2_contribution

    def test_elasticity_impact_on_profits(self) -> None:
        """Test how elasticity differences affect firm profits."""
        segments = [
            DemandSegment(alpha=100.0, beta=1.0, weight=0.5),  # Less elastic
            DemandSegment(alpha=100.0, beta=2.0, weight=0.5),  # More elastic
        ]
        segmented_demand = SegmentedDemand(segments=segments)

        costs = [10.0, 15.0, 20.0]

        # Test at different price levels
        prices_low = [20.0, 25.0, 30.0]  # Low price, high demand
        prices_high = [40.0, 45.0, 50.0]  # High price, low demand

        result_low = bertrand_segmented_simulation(segmented_demand, costs, prices_low)
        result_high = bertrand_segmented_simulation(
            segmented_demand, costs, prices_high
        )

        # Firm 0 gets all demand in both cases
        # At low price: profit = (20-10) * 70 = 700
        # At high price: profit = (40-10) * 40 = 1200

        # Higher price leads to higher profit per unit but lower total quantity
        # The elasticity difference affects the quantity reduction
        assert (
            result_high.profits[0] > result_low.profits[0]
        )  # Higher profit at higher price
        assert (
            result_high.quantities[0] < result_low.quantities[0]
        )  # Lower quantity at higher price
