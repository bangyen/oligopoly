#!/usr/bin/env python3
"""Example script demonstrating segmented demand functionality.

This script shows how to use the new segmented demand features in both
Bertrand and Cournot competition models.
"""

from sim.games.bertrand import bertrand_segmented_simulation
from sim.games.cournot import cournot_segmented_simulation
from sim.models.models import DemandSegment, SegmentedDemand

from .utils import format_currency, format_list, print_demo_completion


def main() -> None:
    """Demonstrate segmented demand functionality."""
    print("=== Segmented Demand Example ===\n")

    # Create two consumer segments with different elasticities
    segments = [
        DemandSegment(alpha=100.0, beta=1.0, weight=0.6),  # Less elastic segment
        DemandSegment(alpha=80.0, beta=2.0, weight=0.4),  # More elastic segment
    ]
    segmented_demand = SegmentedDemand(segments=segments)

    print("Market: 2 segments (α₁=100,β₁=1.0,w₁=0.6 | α₂=80,β₂=2.0,w₂=0.4)")

    # Test Bertrand competition
    print("\n=== Bertrand Competition ===")
    costs = [10.0, 15.0, 20.0]
    prices = [20.0, 25.0, 30.0]  # Firm 0 has lowest price

    bertrand_result = bertrand_segmented_simulation(segmented_demand, costs, prices)

    print(f"Prices: {bertrand_result.prices}")
    print(f"Quantities: {format_list(bertrand_result.quantities, 'numeric')}")
    print(f"Profits: {format_list(bertrand_result.profits)}")
    print(f"Total demand: {bertrand_result.total_demand:.1f}")

    # Test Cournot competition
    print("\n=== Cournot Competition ===")
    quantities = [15.0, 20.0, 25.0]  # Total quantity: 60

    cournot_result = cournot_segmented_simulation(segmented_demand, costs, quantities)

    print(f"Quantities: {cournot_result.quantities}")
    print(f"Market price: {format_currency(cournot_result.price)}")
    print(f"Profits: {format_list(cournot_result.profits)}")

    # Demonstrate elasticity differences
    print("\n=== Elasticity Effects ===")
    print("Price sensitivity comparison:")
    for i, segment in enumerate(segments):
        demand_20 = segment.demand(20.0)
        demand_30 = segment.demand(30.0)
        change_pct = (demand_30 - demand_20) / demand_20 * 100
        print(
            f"  Segment {i + 1}: Q(20)={demand_20:.1f}, Q(30)={demand_30:.1f} ({change_pct:+.0f}%)"
        )

    print_demo_completion(
        "Segmented demand",
        "Multi-segment markets, different elasticities, Bertrand & Cournot models",
    )


if __name__ == "__main__":
    main()
