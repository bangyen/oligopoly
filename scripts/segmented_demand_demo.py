#!/usr/bin/env python3
"""Example script demonstrating segmented demand functionality.

This script shows how to use the new segmented demand features in both
Bertrand and Cournot competition models.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.sim.games.bertrand import bertrand_segmented_simulation
from src.sim.games.cournot import cournot_segmented_simulation
from src.sim.models.models import DemandSegment, SegmentedDemand


def main() -> None:
    """Demonstrate segmented demand functionality."""
    print("=== Segmented Demand Example ===\n")

    # Create two consumer segments with different elasticities
    segments = [
        DemandSegment(alpha=100.0, beta=1.0, weight=0.6),  # Less elastic segment
        DemandSegment(alpha=80.0, beta=2.0, weight=0.4),  # More elastic segment
    ]
    segmented_demand = SegmentedDemand(segments=segments)

    print("Market Configuration:")
    print("Segment 1: α=100, β=1.0, weight=0.6 (less elastic)")
    print("Segment 2: α=80, β=2.0, weight=0.4 (more elastic)")
    print()

    # Test Bertrand competition
    print("=== Bertrand Competition ===")
    costs = [10.0, 15.0, 20.0]
    prices = [20.0, 25.0, 30.0]  # Firm 0 has lowest price

    bertrand_result = bertrand_segmented_simulation(segmented_demand, costs, prices)

    print(f"Firm prices: {bertrand_result.prices}")
    print(f"Firm quantities: {bertrand_result.quantities}")
    print(f"Firm profits: {bertrand_result.profits}")
    print(f"Total market demand: {bertrand_result.total_demand}")
    print()

    # Test Cournot competition
    print("=== Cournot Competition ===")
    quantities = [15.0, 20.0, 25.0]  # Total quantity: 60

    cournot_result = cournot_segmented_simulation(segmented_demand, costs, quantities)

    print(f"Firm quantities: {cournot_result.quantities}")
    print(f"Market price: {cournot_result.price}")
    print(f"Firm profits: {cournot_result.profits}")
    print()

    # Demonstrate elasticity differences
    print("=== Elasticity Differences ===")
    print("At price 20:")
    for i, segment in enumerate(segments):
        demand = segment.demand(20.0)
        weighted_demand = segment.weight * demand
        print(f"Segment {i+1}: Q = {demand:.1f}, weighted = {weighted_demand:.1f}")

    print("\nAt price 30:")
    for i, segment in enumerate(segments):
        demand = segment.demand(30.0)
        weighted_demand = segment.weight * demand
        print(f"Segment {i+1}: Q = {demand:.1f}, weighted = {weighted_demand:.1f}")

    print(
        "\nNotice how the more elastic segment (higher β) reduces demand more as price increases!"
    )


if __name__ == "__main__":
    main()
