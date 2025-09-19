#!/usr/bin/env python3
"""Demonstration of enhanced economic models in oligopoly simulation.

This script demonstrates the new economic features including:
- Capacity constraints
- Fixed costs
- Economies of scale
- Non-linear demand functions
- Enhanced profit calculations
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sim.games.enhanced_simulation import (
    EnhancedSimulationConfig,
    enhanced_bertrand_simulation,
    enhanced_cournot_simulation,
)
from sim.models.models import CostStructure


def demonstrate_capacity_constraints() -> None:
    """Demonstrate how capacity constraints affect market outcomes."""
    print("=" * 60)
    print("DEMONSTRATION: Capacity Constraints")
    print("=" * 60)

    # Create cost structures with different capacity limits
    cost_structures = [
        CostStructure(marginal_cost=10.0, capacity_limit=20.0),  # Limited capacity
        CostStructure(marginal_cost=15.0, capacity_limit=10.0),  # More limited capacity
    ]

    config = EnhancedSimulationConfig(
        demand_type="linear",
        demand_params={"a": 100.0, "b": 1.0},
        cost_structures=cost_structures,
    )

    # Try to produce more than capacity allows
    quantities = [30.0, 15.0]  # Exceed capacity limits
    print(f"Requested quantities: {quantities}")
    print("Capacity limits: [20.0, 10.0]")

    result = enhanced_cournot_simulation(config, quantities)

    print(f"Actual quantities: {[f'{q:.1f}' for q in result.quantities]}")
    print(f"Market price: {result.price:.2f}")
    print(f"Profits: {[f'{p:.2f}' for p in result.profits]}")
    print()


def demonstrate_fixed_costs() -> None:
    """Demonstrate how fixed costs affect profitability and market structure."""
    print("=" * 60)
    print("DEMONSTRATION: Fixed Costs")
    print("=" * 60)

    # Compare scenarios with and without fixed costs
    scenarios = [
        (
            "No Fixed Costs",
            [
                CostStructure(marginal_cost=10.0, fixed_cost=0.0),
                CostStructure(marginal_cost=15.0, fixed_cost=0.0),
            ],
        ),
        (
            "With Fixed Costs",
            [
                CostStructure(marginal_cost=10.0, fixed_cost=100.0),
                CostStructure(marginal_cost=15.0, fixed_cost=50.0),
            ],
        ),
    ]

    quantities = [20.0, 15.0]

    for scenario_name, cost_structures in scenarios:
        print(f"\n{scenario_name}:")
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"a": 100.0, "b": 1.0},
            cost_structures=cost_structures,
        )

        result = enhanced_cournot_simulation(config, quantities)

        print(f"  Market price: {result.price:.2f}")
        print(f"  Quantities: {[f'{q:.1f}' for q in result.quantities]}")
        print(f"  Profits: {[f'{p:.2f}' for p in result.profits]}")
        print(f"  Total profit: {sum(result.profits):.2f}")
    print()


def demonstrate_economies_of_scale() -> None:
    """Demonstrate how economies of scale affect cost structure."""
    print("=" * 60)
    print("DEMONSTRATION: Economies of Scale")
    print("=" * 60)

    # Compare firms with and without economies of scale
    cost_structures = [
        CostStructure(marginal_cost=10.0, economies_of_scale=1.0),  # No economies
        CostStructure(marginal_cost=10.0, economies_of_scale=0.8),  # With economies
    ]

    quantities = [20.0, 20.0]

    print("Cost comparison for different production levels:")
    print("Quantity | No Economies | With Economies | Difference")
    print("-" * 55)

    for qty in [5, 10, 20, 30]:
        cost_no_economies = cost_structures[0].total_cost(qty)
        cost_with_economies = cost_structures[1].total_cost(qty)
        difference = cost_no_economies - cost_with_economies

        print(
            f"{qty:8.0f} | {cost_no_economies:11.2f} | {cost_with_economies:13.2f} | {difference:9.2f}"
        )

    print()

    # Run simulation
    config = EnhancedSimulationConfig(
        demand_type="linear",
        demand_params={"a": 100.0, "b": 1.0},
        cost_structures=cost_structures,
    )

    result = enhanced_cournot_simulation(config, quantities)

    print("Simulation results:")
    print(f"  Market price: {result.price:.2f}")
    print(f"  Quantities: {[f'{q:.1f}' for q in result.quantities]}")
    print(f"  Profits: {[f'{p:.2f}' for p in result.profits]}")
    print()


def demonstrate_isoelastic_demand() -> None:
    """Demonstrate isoelastic demand function."""
    print("=" * 60)
    print("DEMONSTRATION: Isoelastic Demand")
    print("=" * 60)

    # Compare linear vs isoelastic demand
    scenarios = [
        ("Linear Demand", "linear", {"a": 100.0, "b": 1.0}),
        ("Isoelastic Demand", "isoelastic", {"A": 100.0, "elasticity": 2.0}),
    ]

    cost_structures = [
        CostStructure(marginal_cost=10.0, fixed_cost=50.0),
        CostStructure(marginal_cost=15.0, fixed_cost=30.0),
    ]

    quantities = [20.0, 15.0]

    for scenario_name, demand_type, demand_params in scenarios:
        print(f"\n{scenario_name}:")
        config = EnhancedSimulationConfig(
            demand_type=demand_type,
            demand_params=demand_params,
            cost_structures=cost_structures,
        )

        result = enhanced_cournot_simulation(config, quantities)

        print(f"  Market price: {result.price:.2f}")
        print(f"  Total quantity: {sum(result.quantities):.1f}")
        print(f"  Profits: {[f'{p:.2f}' for p in result.profits]}")

        # Show demand elasticity
        if demand_type == "isoelastic":
            elasticity = demand_params["elasticity"]
            print(f"  Price elasticity: {elasticity}")
    print()


def demonstrate_bertrand_with_enhanced_features() -> None:
    """Demonstrate Bertrand competition with enhanced features."""
    print("=" * 60)
    print("DEMONSTRATION: Enhanced Bertrand Competition")
    print("=" * 60)

    cost_structures = [
        CostStructure(marginal_cost=10.0, fixed_cost=100.0, capacity_limit=30.0),
        CostStructure(marginal_cost=12.0, fixed_cost=80.0, capacity_limit=25.0),
        CostStructure(marginal_cost=15.0, fixed_cost=60.0, capacity_limit=20.0),
    ]

    # Test different pricing strategies
    price_scenarios = [
        ("Competitive Pricing", [11.0, 12.0, 15.0]),
        ("Aggressive Pricing", [10.5, 11.5, 14.5]),
        ("High Pricing", [20.0, 25.0, 30.0]),
    ]

    for scenario_name, prices in price_scenarios:
        print(f"\n{scenario_name}: {prices}")

        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"alpha": 100.0, "beta": 1.0},
            cost_structures=cost_structures,
        )

        result = enhanced_bertrand_simulation(config, prices)

        print(f"  Market price: {min(result.prices):.2f}")
        print(f"  Quantities: {[f'{q:.1f}' for q in result.quantities]}")
        print(f"  Profits: {[f'{p:.2f}' for p in result.profits]}")
        print(f"  Total profit: {sum(result.profits):.2f}")
    print()


def main() -> None:
    """Run all demonstrations."""
    print("ENHANCED OLIGOPOLY SIMULATION DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows the new economic features:")
    print("- Capacity constraints")
    print("- Fixed costs")
    print("- Economies of scale")
    print("- Non-linear demand functions")
    print("- Enhanced profit calculations")
    print()

    try:
        demonstrate_capacity_constraints()
        demonstrate_fixed_costs()
        demonstrate_economies_of_scale()
        demonstrate_isoelastic_demand()
        demonstrate_bertrand_with_enhanced_features()

        print("=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The enhanced economic models provide more realistic")
        print("market simulations with:")
        print("• Natural market concentration through fixed costs")
        print("• Realistic production limits through capacity constraints")
        print("• Cost advantages for larger firms through economies of scale")
        print("• More flexible demand modeling through non-linear functions")
        print("• Comprehensive profit calculations including all cost components")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
