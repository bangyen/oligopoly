#!/usr/bin/env python3
"""Demonstration of advanced economic models in oligopoly simulation.

This script demonstrates the new advanced features including:
- Advanced learning strategies (Fictitious Play, Deep Q-Learning, Behavioral)
- Product differentiation (Horizontal, Vertical, Logit demand)
- Market evolution (Entry/Exit, Innovation, Growth)
- Enhanced demand functions (CES, Network Effects, Dynamic)
"""

import json
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sim.models.enhanced_demand import (
    CESDemand,
    DynamicDemand,
    MultiSegmentDemand,
    NetworkEffectsDemand,
)
from sim.models.market_evolution import (
    MarketEvolutionConfig,
    create_market_evolution_engine,
)
from sim.models.product_differentiation import (
    HotellingDemand,
    ProductCharacteristics,
    VerticalDifferentiation,
    calculate_differentiated_nash_equilibrium,
    differentiated_bertrand_simulation,
)
from sim.strategies.advanced_strategies import (
    BehavioralStrategy,
    DeepQLearningStrategy,
    FictitiousPlayStrategy,
    MarketState,
)


def demonstrate_advanced_strategies() -> None:
    """Demonstrate advanced learning strategies."""
    print("=" * 60)
    print("DEMONSTRATION: Advanced Learning Strategies")
    print("=" * 60)

    # Create sample market state
    market_state = MarketState(
        prices=[20.0, 22.0, 18.0],
        quantities=[15.0, 12.0, 18.0],
        market_shares=[0.33, 0.27, 0.40],
        total_demand=45.0,
        market_growth=0.02,
        innovation_level=0.1,
        round_num=5,
    )

    # Test Fictitious Play Strategy
    print("\n1. Fictitious Play Strategy:")
    fp_strategy = FictitiousPlayStrategy(
        belief_decay=0.9, exploration_rate=0.1, seed=42
    )

    # Simulate a few rounds
    for round_num in range(3):
        action = fp_strategy.next_action(
            round_num=round_num,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=(10.0, 50.0),
            market_params={"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0},
        )
        print(f"  Round {round_num}: Action = {action:.2f}")

    # Test Deep Q-Learning Strategy
    print("\n2. Deep Q-Learning Strategy:")
    dqn_strategy = DeepQLearningStrategy(
        learning_rate=0.01, discount_factor=0.95, epsilon_0=0.3, seed=42
    )

    for round_num in range(3):
        action = dqn_strategy.next_action(
            round_num=round_num,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=(10.0, 50.0),
            market_params={"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0},
        )
        print(f"  Round {round_num}: Action = {action:.2f}")

    # Test Behavioral Strategy
    print("\n3. Behavioral Strategy:")
    behavioral_strategy = BehavioralStrategy(
        rationality_level=0.8, loss_aversion=2.0, fairness_weight=0.1, seed=42
    )

    for round_num in range(3):
        action = behavioral_strategy.next_action(
            round_num=round_num,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=(10.0, 50.0),
            market_params={"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0},
        )
        print(f"  Round {round_num}: Action = {action:.2f}")


def demonstrate_product_differentiation() -> None:
    """Demonstrate product differentiation models."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Product Differentiation")
    print("=" * 60)

    # Create differentiated products
    products = [
        ProductCharacteristics(
            quality=1.0, location=0.2, brand_strength=1.2, innovation_level=0.1
        ),
        ProductCharacteristics(
            quality=1.2, location=0.5, brand_strength=1.0, innovation_level=0.2
        ),
        ProductCharacteristics(
            quality=0.8, location=0.8, brand_strength=0.8, innovation_level=0.0
        ),
    ]

    costs = [10.0, 12.0, 8.0]
    prices = [20.0, 25.0, 18.0]

    print("\n1. Logit Demand Model:")
    logit_result = differentiated_bertrand_simulation(
        prices, products, costs, "logit", {}, 100.0
    )
    print(f"  Prices: {[f'{p:.2f}' for p in logit_result.prices]}")
    print(f"  Quantities: {[f'{q:.2f}' for q in logit_result.quantities]}")
    print(f"  Market Shares: {[f'{s:.3f}' for s in logit_result.market_shares]}")
    print(f"  Profits: {[f'{p:.2f}' for p in logit_result.profits]}")

    print("\n2. Hotelling Model:")
    hotelling_demand = HotellingDemand(transportation_cost=1.0, consumer_density=1.0)
    locations = [p.location for p in products]
    hotelling_quantities = hotelling_demand.calculate_demand(prices, locations)
    print(f"  Locations: {[f'{loc:.2f}' for loc in locations]}")
    print(f"  Quantities: {[f'{q:.2f}' for q in hotelling_quantities]}")

    print("\n3. Vertical Differentiation:")
    vertical_demand = VerticalDifferentiation(consumer_heterogeneity=1.0)
    qualities = [p.quality for p in products]
    vertical_quantities = vertical_demand.calculate_demand(prices, qualities, 100.0)
    print(f"  Qualities: {[f'{q:.2f}' for q in qualities]}")
    print(f"  Quantities: {[f'{q:.2f}' for q in vertical_quantities]}")

    print("\n4. Nash Equilibrium for Differentiated Products:")
    equilibrium_prices, equilibrium_result = calculate_differentiated_nash_equilibrium(
        products, costs, "logit", {}, 100.0
    )
    print(f"  Equilibrium Prices: {[f'{p:.2f}' for p in equilibrium_prices]}")
    print(
        f"  Equilibrium Quantities: {[f'{q:.2f}' for q in equilibrium_result.quantities]}"
    )
    print(f"  Equilibrium Profits: {[f'{p:.2f}' for p in equilibrium_result.profits]}")


def demonstrate_market_evolution() -> None:
    """Demonstrate market evolution dynamics."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Market Evolution")
    print("=" * 60)

    # Create market evolution engine
    config = MarketEvolutionConfig(
        entry_cost=100.0,
        exit_threshold=-50.0,
        growth_rate=0.02,
        innovation_cost=50.0,
        innovation_success_rate=0.3,
    )

    evolution_engine = create_market_evolution_engine(config, seed=42)

    # Initial market state
    current_firms = [0, 1, 2]
    current_profits = [50.0, 30.0, -60.0]  # Firm 2 is unprofitable
    current_market_shares = [0.4, 0.35, 0.25]
    current_costs = [10.0, 12.0, 15.0]
    current_qualities = [1.0, 1.1, 0.9]
    demand_params = {"a": 100.0, "b": 1.0}

    print("\nInitial Market State:")
    print(f"  Firms: {current_firms}")
    print(f"  Profits: {[f'{p:.2f}' for p in current_profits]}")
    print(f"  Market Shares: {[f'{s:.3f}' for s in current_market_shares]}")
    print(f"  Costs: {[f'{c:.2f}' for c in current_costs]}")
    print(f"  Qualities: {[f'{q:.2f}' for q in current_qualities]}")

    # Evolve market for several rounds
    for round_num in range(5):
        print(f"\nRound {round_num + 1}:")

        new_firms, new_costs, new_qualities, new_demand_params = (
            evolution_engine.evolve_market(
                current_firms,
                current_profits,
                current_market_shares,
                current_costs,
                current_qualities,
                demand_params,
            )
        )

        # Update state
        current_firms = new_firms
        current_costs = new_costs
        current_qualities = new_qualities
        demand_params = new_demand_params

        # Simulate profits for new state (simplified)
        current_profits = [max(-100.0, 50.0 - cost * 2) for cost in current_costs]
        total_profit = sum(current_profits)
        current_market_shares = [
            p / total_profit if total_profit > 0 else 1.0 / len(current_firms)
            for p in current_profits
        ]

        print(f"  Firms: {current_firms}")
        print(f"  Profits: {[f'{p:.2f}' for p in current_profits]}")
        print(f"  Market Shares: {[f'{s:.3f}' for s in current_market_shares]}")
        print(f"  Costs: {[f'{c:.2f}' for c in current_costs]}")
        print(f"  Qualities: {[f'{q:.2f}' for q in current_qualities]}")

        # Get evolution metrics
        metrics = evolution_engine.get_evolution_metrics()
        print(f"  Market Size: {metrics['total_market_size']:.2f}")
        print(f"  Technology Level: {metrics['technology_level']:.3f}")
        print(f"  Net Entries: {metrics['net_entries']}")


def demonstrate_enhanced_demand() -> None:
    """Demonstrate enhanced demand functions."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Enhanced Demand Functions")
    print("=" * 60)

    prices = [20.0, 25.0, 18.0]
    qualities = [1.0, 1.2, 0.8]

    print("\n1. CES Demand:")
    ces_demand = CESDemand(elasticity=2.0, scale_parameter=1.0, market_size=100.0)
    ces_quantities = ces_demand.calculate_demand(prices, qualities)
    ces_shares = ces_demand.calculate_market_shares(prices, qualities)
    print(f"  Prices: {[f'{p:.2f}' for p in prices]}")
    print(f"  Qualities: {[f'{q:.2f}' for q in qualities]}")
    print(f"  Quantities: {[f'{q:.2f}' for q in ces_quantities]}")
    print(f"  Market Shares: {[f'{s:.3f}' for s in ces_shares]}")

    print("\n2. Network Effects Demand:")
    network_demand = NetworkEffectsDemand(
        network_strength=0.1, base_demand=100.0, critical_mass=10.0
    )
    current_users = [15.0, 12.0, 8.0]
    network_quantities = network_demand.calculate_demand(
        prices, current_users, qualities
    )
    network_values = network_demand.calculate_network_value(current_users, qualities)
    print(f"  Current Users: {[f'{u:.1f}' for u in current_users]}")
    print(f"  Quantities: {[f'{q:.2f}' for q in network_quantities]}")
    print(f"  Network Values: {[f'{v:.2f}' for v in network_values]}")

    print("\n3. Dynamic Demand:")
    dynamic_demand = DynamicDemand(
        base_demand=100.0, growth_rate=0.02, volatility=0.1, seasonal_amplitude=0.1
    )
    for round_num in range(3):
        quantities, market_size = dynamic_demand.calculate_demand(
            round_num, prices, qualities
        )
        print(
            f"  Round {round_num}: Market Size = {market_size:.2f}, Quantities = {[f'{q:.2f}' for q in quantities]}"
        )

    print("\n4. Multi-Segment Demand:")
    segments = [
        {"weight": 0.4, "price_sensitivity": 1.0, "quality_preference": 1.0},
        {"weight": 0.6, "price_sensitivity": 0.5, "quality_preference": 1.5},
    ]
    multi_segment_demand = MultiSegmentDemand(segments)
    multi_quantities = multi_segment_demand.calculate_demand(prices, qualities, 100.0)
    multi_shares = multi_segment_demand.calculate_market_shares(
        prices, qualities, 100.0
    )
    print(f"  Quantities: {[f'{q:.2f}' for q in multi_quantities]}")
    print(f"  Market Shares: {[f'{s:.3f}' for s in multi_shares]}")


def demonstrate_api_usage() -> None:
    """Demonstrate how to use the new API features."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: API Usage Examples")
    print("=" * 60)

    print("\n1. Differentiated Bertrand Competition API Call:")
    api_request = {
        "model": "differentiated_bertrand",
        "rounds": 1,
        "params": {
            "demand_model": "logit",
            "demand_params": {"price_sensitivity": 1.0, "quality_sensitivity": 1.0},
            "total_market_size": 100.0,
        },
        "firms": [
            {
                "cost": 10.0,
                "product_characteristics": {
                    "quality": 1.0,
                    "location": 0.2,
                    "brand_strength": 1.2,
                    "innovation_level": 0.1,
                },
            },
            {
                "cost": 12.0,
                "product_characteristics": {
                    "quality": 1.2,
                    "location": 0.5,
                    "brand_strength": 1.0,
                    "innovation_level": 0.2,
                },
            },
        ],
    }

    print("  API Request:")
    print(json.dumps(api_request, indent=2))

    print("\n2. Advanced Strategy Configuration:")
    strategy_config = {
        "advanced_strategies": [
            {
                "strategy_type": "fictitious_play",
                "learning_rate": 0.1,
                "exploration_rate": 0.1,
                "memory_length": 20,
            },
            {
                "strategy_type": "deep_q_learning",
                "learning_rate": 0.01,
                "exploration_rate": 0.3,
                "memory_length": 50,
            },
        ]
    }

    print("  Strategy Configuration:")
    print(json.dumps(strategy_config, indent=2))

    print("\n3. Market Evolution Configuration:")
    evolution_config = {
        "market_evolution": {
            "enable_evolution": True,
            "entry_cost": 100.0,
            "exit_threshold": -50.0,
            "growth_rate": 0.02,
            "innovation_cost": 50.0,
            "innovation_success_rate": 0.3,
        }
    }

    print("  Evolution Configuration:")
    print(json.dumps(evolution_config, indent=2))

    print("\n4. Enhanced Demand Configuration:")
    demand_config = {
        "enhanced_demand": {
            "demand_type": "ces",
            "elasticity": 2.0,
            "network_strength": 0.1,
            "growth_rate": 0.02,
            "volatility": 0.1,
        }
    }

    print("  Demand Configuration:")
    print(json.dumps(demand_config, indent=2))


def main() -> None:
    """Run all demonstrations."""
    print("ADVANCED ECONOMICS DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows the new advanced economic features")
    print("implemented in the oligopoly simulation.")

    try:
        demonstrate_advanced_strategies()
        demonstrate_product_differentiation()
        demonstrate_market_evolution()
        demonstrate_enhanced_demand()
        demonstrate_api_usage()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("All advanced economic features have been demonstrated.")
        print("These features significantly enhance the realism and")
        print("applicability of the oligopoly simulation for research")
        print("and policy analysis.")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
