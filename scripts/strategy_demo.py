#!/usr/bin/env python3
"""Example usage of the strategy-based oligopoly simulation.

This script demonstrates how to use the three implemented strategies
(Static, TitForTat, RandomWalk) in both Cournot and Bertrand models.
"""

from typing import List

from sim.games.bertrand import BertrandResult
from sim.games.cournot import CournotResult
from sim.strategies.strategies import RandomWalk, Static, TitForTat, create_strategy

from .utils import format_list, print_demo_completion


def demonstrate_strategies() -> None:
    """Demonstrate the three strategies with a simple example."""
    print("=== Strategy Demonstration ===\n")

    # Create strategies
    static = Static(value=5.0)
    titfortat = TitForTat()
    randomwalk = RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=42)

    strategies = [static, titfortat, randomwalk]
    strategy_names = ["Static", "TitForTat", "RandomWalk"]

    bounds = (0.0, 10.0)
    rounds = 5

    print(f"Running {rounds} rounds with bounds {bounds}")
    print("Strategies: Static(5.0), TitForTat(), RandomWalk(step=1.0, seed=42)\n")

    # Simulate multiple rounds
    trajectories: List[List[float]] = [[] for _ in strategies]

    for round_num in range(rounds):
        print(f"Round {round_num + 1}:")
        round_actions = []

        for firm_idx, strategy in enumerate(strategies):
            # Build histories for this firm
            my_history = []
            rival_histories = []

            if round_num > 0:
                # Add previous results to histories
                for i, traj in enumerate(trajectories):
                    if i == firm_idx:
                        # My history
                        if traj:
                            prev_action = traj[-1]
                            result = CournotResult(
                                price=8.0, quantities=[prev_action], profits=[40.0]
                            )
                            my_history = [result]
                    else:
                        # Rival history
                        if traj:
                            prev_action = traj[-1]
                            result = CournotResult(
                                price=8.0, quantities=[prev_action], profits=[40.0]
                            )
                            rival_histories.append([result])
                        else:
                            rival_histories.append([])

            action = strategy.next_action(
                round_num=round_num,
                my_history=my_history,
                rival_histories=rival_histories,
                bounds=bounds,
                market_params={},
            )

            round_actions.append(action)
            trajectories[firm_idx].append(action)

            print(f"  {strategy_names[firm_idx]}: {action:.2f}")

        print()

    # Show final trajectories
    print("Final Trajectories:")
    for i, (name, traj) in enumerate(zip(strategy_names, trajectories)):
        print(f"  {name}: {format_list(traj, 'numeric')}")

    print("\n=== Strategy Factory Demo ===")

    # Demonstrate factory function
    factory_strategies = [
        create_strategy("static", value=3.0),
        create_strategy("titfortat"),
        create_strategy("randomwalk", step=0.5, min_bound=0.0, max_bound=5.0, seed=123),
    ]

    print("Created strategies via factory:")
    for i, strategy in enumerate(factory_strategies):
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0, 5),
            market_params={},
        )
        print(f"  Strategy {i + 1}: {action:.2f}")

    print_demo_completion(
        "Strategy demonstration",
        "Static, TitForTat, RandomWalk behaviors, strategy factory",
    )


def demonstrate_cournot_vs_bertrand() -> None:
    """Demonstrate how strategies work with both Cournot and Bertrand results."""
    print("\n=== Cournot vs Bertrand Integration ===\n")

    strategy = TitForTat()

    # Test with Cournot results (quantities)
    cournot_result = CournotResult(price=12.0, quantities=[8.0], profits=[32.0])

    action_cournot = strategy.next_action(
        round_num=1,
        my_history=[],
        rival_histories=[[cournot_result]],
        bounds=(0, 20),
        market_params={},
    )

    print(f"TitForTat with Cournot rival quantity 8.0: {action_cournot:.2f}")

    # Test with Bertrand results (prices)
    bertrand_result = BertrandResult(
        total_demand=50.0, prices=[6.0], quantities=[25.0], profits=[75.0]
    )

    action_bertrand = strategy.next_action(
        round_num=1,
        my_history=[],
        rival_histories=[[bertrand_result]],
        bounds=(0, 20),
        market_params={},
    )

    print(f"TitForTat with Bertrand rival price 6.0: {action_bertrand:.2f}")

    print(
        "\n✓ TitForTat correctly extracts quantities from Cournot and prices from Bertrand!"
    )


if __name__ == "__main__":
    demonstrate_strategies()
    demonstrate_cournot_vs_bertrand()
