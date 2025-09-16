#!/usr/bin/env python3
"""Demo script for ε-greedy agents in oligopoly simulation.

This script demonstrates the ε-greedy strategy implementation with:
- 3 firms using ε-greedy strategy in Cournot competition
- 20 rounds of simulation
- 4th firm enters at round 10 with specified cost structure
- Comparison with baseline strategies (Static, TitForTat)

The script shows how ε-greedy agents learn to optimize their actions
based on immediate rewards (profits) while balancing exploration and exploitation.
"""


from typing import List, TypedDict, cast

from sim.runners.strategy_runner import get_strategy_run_results, run_strategy_game
from sim.strategies.strategies import EpsilonGreedy, Static, TitForTat

from .utils import (
    calculate_consumer_surplus,
    calculate_hhi,
    create_demo_database,
    format_currency,
    format_list,
    print_demo_completion,
)


class RoundData(TypedDict):
    """Type definition for round data structure."""

    round: int
    quantities: List[float]
    price: float
    profits: List[float]
    hhi: float
    cs: float


def run_epsilon_greedy_demo() -> None:
    """Run the ε-greedy demo scenario."""
    print("=== ε-Greedy Oligopoly Simulation Demo ===\n")

    # Market parameters for Cournot competition
    market_params = {
        "a": 100.0,  # Maximum price
        "b": 1.0,  # Price sensitivity
    }

    # Action bounds for quantities
    bounds = (0.0, 50.0)  # Min and max quantities

    # Grid parameters for ε-greedy
    grid_params = {
        "min_action": 0.0,
        "max_action": 50.0,
        "step_size": 2.0,  # Discrete steps of 2 units
        "epsilon_0": 0.2,  # Initial exploration rate
        "epsilon_min": 0.01,  # Minimum exploration rate
        "learning_rate": 0.1,  # Q-learning rate
        "decay_rate": 0.95,  # ε decay rate
    }

    # Initial firm costs (3 firms)
    initial_costs = [10.0, 12.0, 15.0]

    # Create strategies for initial 3 firms
    strategies = [
        EpsilonGreedy(**grid_params, seed=42),
        EpsilonGreedy(**grid_params, seed=43),
        EpsilonGreedy(**grid_params, seed=44),
    ]

    print(
        f"Setup: {len(strategies)} firms, costs {initial_costs}, ε₀={grid_params['epsilon_0']}, α={grid_params['learning_rate']}"
    )

    # Run simulation with database
    db = create_demo_database()

    try:
        # Run first 10 rounds with 3 firms
        print("Running rounds 0-9 with 3 firms...")
        run_id = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=initial_costs,
            params=market_params,
            bounds=bounds,
            db=db,
            seed=42,
        )

        # Get results for first 10 rounds
        results = get_strategy_run_results(run_id, db)

        # Calculate metrics for rounds 0-9
        rounds_0_9 = []
        for round_idx in range(10):
            round_data = results["results"][round_idx]
            quantities = [
                round_data[firm_id]["quantity"] for firm_id in sorted(round_data.keys())
            ]
            prices = [
                round_data[firm_id]["price"] for firm_id in sorted(round_data.keys())
            ]
            profits = [
                round_data[firm_id]["profit"] for firm_id in sorted(round_data.keys())
            ]

            total_qty = sum(quantities)
            avg_price = (
                prices[0] if prices else 0.0
            )  # All firms have same price in Cournot
            hhi = calculate_hhi(quantities)
            cs = calculate_consumer_surplus(
                avg_price, total_qty, market_params["a"], market_params["b"]
            )

            rounds_0_9.append(
                {
                    "round": round_idx,
                    "quantities": quantities,
                    "price": avg_price,
                    "profits": profits,
                    "hhi": hhi,
                    "cs": cs,
                }
            )

        # Now add 4th firm and continue
        print("Adding 4th firm (cost: 8.0) at round 10...")
        new_cost = 8.0  # Lower cost than existing firms

        # Create new strategy for 4th firm
        new_strategy = EpsilonGreedy(**grid_params, seed=45)

        # Add to existing strategies and costs
        strategies.append(new_strategy)
        all_costs = initial_costs + [new_cost]

        # Run remaining 10 rounds (10-19) with 4 firms
        print("Running rounds 10-19 with 4 firms...")
        run_id_2 = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=all_costs,
            params=market_params,
            bounds=bounds,
            db=db,
            seed=42,
        )

        # Get results for rounds 10-19
        results_2 = get_strategy_run_results(run_id_2, db)

        # Calculate metrics for rounds 10-19
        rounds_10_19 = []
        for round_idx in range(10):
            round_data = results_2["results"][round_idx]
            quantities = [
                round_data[firm_id]["quantity"] for firm_id in sorted(round_data.keys())
            ]
            prices = [
                round_data[firm_id]["price"] for firm_id in sorted(round_data.keys())
            ]
            profits = [
                round_data[firm_id]["profit"] for firm_id in sorted(round_data.keys())
            ]

            total_qty = sum(quantities)
            avg_price = prices[0] if prices else 0.0
            hhi = calculate_hhi(quantities)
            cs = calculate_consumer_surplus(
                avg_price, total_qty, market_params["a"], market_params["b"]
            )

            rounds_10_19.append(
                {
                    "round": round_idx + 10,  # Adjust round number
                    "quantities": quantities,
                    "price": avg_price,
                    "profits": profits,
                    "hhi": hhi,
                    "cs": cs,
                }
            )

        # Calculate averages for different periods
        pre_entry = rounds_0_9[-3:]  # Last 3 rounds before entry
        post_entry = rounds_10_19[3:]  # Rounds 13-19 (after entry effects)

        if pre_entry and post_entry:
            # Cast to proper types for MyPy
            pre_entry_typed = [cast(RoundData, r) for r in pre_entry]
            post_entry_typed = [cast(RoundData, r) for r in post_entry]

            pre_avg_price = sum(r["price"] for r in pre_entry_typed) / len(
                pre_entry_typed
            )
            post_avg_price = sum(r["price"] for r in post_entry_typed) / len(
                post_entry_typed
            )
            pre_avg_hhi = sum(r["hhi"] for r in pre_entry_typed) / len(pre_entry_typed)
            post_avg_hhi = sum(r["hhi"] for r in post_entry_typed) / len(
                post_entry_typed
            )
            pre_avg_cs = sum(r["cs"] for r in pre_entry_typed) / len(pre_entry_typed)
            post_avg_cs = sum(r["cs"] for r in post_entry_typed) / len(post_entry_typed)

            print("\nResults: 20 rounds (3 firms → 4 firms)")
            print(
                f"Pre-entry: Price {format_currency(pre_avg_price)}, HHI {pre_avg_hhi:.0f}, CS {format_currency(pre_avg_cs)}"
            )
            print(
                f"Post-entry: Price {format_currency(post_avg_price)}, HHI {post_avg_hhi:.0f}, CS {format_currency(post_avg_cs)}"
            )
            print(
                f"Changes: Price {post_avg_price - pre_avg_price:+.1f}, HHI {post_avg_hhi - pre_avg_hhi:+.0f}, CS {post_avg_cs - pre_avg_cs:+.0f}"
            )

        # Show learning progress for first firm
        firm_0_strategy = strategies[0]
        q_values = firm_0_strategy.get_q_values()
        action_grid = firm_0_strategy.get_action_grid()
        best_idx = q_values.index(max(q_values))
        best_action = action_grid[best_idx]

        print(
            f"\nLearning (Firm 0): Final ε={firm_0_strategy.get_current_epsilon():.3f}, Best action={best_action:.1f} (Q={q_values[best_idx]:.1f})"
        )

    finally:
        db.close()


def run_baseline_comparison() -> None:
    """Run baseline strategies for comparison."""
    print("\n=== Baseline Comparison ===")

    # Same market parameters
    market_params = {"a": 100.0, "b": 1.0}
    bounds = (0.0, 50.0)
    costs = [10.0, 12.0, 15.0]

    # Create baseline strategies
    baseline_strategies = [
        Static(value=20.0),  # Static quantity
        TitForTat(),  # Tit-for-tat
        Static(value=25.0),  # Different static quantity
    ]

    db = create_demo_database()
    try:
        run_id = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=baseline_strategies,
            costs=costs,
            params=market_params,
            bounds=bounds,
            db=db,
            seed=42,
        )

        results = get_strategy_run_results(run_id, db)

        # Calculate final round metrics
        final_round = results["results"][9]  # Round 9
        quantities = [
            final_round[firm_id]["quantity"] for firm_id in sorted(final_round.keys())
        ]
        profits = [
            final_round[firm_id]["profit"] for firm_id in sorted(final_round.keys())
        ]
        price = final_round[0]["price"]

        print(
            f"Baseline (Static/TitForTat): Price {format_currency(price)}, HHI {calculate_hhi(quantities):.0f}"
        )
        print(f"Profits: {format_list(profits)}")

    finally:
        db.close()


if __name__ == "__main__":
    run_epsilon_greedy_demo()
    run_baseline_comparison()

    print_demo_completion(
        "ε-Greedy learning",
        "Q-learning, exploration-exploitation, firm entry dynamics, baseline comparison",
    )
    print("Expected: Price ↓, HHI ↓, Consumer Surplus ↑ after entry")
