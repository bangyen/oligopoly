#!/usr/bin/env python3
# type: ignore
"""Demo script for ε-greedy agents in oligopoly simulation.

This script demonstrates the ε-greedy strategy implementation with:
- 3 firms using ε-greedy strategy in Cournot competition
- 20 rounds of simulation
- 4th firm enters at round 10 with specified cost structure
- Comparison with baseline strategies (Static, TitForTat)

The script shows how ε-greedy agents learn to optimize their actions
based on immediate rewards (profits) while balancing exploration and exploitation.
"""

import sys
from typing import List

# Add src to path for imports
sys.path.insert(0, "src")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.models import Base
from sim.strategies import EpsilonGreedy, Static, TitForTat
from sim.strategy_runner import get_strategy_run_results, run_strategy_game

# Database setup (in-memory SQLite for demo)
engine = create_engine("sqlite:///:memory:", echo=False)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def calculate_hhi(quantities: List[float]) -> float:
    """Calculate Herfindahl-Hirschman Index (HHI) for market concentration."""
    total_quantity = sum(quantities)
    if total_quantity == 0:
        return 0.0
    market_shares = [q / total_quantity for q in quantities]
    return sum(share**2 for share in market_shares) * 10000


def calculate_consumer_surplus(
    price: float, total_quantity: float, a: float, b: float
) -> float:
    """Calculate consumer surplus for linear demand curve."""
    if total_quantity == 0:
        return 0.0
    # CS = 0.5 * (a - price) * total_quantity
    return 0.5 * max(0, a - price) * total_quantity


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

    print("Initial setup:")
    print(f"- Market parameters: a={market_params['a']}, b={market_params['b']}")
    print(f"- Action bounds: {bounds}")
    print(
        f"- Grid: {grid_params['min_action']} to {grid_params['max_action']} step {grid_params['step_size']}"
    )
    print(f"- Initial firms: {len(strategies)} firms with costs {initial_costs}")
    print(
        f"- Learning parameters: ε₀={grid_params['epsilon_0']}, α={grid_params['learning_rate']}, γ={grid_params['decay_rate']}"
    )
    print()

    # Run simulation with database
    db = SessionLocal()

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

        print(f"Completed rounds 0-9. Run ID: {run_id}")

        # Now add 4th firm and continue
        print("\nAdding 4th firm at round 10...")
        new_cost = 8.0  # Lower cost than existing firms
        print(f"- New firm cost: {new_cost}")

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

        print(f"Completed rounds 10-19. Run ID: {run_id_2}")

        # Combine all results
        all_rounds = rounds_0_9 + rounds_10_19

        # Print summary statistics
        print("\n=== Simulation Results ===")
        print(f"Total rounds: {len(all_rounds)}")
        print("Firms: 3 (rounds 0-9), 4 (rounds 10-19)")

        # Calculate averages for different periods
        pre_entry = rounds_0_9[-3:]  # Last 3 rounds before entry
        post_entry = rounds_10_19[3:]  # Rounds 13-19 (after entry effects)

        if pre_entry and post_entry:
            pre_avg_price = sum(float(r["price"]) for r in pre_entry) / len(pre_entry)
            post_avg_price = sum(float(r["price"]) for r in post_entry) / len(
                post_entry
            )

            pre_avg_hhi = sum(float(r["hhi"]) for r in pre_entry) / len(pre_entry)
            post_avg_hhi = sum(float(r["hhi"]) for r in post_entry) / len(post_entry)

            pre_avg_cs = sum(float(r["cs"]) for r in pre_entry) / len(pre_entry)
            post_avg_cs = sum(float(r["cs"]) for r in post_entry) / len(post_entry)

            print("\nPre-entry (rounds 7-9) averages:")
            print(f"  Price: {pre_avg_price:.2f}")
            print(f"  HHI: {pre_avg_hhi:.0f}")
            print(f"  Consumer Surplus: {pre_avg_cs:.2f}")

            print("\nPost-entry (rounds 13-19) averages:")
            print(f"  Price: {post_avg_price:.2f}")
            print(f"  HHI: {post_avg_hhi:.0f}")
            print(f"  Consumer Surplus: {post_avg_cs:.2f}")

            print("\nChanges after entry:")
            print(f"  Price change: {post_avg_price - pre_avg_price:+.2f}")
            print(f"  HHI change: {post_avg_hhi - pre_avg_hhi:+.0f}")
            print(f"  CS change: {post_avg_cs - pre_avg_cs:+.2f}")

        # Show Q-values evolution for first firm
        print("\n=== Learning Progress (Firm 0) ===")
        firm_0_strategy = strategies[0]
        print(f"Final ε: {firm_0_strategy.get_current_epsilon():.4f}")
        print(f"Final Q-values: {[f'{q:.2f}' for q in firm_0_strategy.get_q_values()]}")

        # Show action grid
        action_grid = firm_0_strategy.get_action_grid()
        print(
            f"Action grid: {action_grid[:5]}...{action_grid[-3:]} ({len(action_grid)} actions)"
        )

        # Find best action
        q_values = firm_0_strategy.get_q_values()
        best_idx = q_values.index(max(q_values))
        best_action = action_grid[best_idx]
        print(
            f"Best learned action: {best_action:.1f} (Q-value: {q_values[best_idx]:.2f})"
        )

    finally:
        db.close()


def run_baseline_comparison() -> None:
    """Run baseline strategies for comparison."""
    print("\n=== Baseline Strategy Comparison ===\n")

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

    print("Running baseline comparison with Static and TitForTat strategies...")

    db = SessionLocal()
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

        print("Baseline results (round 9):")
        print(f"  Quantities: {[f'{q:.1f}' for q in quantities]}")
        print(f"  Profits: {[f'{p:.1f}' for p in profits]}")
        print(f"  Price: {price:.2f}")
        print(f"  HHI: {calculate_hhi(quantities):.0f}")

    finally:
        db.close()


if __name__ == "__main__":
    run_epsilon_greedy_demo()
    run_baseline_comparison()

    print("\n=== Demo Commands ===")
    print("To run this demo:")
    print("  python scripts/epsilon_greedy_demo.py")
    print("\nExpected directional changes after entry:")
    print("  - Price: Should decrease (more competition)")
    print("  - HHI: Should decrease (less concentration)")
    print("  - Consumer Surplus: Should increase (lower prices)")
    print("\nTo run tests:")
    print("  pytest tests/unit/test_bandit_update.py -v")
    print("  pytest tests/unit/test_epsilon_decay.py -v")
    print("  pytest tests/unit/test_grid_bounds.py -v")
    print("  pytest tests/integration/test_demo_outcomes.py -v")
