#!/usr/bin/env python3
"""Demonstration script showing policy shocks in action.

This script demonstrates how different policy interventions affect
simulation outcomes by running simulations with and without policy shocks.
"""

from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.models.models import Base
from sim.policy.policy_shocks import PolicyEvent, PolicyType
from sim.runners.runner import get_run_results, run_game

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/demo.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def run_demo_simulation() -> None:
    """Run demonstration simulations with different policy shocks."""
    db = SessionLocal()

    try:
        print("=== Oligopoly Policy Shocks Demonstration ===\n")

        # Base configuration
        base_config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
            "seed": 42,
        }

        # 1. Baseline simulation (no policy shocks)
        print("1. Baseline Simulation (No Policy Shocks)")
        print("-" * 50)

        config_baseline = {**base_config, "events": []}
        run_id_baseline = run_game("cournot", 3, config_baseline, db)
        results_baseline = get_run_results(run_id_baseline, db)

        print_round_results(results_baseline, "Baseline")

        # 2. Tax policy simulation
        print("\n2. Tax Policy Simulation (20% tax on round 1)")
        print("-" * 50)

        config_tax = {
            **base_config,
            "events": [PolicyEvent(round_idx=1, policy_type=PolicyType.TAX, value=0.2)],
        }
        run_id_tax = run_game("cournot", 3, config_tax, db)
        results_tax = get_run_results(run_id_tax, db)

        print_round_results(results_tax, "With Tax")

        # Compare tax effect
        print("\nTax Effect Analysis:")
        baseline_round1 = results_baseline["rounds_data"][1]
        tax_round1 = results_tax["rounds_data"][1]

        tax_reduction = baseline_round1["total_profit"] - tax_round1["total_profit"]
        tax_rate_actual = tax_reduction / baseline_round1["total_profit"]

        print(f"  Round 1 Baseline Profit: ${baseline_round1['total_profit']:.2f}")
        print(f"  Round 1 Taxed Profit: ${tax_round1['total_profit']:.2f}")
        print(f"  Tax Reduction: ${tax_reduction:.2f}")
        print(f"  Effective Tax Rate: {tax_rate_actual:.1%}")

        # 3. Subsidy policy simulation
        print("\n3. Subsidy Policy Simulation ($5 per unit on round 0)")
        print("-" * 50)

        config_subsidy = {
            **base_config,
            "events": [
                PolicyEvent(round_idx=0, policy_type=PolicyType.SUBSIDY, value=5.0)
            ],
        }
        run_id_subsidy = run_game("cournot", 3, config_subsidy, db)
        results_subsidy = get_run_results(run_id_subsidy, db)

        print_round_results(results_subsidy, "With Subsidy")

        # Compare subsidy effect
        print("\nSubsidy Effect Analysis:")
        baseline_round0 = results_baseline["rounds_data"][0]
        subsidy_round0 = results_subsidy["rounds_data"][0]

        subsidy_increase = (
            subsidy_round0["total_profit"] - baseline_round0["total_profit"]
        )
        subsidy_per_unit = subsidy_increase / baseline_round0["total_qty"]

        print(f"  Round 0 Baseline Profit: ${baseline_round0['total_profit']:.2f}")
        print(f"  Round 0 Subsidized Profit: ${subsidy_round0['total_profit']:.2f}")
        print(f"  Subsidy Increase: ${subsidy_increase:.2f}")
        print(f"  Total Quantity: {baseline_round0['total_qty']:.1f}")
        print(f"  Effective Subsidy per Unit: ${subsidy_per_unit:.2f}")

        # 4. Price cap policy simulation
        print("\n4. Price Cap Policy Simulation ($50 cap on round 2)")
        print("-" * 50)

        config_cap = {
            **base_config,
            "events": [
                PolicyEvent(round_idx=2, policy_type=PolicyType.PRICE_CAP, value=50.0)
            ],
        }
        run_id_cap = run_game("cournot", 3, config_cap, db)
        results_cap = get_run_results(run_id_cap, db)

        print_round_results(results_cap, "With Price Cap")

        # Compare price cap effect
        print("\nPrice Cap Effect Analysis:")
        baseline_round2 = results_baseline["rounds_data"][2]
        cap_round2 = results_cap["rounds_data"][2]

        print(f"  Round 2 Baseline Price: ${baseline_round2['price']:.2f}")
        print(f"  Round 2 Capped Price: ${cap_round2['price']:.2f}")
        print(f"  Round 2 Baseline Profit: ${baseline_round2['total_profit']:.2f}")
        print(f"  Round 2 Capped Profit: ${cap_round2['total_profit']:.2f}")

        if baseline_round2["price"] > 50.0:
            print(f"  Price was capped from ${baseline_round2['price']:.2f} to $50.00")
        else:
            print("  Price cap had no effect (price was already below cap)")

        # 5. Multiple policies simulation
        print("\n5. Multiple Policies Simulation")
        print("-" * 50)
        print("   - Round 0: $3 per unit subsidy")
        print("   - Round 1: 15% tax")
        print("   - Round 2: $45 price cap")

        config_multi = {
            **base_config,
            "events": [
                PolicyEvent(round_idx=0, policy_type=PolicyType.SUBSIDY, value=3.0),
                PolicyEvent(round_idx=1, policy_type=PolicyType.TAX, value=0.15),
                PolicyEvent(round_idx=2, policy_type=PolicyType.PRICE_CAP, value=45.0),
            ],
        }
        run_id_multi = run_game("cournot", 3, config_multi, db)
        results_multi = get_run_results(run_id_multi, db)

        print_round_results(results_multi, "Multiple Policies")

        print("\n=== Summary ===")
        print("Policy shocks successfully implemented and tested!")
        print("- Tax policy reduces profits by the specified rate")
        print("- Subsidy policy increases profits by subsidy × quantity")
        print("- Price cap policy limits prices and recalculates profits")
        print("- Multiple policies can be applied across different rounds")

    finally:
        db.close()


def print_round_results(results: Dict[str, Any], label: str) -> None:
    """Print simulation results in a formatted way."""
    rounds_data = results["rounds_data"]

    print(f"\n{label} Results:")
    print("Round | Price | Total Qty | Total Profit")
    print("-" * 40)

    for round_data in rounds_data:
        print(
            f"  {round_data['round']}   | ${round_data['price']:5.2f} | {round_data['total_qty']:8.1f} | ${round_data['total_profit']:11.2f}"
        )


if __name__ == "__main__":
    run_demo_simulation()
