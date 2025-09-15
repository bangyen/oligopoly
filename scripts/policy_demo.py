#!/usr/bin/env python3
"""Demonstration script showing policy shocks in action.

This script demonstrates how different policy interventions affect
simulation outcomes by running simulations with and without policy shocks.
"""

from typing import Any, Dict

from sim.policy.policy_shocks import PolicyEvent, PolicyType
from sim.runners.runner import get_run_results, run_game

from utils import create_demo_database, format_currency, print_demo_completion, print_round_results


def run_demo_simulation() -> None:
    """Run demonstration simulations with different policy shocks."""
    db = create_demo_database("sqlite:///./data/demo.db")

    try:
        print("=== Policy Shocks Demonstration ===\n")

        # Base configuration
        base_config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
            "seed": 42,
        }

        # 1. Baseline simulation (no policy shocks)
        config_baseline = {**base_config, "events": []}
        run_id_baseline = run_game("cournot", 3, config_baseline, db)
        results_baseline = get_run_results(run_id_baseline, db)

        print("1. Baseline (no policies):")
        print_round_results(results_baseline, "Baseline")

        # 2. Tax policy simulation
        config_tax = {
            **base_config,
            "events": [PolicyEvent(round_idx=1, policy_type=PolicyType.TAX, value=0.2)],
        }
        run_id_tax = run_game("cournot", 3, config_tax, db)
        results_tax = get_run_results(run_id_tax, db)

        print("\n2. Tax Policy (20% tax on round 1):")
        print_round_results(results_tax, "With Tax")

        # Compare tax effect
        baseline_round1 = results_baseline["rounds_data"][1]
        tax_round1 = results_tax["rounds_data"][1]
        tax_reduction = baseline_round1["total_profit"] - tax_round1["total_profit"]
        tax_rate_actual = tax_reduction / baseline_round1["total_profit"]
        print(f"   Tax effect: {format_currency(baseline_round1['total_profit'])} → {format_currency(tax_round1['total_profit'])} ({tax_rate_actual:.0%} reduction)")

        # 3. Subsidy policy simulation
        config_subsidy = {
            **base_config,
            "events": [
                PolicyEvent(round_idx=0, policy_type=PolicyType.SUBSIDY, value=5.0)
            ],
        }
        run_id_subsidy = run_game("cournot", 3, config_subsidy, db)
        results_subsidy = get_run_results(run_id_subsidy, db)

        print("\n3. Subsidy Policy ($5/unit on round 0):")
        print_round_results(results_subsidy, "With Subsidy")

        # Compare subsidy effect
        baseline_round0 = results_baseline["rounds_data"][0]
        subsidy_round0 = results_subsidy["rounds_data"][0]
        subsidy_increase = subsidy_round0["total_profit"] - baseline_round0["total_profit"]
        subsidy_per_unit = subsidy_increase / baseline_round0["total_qty"]
        print(f"   Subsidy effect: {format_currency(baseline_round0['total_profit'])} → {format_currency(subsidy_round0['total_profit'])} (+{format_currency(subsidy_per_unit)}/unit)")

        # 4. Price cap policy simulation
        config_cap = {
            **base_config,
            "events": [
                PolicyEvent(round_idx=2, policy_type=PolicyType.PRICE_CAP, value=50.0)
            ],
        }
        run_id_cap = run_game("cournot", 3, config_cap, db)
        results_cap = get_run_results(run_id_cap, db)

        print("\n4. Price Cap ($50 cap on round 2):")
        print_round_results(results_cap, "With Price Cap")

        # Compare price cap effect
        baseline_round2 = results_baseline["rounds_data"][2]
        cap_round2 = results_cap["rounds_data"][2]
        if baseline_round2["price"] > 50.0:
            print(f"   Price cap effect: {format_currency(baseline_round2['price'])} → {format_currency(cap_round2['price'])}")
        else:
            print(f"   Price cap effect: No change (price {format_currency(baseline_round2['price'])} < $50)")

        # 5. Multiple policies simulation
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

        print("\n5. Multiple Policies (subsidy→tax→price cap):")
        print_round_results(results_multi, "Multiple Policies")

        print_demo_completion(
            "Policy shocks",
            "Tax reduction, subsidy increase, price caps, multi-round policies"
        )

    finally:
        db.close()


if __name__ == "__main__":
    run_demo_simulation()
