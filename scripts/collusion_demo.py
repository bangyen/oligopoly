#!/usr/bin/env python3
"""Demo script for collusion and regulator dynamics.

This script demonstrates the collusion features including cartel formation,
defection detection, regulatory intervention, and event logging.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Any

from sim.collusion import CollusionEventType, CollusionManager, RegulatorState
from sim.games.bertrand import bertrand_simulation
from sim.strategies.collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_event(event: Any) -> None:
    """Print a formatted event."""
    print(f"  Round {event.round_idx}: {event.description}")
    if event.data:
        for key, value in event.data.items():
            print(f"    {key}: {value}")


def demo_cartel_stability() -> None:
    """Demonstrate cartel formation and stability."""
    print_header("CARTEL STABILITY DEMONSTRATION")

    manager = CollusionManager()

    # Form a cartel
    print("1. Forming cartel with 3 firms...")
    manager.form_cartel(
        round_idx=0,
        collusive_price=60.0,
        collusive_quantity=8.0,
        participating_firms=[0, 1, 2],
    )

    print(f"   Cartel active: {manager.is_cartel_active()}")
    print(f"   Collusive price: ${manager.current_cartel.collusive_price}")
    print(
        f"   Collusive quantity per firm: {manager.current_cartel.collusive_quantity}"
    )

    # Simulate cartel compliance
    print("\n2. Simulating cartel compliance for 3 rounds...")
    costs = [20.0, 25.0, 22.0]

    for round_idx in range(1, 4):
        # All firms follow cartel agreement
        prices = [60.0, 60.0, 60.0]
        quantities = [8.0, 8.0, 8.0]

        # Run Bertrand simulation (result not used, just for completeness)
        bertrand_simulation(alpha=100.0, beta=1.0, costs=costs, prices=prices)

        # Calculate profits
        profits = [
            (price - cost) * qty for price, cost, qty in zip(prices, costs, quantities)
        ]

        print(f"   Round {round_idx}:")
        print(f"     Prices: {prices}")
        print(f"     Quantities: {quantities}")
        print(f"     Profits: {[f'${p:.1f}' for p in profits]}")

        # Calculate HHI
        total_quantity = sum(quantities)
        market_shares = [q / total_quantity for q in quantities]
        hhi = manager.calculate_hhi(market_shares)
        print(f"     HHI: {hhi:.3f}")

    print("\n3. Cartel stability results:")
    print("   ✓ High profits for all firms")
    print("   ✓ High market concentration (HHI)")
    print("   ✓ Stable prices")


def demo_defection() -> None:
    """Demonstrate defection behavior."""
    print_header("DEFECTION DEMONSTRATION")

    manager = CollusionManager()

    # Form cartel
    manager.form_cartel(
        round_idx=0,
        collusive_price=50.0,
        collusive_quantity=10.0,
        participating_firms=[0, 1, 2],
    )

    print("1. Cartel formed with price $50 and quantity 10 per firm")

    # Simulate compliance for 2 rounds
    print("\n2. Rounds 1-2: All firms comply with cartel")
    costs = [20.0, 20.0, 20.0]

    for round_idx in range(1, 3):
        prices = [50.0, 50.0, 50.0]
        quantities = [10.0, 10.0, 10.0]

        # Run Bertrand simulation (result not used, just for completeness)
        bertrand_simulation(alpha=100.0, beta=1.0, costs=costs, prices=prices)
        profits = [
            (price - cost) * qty for price, cost, qty in zip(prices, costs, quantities)
        ]

        print(f"   Round {round_idx}: Profits = {[f'${p:.1f}' for p in profits]}")

    # Firm 1 defects
    print("\n3. Round 3: Firm 1 defects by undercutting price")

    # Check for defection
    defected = manager.detect_defection(
        round_idx=3,
        firm_id=1,
        firm_price=45.0,  # 10% undercut
        firm_quantity=10.0,
        cartel_price=50.0,
        cartel_quantity=10.0,
    )

    if defected:
        print("   ✓ Defection detected!")
        print(f"   Defection count for Firm 1: {manager.get_firm_defection_count(1)}")

        # Show defection profit advantage
        compliant_profit = (50.0 - 20.0) * 10.0  # $300
        defection_profit = (45.0 - 20.0) * 15.0  # Assume captures more demand: $375

        print(f"   Compliant profit: ${compliant_profit}")
        print(f"   Defection profit: ${defection_profit}")
        print(f"   Profit advantage: ${defection_profit - compliant_profit:.1f}")

    # Show events
    print("\n4. Events logged:")
    events = manager.get_events_for_round(3)
    for event in events:
        print_event(event)


def demo_regulator_intervention() -> None:
    """Demonstrate regulator intervention."""
    print_header("REGULATOR INTERVENTION DEMONSTRATION")

    # Configure regulator
    regulator_state = RegulatorState(
        hhi_threshold=0.8,
        price_threshold_multiplier=1.5,
        baseline_price=30.0,
        intervention_probability=1.0,
        penalty_amount=100.0,
    )
    manager = CollusionManager(regulator_state)

    print("1. Regulator configured:")
    print(f"   HHI threshold: {regulator_state.hhi_threshold}")
    print(
        f"   Price threshold: {regulator_state.baseline_price * regulator_state.price_threshold_multiplier}"
    )
    print(f"   Penalty amount: ${regulator_state.penalty_amount}")

    # Form cartel
    manager.form_cartel(
        round_idx=0,
        collusive_price=50.0,
        collusive_quantity=10.0,
        participating_firms=[0, 1, 2],
    )

    print("\n2. Cartel formed with high prices")

    # Simulate high concentration and high prices
    print("\n3. Simulating high market concentration...")

    # High concentration market (one dominant firm)
    market_shares = [0.85, 0.10, 0.05]  # HHI = 0.85² + 0.10² + 0.05² = 0.735
    # Let's make it even higher
    market_shares = [0.9, 0.08, 0.02]  # HHI = 0.9² + 0.08² + 0.02² = 0.8168

    prices = [50.0, 50.0, 50.0]  # Above threshold of 30.0 * 1.5 = 45.0
    quantities = [18.0, 1.6, 0.4]  # Proportional to market shares

    print(f"   Market shares: {market_shares}")
    print(f"   Prices: {prices}")
    print(f"   Quantities: {quantities}")

    # Calculate HHI and average price
    hhi = manager.calculate_hhi(market_shares)
    avg_price = manager.calculate_average_price(prices, quantities)

    print(f"   HHI: {hhi:.3f} (threshold: {regulator_state.hhi_threshold})")
    print(
        f"   Average price: ${avg_price:.1f} (threshold: ${regulator_state.baseline_price * regulator_state.price_threshold_multiplier})"
    )

    # Check for intervention
    should_intervene, intervention_type, intervention_value = (
        manager.check_regulator_intervention(
            round_idx=5,
            market_shares=market_shares,
            prices=prices,
            quantities=quantities,
        )
    )

    if should_intervene:
        print("\n4. ✓ Regulator intervenes!")
        print(f"   Intervention type: {intervention_type}")
        print(f"   Intervention value: {intervention_value}")

        # Apply intervention
        original_profits = [300.0, 250.0, 200.0]
        modified_profits = manager.apply_regulator_intervention(
            round_idx=5,
            intervention_type=intervention_type,
            intervention_value=intervention_value,
            firm_profits=original_profits,
        )

        print(f"   Original profits: {[f'${p:.1f}' for p in original_profits]}")
        print(f"   Modified profits: {[f'${p:.1f}' for p in modified_profits]}")

    # Show events
    print("\n5. Events logged:")
    events = manager.get_events_for_round(5)
    for event in events:
        print_event(event)


def demo_event_feed() -> None:
    """Demonstrate comprehensive event logging."""
    print_header("EVENT FEED DEMONSTRATION")

    manager = CollusionManager()

    print("1. Complete collusion scenario with event logging:")

    # Round 0: Cartel formation
    manager.form_cartel(
        round_idx=0,
        collusive_price=55.0,
        collusive_quantity=9.0,
        participating_firms=[0, 1, 2],
    )

    # Rounds 1-2: Compliance
    for round_idx in range(1, 3):
        manager.detect_defection(
            round_idx=round_idx,
            firm_id=0,
            firm_price=55.0,
            firm_quantity=9.0,
            cartel_price=55.0,
            cartel_quantity=9.0,
        )

    # Round 3: Defection
    manager.detect_defection(
        round_idx=3,
        firm_id=1,
        firm_price=50.0,  # Defects
        firm_quantity=9.0,
        cartel_price=55.0,
        cartel_quantity=9.0,
    )

    # Round 4: Another defection
    manager.detect_defection(
        round_idx=4,
        firm_id=2,
        firm_price=48.0,  # Also defects
        firm_quantity=9.0,
        cartel_price=55.0,
        cartel_quantity=9.0,
    )

    # Round 5: Regulator intervention
    market_shares = [0.9, 0.08, 0.02]
    prices = [55.0, 50.0, 48.0]
    quantities = [9.0, 9.0, 9.0]

    should_intervene, intervention_type, intervention_value = (
        manager.check_regulator_intervention(
            round_idx=5,
            market_shares=market_shares,
            prices=prices,
            quantities=quantities,
        )
    )

    if should_intervene:
        manager.apply_regulator_intervention(
            round_idx=5,
            intervention_type=intervention_type,
            intervention_value=intervention_value,
            firm_profits=[300.0, 250.0, 200.0],
        )

    # Display all events
    print("\n2. Event timeline:")
    for round_idx in range(6):
        events = manager.get_events_for_round(round_idx)
        if events:
            print(f"\n   Round {round_idx}:")
            for event in events:
                print_event(event)

    # Summary statistics
    print("\n3. Summary statistics:")
    total_events = len(manager.events)
    cartel_events = len(
        [e for e in manager.events if e.event_type == CollusionEventType.CARTEL_FORMED]
    )
    defection_events = len(
        [e for e in manager.events if e.event_type == CollusionEventType.FIRM_DEFECTED]
    )
    intervention_events = len(
        [
            e
            for e in manager.events
            if e.event_type == CollusionEventType.REGULATOR_INTERVENED
        ]
    )

    print(f"   Total events: {total_events}")
    print(f"   Cartel formations: {cartel_events}")
    print(f"   Defections: {defection_events}")
    print(f"   Regulator interventions: {intervention_events}")

    # Defection counts by firm
    print("   Defection counts by firm:")
    for firm_id in range(3):
        count = manager.get_firm_defection_count(firm_id)
        print(f"     Firm {firm_id}: {count} defections")


def demo_strategies() -> None:
    """Demonstrate collusion-aware strategies."""
    print_header("COLLUSION STRATEGIES DEMONSTRATION")

    manager = CollusionManager()

    # Form cartel
    manager.form_cartel(
        round_idx=0,
        collusive_price=50.0,
        collusive_quantity=10.0,
        participating_firms=[0, 1, 2],
    )

    print("1. Cartel formed with price $50 and quantity 10")

    # Test different strategies
    strategies = [
        ("Cartel Strategy", CartelStrategy()),
        ("Collusive Strategy", CollusiveStrategy(defection_probability=0.3)),
        (
            "Opportunistic Strategy",
            OpportunisticStrategy(profit_threshold_multiplier=1.2),
        ),
    ]

    print("\n2. Testing different strategies:")

    for name, strategy in strategies:
        print(f"\n   {name}:")

        if isinstance(strategy, CartelStrategy):
            action = strategy.next_action(
                round_num=1,
                my_history=[],
                rival_histories=[],
                bounds=(0, 100),
                market_params={"model_type": "bertrand"},
                collusion_manager=manager,
            )
            print(f"     Action: ${action:.1f} (always follows cartel)")

        elif isinstance(strategy, CollusiveStrategy):
            prob = strategy.calculate_defection_probability(
                round_num=1,
                my_history=[],
                rival_histories=[],
                collusion_manager=manager,
            )
            print(f"     Defection probability: {prob:.2f}")

        elif isinstance(strategy, OpportunisticStrategy):
            cartel_profit = strategy.estimate_cartel_profit(
                cartel_price=50.0,
                cartel_quantity=10.0,
                my_cost=20.0,
                model_type="bertrand",
            )
            defection_profit = strategy.estimate_defection_profit(
                cartel_price=50.0,
                cartel_quantity=10.0,
                my_cost=20.0,
                market_params={"alpha": 100.0, "beta": 1.0},
                model_type="bertrand",
            )
            print(f"     Cartel profit estimate: ${cartel_profit:.1f}")
            print(f"     Defection profit estimate: ${defection_profit:.1f}")
            print(f"     Profit advantage: {defection_profit/cartel_profit:.2f}x")


def main() -> None:
    """Run all demonstrations."""
    print("OLIGOPOLY COLLUSION & REGULATOR DYNAMICS DEMO")
    print("=" * 60)

    try:
        demo_cartel_stability()
        demo_defection()
        demo_regulator_intervention()
        demo_event_feed()
        demo_strategies()

        print_header("DEMO COMPLETE")
        print(
            "All collusion and regulator dynamics features demonstrated successfully!"
        )
        print("\nKey features showcased:")
        print("✓ Cartel formation and stability")
        print("✓ Defection detection and profit advantages")
        print("✓ Regulator monitoring (HHI and price thresholds)")
        print("✓ Regulatory interventions (penalties and price caps)")
        print("✓ Comprehensive event logging")
        print("✓ Collusion-aware strategies")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
