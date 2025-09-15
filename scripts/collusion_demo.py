#!/usr/bin/env python3
"""Demo script for collusion and regulator dynamics.

This script demonstrates the collusion features including cartel formation,
defection detection, regulatory intervention, and event logging.
"""


from typing import Any

from sim.collusion import CollusionEventType, CollusionManager, RegulatorState
from sim.games.bertrand import bertrand_simulation
from sim.strategies.collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
)

from utils import (
    format_currency,
    format_list,
    print_demo_completion,
    print_header,
    print_summary,
)


def print_event(event: Any) -> None:
    """Print a formatted event."""
    print(f"  Round {event.round_idx}: {event.description}")
    if event.data:
        for key, value in event.data.items():
            print(f"    {key}: {value}")


def demo_cartel_stability() -> None:
    """Demonstrate cartel formation and stability."""
    print("\n=== CARTEL STABILITY ===")

    manager = CollusionManager()

    # Form a cartel
    manager.form_cartel(
        round_idx=0,
        collusive_price=60.0,
        collusive_quantity=8.0,
        participating_firms=[0, 1, 2],
    )

    print(f"Cartel formed: {manager.current_cartel.collusive_price} price, {manager.current_cartel.collusive_quantity} qty per firm")

    # Simulate cartel compliance
    print("\nSimulating 3 rounds of compliance...")
    costs = [20.0, 25.0, 22.0]
    profits_by_round = []

    for round_idx in range(1, 4):
        prices = [60.0, 60.0, 60.0]
        quantities = [8.0, 8.0, 8.0]
        bertrand_simulation(alpha=100.0, beta=1.0, costs=costs, prices=prices)
        
        profits = [(price - cost) * qty for price, cost, qty in zip(prices, costs, quantities)]
        profits_by_round.append(profits)

    # Show summary
    avg_profits = [sum(round_profits[i] for round_profits in profits_by_round) / len(profits_by_round) 
                   for i in range(3)]
    print(f"Average profits: {format_list(avg_profits)}")
    
    total_quantity = sum([8.0, 8.0, 8.0])
    market_shares = [8.0 / total_quantity] * 3
    hhi = manager.calculate_hhi(market_shares)
    print(f"Market concentration (HHI): {hhi:.3f}")

    print_summary("Results", [
        "Stable collusive pricing maintained",
        "High profits for all participants", 
        "Perfect market concentration achieved"
    ])


def demo_defection() -> None:
    """Demonstrate defection behavior."""
    print("\n=== DEFECTION DETECTION ===")

    manager = CollusionManager()

    # Form cartel
    manager.form_cartel(
        round_idx=0,
        collusive_price=50.0,
        collusive_quantity=10.0,
        participating_firms=[0, 1, 2],
    )

    print(f"Cartel formed: {format_currency(50.0)} price, 10 qty per firm")

    # Simulate compliance for 2 rounds
    costs = [20.0, 20.0, 20.0]
    compliant_profit = (50.0 - 20.0) * 10.0
    print(f"Rounds 1-2: All firms comply (profit: {format_currency(compliant_profit)} each)")

    # Firm 1 defects
    defected = manager.detect_defection(
        round_idx=3,
        firm_id=1,
        firm_price=45.0,  # 10% undercut
        firm_quantity=10.0,
        cartel_price=50.0,
        cartel_quantity=10.0,
    )

    if defected:
        defection_profit = (45.0 - 20.0) * 15.0  # Assume captures more demand
        profit_advantage = defection_profit - compliant_profit
        
        print(f"\nRound 3: Firm 1 defects (price: {format_currency(45.0)})")
        print(f"Defection profit: {format_currency(defection_profit)} (+{format_currency(profit_advantage)})")
        print(f"Defection count: {manager.get_firm_defection_count(1)}")

    print_summary("Key Events", [
        "Defection detected and logged",
        "Significant profit advantage from undercutting",
        "Event tracking system activated"
    ])


def demo_regulator_intervention() -> None:
    """Demonstrate regulator intervention."""
    print("\n=== REGULATOR INTERVENTION ===")

    # Configure regulator
    regulator_state = RegulatorState(
        hhi_threshold=0.8,
        price_threshold_multiplier=1.5,
        baseline_price=30.0,
        intervention_probability=1.0,
        penalty_amount=100.0,
    )
    manager = CollusionManager(regulator_state)

    price_threshold = regulator_state.baseline_price * regulator_state.price_threshold_multiplier
    print(f"Regulator thresholds: HHI > {regulator_state.hhi_threshold}, Price > {format_currency(price_threshold)}")

    # Form cartel
    manager.form_cartel(
        round_idx=0,
        collusive_price=50.0,
        collusive_quantity=10.0,
        participating_firms=[0, 1, 2],
    )

    # Simulate high concentration and high prices
    market_shares = [0.9, 0.08, 0.02]  # HHI = 0.8168
    prices = [50.0, 50.0, 50.0]  # Above threshold
    quantities = [18.0, 1.6, 0.4]

    hhi = manager.calculate_hhi(market_shares)
    avg_price = manager.calculate_average_price(prices, quantities)

    print(f"\nMarket conditions: HHI = {hhi:.3f}, Avg price = {format_currency(avg_price)}")

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
        print(f"✓ Regulator intervenes: {intervention_type} = {intervention_value}")
        
        original_profits = [300.0, 250.0, 200.0]
        modified_profits = manager.apply_regulator_intervention(
            round_idx=5,
            intervention_type=intervention_type,
            intervention_value=intervention_value,
            firm_profits=original_profits,
        )
        print(f"Profit impact: {format_list(original_profits)} → {format_list(modified_profits)}")

    print_summary("Intervention Results", [
        "High concentration detected (HHI > threshold)",
        "Price cap intervention triggered",
        "Market power constrained by regulation"
    ])


def demo_event_feed() -> None:
    """Demonstrate comprehensive event logging."""
    print("\n=== EVENT LOGGING ===")

    manager = CollusionManager()

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

    # Show key events only
    print("Key events logged:")
    key_rounds = [0, 3, 4, 5]
    for round_idx in key_rounds:
        events = manager.get_events_for_round(round_idx)
        if events:
            print(f"\nRound {round_idx}:")
            for event in events:
                print(f"  • {event.description}")

    # Summary statistics
    total_events = len(manager.events)
    cartel_events = len([e for e in manager.events if e.event_type == CollusionEventType.CARTEL_FORMED])
    defection_events = len([e for e in manager.events if e.event_type == CollusionEventType.FIRM_DEFECTED])
    intervention_events = len([e for e in manager.events if e.event_type == CollusionEventType.REGULATOR_INTERVENED])

    print(f"\nEvent summary: {total_events} total ({cartel_events} formations, {defection_events} defections, {intervention_events} interventions)")

    print_summary("Event Tracking Features", [
        "Comprehensive event logging across all rounds",
        "Defection detection and counting by firm",
        "Regulatory intervention monitoring",
        "Timeline reconstruction capability"
    ])


def demo_strategies() -> None:
    """Demonstrate collusion-aware strategies."""
    print("\n=== COLLUSION STRATEGIES ===")

    manager = CollusionManager()

    # Form cartel
    manager.form_cartel(
        round_idx=0,
        collusive_price=50.0,
        collusive_quantity=10.0,
        participating_firms=[0, 1, 2],
    )

    print(f"Cartel formed: {format_currency(50.0)} price, 10 qty per firm")

    # Test different strategies
    strategies = [
        ("Cartel Strategy", CartelStrategy()),
        ("Collusive Strategy", CollusiveStrategy(defection_probability=0.3)),
        ("Opportunistic Strategy", OpportunisticStrategy(profit_threshold_multiplier=1.2)),
    ]

    print("\nStrategy behaviors:")
    for name, strategy in strategies:
        if isinstance(strategy, CartelStrategy):
            action = strategy.next_action(
                round_num=1,
                my_history=[],
                rival_histories=[],
                bounds=(0, 100),
                market_params={"model_type": "bertrand"},
                collusion_manager=manager,
            )
            print(f"  {name}: Always follows cartel ({format_currency(action)})")

        elif isinstance(strategy, CollusiveStrategy):
            prob = strategy.calculate_defection_probability(
                round_num=1,
                my_history=[],
                rival_histories=[],
                collusion_manager=manager,
            )
            print(f"  {name}: {prob:.0%} defection probability")

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
            advantage = defection_profit / cartel_profit
            print(f"  {name}: {advantage:.1f}x profit advantage from defection")

    print_summary("Strategy Features", [
        "Cartel-compliant behavior modeling",
        "Probabilistic defection strategies",
        "Profit-based opportunistic decisions",
        "Integration with collusion management"
    ])


def main() -> None:
    """Run all demonstrations."""
    print("=== OLIGOPOLY COLLUSION & REGULATOR DYNAMICS DEMO ===")

    try:
        demo_cartel_stability()
        demo_defection()
        demo_regulator_intervention()
        demo_event_feed()
        demo_strategies()

        print("\n=== DEMO COMPLETE ===")
        print_demo_completion(
            "Collusion and regulator dynamics",
            "Cartel formation, defection detection, regulatory intervention, event logging, collusion strategies"
        )

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
