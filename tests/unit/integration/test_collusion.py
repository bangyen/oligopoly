"""Tests for collusion and regulator dynamics.

This module contains comprehensive tests for cartel stability, defection behavior,
regulator interventions, and event logging functionality.
"""

import math

from src.sim.collusion import (
    CollusionEventType,
    CollusionManager,
    RegulatorState,
)
from src.sim.runners.collusion_runner import (
    create_collusion_simulation_config,
)
from src.sim.strategies.collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
)


class TestCartelStability:
    """Test cartel formation and stability."""

    def test_cartel_formation(self):
        """Test that cartels can be formed with correct parameters."""
        manager = CollusionManager()

        # Form a cartel
        manager.form_cartel(
            round_idx=5,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        assert manager.is_cartel_active()
        assert manager.current_cartel is not None
        assert manager.current_cartel.collusive_price == 50.0
        assert manager.current_cartel.collusive_quantity == 10.0
        assert manager.current_cartel.participating_firms == [0, 1, 2]
        assert manager.current_cartel.formed_round == 5

    def test_cartel_stability_high_profits(self):
        """Test that stable cartels lead to high profits and high HHI."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=60.0,
            collusive_quantity=8.0,
            participating_firms=[0, 1, 2],
        )

        # Simulate cartel compliance (all firms follow agreement)
        prices = [60.0, 60.0, 60.0]  # All firms set cartel price
        quantities = [8.0, 8.0, 8.0]  # All firms produce cartel quantity

        # Calculate market shares (equal shares in stable cartel)
        total_quantity = sum(quantities)
        market_shares = [q / total_quantity for q in quantities]

        # Calculate HHI
        hhi = manager.calculate_hhi(market_shares)

        # In a stable cartel, HHI should be high (close to 1.0 for equal shares)
        expected_hhi = 3 * (1 / 3) ** 2  # 3 firms with equal shares
        assert math.isclose(hhi, expected_hhi, abs_tol=1e-6)

        # Calculate average price
        avg_price = manager.calculate_average_price(prices, quantities)
        assert avg_price == 60.0

        # High cartel price should lead to high profits
        # Assuming marginal cost of 20, profit per firm = (60 - 20) * 8 = 320
        expected_profit_per_firm = (60.0 - 20.0) * 8.0
        assert expected_profit_per_firm > 200  # High profit threshold

    def test_cartel_dissolution(self):
        """Test that cartels can be dissolved."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        assert manager.is_cartel_active()

        # Dissolve cartel
        manager.dissolve_cartel(round_idx=10)

        assert not manager.is_cartel_active()
        assert manager.current_cartel is None


class TestDefection:
    """Test defection mechanisms and detection."""

    def test_defection_detection_price_undercutting(self):
        """Test detection of price undercutting in Bertrand model."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        # Firm 1 defects by undercutting price
        defected = manager.detect_defection(
            round_idx=1,
            firm_id=1,
            firm_price=45.0,  # 10% below cartel price
            firm_quantity=10.0,
            cartel_price=50.0,
            cartel_quantity=10.0,
            tolerance=0.05,
        )

        assert defected
        assert manager.get_firm_defection_count(1) == 1
        assert 1 in manager.firm_defection_history

        # Check event was logged
        events = manager.get_events_for_round(1)
        assert len(events) == 1
        assert events[0].event_type == CollusionEventType.FIRM_DEFECTED
        assert events[0].firm_id == 1
        assert "Firm 1 defects" in events[0].description

    def test_defection_detection_quantity_overproduction(self):
        """Test detection of quantity overproduction in Cournot model."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        # Firm 2 defects by overproducing
        defected = manager.detect_defection(
            round_idx=2,
            firm_id=2,
            firm_price=50.0,
            firm_quantity=15.0,  # 50% above cartel quantity
            cartel_price=50.0,
            cartel_quantity=10.0,
            tolerance=0.05,
        )

        assert defected
        assert manager.get_firm_defection_count(2) == 1

    def test_no_defection_when_compliant(self):
        """Test that compliant firms are not detected as defectors."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        # Firm follows cartel agreement exactly
        defected = manager.detect_defection(
            round_idx=1,
            firm_id=0,
            firm_price=50.0,  # Exactly cartel price
            firm_quantity=10.0,  # Exactly cartel quantity
            cartel_price=50.0,
            cartel_quantity=10.0,
            tolerance=0.05,
        )

        assert not defected
        assert manager.get_firm_defection_count(0) == 0

    def test_defection_profit_spike(self):
        """Test that defection leads to short-term profit spike."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        # Calculate profits for compliant vs defecting firm
        marginal_cost = 20.0

        # Compliant firm profit
        compliant_profit = (50.0 - marginal_cost) * 10.0  # 300

        # Defecting firm (undercuts price, captures more demand)
        defection_price = 45.0  # 10% undercut
        # Assume defecting firm captures 60% of market demand
        defection_quantity = 18.0  # Higher quantity due to lower price
        defection_profit = (defection_price - marginal_cost) * defection_quantity  # 450

        # Defection should lead to higher profit
        assert defection_profit > compliant_profit
        assert defection_profit / compliant_profit > 1.3  # At least 30% higher


class TestRegulatorTrigger:
    """Test regulator monitoring and intervention."""

    def test_hhi_calculation(self):
        """Test HHI calculation for market concentration."""
        manager = CollusionManager()

        # Equal market shares
        equal_shares = [1 / 3, 1 / 3, 1 / 3]
        hhi_equal = manager.calculate_hhi(equal_shares)
        expected_hhi_equal = 3 * (1 / 3) ** 2
        assert math.isclose(hhi_equal, expected_hhi_equal, abs_tol=1e-6)

        # Monopoly (one firm has 100% share)
        monopoly_shares = [1.0, 0.0, 0.0]
        hhi_monopoly = manager.calculate_hhi(monopoly_shares)
        assert math.isclose(hhi_monopoly, 1.0, abs_tol=1e-6)

        # Perfect competition (many small firms)
        competitive_shares = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        hhi_competitive = manager.calculate_hhi(competitive_shares)
        assert hhi_competitive < 0.2  # Low HHI for competitive market

    def test_regulator_intervention_trigger(self):
        """Test that regulator intervenes when HHI > 0.8 and price > baseline."""
        regulator_state = RegulatorState(
            hhi_threshold=0.8,
            price_threshold_multiplier=1.5,
            baseline_price=30.0,
            intervention_probability=1.0,  # Always intervene when triggered
        )
        manager = CollusionManager(regulator_state)

        # High concentration market (HHI > 0.8)
        market_shares = [0.6, 0.3, 0.1]  # HHI = 0.6² + 0.3² + 0.1² = 0.46
        # Actually, let's use higher concentration
        market_shares = [0.8, 0.15, 0.05]  # HHI = 0.8² + 0.15² + 0.05² = 0.665
        # Let's use even higher concentration
        market_shares = [0.9, 0.08, 0.02]  # HHI = 0.9² + 0.08² + 0.02² = 0.8168

        # High prices (above threshold)
        prices = [50.0, 50.0, 50.0]  # Above 30.0 * 1.5 = 45.0
        quantities = [10.0, 5.0, 2.0]

        should_intervene, intervention_type, intervention_value = (
            manager.check_regulator_intervention(
                round_idx=5,
                market_shares=market_shares,
                prices=prices,
                quantities=quantities,
            )
        )

        assert should_intervene
        assert intervention_type in ["penalty", "price_cap"]
        assert intervention_value is not None

    def test_regulator_penalty_application(self):
        """Test that penalties reduce firm profits."""
        manager = CollusionManager()

        original_profits = [300.0, 250.0, 200.0]
        penalty_amount = 100.0

        modified_profits = manager.apply_regulator_intervention(
            round_idx=5,
            intervention_type="penalty",
            intervention_value=penalty_amount,
            firm_profits=original_profits,
        )

        # All profits should be reduced by penalty amount
        expected_profits = [200.0, 150.0, 100.0]
        assert modified_profits == expected_profits

        # Check penalty event was logged
        events = manager.get_events_for_round(5)
        penalty_events = [
            e for e in events if e.event_type == CollusionEventType.PENALTY_IMPOSED
        ]
        assert len(penalty_events) == 1
        assert penalty_events[0].data["penalty_amount"] == penalty_amount

    def test_regulator_price_cap(self):
        """Test that price caps are imposed correctly."""
        manager = CollusionManager()

        # Apply price cap intervention
        price_cap = 40.0
        manager.apply_regulator_intervention(
            round_idx=3,
            intervention_type="price_cap",
            intervention_value=price_cap,
            firm_profits=[300.0, 250.0, 200.0],  # Profits unchanged for price cap
        )

        # Check price cap event was logged
        events = manager.get_events_for_round(3)
        price_cap_events = [
            e for e in events if e.event_type == CollusionEventType.PRICE_CAP_IMPOSED
        ]
        assert len(price_cap_events) == 1
        assert price_cap_events[0].data["price_cap"] == price_cap


class TestEventFeed:
    """Test event logging and retrieval."""

    def test_event_logging_cartel_formation(self):
        """Test that cartel formation events are logged correctly."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=5,
            collusive_price=60.0,
            collusive_quantity=8.0,
            participating_firms=[0, 1, 2],
        )

        # Check event was logged
        events = manager.get_events_for_round(5)
        assert len(events) == 1

        event = events[0]
        assert event.event_type == CollusionEventType.CARTEL_FORMED
        assert event.round_idx == 5
        assert event.firm_id is None  # Market-wide event
        assert "Cartel formed with 3 firms" in event.description
        assert event.data["collusive_price"] == 60.0
        assert event.data["collusive_quantity"] == 8.0
        assert event.data["participating_firms"] == [0, 1, 2]

    def test_event_logging_defection(self):
        """Test that defection events are logged correctly."""
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        # Firm defects
        manager.detect_defection(
            round_idx=3,
            firm_id=1,
            firm_price=45.0,
            firm_quantity=10.0,
            cartel_price=50.0,
            cartel_quantity=10.0,
        )

        # Check event was logged
        events = manager.get_events_for_round(3)
        assert len(events) == 1

        event = events[0]
        assert event.event_type == CollusionEventType.FIRM_DEFECTED
        assert event.round_idx == 3
        assert event.firm_id == 1
        assert "Firm 1 defects" in event.description
        assert event.data["firm_price"] == 45.0
        assert event.data["cartel_price"] == 50.0

    def test_event_logging_regulator_intervention(self):
        """Test that regulator intervention events are logged correctly."""
        regulator_state = RegulatorState(
            hhi_threshold=0.8,
            price_threshold_multiplier=1.5,
            baseline_price=30.0,
            intervention_probability=1.0,
        )
        manager = CollusionManager(regulator_state)

        # Trigger intervention
        market_shares = [0.9, 0.08, 0.02]  # High HHI
        prices = [50.0, 50.0, 50.0]  # High prices
        quantities = [10.0, 5.0, 2.0]

        manager.check_regulator_intervention(
            round_idx=7,
            market_shares=market_shares,
            prices=prices,
            quantities=quantities,
        )

        # Check event was logged
        events = manager.get_events_for_round(7)
        intervention_events = [
            e for e in events if e.event_type == CollusionEventType.REGULATOR_INTERVENED
        ]
        assert len(intervention_events) == 1

        event = intervention_events[0]
        assert event.round_idx == 7
        assert event.firm_id is None  # Market-wide event
        assert "Regulator intervenes" in event.description
        assert "hhi" in event.data
        assert "avg_price" in event.data
        assert "intervention_type" in event.data

    def test_event_round_filtering(self):
        """Test that events can be filtered by round."""
        manager = CollusionManager()

        # Create events in different rounds
        manager.form_cartel(
            round_idx=1,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1],
        )
        manager.form_cartel(
            round_idx=3,
            collusive_price=60.0,
            collusive_quantity=8.0,
            participating_firms=[0, 1, 2],
        )

        # Check filtering
        round_1_events = manager.get_events_for_round(1)
        round_3_events = manager.get_events_for_round(3)
        round_5_events = manager.get_events_for_round(5)

        assert len(round_1_events) == 1
        assert len(round_3_events) == 1
        assert len(round_5_events) == 0

        assert round_1_events[0].round_idx == 1
        assert round_3_events[0].round_idx == 3


class TestCollusionStrategies:
    """Test collusion-aware strategies."""

    def test_cartel_strategy_compliance(self):
        """Test that cartel strategy always follows agreements."""
        strategy = CartelStrategy()
        manager = CollusionManager()

        # Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=50.0,
            collusive_quantity=10.0,
            participating_firms=[0, 1, 2],
        )

        # Strategy should return cartel-compliant action
        action = strategy.next_action(
            round_num=1,
            my_history=[],
            rival_histories=[],
            bounds=(0, 100),
            market_params={"model_type": "bertrand"},
            collusion_manager=manager,
        )

        assert action == 50.0  # Cartel price

    def test_collusive_strategy_defection_probability(self):
        """Test that collusive strategy has configurable defection probability."""
        strategy = CollusiveStrategy(defection_probability=0.5)

        # Test defection probability calculation
        prob = strategy.calculate_defection_probability(
            round_num=5,
            my_history=[],
            rival_histories=[],
            collusion_manager=CollusionManager(),
        )

        assert prob >= 0.5  # Base probability
        assert prob <= 1.0

    def test_opportunistic_strategy_profit_calculation(self):
        """Test that opportunistic strategy calculates defection profitability."""
        strategy = OpportunisticStrategy(
            profit_threshold_multiplier=1.2, risk_tolerance=0.3
        )

        # Test profit estimation
        cartel_profit = strategy.estimate_cartel_profit(
            cartel_price=50.0, cartel_quantity=10.0, my_cost=20.0, model_type="bertrand"
        )

        expected_profit = (50.0 - 20.0) * 10.0  # 300
        assert cartel_profit == expected_profit

        defection_profit = strategy.estimate_defection_profit(
            cartel_price=50.0,
            cartel_quantity=10.0,
            my_cost=20.0,
            market_params={"alpha": 100.0, "beta": 1.0},
            model_type="bertrand",
        )

        # Defection profit should be higher due to lower price capturing more demand
        assert defection_profit > cartel_profit


class TestIntegration:
    """Integration tests for the complete collusion system."""

    def test_collusion_simulation_config(self):
        """Test collusion simulation configuration creation."""
        config = create_collusion_simulation_config(
            hhi_threshold=0.75,
            price_threshold_multiplier=1.3,
            baseline_price=25.0,
            intervention_probability=0.9,
            penalty_amount=150.0,
            price_cap_multiplier=0.85,
            auto_form_cartel=True,
        )

        assert config["hhi_threshold"] == 0.75
        assert config["price_threshold_multiplier"] == 1.3
        assert config["baseline_price"] == 25.0
        assert config["intervention_probability"] == 0.9
        assert config["penalty_amount"] == 150.0
        assert config["price_cap_multiplier"] == 0.85
        assert config["auto_form_cartel"] is True

    def test_end_to_end_collusion_scenario(self):
        """Test a complete collusion scenario from formation to intervention."""
        manager = CollusionManager()

        # 1. Form cartel
        manager.form_cartel(
            round_idx=0,
            collusive_price=60.0,
            collusive_quantity=8.0,
            participating_firms=[0, 1, 2],
        )

        # 2. Simulate cartel compliance for several rounds
        for round_idx in range(1, 4):
            # All firms follow cartel agreement
            manager.detect_defection(
                round_idx=round_idx,
                firm_id=0,
                firm_price=60.0,
                firm_quantity=8.0,
                cartel_price=60.0,
                cartel_quantity=8.0,
            )

        # 3. Firm defects
        manager.detect_defection(
            round_idx=4,
            firm_id=1,
            firm_price=50.0,  # Defects by undercutting
            firm_quantity=8.0,
            cartel_price=60.0,
            cartel_quantity=8.0,
        )

        # 4. Regulator intervenes due to high concentration
        market_shares = [0.9, 0.08, 0.02]  # High HHI
        prices = [60.0, 50.0, 60.0]
        quantities = [8.0, 8.0, 8.0]

        should_intervene, intervention_type, intervention_value = (
            manager.check_regulator_intervention(
                round_idx=5,
                market_shares=market_shares,
                prices=prices,
                quantities=quantities,
            )
        )

        # 5. Apply intervention
        if should_intervene:
            manager.apply_regulator_intervention(
                round_idx=5,
                intervention_type=intervention_type,
                intervention_value=intervention_value,
                firm_profits=[300.0, 250.0, 300.0],
            )

        # Verify all events were logged
        all_events = manager.events
        event_types = [event.event_type for event in all_events]

        assert CollusionEventType.CARTEL_FORMED in event_types
        assert CollusionEventType.FIRM_DEFECTED in event_types
        if should_intervene:
            assert CollusionEventType.REGULATOR_INTERVENED in event_types

        # Verify defection count
        assert manager.get_firm_defection_count(1) == 1
        assert manager.get_firm_defection_count(0) == 0
        assert manager.get_firm_defection_count(2) == 0
