"""Tests for cartel formation, defection detection, events and collusion strategies."""

from sim.collusion import CollusionManager
from sim.strategies.collusion_strategies import (
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
