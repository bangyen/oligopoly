"""Tests for integrated strategy implementations.

This module contains tests for the strategy implementations that integrate
with the existing Cournot and Bertrand simulation framework.
"""

import math

from src.sim.games.bertrand import BertrandResult
from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import RandomWalk, Static, TitForTat, create_strategy


class TestIntegratedStrategies:
    """Test cases for integrated strategy implementations."""

    def test_static_strategy_integration(self) -> None:
        """Test Static strategy with new interface."""
        strategy = Static(value=10.0)

        # Test with Cournot result
        cournot_result = CournotResult(price=15.0, quantities=[8.0], profits=[56.0])

        action = strategy.next_action(
            round_num=1,
            my_history=[cournot_result],
            rival_histories=[[cournot_result]],
            bounds=(0, 20),
            market_params={},
        )

        assert action == 10.0

    def test_titfortat_strategy_integration(self) -> None:
        """Test TitForTat strategy with new interface."""
        strategy = TitForTat()

        # Test first round (should use midpoint)
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0, 20),
            market_params={},
        )

        assert action == 10.0  # Midpoint of (0, 20)

        # Test second round with rival history
        rival_cournot = CournotResult(price=12.0, quantities=[5.0], profits=[35.0])
        rival_bertrand = BertrandResult(
            total_demand=50.0, prices=[8.0], quantities=[25.0], profits=[125.0]
        )

        action = strategy.next_action(
            round_num=1,
            my_history=[],
            rival_histories=[[rival_cournot], [rival_bertrand]],
            bounds=(0, 20),
            market_params={},
        )

        # Should mirror mean of rival actions: (5.0 + 8.0) / 2 = 6.5
        assert math.isclose(action, 6.5, abs_tol=1e-6)

    def test_randomwalk_strategy_integration(self) -> None:
        """Test RandomWalk strategy with new interface."""
        strategy = RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=42)

        # Test first round (should use midpoint)
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0, 20),
            market_params={},
        )

        assert action == 10.0  # Midpoint of (0, 20)

        # Test second round with history
        my_cournot = CournotResult(price=12.0, quantities=[5.0], profits=[35.0])

        action = strategy.next_action(
            round_num=1,
            my_history=[my_cournot],
            rival_histories=[],
            bounds=(0, 20),
            market_params={},
        )

        # Should be previous action (5.0) plus/minus random step
        assert 4.0 <= action <= 6.0  # 5.0 ± 1.0

    def test_strategy_bounds_clamping(self) -> None:
        """Test that strategies respect bounds."""
        strategy = Static(value=15.0)

        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0, 10),  # Tight bounds
            market_params={},
        )

        assert action == 10.0  # Clamped to max bound

    def test_strategy_with_bertrand_results(self) -> None:
        """Test strategies work with Bertrand results."""
        strategy = RandomWalk(step=0.5, min_bound=0.0, max_bound=10.0, seed=123)

        # Test with Bertrand result
        bertrand_result = BertrandResult(
            total_demand=50.0, prices=[8.0], quantities=[25.0], profits=[125.0]
        )

        action = strategy.next_action(
            round_num=1,
            my_history=[bertrand_result],
            rival_histories=[],
            bounds=(0, 20),
            market_params={},
        )

        # Should be previous price (8.0) plus/minus random step
        assert 7.5 <= action <= 8.5  # 8.0 ± 0.5


class TestStrategyFactoryIntegration:
    """Test cases for strategy factory with new interface."""

    def test_create_strategy_integration(self) -> None:
        """Test strategy factory works with new interface."""
        static = create_strategy("static", value=5.0)
        titfortat = create_strategy("titfortat")
        randomwalk = create_strategy(
            "randomwalk", step=1.0, min_bound=0.0, max_bound=10.0, seed=42
        )

        # Test that all strategies implement the interface
        for strategy in [static, titfortat, randomwalk]:
            action = strategy.next_action(
                round_num=0,
                my_history=[],
                rival_histories=[],
                bounds=(0, 20),
                market_params={},
            )
            assert isinstance(action, float)
            assert 0 <= action <= 20
