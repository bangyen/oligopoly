"""Tests for integrated strategy implementations.

This module contains comprehensive tests for all strategy types including
Static, TitForTat, and RandomWalk strategies that integrate with the
existing Cournot and Bertrand simulation framework.
"""

import math
from typing import List, Sequence, Union

import pytest

from src.sim.games.bertrand import BertrandResult
from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import (
    RandomWalk,
    Static,
    Strategy,
    TitForTat,
    create_strategy,
)


class TestStaticStrategy:
    """Test cases for Static strategy."""

    def test_static_basic(self) -> None:
        """Test basic static strategy functionality."""
        strategy = Static(value=10.0)

        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0, 20),
            market_params={},
        )

        assert action == 10.0

    def test_static_with_bounds_clamping(self) -> None:
        """Test static strategy respects bounds."""
        strategy = Static(value=15.0)

        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0, 10),
            market_params={},
        )

        assert action == 10.0  # Clamped to max

    def test_static_negative_value_validation(self) -> None:
        """Test static strategy rejects negative values."""
        with pytest.raises(ValueError, match="Static value -5.0 must be non-negative"):
            Static(value=-5.0)


class TestTitForTatStrategy:
    """Test cases for TitForTat strategy."""

    def test_titfortat_first_round(self) -> None:
        """Test TitForTat first round uses midpoint of bounds."""
        strategy = TitForTat()

        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0, 20),
            market_params={},
        )

        expected = (0 + 20) / 2.0  # Midpoint
        assert action == expected

    def test_titfortat_follow_cournot(self) -> None:
        """Test TitForTat mirrors rival mean from previous round with Cournot results."""
        strategy = TitForTat()

        # Create rival Cournot results with quantities [5.0, 15.0] -> mean = 10.0
        rival_result1 = CournotResult(price=12.0, quantities=[5.0], profits=[35.0])
        rival_result2 = CournotResult(price=8.0, quantities=[15.0], profits=[45.0])

        action = strategy.next_action(
            round_num=1,
            my_history=[],
            rival_histories=[[rival_result1], [rival_result2]],
            bounds=(0, 20),
            market_params={},
        )

        expected = 10.0  # Mean of rival quantities
        assert math.isclose(action, expected, abs_tol=1e-6)

    def test_titfortat_follow_bertrand(self) -> None:
        """Test TitForTat mirrors rival mean from previous round with Bertrand results."""
        strategy = TitForTat()

        # Create rival Bertrand results with prices [8.0, 12.0] -> mean = 10.0
        rival_result1 = BertrandResult(
            total_demand=50.0, prices=[8.0], quantities=[25.0], profits=[125.0]
        )
        rival_result2 = BertrandResult(
            total_demand=40.0, prices=[12.0], quantities=[20.0], profits=[160.0]
        )

        action = strategy.next_action(
            round_num=1,
            my_history=[],
            rival_histories=[[rival_result1], [rival_result2]],
            bounds=(0, 20),
            market_params={},
        )

        expected = 10.0  # Mean of rival prices
        assert math.isclose(action, expected, abs_tol=1e-6)

    def test_titfortat_no_rival_history(self) -> None:
        """Test TitForTat uses midpoint when no rival history."""
        strategy = TitForTat()

        action = strategy.next_action(
            round_num=1,
            my_history=[],
            rival_histories=[],
            bounds=(0, 20),
            market_params={},
        )

        expected = (0 + 20) / 2.0  # Midpoint
        assert action == expected

    def test_titfortat_with_bounds_clamping(self) -> None:
        """Test TitForTat respects bounds when mirroring."""
        strategy = TitForTat()

        # Create rival results with high quantities that exceed bounds
        rival_result1 = CournotResult(price=5.0, quantities=[15.0], profits=[0.0])
        rival_result2 = CournotResult(price=5.0, quantities=[20.0], profits=[0.0])

        action = strategy.next_action(
            round_num=1,
            my_history=[],
            rival_histories=[[rival_result1], [rival_result2]],
            bounds=(0, 10),  # Tight bounds
            market_params={},
        )

        assert action == 10.0  # Clamped to max bound


class TestRandomWalkStrategy:
    """Test cases for RandomWalk strategy."""

    def test_randomwalk_bounds(self) -> None:
        """Test RandomWalk always stays within bounds over many steps."""
        strategy = RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=42)

        # Run 100 steps and verify all actions are within bounds
        current_action = None
        for round_num in range(100):
            my_history: List[Union[CournotResult, BertrandResult]] = []
            if current_action is not None:
                # Create a mock result with the current action
                result: Union[CournotResult, BertrandResult]
                if round_num % 2 == 0:
                    result = CournotResult(
                        price=10.0, quantities=[current_action], profits=[50.0]
                    )
                else:
                    result = BertrandResult(
                        total_demand=50.0,
                        prices=[current_action],
                        quantities=[25.0],
                        profits=[125.0],
                    )
                my_history = [result]

            current_action = strategy.next_action(
                round_num=round_num,
                my_history=my_history,
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

            assert 0.0 <= current_action <= 10.0

    def test_randomwalk_first_round(self) -> None:
        """Test RandomWalk first round uses midpoint."""
        strategy = RandomWalk(step=1.0, min_bound=0.0, max_bound=20.0, seed=42)

        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0.0, 20.0),
            market_params={},
        )

        expected = (0.0 + 20.0) / 2.0  # Midpoint
        assert action == expected

    def test_randomwalk_reproducible_with_seed(self) -> None:
        """Test RandomWalk produces same sequence with same seed."""
        strategy1 = RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=123)
        strategy2 = RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=123)

        # Run both strategies and compare results
        actions1: List[float] = []
        actions2: List[float] = []

        for round_num in range(10):
            my_history1 = []
            my_history2 = []

            if round_num > 0:
                # Create mock results with previous actions
                result1 = CournotResult(
                    price=10.0, quantities=[actions1[-1]], profits=[50.0]
                )
                result2 = CournotResult(
                    price=10.0, quantities=[actions2[-1]], profits=[50.0]
                )
                my_history1 = [result1]
                my_history2 = [result2]

            action1 = strategy1.next_action(
                round_num=round_num,
                my_history=my_history1,
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )
            actions1.append(action1)

            action2 = strategy2.next_action(
                round_num=round_num,
                my_history=my_history2,
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )
            actions2.append(action2)

        assert actions1 == actions2

    def test_randomwalk_parameter_validation(self) -> None:
        """Test RandomWalk parameter validation."""
        with pytest.raises(ValueError, match="Step size -1.0 must be positive"):
            RandomWalk(step=-1.0, min_bound=0.0, max_bound=10.0)

        with pytest.raises(
            ValueError, match="Min bound 10.0 must be less than max bound 5.0"
        ):
            RandomWalk(step=1.0, min_bound=10.0, max_bound=5.0)


class TestStrategyFactory:
    """Test cases for strategy factory function."""

    def test_create_static_strategy(self) -> None:
        """Test creating Static strategy via factory."""
        strategy: Strategy = create_strategy("static", value=5.0)
        assert isinstance(strategy, Static)
        assert strategy.value == 5.0

    def test_create_titfortat_strategy(self) -> None:
        """Test creating TitForTat strategy via factory."""
        strategy: Strategy = create_strategy("titfortat")
        assert isinstance(strategy, TitForTat)

    def test_create_randomwalk_strategy(self) -> None:
        """Test creating RandomWalk strategy via factory."""
        strategy: Strategy = create_strategy(
            "randomwalk", step=2.0, min_bound=0.0, max_bound=10.0, seed=42
        )
        assert isinstance(strategy, RandomWalk)
        assert strategy.step == 2.0
        assert strategy.min_bound == 0.0
        assert strategy.max_bound == 10.0
        assert strategy.seed == 42

    def test_create_strategy_missing_params(self) -> None:
        """Test factory raises error for missing parameters."""
        with pytest.raises(
            ValueError, match="Static strategy requires 'value' parameter"
        ):
            create_strategy("static")

        with pytest.raises(
            ValueError, match="RandomWalk strategy requires 'step' parameter"
        ):
            create_strategy("randomwalk", min_bound=0.0, max_bound=10.0)

    def test_create_strategy_unknown_type(self) -> None:
        """Test factory raises error for unknown strategy type."""
        with pytest.raises(ValueError, match="Unknown strategy type: unknown"):
            create_strategy("unknown")


class TestStrategyIntegration:
    """Integration tests for strategy combinations."""

    def test_strategy_integration_basic(self) -> None:
        """Test that different strategies produce different actions."""
        strategies: List[Strategy] = [
            Static(value=5.0),
            TitForTat(),
            RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=42),
        ]

        bounds = (0.0, 10.0)

        # Test first round - Static and midpoint strategies should be different
        actions = []
        for strategy in strategies:
            action = strategy.next_action(
                round_num=0,
                my_history=[],
                rival_histories=[],
                bounds=bounds,
                market_params={},
            )
            actions.append(action)

        # Static should be 5.0, others should be 5.0 (midpoint)
        assert actions[0] == 5.0  # Static
        assert actions[1] == 5.0  # TitForTat midpoint
        assert actions[2] == 5.0  # RandomWalk midpoint

        # Test second round with some history
        rival_result = CournotResult(price=8.0, quantities=[7.0], profits=[21.0])

        actions_round2 = []
        for i, strategy in enumerate(strategies):
            my_history: List[Union[CournotResult, BertrandResult]] = []
            rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]] = []

            if i == 0:  # Static - no history needed
                pass
            elif i == 1:  # TitForTat - needs rival history
                rival_histories = [[rival_result]]
            else:  # RandomWalk - needs own history
                my_history = [
                    CournotResult(price=8.0, quantities=[5.0], profits=[15.0])
                ]

            action = strategy.next_action(
                round_num=1,
                my_history=my_history,
                rival_histories=rival_histories,
                bounds=bounds,
                market_params={},
            )
            actions_round2.append(action)

        # Now we should see different behaviors
        assert actions_round2[0] == 5.0  # Static unchanged
        assert actions_round2[1] == 7.0  # TitForTat mirrors rival quantity
        assert 4.0 <= actions_round2[2] <= 6.0  # RandomWalk: 5.0 ± 1.0
