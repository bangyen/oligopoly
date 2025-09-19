"""Tests for advanced strategy implementations.

This module tests the advanced learning strategies including
Fictitious Play, Deep Q-Learning, and Behavioral strategies.
"""

import math
from typing import List

import pytest

from src.sim.strategies.advanced_strategies import (
    AdvancedStrategy,
    BehavioralStrategy,
    DeepQLearningStrategy,
    FictitiousPlayStrategy,
    MarketState,
    StrategyBelief,
    create_advanced_strategy,
)


class TestMarketState:
    """Test MarketState class."""

    def test_market_state_creation(self) -> None:
        """Test creating a market state."""
        market_state = MarketState(
            prices=[20.0, 22.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            total_demand=27.0,
        )

        assert market_state.prices == [20.0, 22.0]
        assert market_state.quantities == [15.0, 12.0]
        assert market_state.market_shares == [0.6, 0.4]
        assert market_state.total_demand == 27.0

    def test_market_state_validation(self) -> None:
        """Test market state validation."""
        # Test mismatched lengths
        with pytest.raises(
            ValueError, match="Prices and quantities must have same length"
        ):
            MarketState(
                prices=[20.0, 22.0],
                quantities=[15.0],  # Different length
                market_shares=[0.6, 0.4],
                total_demand=27.0,
            )

        # Test market shares don't sum to 1
        with pytest.raises(ValueError, match="Market shares must sum to 1.0"):
            MarketState(
                prices=[20.0, 22.0],
                quantities=[15.0, 12.0],
                market_shares=[0.6, 0.5],  # Sums to 1.1
                total_demand=27.0,
            )


class TestStrategyBelief:
    """Test StrategyBelief class."""

    def test_belief_creation(self) -> None:
        """Test creating a strategy belief."""
        belief = StrategyBelief(firm_id=0)

        assert belief.firm_id == 0
        assert belief.action_history == []
        assert belief.confidence == 0.5

    def test_belief_update(self) -> None:
        """Test updating beliefs."""
        belief = StrategyBelief(firm_id=0)

        # Update with first action
        belief.update_belief(20.0, 1, 0.9)

        assert belief.action_history == [20.0]
        assert belief.belief_weights == [1.0]
        assert belief.last_update == 1

        # Update with second action
        belief.update_belief(22.0, 2, 0.9)

        assert belief.action_history == [20.0, 22.0]
        assert len(belief.belief_weights) == 2
        assert belief.last_update == 2

    def test_belief_prediction(self) -> None:
        """Test action prediction."""
        belief = StrategyBelief(firm_id=0)

        # No history
        assert belief.predict_action() == 0.0

        # With history
        belief.update_belief(20.0, 1, 0.9)
        belief.update_belief(22.0, 2, 0.9)

        predicted = belief.predict_action()
        assert 20.0 <= predicted <= 22.0  # Should be between the two actions


class TestFictitiousPlayStrategy:
    """Test FictitiousPlayStrategy class."""

    def test_strategy_creation(self) -> None:
        """Test creating fictitious play strategy."""
        strategy = FictitiousPlayStrategy(
            belief_decay=0.9, exploration_rate=0.1, seed=42
        )

        assert strategy.belief_decay == 0.9
        assert strategy.exploration_rate == 0.1
        assert strategy._beliefs == {}

    def test_strategy_validation(self) -> None:
        """Test strategy parameter validation."""
        with pytest.raises(ValueError, match="Belief decay must be in"):
            FictitiousPlayStrategy(belief_decay=1.5)

        with pytest.raises(ValueError, match="Exploration rate must be in"):
            FictitiousPlayStrategy(exploration_rate=-0.1)

    def test_cournot_best_response(self) -> None:
        """Test Cournot best response calculation."""
        strategy = FictitiousPlayStrategy(seed=42)

        # Test with no rivals
        best_response = strategy._cournot_best_response(
            [], {"a": 100.0, "b": 1.0, "my_cost": 10.0}, 0.0, 100.0
        )
        expected = (100.0 - 10.0) / (2 * 1.0)  # Monopoly quantity
        assert abs(best_response - expected) < 1e-6

        # Test with rivals
        best_response = strategy._cournot_best_response(
            [20.0, 15.0], {"a": 100.0, "b": 1.0, "my_cost": 10.0}, 0.0, 100.0
        )
        expected = (100.0 - 10.0 - 1.0 * 35.0) / (2 * 1.0)
        assert abs(best_response - expected) < 1e-6

    def test_bertrand_best_response(self) -> None:
        """Test Bertrand best response calculation."""
        strategy = FictitiousPlayStrategy(seed=42)

        # Test with no rivals
        best_response = strategy._bertrand_best_response(
            [], {"alpha": 100.0, "beta": 1.0, "my_cost": 10.0}, 0.0, 100.0
        )
        expected = (100.0 + 10.0) / 2  # Monopoly price
        assert abs(best_response - expected) < 1e-6

        # Test with rivals
        best_response = strategy._bertrand_best_response(
            [25.0, 30.0], {"alpha": 100.0, "beta": 1.0, "my_cost": 10.0}, 0.0, 100.0
        )
        expected = 25.0 - 0.1  # Undercut by small amount
        assert abs(best_response - expected) < 1e-6

    def test_next_action(self) -> None:
        """Test next action calculation."""
        strategy = FictitiousPlayStrategy(seed=42)

        market_state = MarketState(
            prices=[20.0, 22.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            total_demand=27.0,
        )

        action = strategy.next_action(
            round_num=0,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=(0.0, 100.0),
            market_params={"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0},
        )

        assert 0.0 <= action <= 100.0


class TestDeepQLearningStrategy:
    """Test DeepQLearningStrategy class."""

    def test_strategy_creation(self) -> None:
        """Test creating deep Q-learning strategy."""
        strategy = DeepQLearningStrategy(
            learning_rate=0.01, discount_factor=0.95, epsilon_0=0.3, seed=42
        )

        assert strategy.learning_rate == 0.01
        assert strategy.discount_factor == 0.95
        assert strategy._epsilon == 0.3
        assert strategy._weights.shape == (strategy.feature_dim, strategy.action_dim)

    def test_strategy_validation(self) -> None:
        """Test strategy parameter validation."""
        with pytest.raises(ValueError, match="Learning rate must be in"):
            DeepQLearningStrategy(learning_rate=1.5)

        with pytest.raises(ValueError, match="Discount factor must be in"):
            DeepQLearningStrategy(discount_factor=1.5)

    def test_feature_extraction(self) -> None:
        """Test feature extraction."""
        strategy = DeepQLearningStrategy(seed=42)

        market_state = MarketState(
            prices=[20.0, 22.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            total_demand=27.0,
        )

        features = strategy._extract_features(market_state, [], (0.0, 100.0))

        assert features.shape == (strategy.feature_dim,)
        assert all(not math.isnan(f) for f in features)
        assert all(not math.isinf(f) for f in features)

    def test_q_values_calculation(self) -> None:
        """Test Q-values calculation."""
        strategy = DeepQLearningStrategy(seed=42)

        features = strategy._extract_features(
            MarketState([20.0], [15.0], [1.0], 15.0), [], (0.0, 100.0)
        )

        q_values = strategy._q_values(features)

        assert q_values.shape == (strategy.action_dim,)
        assert all(not math.isnan(q) for q in q_values)
        assert all(not math.isinf(q) for q in q_values)

    def test_next_action(self) -> None:
        """Test next action calculation."""
        strategy = DeepQLearningStrategy(seed=42)

        market_state = MarketState(
            prices=[20.0, 22.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            total_demand=27.0,
        )

        action = strategy.next_action(
            round_num=0,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=(0.0, 100.0),
            market_params={"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0},
        )

        assert 0.0 <= action <= 100.0


class TestBehavioralStrategy:
    """Test BehavioralStrategy class."""

    def test_strategy_creation(self) -> None:
        """Test creating behavioral strategy."""
        strategy = BehavioralStrategy(
            rationality_level=0.8, loss_aversion=2.0, fairness_weight=0.1, seed=42
        )

        assert strategy.rationality_level == 0.8
        assert strategy.loss_aversion == 2.0
        assert strategy.fairness_weight == 0.1

    def test_strategy_validation(self) -> None:
        """Test strategy parameter validation."""
        with pytest.raises(ValueError, match="Rationality level must be in"):
            BehavioralStrategy(rationality_level=1.5)

        with pytest.raises(ValueError, match="Loss aversion must be >= 1"):
            BehavioralStrategy(loss_aversion=0.5)

    def test_utility_calculation(self) -> None:
        """Test utility calculation with loss aversion."""
        strategy = BehavioralStrategy(loss_aversion=2.0, reference_point=10.0)

        # Gain
        utility_gain = strategy._calculate_utility(15.0)
        assert utility_gain == 5.0

        # Loss
        utility_loss = strategy._calculate_utility(5.0)
        assert utility_loss == -10.0  # 2.0 * (5.0 - 10.0)

    def test_fairness_utility(self) -> None:
        """Test fairness utility calculation."""
        strategy = BehavioralStrategy(fairness_weight=0.1)

        # Above average profit
        fairness_util = strategy._calculate_fairness_utility(100.0, [50.0, 60.0])
        assert fairness_util < 0  # Negative utility from being above average

        # Below average profit
        fairness_util = strategy._calculate_fairness_utility(40.0, [50.0, 60.0])
        assert fairness_util == 0  # No penalty for being below average

    def test_next_action(self) -> None:
        """Test next action calculation."""
        strategy = BehavioralStrategy(seed=42)

        market_state = MarketState(
            prices=[20.0, 22.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            total_demand=27.0,
        )

        action = strategy.next_action(
            round_num=0,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=(0.0, 100.0),
            market_params={"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0},
        )

        assert 0.0 <= action <= 100.0


class TestStrategyFactory:
    """Test strategy factory function."""

    def test_create_fictitious_play(self) -> None:
        """Test creating fictitious play strategy."""
        strategy = create_advanced_strategy("fictitious_play", seed=42)
        assert isinstance(strategy, FictitiousPlayStrategy)

    def test_create_deep_q_learning(self) -> None:
        """Test creating deep Q-learning strategy."""
        strategy = create_advanced_strategy("deep_q_learning", seed=42)
        assert isinstance(strategy, DeepQLearningStrategy)

    def test_create_behavioral(self) -> None:
        """Test creating behavioral strategy."""
        strategy = create_advanced_strategy("behavioral", seed=42)
        assert isinstance(strategy, BehavioralStrategy)

    def test_unknown_strategy_type(self) -> None:
        """Test error for unknown strategy type."""
        with pytest.raises(ValueError, match="Unknown advanced strategy type"):
            create_advanced_strategy("unknown_strategy")


class TestStrategyIntegration:
    """Test strategy integration and interaction."""

    def test_multiple_strategies_interaction(self) -> None:
        """Test multiple strategies interacting."""
        strategies: List[AdvancedStrategy] = [
            FictitiousPlayStrategy(seed=42),
            DeepQLearningStrategy(seed=43),
            BehavioralStrategy(seed=44),
        ]

        market_state = MarketState(
            prices=[20.0, 22.0, 18.0],
            quantities=[15.0, 12.0, 18.0],
            market_shares=[0.33, 0.27, 0.40],
            total_demand=45.0,
        )

        actions = []
        for strategy in strategies:
            action = strategy.next_action(
                round_num=0,
                market_state=market_state,
                my_history=[],
                rival_histories=[],
                beliefs={},
                bounds=(0.0, 100.0),
                market_params={
                    "model": "cournot",
                    "a": 100.0,
                    "b": 1.0,
                    "my_cost": 10.0,
                },
            )
            actions.append(action)

        # All actions should be within bounds
        for action in actions:
            assert 0.0 <= action <= 100.0

        # Actions should be different (due to different strategies)
        assert len(set(actions)) > 1

    def test_strategy_learning_over_time(self) -> None:
        """Test that strategies learn and adapt over time."""
        strategy = FictitiousPlayStrategy(seed=42)

        market_state = MarketState(
            prices=[20.0, 22.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            total_demand=27.0,
        )

        # Simulate multiple rounds
        actions = []
        for round_num in range(5):
            action = strategy.next_action(
                round_num=round_num,
                market_state=market_state,
                my_history=[],
                rival_histories=[],
                beliefs={},
                bounds=(0.0, 100.0),
                market_params={
                    "model": "cournot",
                    "a": 100.0,
                    "b": 1.0,
                    "my_cost": 10.0,
                },
            )
            actions.append(action)

        # Actions should be reasonable
        for action in actions:
            assert 0.0 <= action <= 100.0

        # Strategy should maintain some consistency
        assert len(set(actions)) < len(actions)  # Not all actions are different
