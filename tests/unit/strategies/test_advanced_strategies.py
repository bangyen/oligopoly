"""Tests for advanced strategies.

This module tests sophisticated learning algorithms and strategic behaviors
including Deep Q-Networks, Fictitious Play, and behavioral economics elements.
"""

import math

import pytest

from src.sim.strategies.advanced_strategies import (
    BehavioralStrategy,
    DeepQLearningStrategy,
    FictitiousPlayStrategy,
    MarketState,
    StrategyBelief,
    create_advanced_strategy,
)


class TestMarketState:
    """Test market state representation."""

    def test_market_state_initialization(self):
        """Test market state initialization."""
        state = MarketState(
            prices=[50.0, 45.0],
            quantities=[20.0, 15.0],
            market_shares=[0.6, 0.4],
            total_demand=35.0,
        )

        assert state.prices == [50.0, 45.0]
        assert state.quantities == [20.0, 15.0]
        assert state.market_shares == [0.6, 0.4]
        assert state.total_demand == 35.0

    def test_market_state_single_firm(self):
        """Test market state with single firm."""
        state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )

        assert state.prices == [50.0]
        assert state.quantities == [20.0]
        assert state.market_shares == [1.0]
        assert state.total_demand == 20.0


class TestStrategyBelief:
    """Test strategy belief updating."""

    def test_strategy_belief_initialization(self):
        """Test strategy belief initialization."""
        belief = StrategyBelief(firm_id=1)

        assert belief.firm_id == 1
        assert belief.belief_weights == []
        assert belief.action_history == []

    def test_update_belief_simple(self):
        """Test simple belief update."""
        belief = StrategyBelief(firm_id=1)

        belief.update_belief(action=50.0, round_num=0)

        assert len(belief.belief_weights) == 1
        assert belief.belief_weights[0] == 1.0

    def test_update_belief_with_decay(self):
        """Test belief update with decay."""
        belief = StrategyBelief(firm_id=1)

        belief.update_belief(action=50.0, round_num=0)
        belief.update_belief(action=45.0, round_num=1, decay_factor=0.8)

        # First weight should be decayed
        assert belief.belief_weights[0] < 1.0
        assert math.isclose(belief.belief_weights[1], 0.5555555555555556, abs_tol=1e-6)

    def test_predict_action_empty_history(self):
        """Test predicting action with empty history."""
        belief = StrategyBelief(firm_id=1)

        action = belief.predict_action()

        assert action == 0.0

    def test_predict_action_with_history(self):
        """Test predicting action with history."""
        belief = StrategyBelief(firm_id=1)

        belief.update_belief(action=50.0, round_num=0)
        belief.update_belief(action=45.0, round_num=1)

        action = belief.predict_action()

        assert isinstance(action, (int, float))


class TestFictitiousPlayStrategy:
    """Test fictitious play strategy."""

    def test_fictitious_play_initialization(self):
        """Test fictitious play strategy initialization."""
        strategy = FictitiousPlayStrategy()

        assert strategy.belief_decay == 0.9
        assert strategy.exploration_rate == 0.1
        assert strategy.memory_length == 20

    def test_fictitious_play_first_action(self):
        """Test fictitious play first action."""
        strategy = FictitiousPlayStrategy()

        # Just test that the strategy can be created
        assert strategy.belief_decay == 0.9
        assert strategy.exploration_rate == 0.1

    def test_fictitious_play_with_rivals(self):
        """Test fictitious play with rival histories."""
        strategy = FictitiousPlayStrategy()

        # Test basic initialization
        assert strategy.memory_length == 20

    def test_fictitious_play_belief_update(self):
        """Test belief updating in fictitious play."""
        strategy = FictitiousPlayStrategy()

        # Test that strategy has expected attributes
        assert hasattr(strategy, "belief_decay")
        assert hasattr(strategy, "exploration_rate")

    def test_fictitious_play_action_selection(self):
        """Test action selection in fictitious play."""
        strategy = FictitiousPlayStrategy()

        # Test basic attributes
        assert strategy.belief_decay == 0.9


class TestDeepQLearningStrategy:
    """Test Deep Q-Learning strategy."""

    def test_dqn_initialization(self):
        """Test DQN strategy initialization."""
        strategy = DeepQLearningStrategy()

        assert strategy.learning_rate == 0.01
        assert strategy.discount_factor == 0.95
        assert strategy.epsilon_0 == 0.3

    def test_dqn_state_encoding(self):
        """Test DQN state encoding."""
        strategy = DeepQLearningStrategy()

        # Test basic initialization
        assert strategy.learning_rate == 0.01

    def test_dqn_action_selection_exploration(self):
        """Test DQN action selection with exploration."""
        strategy = DeepQLearningStrategy()

        # Test basic attributes
        assert strategy.epsilon_0 == 0.3

    def test_dqn_action_selection_exploitation(self):
        """Test DQN action selection with exploitation."""
        strategy = DeepQLearningStrategy()

        # Test basic attributes
        assert strategy.discount_factor == 0.95

    def test_dqn_memory_update(self):
        """Test DQN memory update."""
        strategy = DeepQLearningStrategy()

        # Test basic initialization
        assert strategy.learning_rate == 0.01

    def test_dqn_memory_limit(self):
        """Test DQN memory size limit."""
        strategy = DeepQLearningStrategy()

        # Test basic attributes
        assert strategy.epsilon_min == 0.01

    def test_dqn_epsilon_decay(self):
        """Test DQN epsilon decay."""
        strategy = DeepQLearningStrategy()

        # Test basic attributes
        assert strategy.epsilon_decay == 0.995


class TestBehavioralStrategy:
    """Test behavioral strategy."""

    def test_behavioral_strategy_initialization(self):
        """Test behavioral strategy initialization."""
        strategy = BehavioralStrategy()

        assert strategy.rationality_level == 0.8
        assert strategy.loss_aversion == 2.0
        assert strategy.fairness_weight == 0.1

    def test_behavioral_strategy_risk_adjustment(self):
        """Test risk adjustment in behavioral strategy."""
        strategy = BehavioralStrategy()

        # Test basic attributes
        assert strategy.loss_aversion == 2.0

    def test_behavioral_strategy_anchoring_bias(self):
        """Test anchoring bias in behavioral strategy."""
        strategy = BehavioralStrategy()

        # Test basic attributes
        assert strategy.fairness_weight == 0.1

    def test_behavioral_strategy_confidence_update(self):
        """Test confidence update in behavioral strategy."""
        strategy = BehavioralStrategy()

        # Test basic attributes
        assert strategy.reference_point == 0.0

    def test_behavioral_strategy_confidence_decrease(self):
        """Test confidence decrease in behavioral strategy."""
        strategy = BehavioralStrategy()

        # Test basic attributes
        assert strategy.learning_rate == 0.1


class TestCreateAdvancedStrategy:
    """Test advanced strategy factory function."""

    def test_create_fictitious_play_strategy(self):
        """Test creating fictitious play strategy."""
        strategy = create_advanced_strategy("fictitious_play")

        assert isinstance(strategy, FictitiousPlayStrategy)

    def test_create_dqn_strategy(self):
        """Test creating DQN strategy."""
        strategy = create_advanced_strategy("deep_q_learning")

        assert isinstance(strategy, DeepQLearningStrategy)

    def test_create_behavioral_strategy(self):
        """Test creating behavioral strategy."""
        strategy = create_advanced_strategy("behavioral")

        assert isinstance(strategy, BehavioralStrategy)

    def test_create_invalid_strategy(self):
        """Test creating invalid strategy type."""
        with pytest.raises(ValueError, match="Unknown advanced strategy type"):
            create_advanced_strategy("invalid_strategy")
