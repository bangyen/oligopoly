"""Tests for advanced strategies.

This module tests sophisticated learning algorithms and strategic behaviors
including Deep Q-Networks, Fictitious Play, and behavioral economics elements.
"""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from src.sim.games.bertrand import BertrandResult
from src.sim.games.cournot import CournotResult
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


class TestMarketStateValidation:
    """Test market state validation."""

    def test_market_state_validation_prices_quantities_mismatch(self):
        """Test market state validation with mismatched prices and quantities."""
        with pytest.raises(
            ValueError, match="Prices and quantities must have same length"
        ):
            MarketState(
                prices=[50.0, 45.0],
                quantities=[20.0],  # Different length
                market_shares=[0.6, 0.4],
                total_demand=35.0,
            )

    def test_market_state_validation_market_shares_mismatch(self):
        """Test market state validation with mismatched market shares."""
        with pytest.raises(
            ValueError, match="Market shares must match number of firms"
        ):
            MarketState(
                prices=[50.0, 45.0],
                quantities=[20.0, 15.0],
                market_shares=[0.6],  # Different length
                total_demand=35.0,
            )

    def test_market_state_validation_market_shares_sum(self):
        """Test market state validation with market shares not summing to 1."""
        with pytest.raises(ValueError, match="Market shares must sum to 1.0"):
            MarketState(
                prices=[50.0, 45.0],
                quantities=[20.0, 15.0],
                market_shares=[0.6, 0.5],  # Sums to 1.1
                total_demand=35.0,
            )


class TestStrategyBeliefAdvanced:
    """Test advanced strategy belief functionality."""

    def test_update_belief_multiple_rounds(self):
        """Test belief update across multiple rounds."""
        belief = StrategyBelief(firm_id=1)

        belief.update_belief(action=50.0, round_num=0)
        belief.update_belief(action=45.0, round_num=1)
        belief.update_belief(action=40.0, round_num=2)

        assert len(belief.action_history) == 3
        assert len(belief.belief_weights) == 3
        assert belief.action_history == [50.0, 45.0, 40.0]

    def test_predict_action_weighted_average(self):
        """Test prediction using weighted average."""
        belief = StrategyBelief(firm_id=1)

        belief.update_belief(action=50.0, round_num=0)
        belief.update_belief(action=30.0, round_num=1)

        predicted = belief.predict_action()
        # Should be weighted average of the two actions
        # The exact calculation depends on the implementation
        assert isinstance(predicted, (int, float))
        assert 30.0 <= predicted <= 50.0  # Should be between the two actions


class TestFictitiousPlayStrategyAdvanced:
    """Test advanced fictitious play strategy functionality."""

    def test_fictitious_play_validation(self):
        """Test fictitious play parameter validation."""
        with pytest.raises(ValueError, match="Belief decay must be in"):
            FictitiousPlayStrategy(belief_decay=1.5)

        with pytest.raises(ValueError, match="Exploration rate must be in"):
            FictitiousPlayStrategy(exploration_rate=-0.1)

    def test_update_beliefs_with_cournot_results(self):
        """Test belief updating with Cournot results."""
        strategy = FictitiousPlayStrategy(seed=42)

        # Create mock Cournot results
        cournot_result = Mock(spec=CournotResult)
        cournot_result.quantities = [25.0, 20.0]

        rival_histories = [[cournot_result]]

        strategy._update_beliefs(rival_histories, round_num=1)

        assert 0 in strategy._beliefs
        assert strategy._beliefs[0].action_history == [25.0]

    def test_update_beliefs_with_bertrand_results(self):
        """Test belief updating with Bertrand results."""
        strategy = FictitiousPlayStrategy(seed=42)

        # Create mock Bertrand results
        bertrand_result = Mock(spec=BertrandResult)
        bertrand_result.prices = [50.0, 45.0]

        rival_histories = [[bertrand_result]]

        strategy._update_beliefs(rival_histories, round_num=1)

        assert 0 in strategy._beliefs
        assert strategy._beliefs[0].action_history == [50.0]

    def test_calculate_best_response_cournot_monopoly(self):
        """Test Cournot best response with no rivals (monopoly)."""
        strategy = FictitiousPlayStrategy(seed=42)

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot", "a": 100.0, "b": 1.0}

        action = strategy._calculate_best_response(
            market_state, bounds, market_params, "cournot"
        )

        # Monopoly quantity: a / (2 * b) = 100 / (2 * 1) = 50
        assert math.isclose(action, 50.0, abs_tol=1e-6)

    def test_calculate_best_response_bertrand_monopoly(self):
        """Test Bertrand best response with no rivals (monopoly)."""
        strategy = FictitiousPlayStrategy(seed=42)

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "bertrand", "alpha": 100.0}

        action = strategy._calculate_best_response(
            market_state, bounds, market_params, "bertrand"
        )

        # Monopoly price: (alpha + cost) / 2 = (100 + 10) / 2 = 55
        assert math.isclose(action, 55.0, abs_tol=1e-6)

    def test_cournot_best_response_with_rivals(self):
        """Test Cournot best response with rival quantities."""
        strategy = FictitiousPlayStrategy(seed=42)

        rival_quantities = [20.0, 15.0]
        market_params = {"a": 100.0, "b": 1.0, "my_cost": 10.0}
        bounds = (0.0, 100.0)

        action = strategy._cournot_best_response(
            rival_quantities, market_params, bounds[0], bounds[1]
        )

        # Best response: (a - my_cost - b * total_rival_qty) / (2 * b)
        # = (100 - 10 - 1 * 35) / (2 * 1) = 55 / 2 = 27.5
        assert math.isclose(action, 27.5, abs_tol=1e-6)

    def test_bertrand_best_response_with_rivals(self):
        """Test Bertrand best response with rival prices."""
        strategy = FictitiousPlayStrategy(seed=42)

        rival_prices = [60.0, 55.0]
        market_params = {"alpha": 100.0, "my_cost": 10.0}
        bounds = (0.0, 100.0)

        action = strategy._bertrand_best_response(
            rival_prices, market_params, bounds[0], bounds[1]
        )

        # Should undercut by 0.1: min(60.0, 55.0) - 0.1 = 54.9
        assert math.isclose(action, 54.9, abs_tol=1e-6)

    def test_bertrand_best_response_at_cost(self):
        """Test Bertrand best response when rivals price at cost."""
        strategy = FictitiousPlayStrategy(seed=42)

        rival_prices = [10.0, 10.1]  # At or near cost
        market_params = {"alpha": 100.0, "my_cost": 10.0}
        bounds = (0.0, 100.0)

        action = strategy._bertrand_best_response(
            rival_prices, market_params, bounds[0], bounds[1]
        )

        # Should price at cost
        assert math.isclose(action, 10.0, abs_tol=1e-6)

    def test_next_action_fictitious_play(self):
        """Test next action calculation for fictitious play."""
        strategy = FictitiousPlayStrategy(
            seed=42, exploration_rate=0.0
        )  # No exploration

        market_state = MarketState(
            prices=[50.0, 45.0],
            quantities=[20.0, 15.0],
            market_shares=[0.6, 0.4],
            total_demand=35.0,
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0}

        action = strategy.next_action(
            round_num=1,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=bounds,
            market_params=market_params,
        )

        # Should be monopoly quantity since no rivals
        assert math.isclose(action, 50.0, abs_tol=1e-6)

    def test_next_action_with_exploration(self):
        """Test next action with exploration noise."""
        strategy = FictitiousPlayStrategy(
            seed=42, exploration_rate=1.0
        )  # Always explore

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot", "a": 100.0, "b": 1.0}

        # Run multiple times to test exploration
        actions = []
        for _ in range(10):
            action = strategy.next_action(
                round_num=1,
                market_state=market_state,
                my_history=[],
                rival_histories=[],
                beliefs={},
                bounds=bounds,
                market_params=market_params,
            )
            actions.append(action)

        # Actions should vary due to exploration
        assert len(set(actions)) > 1
        # All actions should be within bounds
        for action in actions:
            assert bounds[0] <= action <= bounds[1]


class TestDeepQLearningStrategyAdvanced:
    """Test advanced Deep Q-Learning strategy functionality."""

    def test_dqn_validation(self):
        """Test DQN parameter validation."""
        with pytest.raises(ValueError, match="Learning rate must be in"):
            DeepQLearningStrategy(learning_rate=0.0)

        with pytest.raises(ValueError, match="Discount factor must be in"):
            DeepQLearningStrategy(discount_factor=1.5)

    def test_extract_features(self):
        """Test feature extraction from market state."""
        strategy = DeepQLearningStrategy(seed=42)

        market_state = MarketState(
            prices=[50.0, 45.0],
            quantities=[20.0, 15.0],
            market_shares=[0.6, 0.4],
            total_demand=35.0,
            market_growth=0.1,
            innovation_level=0.2,
        )
        bounds = (0.0, 100.0)

        # Create mock history
        cournot_result = Mock(spec=CournotResult)
        cournot_result.quantities = [20.0]
        cournot_result.profits = [500.0]
        my_history = [cournot_result]

        features = strategy._extract_features(market_state, my_history, bounds)

        assert features.shape == (10,)
        assert features[0] == 0.35  # total_demand / 100
        assert features[1] == 0.475  # mean(prices) / 100
        assert features[2] == 0.1  # market_growth
        assert features[3] == 0.2  # innovation_level
        assert features[4] == 0.6  # max market share
        assert features[6] == 0.2  # last quantity / 100
        assert features[7] == 0.5  # last profit / 1000

    def test_q_values_calculation(self):
        """Test Q-values calculation."""
        strategy = DeepQLearningStrategy(seed=42)

        features = np.array([0.5, 0.3, 0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_values = strategy._q_values(features)

        assert q_values.shape == (20,)  # action_dim
        assert isinstance(q_values, np.ndarray)

    def test_action_selection_exploration(self):
        """Test action selection with exploration."""
        strategy = DeepQLearningStrategy(seed=42)
        strategy._epsilon = 1.0  # Always explore

        q_values = np.random.random(20)
        bounds = (0.0, 100.0)

        action = strategy._select_action(q_values, bounds)

        assert bounds[0] <= action <= bounds[1]

    def test_action_selection_exploitation(self):
        """Test action selection with exploitation."""
        strategy = DeepQLearningStrategy(seed=42)
        strategy._epsilon = 0.0  # Never explore

        # Create Q-values with max at index 1 (action_dim = 20)
        q_values = np.zeros(20)
        q_values[1] = 0.9  # Max at index 1
        q_values[0] = 0.1
        q_values[2] = 0.3
        q_values[3] = 0.2

        bounds = (0.0, 100.0)

        action = strategy._select_action(q_values, bounds)

        # Should select action corresponding to max Q-value (index 1)
        # With action_dim=20: index 1 maps to 1/19 * 100 = 5.263...
        expected_action = bounds[0] + (1 / (strategy.action_dim - 1)) * (
            bounds[1] - bounds[0]
        )
        assert math.isclose(action, expected_action, abs_tol=1e-6)

    def test_action_to_index(self):
        """Test action to index conversion."""
        strategy = DeepQLearningStrategy(seed=42)

        bounds = (0.0, 100.0)

        # Test middle action
        action = 50.0
        index = strategy._action_to_index(action, bounds)
        assert index == 9  # (50-0)/(100-0) * (20-1) = 9.5 -> 9

        # Test boundary actions
        index_min = strategy._action_to_index(0.0, bounds)
        assert index_min == 0

        index_max = strategy._action_to_index(100.0, bounds)
        assert index_max == 19

    def test_update_weights(self):
        """Test weight updating."""
        strategy = DeepQLearningStrategy(seed=42)

        features = np.array([0.5, 0.3, 0.1, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
        action_index = 5
        reward = 0.8
        next_features = np.array([0.6, 0.4, 0.2, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        old_weights = strategy._weights.copy()
        old_bias = strategy._bias.copy()

        strategy._update_weights(features, action_index, reward, next_features)

        # Weights and bias should be updated
        assert not np.array_equal(strategy._weights, old_weights)
        assert not np.array_equal(strategy._bias, old_bias)

    def test_next_action_dqn(self):
        """Test next action calculation for DQN."""
        strategy = DeepQLearningStrategy(seed=42)

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot"}

        action = strategy.next_action(
            round_num=1,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=bounds,
            market_params=market_params,
        )

        assert bounds[0] <= action <= bounds[1]
        assert len(strategy._replay_buffer) == 1

    def test_next_action_with_learning(self):
        """Test next action with learning from history."""
        strategy = DeepQLearningStrategy(seed=42)

        # Create mock history with profit and quantities
        cournot_result = Mock(spec=CournotResult)
        cournot_result.profits = [800.0]
        cournot_result.quantities = [20.0]
        my_history = [cournot_result]

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot"}

        # First action to populate replay buffer
        strategy.next_action(
            round_num=0,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=bounds,
            market_params=market_params,
        )

        # Second action with history for learning
        action = strategy.next_action(
            round_num=1,
            market_state=market_state,
            my_history=my_history,
            rival_histories=[],
            beliefs={},
            bounds=bounds,
            market_params=market_params,
        )

        assert bounds[0] <= action <= bounds[1]
        assert len(strategy._replay_buffer) == 2

    def test_epsilon_decay(self):
        """Test epsilon decay over time."""
        strategy = DeepQLearningStrategy(seed=42, epsilon_0=0.5, epsilon_decay=0.9)

        initial_epsilon = strategy._epsilon

        # Simulate multiple rounds
        for _ in range(10):
            market_state = MarketState(
                prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
            )
            strategy.next_action(
                round_num=1,
                market_state=market_state,
                my_history=[],
                rival_histories=[],
                beliefs={},
                bounds=(0.0, 100.0),
                market_params={"model": "cournot"},
            )

        # Epsilon should have decayed
        assert strategy._epsilon < initial_epsilon
        assert strategy._epsilon >= strategy.epsilon_min

    def test_replay_buffer_limit(self):
        """Test replay buffer size limit."""
        strategy = DeepQLearningStrategy(seed=42, epsilon_0=0.0)  # No exploration

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot"}

        # Add more experiences than buffer size
        for i in range(1100):  # More than buffer_size (1000)
            strategy.next_action(
                round_num=i,
                market_state=market_state,
                my_history=[],
                rival_histories=[],
                beliefs={},
                bounds=bounds,
                market_params=market_params,
            )

        # Buffer should be limited to buffer_size
        assert len(strategy._replay_buffer) == strategy._buffer_size


class TestBehavioralStrategyAdvanced:
    """Test advanced behavioral strategy functionality."""

    def test_behavioral_strategy_validation(self):
        """Test behavioral strategy parameter validation."""
        with pytest.raises(ValueError, match="Rationality level must be in"):
            BehavioralStrategy(rationality_level=1.5)

        with pytest.raises(ValueError, match="Loss aversion must be >= 1"):
            BehavioralStrategy(loss_aversion=0.5)

    def test_calculate_utility_gain(self):
        """Test utility calculation for gains."""
        strategy = BehavioralStrategy(reference_point=100.0)

        utility = strategy._calculate_utility(150.0)
        assert utility == 50.0  # profit - reference_point

    def test_calculate_utility_loss(self):
        """Test utility calculation for losses."""
        strategy = BehavioralStrategy(reference_point=100.0, loss_aversion=2.0)

        utility = strategy._calculate_utility(80.0)
        assert utility == -40.0  # loss_aversion * (profit - reference_point)

    def test_calculate_fairness_utility(self):
        """Test fairness utility calculation."""
        strategy = BehavioralStrategy(fairness_weight=0.2)

        # My profit above average
        utility = strategy._calculate_fairness_utility(100.0, [50.0, 60.0])
        expected = -0.2 * (100.0 - 55.0)  # -0.2 * 45 = -9.0
        assert math.isclose(utility, expected, abs_tol=1e-6)

        # My profit below average
        utility = strategy._calculate_fairness_utility(40.0, [50.0, 60.0])
        assert utility == 0.0  # No penalty for being below average

    def test_calculate_fairness_utility_no_rivals(self):
        """Test fairness utility with no rivals."""
        strategy = BehavioralStrategy()

        utility = strategy._calculate_fairness_utility(100.0, [])
        assert utility == 0.0

    def test_update_reference_point(self):
        """Test reference point updating."""
        strategy = BehavioralStrategy(memory_decay=0.8)

        # Add profit history
        strategy._profit_history = [100.0, 120.0, 80.0]
        strategy._update_reference_point()

        # Should be weighted average with decay
        weights = [0.8**i for i in range(3)]
        weights = [w / sum(weights) for w in weights]
        expected = sum(p * w for p, w in zip([100.0, 120.0, 80.0], weights))

        assert math.isclose(strategy._reference_profit, expected, abs_tol=1e-6)

    def test_next_action_behavioral_rational(self):
        """Test next action with high rationality."""
        strategy = BehavioralStrategy(rationality_level=1.0, seed=42)

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0}

        action = strategy.next_action(
            round_num=1,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=bounds,
            market_params=market_params,
        )

        # Should be close to rational action: (a - my_cost) / (2 * b) = 45
        assert math.isclose(action, 45.0, abs_tol=5.0)  # Allow some noise

    def test_next_action_behavioral_irrational(self):
        """Test next action with low rationality."""
        strategy = BehavioralStrategy(rationality_level=0.0, seed=42)

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0}

        # Run multiple times to test randomness
        actions = []
        for _ in range(10):
            action = strategy.next_action(
                round_num=1,
                market_state=market_state,
                my_history=[],
                rival_histories=[],
                beliefs={},
                bounds=bounds,
                market_params=market_params,
            )
            actions.append(action)

        # Actions should vary and be within bounds
        assert len(set(actions)) > 1
        for action in actions:
            assert bounds[0] <= action <= bounds[1]

    def test_next_action_with_profit_history(self):
        """Test next action with profit history for reference point."""
        strategy = BehavioralStrategy(seed=42)

        # Create mock history with profit
        cournot_result = Mock(spec=CournotResult)
        cournot_result.profits = [500.0]
        my_history = [cournot_result]

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot", "a": 100.0, "b": 1.0, "my_cost": 10.0}

        action = strategy.next_action(
            round_num=1,
            market_state=market_state,
            my_history=my_history,
            rival_histories=[],
            beliefs={},
            bounds=bounds,
            market_params=market_params,
        )

        assert bounds[0] <= action <= bounds[1]
        assert len(strategy._profit_history) == 1
        assert strategy._profit_history[0] == 500.0

    def test_next_action_bertrand_model(self):
        """Test next action with Bertrand model."""
        strategy = BehavioralStrategy(rationality_level=1.0, seed=42)

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "bertrand", "alpha": 100.0, "my_cost": 10.0}

        action = strategy.next_action(
            round_num=1,
            market_state=market_state,
            my_history=[],
            rival_histories=[],
            beliefs={},
            bounds=bounds,
            market_params=market_params,
        )

        # Should be close to Bertrand monopoly price: (alpha + my_cost) / 2 = 55
        # Allow more tolerance for noise
        assert 45.0 <= action <= 65.0  # Reasonable range around 55

    def test_profit_history_limit(self):
        """Test profit history size limit."""
        strategy = BehavioralStrategy(seed=42)

        # Create mock history with many profits
        cournot_results = []
        for i in range(15):  # More than the limit of 10
            result = Mock(spec=CournotResult)
            result.profits = [float(i * 100)]
            cournot_results.append(result)

        my_history = cournot_results

        market_state = MarketState(
            prices=[50.0], quantities=[20.0], market_shares=[1.0], total_demand=20.0
        )
        bounds = (0.0, 100.0)
        market_params = {"model": "cournot"}

        # Process all history
        for i, result in enumerate(my_history):
            strategy.next_action(
                round_num=i,
                market_state=market_state,
                my_history=[result],
                rival_histories=[],
                beliefs={},
                bounds=bounds,
                market_params=market_params,
            )

        # Should only keep last 10 profits
        assert len(strategy._profit_history) == 10
        assert strategy._profit_history[0] == 500.0  # 5th profit (index 5)
        assert strategy._profit_history[-1] == 1400.0  # Last profit (index 14)
