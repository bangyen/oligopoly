"""Advanced strategy implementations for oligopoly market simulation.

This module implements sophisticated learning algorithms and strategic behaviors
including Deep Q-Networks, Fictitious Play, and behavioral economics elements.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

from ..games.bertrand import BertrandResult
from ..games.cournot import CournotResult


@dataclass
class MarketState:
    """Rich market state information for advanced strategies."""

    prices: List[float]
    quantities: List[float]
    market_shares: List[float]
    total_demand: float
    market_growth: float = 0.0
    innovation_level: float = 0.0
    regulatory_environment: str = "competitive"
    round_num: int = 0

    def __post_init__(self) -> None:
        """Validate market state data."""
        if len(self.prices) != len(self.quantities):
            raise ValueError("Prices and quantities must have same length")
        if len(self.market_shares) != len(self.prices):
            raise ValueError("Market shares must match number of firms")
        if not math.isclose(sum(self.market_shares), 1.0, abs_tol=1e-6):
            raise ValueError(
                f"Market shares must sum to 1.0, got {sum(self.market_shares)}"
            )


@dataclass
class StrategyBelief:
    """Belief about a rival firm's strategy."""

    firm_id: int
    action_history: List[float] = field(default_factory=list)
    belief_weights: List[float] = field(default_factory=list)
    confidence: float = 0.5
    last_update: int = 0

    def update_belief(
        self, action: float, round_num: int, decay_factor: float = 0.9
    ) -> None:
        """Update belief about rival's strategy."""
        self.action_history.append(action)
        self.last_update = round_num

        # Update weights with decay
        if self.belief_weights:
            self.belief_weights = [w * decay_factor for w in self.belief_weights]
            self.belief_weights.append(1.0)
        else:
            self.belief_weights = [1.0]

        # Normalize weights
        total_weight = sum(self.belief_weights)
        self.belief_weights = [w / total_weight for w in self.belief_weights]

        # Update confidence based on consistency
        if len(self.action_history) > 1:
            recent_actions = self.action_history[-5:]  # Last 5 actions
            variance = np.var(recent_actions) if len(recent_actions) > 1 else 0.0
            self.confidence = float(
                max(0.1, 1.0 - variance / 100.0)
            )  # Normalize variance

    def predict_action(self) -> float:
        """Predict rival's next action based on beliefs."""
        if not self.action_history:
            return 0.0

        # Weighted average of historical actions
        weighted_sum = sum(
            action * weight
            for action, weight in zip(self.action_history, self.belief_weights)
        )
        return float(weighted_sum)


class AdvancedStrategy(Protocol):
    """Protocol for advanced firm strategies with rich market information."""

    def next_action(
        self,
        round_num: int,
        market_state: MarketState,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        beliefs: Dict[int, StrategyBelief],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action with advanced market information."""
        ...


@dataclass
class FictitiousPlayStrategy:
    """Fictitious play strategy with belief updating and best response.

    Firms maintain beliefs about rivals' strategies and play best response
    to their beliefs about what rivals will do.
    """

    # Learning parameters
    belief_decay: float = 0.9
    exploration_rate: float = 0.1
    memory_length: int = 20

    # Strategy parameters
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize strategy parameters."""
        if not 0 <= self.belief_decay <= 1:
            raise ValueError(f"Belief decay must be in [0, 1], got {self.belief_decay}")
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError(
                f"Exploration rate must be in [0, 1], got {self.exploration_rate}"
            )

        self._rng = random.Random(self.seed)
        self._beliefs: Dict[int, StrategyBelief] = {}

    def _update_beliefs(
        self,
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        round_num: int,
    ) -> None:
        """Update beliefs about rival strategies."""
        for i, history in enumerate(rival_histories):
            if not history:
                continue

            last_result = history[-1]
            if isinstance(last_result, CournotResult):
                action = last_result.quantities[0] if last_result.quantities else 0.0
            elif isinstance(last_result, BertrandResult):
                action = last_result.prices[0] if last_result.prices else 0.0
            else:
                continue

            if i not in self._beliefs:
                self._beliefs[i] = StrategyBelief(firm_id=i)

            self._beliefs[i].update_belief(action, round_num, self.belief_decay)

    def _calculate_best_response(
        self,
        market_state: MarketState,
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
        model: str,
    ) -> float:
        """Calculate best response to beliefs about rivals."""
        min_bound, max_bound = bounds

        # Predict rivals' actions
        predicted_rival_actions = []
        for belief in self._beliefs.values():
            predicted_rival_actions.append(belief.predict_action())

        if not predicted_rival_actions:
            # No rivals, choose monopoly quantity/price
            if model == "cournot":
                a = market_params.get("a", 100.0)
                b = market_params.get("b", 1.0)
                monopoly_qty = a / (2 * b)
                return float(max(min_bound, min(max_bound, monopoly_qty)))
            else:  # bertrand
                alpha = market_params.get("alpha", 100.0)
                # Assume marginal cost of 10 for monopoly pricing
                monopoly_price = (alpha + 10) / 2
                return float(max(min_bound, min(max_bound, monopoly_price)))

        # Calculate best response to predicted actions
        if model == "cournot":
            return self._cournot_best_response(
                predicted_rival_actions, market_params, min_bound, max_bound
            )
        else:  # bertrand
            return self._bertrand_best_response(
                predicted_rival_actions, market_params, min_bound, max_bound
            )

    def _cournot_best_response(
        self,
        rival_quantities: List[float],
        market_params: Dict[str, Any],
        min_bound: float,
        max_bound: float,
    ) -> float:
        """Calculate Cournot best response to predicted rival quantities."""
        a = market_params.get("a", 100.0)
        b = market_params.get("b", 1.0)
        my_cost = market_params.get("my_cost", 10.0)

        total_rival_qty = sum(rival_quantities)
        best_response = (a - my_cost - b * total_rival_qty) / (2 * b)

        return float(max(min_bound, min(max_bound, best_response)))

    def _bertrand_best_response(
        self,
        rival_prices: List[float],
        market_params: Dict[str, Any],
        min_bound: float,
        max_bound: float,
    ) -> float:
        """Calculate Bertrand best response to predicted rival prices."""
        alpha = market_params.get("alpha", 100.0)
        my_cost = market_params.get("my_cost", 10.0)

        if not rival_prices:
            # No rivals, monopoly pricing
            return float((alpha + my_cost) / 2)

        min_rival_price = min(rival_prices)

        # Best response: undercut by small amount or price at cost
        if min_rival_price > my_cost + 0.1:
            best_response = min_rival_price - 0.1
        else:
            best_response = my_cost

        return float(max(min_bound, min(max_bound, best_response)))

    def next_action(
        self,
        round_num: int,
        market_state: MarketState,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        beliefs: Dict[int, StrategyBelief],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action using fictitious play."""
        # Update beliefs about rivals
        self._update_beliefs(rival_histories, round_num)

        # Determine model type from market params
        model = market_params.get("model", "cournot")

        # Calculate best response
        best_response = self._calculate_best_response(
            market_state, bounds, market_params, model
        )

        # Add exploration noise
        if self._rng.random() < self.exploration_rate:
            noise = self._rng.uniform(-0.1, 0.1) * best_response
            best_response += noise

        # Ensure within bounds
        min_bound, max_bound = bounds
        return float(max(min_bound, min(max_bound, best_response)))


@dataclass
class DeepQLearningStrategy:
    """Deep Q-Learning strategy with neural network approximation.

    This is a simplified version that uses function approximation
    instead of a full neural network for computational efficiency.
    """

    # Learning parameters
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    epsilon_0: float = 0.3
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995

    # Feature parameters
    feature_dim: int = 10
    action_dim: int = 20

    # Strategy parameters
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize DQN parameters."""
        if not 0 < self.learning_rate <= 1:
            raise ValueError(
                f"Learning rate must be in (0, 1], got {self.learning_rate}"
            )
        if not 0 <= self.discount_factor <= 1:
            raise ValueError(
                f"Discount factor must be in [0, 1], got {self.discount_factor}"
            )

        self._rng = random.Random(self.seed)
        self._epsilon = self.epsilon_0

        # Initialize Q-function approximation weights
        self._weights = np.random.normal(0, 0.1, (self.feature_dim, self.action_dim))
        self._bias = np.zeros(self.action_dim)

        # Experience replay buffer
        self._replay_buffer: List[Tuple] = []
        self._buffer_size = 1000

    def _extract_features(
        self,
        market_state: MarketState,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        bounds: Tuple[float, float],
    ) -> np.ndarray[np.float64, Any]:
        """Extract features from market state for Q-function."""
        features: np.ndarray[np.float64, Any] = np.zeros(
            self.feature_dim, dtype=np.float64
        )

        # Market features
        features[0] = market_state.total_demand / 100.0  # Normalized demand
        features[1] = np.mean(market_state.prices) / 100.0  # Normalized average price
        features[2] = market_state.market_growth
        features[3] = market_state.innovation_level

        # Competitive features
        if len(market_state.market_shares) > 1:
            features[4] = max(market_state.market_shares)  # Max market share
            features[5] = np.std(market_state.market_shares)  # Market share inequality

        # Historical features
        if my_history:
            last_result = my_history[-1]
            if isinstance(last_result, CournotResult):
                features[6] = (
                    last_result.quantities[0] / 100.0 if last_result.quantities else 0.0
                )
                features[7] = (
                    last_result.profits[0] / 1000.0 if last_result.profits else 0.0
                )
            elif isinstance(last_result, BertrandResult):
                features[6] = (
                    last_result.prices[0] / 100.0 if last_result.prices else 0.0
                )
                features[7] = (
                    last_result.profits[0] / 1000.0 if last_result.profits else 0.0
                )

        # Bounds features
        min_bound, max_bound = bounds
        features[8] = min_bound / 100.0
        features[9] = max_bound / 100.0

        return features

    def _q_values(self, features: np.ndarray) -> np.ndarray:
        """Calculate Q-values for all actions given features."""
        return np.dot(features, self._weights) + self._bias  # type: ignore

    def _select_action(
        self, q_values: np.ndarray, bounds: Tuple[float, float]
    ) -> float:
        """Select action using epsilon-greedy policy."""
        min_bound, max_bound = bounds

        if self._rng.random() < self._epsilon:
            # Explore: random action
            action = self._rng.uniform(min_bound, max_bound)
        else:
            # Exploit: best action
            action_index = np.argmax(q_values)
            # Map action index to actual action value
            action_range = max_bound - min_bound
            action = (
                min_bound + float(action_index / (self.action_dim - 1)) * action_range
            )

        return float(action)

    def _update_weights(
        self,
        features: np.ndarray,
        action_index: int,
        reward: float,
        next_features: np.ndarray,
    ) -> None:
        """Update Q-function weights using TD learning."""
        current_q = self._q_values(features)[action_index]
        next_q_values = self._q_values(next_features)
        target_q = reward + self.discount_factor * np.max(next_q_values)

        # TD error
        td_error = target_q - current_q

        # Update weights
        self._weights[:, action_index] += self.learning_rate * td_error * features
        self._bias[action_index] += self.learning_rate * td_error

    def _action_to_index(self, action: float, bounds: Tuple[float, float]) -> int:
        """Convert action value to action index."""
        min_bound, max_bound = bounds
        normalized = (action - min_bound) / (max_bound - min_bound)
        return int(np.clip(normalized * (self.action_dim - 1), 0, self.action_dim - 1))

    def next_action(
        self,
        round_num: int,
        market_state: MarketState,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        beliefs: Dict[int, StrategyBelief],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action using Deep Q-Learning."""
        # Extract features
        features = self._extract_features(market_state, my_history, bounds)

        # Calculate Q-values
        q_values = self._q_values(features)

        # Select action
        action = self._select_action(q_values, bounds)

        # Update weights if we have previous experience
        if my_history and len(self._replay_buffer) > 0:
            last_experience = self._replay_buffer[-1]
            prev_features, prev_action, prev_reward = last_experience

            # Calculate reward from last round
            last_result = my_history[-1]
            if isinstance(last_result, CournotResult):
                reward = last_result.profits[0] if last_result.profits else 0.0
            elif isinstance(last_result, BertrandResult):
                reward = last_result.profits[0] if last_result.profits else 0.0
            else:
                reward = 0.0

            # Normalize reward
            reward = reward / 1000.0

            # Update weights
            prev_action_index = self._action_to_index(prev_action, bounds)
            self._update_weights(prev_features, prev_action_index, reward, features)

        # Store experience
        self._replay_buffer.append(
            (features, action, 0.0)
        )  # Reward will be filled next round

        # Limit buffer size
        if len(self._replay_buffer) > self._buffer_size:
            self._replay_buffer.pop(0)

        # Decay epsilon
        self._epsilon = float(max(self.epsilon_min, self._epsilon * self.epsilon_decay))

        return float(action)


@dataclass
class BehavioralStrategy:
    """Strategy incorporating behavioral economics elements.

    Includes bounded rationality, loss aversion, and social preferences.
    """

    # Behavioral parameters
    rationality_level: float = 0.8  # 0 = random, 1 = fully rational
    loss_aversion: float = 2.0  # Loss aversion coefficient
    fairness_weight: float = 0.1  # Weight on fairness concerns
    reference_point: float = 0.0  # Reference point for loss aversion

    # Learning parameters
    learning_rate: float = 0.1
    memory_decay: float = 0.9

    # Strategy parameters
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize behavioral parameters."""
        if not 0 <= self.rationality_level <= 1:
            raise ValueError(
                f"Rationality level must be in [0, 1], got {self.rationality_level}"
            )
        if self.loss_aversion < 1:
            raise ValueError(f"Loss aversion must be >= 1, got {self.loss_aversion}")

        self._rng = random.Random(self.seed)
        self._profit_history: List[float] = []
        self._reference_profit = 0.0

    def _calculate_utility(self, profit: float) -> float:
        """Calculate utility incorporating loss aversion."""
        if profit >= self.reference_point:
            return profit - self.reference_point
        else:
            return self.loss_aversion * (profit - self.reference_point)

    def _calculate_fairness_utility(
        self, my_profit: float, rival_profits: List[float]
    ) -> float:
        """Calculate utility from fairness concerns."""
        if not rival_profits:
            return 0.0

        avg_rival_profit = float(np.mean(rival_profits))
        profit_difference = my_profit - avg_rival_profit

        # Fairness utility: negative if too much above average
        if profit_difference > 0:
            return -self.fairness_weight * profit_difference
        else:
            return 0.0

    def _update_reference_point(self) -> None:
        """Update reference point based on profit history."""
        if self._profit_history:
            # Use weighted average of recent profits
            weights = [self.memory_decay**i for i in range(len(self._profit_history))]
            weights = [w / sum(weights) for w in weights]
            self._reference_profit = sum(
                p * w for p, w in zip(self._profit_history, weights)
            )

    def next_action(
        self,
        round_num: int,
        market_state: MarketState,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        beliefs: Dict[int, StrategyBelief],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action incorporating behavioral elements."""
        min_bound, max_bound = bounds

        # Update reference point
        if my_history:
            last_result = my_history[-1]
            if isinstance(last_result, CournotResult):
                profit = last_result.profits[0] if last_result.profits else 0.0
            elif isinstance(last_result, BertrandResult):
                profit = last_result.profits[0] if last_result.profits else 0.0
            else:
                profit = 0.0

            self._profit_history.append(profit)
            if len(self._profit_history) > 10:  # Keep last 10 profits
                self._profit_history.pop(0)

            self._update_reference_point()

        # Calculate rational action (simplified)
        if market_params.get("model") == "cournot":
            a = market_params.get("a", 100.0)
            b = market_params.get("b", 1.0)
            my_cost = market_params.get("my_cost", 10.0)
            rational_action = (a - my_cost) / (2 * b)
        else:  # bertrand
            alpha = market_params.get("alpha", 100.0)
            my_cost = market_params.get("my_cost", 10.0)
            rational_action = (alpha + my_cost) / 2

        # Apply bounded rationality
        if self._rng.random() > self.rationality_level:
            # Random action
            action = self._rng.uniform(min_bound, max_bound)
        else:
            # Rational action with some noise
            noise = self._rng.uniform(-0.1, 0.1) * rational_action
            action = rational_action + noise

        # Ensure within bounds
        return float(max(min_bound, min(max_bound, action)))


def create_advanced_strategy(strategy_type: str, **kwargs: Any) -> AdvancedStrategy:
    """Factory function to create advanced strategies.

    Args:
        strategy_type: Type of strategy to create
        **kwargs: Strategy-specific parameters

    Returns:
        Advanced strategy instance

    Raises:
        ValueError: If strategy type is unknown
    """
    if strategy_type == "fictitious_play":
        return FictitiousPlayStrategy(**kwargs)
    elif strategy_type == "deep_q_learning":
        return DeepQLearningStrategy(**kwargs)
    elif strategy_type == "behavioral":
        return BehavioralStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown advanced strategy type: {strategy_type}")
