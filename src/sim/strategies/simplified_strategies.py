"""Simplified strategy implementations for oligopoly market simulation.

This module provides simplified learning algorithms that are easier to understand
and configure while maintaining economic realism.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..games.bertrand import BertrandResult
from ..games.cournot import CournotResult


@dataclass
class SimpleMarketState:
    """Simplified market state information."""

    prices: List[float]
    quantities: List[float]
    market_shares: List[float]
    total_demand: float
    round_num: int = 0

    def __post_init__(self) -> None:
        """Validate market state data."""
        if len(self.prices) != len(self.quantities):
            raise ValueError("Prices and quantities must have same length")


@dataclass
class SimpleStrategyBelief:
    """Simplified belief about a rival firm's strategy."""

    firm_id: int
    predicted_action: float
    confidence: float = 0.5  # How confident we are in this prediction


class SimpleFictitiousPlayStrategy:
    """Simplified Fictitious Play strategy with reduced complexity."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        memory_length: int = 10,
        seed: Optional[int] = None,
    ):
        """Initialize simplified fictitious play strategy."""
        self.learning_rate = learning_rate
        self.memory_length = memory_length
        self.rng = random.Random(seed)
        self.beliefs: Dict[int, SimpleStrategyBelief] = {}

    def _update_beliefs(
        self,
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
    ) -> None:
        """Update beliefs about rival strategies (simplified)."""
        for i, history in enumerate(rival_histories):
            if not history:
                continue

            # Simple belief: rival will repeat their last action
            last_result = history[-1]
            if isinstance(last_result, CournotResult):
                predicted_action = (
                    last_result.quantities[0] if last_result.quantities else 0.0
                )
            elif isinstance(last_result, BertrandResult):
                predicted_action = last_result.prices[0] if last_result.prices else 0.0
            else:
                predicted_action = 0.0

            # Update belief with simple exponential smoothing
            if i in self.beliefs:
                old_prediction = self.beliefs[i].predicted_action
                self.beliefs[i].predicted_action = (
                    1 - self.learning_rate
                ) * old_prediction + self.learning_rate * predicted_action
            else:
                self.beliefs[i] = SimpleStrategyBelief(
                    firm_id=i, predicted_action=predicted_action
                )

    def _calculate_best_response(
        self,
        market_state: SimpleMarketState,
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
        model: str,
    ) -> float:
        """Calculate best response to predicted rival actions (simplified)."""
        min_bound, max_bound = bounds

        if model == "cournot":
            return self._cournot_best_response(market_params, min_bound, max_bound)
        else:  # bertrand
            return self._bertrand_best_response(market_params, min_bound, max_bound)

    def _cournot_best_response(
        self,
        market_params: Dict[str, Any],
        min_bound: float,
        max_bound: float,
    ) -> float:
        """Calculate Cournot best response (simplified)."""
        a = market_params.get("a", 100.0)
        b = market_params.get("b", 1.0)
        my_cost = market_params.get("my_cost", 10.0)

        # Simple best response: assume rivals produce average quantity
        rival_quantities = [belief.predicted_action for belief in self.beliefs.values()]
        total_rival_qty = sum(rival_quantities) if rival_quantities else 0.0

        best_response = (a - my_cost - b * total_rival_qty) / (2 * b)
        return float(max(min_bound, min(max_bound, best_response)))

    def _bertrand_best_response(
        self,
        market_params: Dict[str, Any],
        min_bound: float,
        max_bound: float,
    ) -> float:
        """Calculate Bertrand best response (simplified)."""
        alpha = market_params.get("alpha", 100.0)
        my_cost = market_params.get("my_cost", 10.0)

        rival_prices = [belief.predicted_action for belief in self.beliefs.values()]
        if not rival_prices:
            # No rivals, monopoly pricing
            return float((alpha + my_cost) / 2)

        min_rival_price = min(rival_prices)

        # Simple best response: undercut by small amount
        if min_rival_price > my_cost + 0.1:
            best_response = min_rival_price - 0.1
        else:
            best_response = my_cost

        return float(max(min_bound, min(max_bound, best_response)))

    def next_action(
        self,
        round_num: int,
        market_state: SimpleMarketState,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action using simplified fictitious play."""
        # Update beliefs about rivals
        self._update_beliefs(rival_histories)

        # Determine model type
        model = "cournot" if "a" in market_params else "bertrand"

        # Calculate best response
        action = self._calculate_best_response(
            market_state, bounds, market_params, model
        )

        # Add small amount of exploration
        if self.rng.random() < 0.1:  # 10% exploration
            min_bound, max_bound = bounds
            noise = self.rng.uniform(-0.1, 0.1) * (max_bound - min_bound)
            action = max(min_bound, min(max_bound, action + noise))

        return float(action)


class SimpleQLearningStrategy:
    """Simplified Q-Learning strategy without neural networks."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        memory_length: int = 10,
        seed: Optional[int] = None,
    ):
        """Initialize simplified Q-learning strategy."""
        self.learning_rate = learning_rate
        self.memory_length = memory_length
        self.rng = random.Random(seed)
        self.q_table: Dict[Tuple, float] = {}
        self.action_history: List[Tuple] = []

    def _discretize_state(
        self, market_state: SimpleMarketState, bounds: Tuple[float, float]
    ) -> Tuple:
        """Discretize market state for Q-table lookup (simplified)."""
        # Simple state discretization
        price_bin = int(market_state.prices[0] / 10) if market_state.prices else 0
        demand_bin = int(market_state.total_demand / 20)
        round_bin = min(9, market_state.round_num // 10)  # 10 round bins

        return (price_bin, demand_bin, round_bin)

    def _discretize_action(self, action: float, bounds: Tuple[float, float]) -> int:
        """Discretize action for Q-table (simplified)."""
        min_bound, max_bound = bounds
        # 10 action bins
        normalized = (action - min_bound) / (max_bound - min_bound)
        return int(float(np.clip(normalized * 9, 0, 9)))

    def _get_q_value(self, state: Tuple, action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def _update_q_value(
        self, state: Tuple, action: int, reward: float, next_state: Tuple
    ) -> None:
        """Update Q-value using simplified Q-learning."""
        current_q = self._get_q_value(state, action)

        # Find best action in next state
        best_next_q = max(self._get_q_value(next_state, a) for a in range(10))

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + 0.9 * best_next_q - current_q
        )

        self.q_table[(state, action)] = new_q

    def next_action(
        self,
        round_num: int,
        market_state: SimpleMarketState,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action using simplified Q-learning."""
        # Discretize current state
        current_state = self._discretize_state(market_state, bounds)

        # Update Q-values if we have history
        if my_history and len(self.action_history) > 0:
            last_result = my_history[-1]
            if isinstance(last_result, CournotResult):
                reward = last_result.profits[0] if last_result.profits else 0.0
            elif isinstance(last_result, BertrandResult):
                reward = last_result.profits[0] if last_result.profits else 0.0
            else:
                reward = 0.0

            # Normalize reward
            reward = reward / 1000.0

            # Update Q-value
            prev_state, prev_action = self.action_history[-1]
            self._update_q_value(prev_state, prev_action, reward, current_state)

        # Choose action (epsilon-greedy)
        if self.rng.random() < 0.1:  # 10% exploration
            action_bin = self.rng.randint(0, 9)
        else:
            # Exploit: choose best action
            q_values = [self._get_q_value(current_state, a) for a in range(10)]
            action_bin = int(np.argmax(q_values))

        # Convert action bin back to continuous value
        min_bound, max_bound = bounds
        action = min_bound + (action_bin / 9.0) * (max_bound - min_bound)

        # Store action for next update
        self.action_history.append((current_state, action_bin))
        if len(self.action_history) > self.memory_length:
            self.action_history.pop(0)

        return float(action)


def create_simple_strategy(
    strategy_type: str,
    learning_rate: float = 0.1,
    memory_length: int = 10,
    seed: Optional[int] = None,
) -> Union[SimpleFictitiousPlayStrategy, SimpleQLearningStrategy]:
    """Create a simplified strategy instance."""
    if strategy_type == "fictitious_play":
        return SimpleFictitiousPlayStrategy(learning_rate, memory_length, seed)
    elif strategy_type == "q_learning":
        return SimpleQLearningStrategy(learning_rate, memory_length, seed)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
