"""Strategy implementations for oligopoly market simulation.

This module defines the Strategy protocol and implements three concrete strategies
that can be used in both Cournot and Bertrand models: Static, TitForTat, and RandomWalk.

The strategies integrate with the existing simulation framework by working with
CournotResult and BertrandResult objects from the cournot and bertrand modules.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union

from .bertrand import BertrandResult
from .cournot import CournotResult


class Strategy(Protocol):
    """Protocol defining the interface for firm strategies.

    Strategies determine how firms choose their actions (quantities in Cournot,
    prices in Bertrand) based on market state and history.
    """

    def next_action(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate the next action based on market history.

        Args:
            round_num: Current round number (0-based)
            my_history: List of previous results for this firm
            rival_histories: List of histories for rival firms
            bounds: Tuple of (min, max) action bounds
            market_params: Additional market parameters

        Returns:
            The action to take (quantity for Cournot, price for Bertrand)
        """
        ...


@dataclass
class Static:
    """Static strategy that always returns the same value.

    This strategy is useful for testing and as a baseline comparison.
    It always returns the specified value regardless of market conditions.
    """

    value: float

    def __post_init__(self) -> None:
        """Validate the static value is non-negative."""
        if self.value < 0:
            raise ValueError(f"Static value {self.value} must be non-negative")

    def next_action(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Return the static value, clamped to bounds.

        Args:
            round_num: Current round number (ignored)
            my_history: Previous results for this firm (ignored)
            rival_histories: Previous results for rival firms (ignored)
            bounds: Tuple of (min, max) action bounds
            market_params: Additional market parameters (ignored)

        Returns:
            The static value, clamped to bounds
        """
        min_bound, max_bound = bounds
        return max(min_bound, min(max_bound, self.value))


@dataclass
class TitForTat:
    """Tit-for-tat strategy that mirrors rival behavior.

    On round 0, uses the midpoint of bounds. On subsequent rounds,
    mirrors the mean of rival actions from the previous round.
    """

    def next_action(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate action based on tit-for-tat logic.

        Args:
            round_num: Current round number
            my_history: Previous results for this firm
            rival_histories: Previous results for rival firms
            bounds: Tuple of (min, max) action bounds
            market_params: Additional market parameters (ignored)

        Returns:
            Action for this round
        """
        min_bound, max_bound = bounds

        if round_num == 0:
            # First round: use midpoint of bounds
            return (min_bound + max_bound) / 2.0

        if not rival_histories or not any(hist for hist in rival_histories):
            # No rival history: use midpoint
            return (min_bound + max_bound) / 2.0

        # Extract rival actions from previous round
        rival_actions = []
        for rival_history in rival_histories:
            if rival_history:  # Has history
                last_result = rival_history[-1]
                if isinstance(last_result, CournotResult):
                    # For Cournot, use quantity
                    rival_actions.append(
                        last_result.quantities[0]
                    )  # Assuming single firm per history
                elif isinstance(last_result, BertrandResult):
                    # For Bertrand, use price
                    rival_actions.append(
                        last_result.prices[0]
                    )  # Assuming single firm per history

        if not rival_actions:
            # No valid rival actions: use midpoint
            return (min_bound + max_bound) / 2.0

        # Mirror the mean of rival actions from previous round
        rival_mean = sum(rival_actions) / len(rival_actions)

        # Clamp to bounds
        return max(min_bound, min(max_bound, rival_mean))


@dataclass
class RandomWalk:
    """Random walk strategy with bounded steps.

    Each action is the previous action plus/minus a random step,
    bounded by the specified min/max values.
    """

    step: float
    min_bound: float
    max_bound: float
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters and initialize random state."""
        if self.step <= 0:
            raise ValueError(f"Step size {self.step} must be positive")
        if self.min_bound >= self.max_bound:
            raise ValueError(
                f"Min bound {self.min_bound} must be less than max bound {self.max_bound}"
            )

        # Initialize private random state
        self._rng = random.Random(self.seed)

    def next_action(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action using random walk.

        Args:
            round_num: Current round number
            my_history: Previous results for this firm
            rival_histories: Previous results for rival firms (ignored)
            bounds: Tuple of (min, max) action bounds
            market_params: Additional market parameters (ignored)

        Returns:
            Action for this round
        """
        min_bound, max_bound = bounds

        if round_num == 0 or not my_history:
            # First round or no history: start at midpoint
            return (min_bound + max_bound) / 2.0

        # Extract previous action from history
        last_result = my_history[-1]
        if isinstance(last_result, CournotResult):
            previous_action = last_result.quantities[
                0
            ]  # Assuming single firm per history
        elif isinstance(last_result, BertrandResult):
            previous_action = last_result.prices[0]  # Assuming single firm per history
        else:
            # Fallback to midpoint
            return (min_bound + max_bound) / 2.0

        # Random walk from previous action
        random_step = self._rng.uniform(-self.step, self.step)
        new_action = previous_action + random_step

        # Clamp to bounds
        return max(min_bound, min(max_bound, new_action))


def create_strategy(strategy_type: str, **kwargs: Any) -> Strategy:
    """Factory function to create strategy instances.

    Args:
        strategy_type: Type of strategy ("static", "titfortat", "randomwalk")
        **kwargs: Strategy-specific parameters

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy_type is not recognized
    """
    strategy_type = strategy_type.lower()

    if strategy_type == "static":
        if "value" not in kwargs:
            raise ValueError("Static strategy requires 'value' parameter")
        return Static(value=kwargs["value"])

    elif strategy_type == "titfortat":
        return TitForTat()

    elif strategy_type == "randomwalk":
        required_params = ["step", "min_bound", "max_bound"]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"RandomWalk strategy requires '{param}' parameter")

        return RandomWalk(
            step=kwargs["step"],
            min_bound=kwargs["min_bound"],
            max_bound=kwargs["max_bound"],
            seed=kwargs.get("seed"),
        )

    elif strategy_type == "epsilongreedy":
        required_params = ["min_action", "max_action", "step_size"]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"EpsilonGreedy strategy requires '{param}' parameter")

        return EpsilonGreedy(
            min_action=kwargs["min_action"],
            max_action=kwargs["max_action"],
            step_size=kwargs["step_size"],
            epsilon_0=kwargs.get("epsilon_0", 0.1),
            epsilon_min=kwargs.get("epsilon_min", 0.01),
            learning_rate=kwargs.get("learning_rate", 0.1),
            decay_rate=kwargs.get("decay_rate", 0.95),
            seed=kwargs.get("seed"),
        )

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


@dataclass
class EpsilonGreedy:
    """ε-greedy strategy with discrete action grid and Q-learning.

    This strategy implements a multi-armed bandit approach where firms learn
    to choose actions from a discrete grid based on Q-values updated by immediate
    rewards (profits). The exploration rate ε decays over time.
    """

    # Grid parameters
    min_action: float
    max_action: float
    step_size: float

    # Learning parameters
    epsilon_0: float = 0.1  # Initial exploration rate
    epsilon_min: float = 0.01  # Minimum exploration rate
    learning_rate: float = 0.1  # Q-learning update rate
    decay_rate: float = 0.95  # ε decay rate per round

    # Optional parameters
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters and initialize Q-values and action grid."""
        if self.min_action >= self.max_action:
            raise ValueError(
                f"Min action {self.min_action} must be less than max action {self.max_action}"
            )
        if self.step_size <= 0:
            raise ValueError(f"Step size {self.step_size} must be positive")
        if not 0 <= self.epsilon_0 <= 1:
            raise ValueError(f"Epsilon_0 {self.epsilon_0} must be in [0, 1]")
        if not 0 <= self.epsilon_min <= self.epsilon_0:
            raise ValueError(
                f"Epsilon_min {self.epsilon_min} must be in [0, epsilon_0]"
            )
        if not 0 < self.learning_rate <= 1:
            raise ValueError(f"Learning rate {self.learning_rate} must be in (0, 1]")
        if not 0 < self.decay_rate <= 1:
            raise ValueError(f"Decay rate {self.decay_rate} must be in (0, 1]")

        # Create discrete action grid
        self.action_grid = self._create_action_grid()
        self.num_actions = len(self.action_grid)

        # Initialize Q-values for each action
        self.q_values = [0.0] * self.num_actions

        # Initialize random number generator
        self._rng = random.Random(self.seed)

        # Track current epsilon
        self._current_epsilon = self.epsilon_0

        # Track previous action for Q-value updates
        self._previous_action_index: Optional[int] = None

    def _create_action_grid(self) -> List[float]:
        """Create discrete action grid from min to max with given step size."""
        actions = []
        current = self.min_action
        while current <= self.max_action:
            actions.append(current)
            current += self.step_size
        return actions

    def _get_action_index(self, action: float) -> int:
        """Get the index of the closest action in the grid."""
        distances = [abs(action - grid_action) for grid_action in self.action_grid]
        return distances.index(min(distances))

    def _update_q_value(self, action_index: int, reward: float) -> None:
        """Update Q-value for the given action using immediate reward."""
        if 0 <= action_index < self.num_actions:
            self.q_values[action_index] += self.learning_rate * (
                reward - self.q_values[action_index]
            )

    def _decay_epsilon(self) -> None:
        """Decay exploration rate while maintaining minimum."""
        self._current_epsilon = max(
            self.epsilon_min, self._current_epsilon * self.decay_rate
        )

    def _choose_action(self) -> float:
        """Choose action using ε-greedy policy."""
        if self._rng.random() < self._current_epsilon:
            # Explore: choose random action
            action_index = self._rng.randint(0, self.num_actions - 1)
        else:
            # Exploit: choose action with highest Q-value
            max_q = max(self.q_values)
            best_actions = [i for i, q in enumerate(self.q_values) if q == max_q]
            action_index = self._rng.choice(best_actions)

        return self.action_grid[action_index]

    def next_action(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
    ) -> float:
        """Calculate next action using ε-greedy policy.

        Args:
            round_num: Current round number
            my_history: Previous results for this firm
            rival_histories: Previous results for rival firms (ignored)
            bounds: Tuple of (min, max) action bounds (ignored - uses internal grid)
            market_params: Additional market parameters (ignored)

        Returns:
            Action for this round
        """
        # Update Q-values based on previous round's reward
        if round_num > 0 and my_history and self._previous_action_index is not None:
            last_result = my_history[-1]

            # Extract profit from last result
            if isinstance(last_result, CournotResult):
                profit = last_result.profits[0] if last_result.profits else 0.0
            elif isinstance(last_result, BertrandResult):
                profit = last_result.profits[0] if last_result.profits else 0.0
            else:
                profit = 0.0

            # Update Q-value for the action we actually chose
            self._update_q_value(self._previous_action_index, profit)

        # Decay epsilon
        self._decay_epsilon()

        # Choose next action
        action = self._choose_action()

        # Store the action index for next round's Q-value update
        self._previous_action_index = self._get_action_index(action)

        # Clamp to bounds (though our grid should already be within bounds)
        min_bound, max_bound = bounds
        return max(min_bound, min(max_bound, action))

    def get_q_values(self) -> List[float]:
        """Get current Q-values for all actions."""
        return self.q_values.copy()

    def get_current_epsilon(self) -> float:
        """Get current exploration rate."""
        return self._current_epsilon

    def get_action_grid(self) -> List[float]:
        """Get the discrete action grid."""
        return self.action_grid.copy()
