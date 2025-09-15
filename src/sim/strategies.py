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

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
