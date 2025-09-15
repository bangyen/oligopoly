"""Collusion-aware strategies for oligopoly simulation.

This module implements strategies that can participate in cartel agreements,
defect from collusion, or respond to regulatory interventions.
"""

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from ..collusion import CollusionEventType, CollusionManager
from ..games.bertrand import BertrandResult
from ..games.cournot import CournotResult


@dataclass
class CollusiveStrategy:
    """Strategy that participates in cartel agreements but may defect.

    This strategy follows cartel agreements but has a probability of defecting
    to gain higher short-term profits. Defection probability can be influenced
    by various factors like profit differentials and regulatory pressure.
    """

    defection_probability: float = 0.1  # Base probability of defecting
    defection_threshold: float = 0.2  # Profit advantage threshold for defection
    regulatory_sensitivity: float = (
        0.5  # How much regulatory pressure affects defection
    )
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters and initialize random state."""
        if not 0 <= self.defection_probability <= 1:
            raise ValueError(
                f"Defection probability {self.defection_probability} must be in [0, 1]"
            )
        if not 0 <= self.defection_threshold <= 1:
            raise ValueError(
                f"Defection threshold {self.defection_threshold} must be in [0, 1]"
            )
        if not 0 <= self.regulatory_sensitivity <= 1:
            raise ValueError(
                f"Regulatory sensitivity {self.regulatory_sensitivity} must be in [0, 1]"
            )

        self._rng = random.Random(self.seed)

    def calculate_defection_probability(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        collusion_manager: CollusionManager,
    ) -> float:
        """Calculate dynamic defection probability based on market conditions.

        Args:
            round_num: Current round number
            my_history: This firm's history
            rival_histories: Rival firms' histories
            collusion_manager: Collusion manager for market state

        Returns:
            Probability of defecting this round
        """
        base_prob = self.defection_probability

        # Increase defection probability if there have been recent regulatory interventions
        recent_interventions = sum(
            1
            for event in collusion_manager.events
            if event.event_type == CollusionEventType.REGULATOR_INTERVENED
            and event.round_idx >= round_num - 3
        )
        regulatory_factor = 1 + (recent_interventions * self.regulatory_sensitivity)

        # Adjust based on profit history (if we've been doing poorly, more likely to defect)
        if my_history and len(my_history) >= 2:
            recent_profit = my_history[-1].profits[0] if my_history[-1].profits else 0
            avg_profit = sum(r.profits[0] for r in my_history[-3:] if r.profits) / min(
                3, len(my_history)
            )

            if (
                recent_profit < avg_profit * 0.8
            ):  # Recent profit significantly below average
                base_prob *= 1.5

        return min(1.0, base_prob * regulatory_factor)

    def should_defect(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        collusion_manager: CollusionManager,
    ) -> bool:
        """Determine if this firm should defect from the cartel.

        Args:
            round_num: Current round number
            my_history: This firm's history
            rival_histories: Rival firms' histories
            collusion_manager: Collusion manager for market state

        Returns:
            True if firm should defect, False otherwise
        """
        if not collusion_manager.is_cartel_active():
            return False

        defection_prob = self.calculate_defection_probability(
            round_num, my_history, rival_histories, collusion_manager
        )

        return self._rng.random() < defection_prob

    def calculate_defection_action(
        self,
        cartel_price: float,
        cartel_quantity: float,
        bounds: Tuple[float, float],
        model_type: str,
    ) -> float:
        """Calculate the action to take when defecting.

        Args:
            cartel_price: Agreed cartel price
            cartel_quantity: Agreed cartel quantity per firm
            bounds: Action bounds (min, max)
            model_type: "cournot" or "bertrand"

        Returns:
            Defection action (undercut price or overproduce quantity)
        """
        min_bound, max_bound = bounds

        if model_type == "bertrand":
            # Defect by undercutting price (5-15% below cartel price)
            undercut_factor = self._rng.uniform(0.85, 0.95)
            defection_price = cartel_price * undercut_factor
            return max(min_bound, min(max_bound, defection_price))

        else:  # cournot
            # Defect by overproducing quantity (10-25% above cartel quantity)
            overproduce_factor = self._rng.uniform(1.10, 1.25)
            defection_quantity = cartel_quantity * overproduce_factor
            return max(min_bound, min(max_bound, defection_quantity))


@dataclass
class CartelStrategy:
    """Strategy that always follows cartel agreements.

    This strategy never defects and always follows the agreed cartel price/quantity.
    Useful for testing cartel stability and as a baseline comparison.
    """

    def next_action(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
        collusion_manager: Optional[CollusionManager] = None,
        my_cost: Optional[float] = None,
    ) -> float:
        """Always follow cartel agreement if active, otherwise use midpoint of bounds.

        Args:
            round_num: Current round number
            my_history: Previous results for this firm
            rival_histories: Previous results for rival firms
            bounds: Tuple of (min, max) action bounds
            market_params: Additional market parameters
            collusion_manager: Collusion manager (optional)

        Returns:
            Cartel-compliant action
        """
        min_bound, max_bound = bounds

        # If cartel is active, follow the agreement
        if collusion_manager and collusion_manager.is_cartel_active():
            cartel = collusion_manager.current_cartel
            if cartel:
                # Determine if this is Cournot (quantity) or Bertrand (price) model
                model_type = market_params.get("model_type", "cournot")

                if model_type == "bertrand":
                    return max(min_bound, min(max_bound, cartel.collusive_price))
                else:  # cournot
                    return max(min_bound, min(max_bound, cartel.collusive_quantity))

        # No cartel active, use midpoint of bounds
        return (min_bound + max_bound) / 2.0


@dataclass
class OpportunisticStrategy:
    """Strategy that opportunistically participates in cartels but defects when profitable.

    This strategy joins cartels when they form but calculates whether defection
    would be more profitable based on market conditions and rival behavior.
    """

    profit_threshold_multiplier: float = 1.3  # Minimum profit advantage to defect
    risk_tolerance: float = 0.5  # Risk tolerance for defection
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters and initialize random state."""
        if self.profit_threshold_multiplier <= 1:
            raise ValueError(
                f"Profit threshold multiplier {self.profit_threshold_multiplier} must be > 1"
            )
        if not 0 <= self.risk_tolerance <= 1:
            raise ValueError(f"Risk tolerance {self.risk_tolerance} must be in [0, 1]")

        self._rng = random.Random(self.seed)

    def estimate_defection_profit(
        self,
        cartel_price: float,
        cartel_quantity: float,
        my_cost: float,
        market_params: Dict[str, Any],
        model_type: str,
    ) -> float:
        """Estimate profit from defection.

        Args:
            cartel_price: Agreed cartel price
            cartel_quantity: Agreed cartel quantity per firm
            my_cost: This firm's marginal cost
            market_params: Market parameters
            model_type: "cournot" or "bertrand"

        Returns:
            Estimated profit from defection
        """
        if model_type == "bertrand":
            # Defect by undercutting price by 10%
            defection_price = cartel_price * 0.9

            # Estimate quantity sold (simplified - assumes we capture most demand)
            alpha = float(market_params.get("alpha", 100.0))
            beta = float(market_params.get("beta", 1.0))
            estimated_quantity = max(0, alpha - beta * defection_price)

            return (defection_price - my_cost) * estimated_quantity

        else:  # cournot
            # Defect by overproducing by 20%
            defection_quantity = cartel_quantity * 1.2

            # Estimate market price (simplified)
            a = float(market_params.get("a", 100.0))
            b = float(market_params.get("b", 1.0))
            # Assume rivals stick to cartel quantity
            total_quantity = (
                defection_quantity + cartel_quantity * 2
            )  # Assuming 3 firms
            estimated_price = max(0, a - b * total_quantity)

            return (estimated_price - my_cost) * defection_quantity

    def estimate_cartel_profit(
        self,
        cartel_price: float,
        cartel_quantity: float,
        my_cost: float,
        model_type: str,
    ) -> float:
        """Estimate profit from following cartel agreement.

        Args:
            cartel_price: Agreed cartel price
            cartel_quantity: Agreed cartel quantity per firm
            my_cost: This firm's marginal cost
            model_type: "cournot" or "bertrand"

        Returns:
            Estimated profit from cartel compliance
        """
        if model_type == "bertrand":
            return (cartel_price - my_cost) * cartel_quantity
        else:  # cournot
            return (cartel_price - my_cost) * cartel_quantity

    def should_defect(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        collusion_manager: CollusionManager,
        my_cost: float,
        market_params: Dict[str, Any],
    ) -> bool:
        """Determine if defection would be profitable.

        Args:
            round_num: Current round number
            my_history: This firm's history
            rival_histories: Rival firms' histories
            collusion_manager: Collusion manager for market state
            my_cost: This firm's marginal cost
            market_params: Market parameters

        Returns:
            True if defection appears profitable
        """
        if not collusion_manager.is_cartel_active():
            return False

        cartel = collusion_manager.current_cartel
        if not cartel:
            return False

        model_type = market_params.get("model_type", "cournot")

        # Estimate profits
        cartel_profit = self.estimate_cartel_profit(
            cartel.collusive_price, cartel.collusive_quantity, my_cost, model_type
        )

        defection_profit = self.estimate_defection_profit(
            cartel.collusive_price,
            cartel.collusive_quantity,
            my_cost,
            market_params,
            model_type,
        )

        # Check if defection profit exceeds threshold
        profit_advantage = defection_profit / cartel_profit if cartel_profit > 0 else 0

        # Add some randomness based on risk tolerance
        risk_factor = self._rng.random() * self.risk_tolerance

        return profit_advantage > self.profit_threshold_multiplier + risk_factor

    def next_action(
        self,
        round_num: int,
        my_history: Sequence[Union[CournotResult, BertrandResult]],
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]],
        bounds: Tuple[float, float],
        market_params: Dict[str, Any],
        collusion_manager: Optional[CollusionManager] = None,
        my_cost: Optional[float] = None,
    ) -> float:
        """Calculate action based on opportunistic defection logic.

        Args:
            round_num: Current round number
            my_history: Previous results for this firm
            rival_histories: Previous results for rival firms
            bounds: Tuple of (min, max) action bounds
            market_params: Additional market parameters
            collusion_manager: Collusion manager (optional)
            my_cost: This firm's marginal cost (optional)

        Returns:
            Action for this round
        """
        min_bound, max_bound = bounds

        # If cartel is active, decide whether to defect
        if (
            collusion_manager
            and collusion_manager.is_cartel_active()
            and my_cost is not None
        ):

            cartel = collusion_manager.current_cartel
            if cartel:
                model_type = market_params.get("model_type", "cournot")

                # Check if we should defect
                if self.should_defect(
                    round_num,
                    my_history,
                    rival_histories,
                    collusion_manager,
                    my_cost,
                    market_params,
                ):
                    # Calculate defection action
                    if model_type == "bertrand":
                        defection_price = cartel.collusive_price * 0.9  # 10% undercut
                        return max(min_bound, min(max_bound, defection_price))
                    else:  # cournot
                        defection_quantity = (
                            cartel.collusive_quantity * 1.2
                        )  # 20% overproduce
                        return max(min_bound, min(max_bound, defection_quantity))

                # Follow cartel agreement
                if model_type == "bertrand":
                    return max(min_bound, min(max_bound, cartel.collusive_price))
                else:  # cournot
                    return max(min_bound, min(max_bound, cartel.collusive_quantity))

        # No cartel active or missing cost info, use midpoint
        return (min_bound + max_bound) / 2.0


def create_collusion_strategy(
    strategy_type: str, **kwargs: Any
) -> Union[CartelStrategy, CollusiveStrategy, OpportunisticStrategy]:
    """Factory function to create collusion-aware strategy instances.

    Args:
        strategy_type: Type of strategy ("cartel", "collusive", "opportunistic")
        **kwargs: Strategy-specific parameters

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy_type is not recognized
    """
    strategy_type = strategy_type.lower()

    if strategy_type == "cartel":
        return CartelStrategy()

    elif strategy_type == "collusive":
        return CollusiveStrategy(
            defection_probability=kwargs.get("defection_probability", 0.1),
            defection_threshold=kwargs.get("defection_threshold", 0.2),
            regulatory_sensitivity=kwargs.get("regulatory_sensitivity", 0.5),
            seed=kwargs.get("seed"),
        )

    elif strategy_type == "opportunistic":
        return OpportunisticStrategy(
            profit_threshold_multiplier=kwargs.get("profit_threshold_multiplier", 1.3),
            risk_tolerance=kwargs.get("risk_tolerance", 0.5),
            seed=kwargs.get("seed"),
        )

    else:
        raise ValueError(f"Unknown collusion strategy type: {strategy_type}")
