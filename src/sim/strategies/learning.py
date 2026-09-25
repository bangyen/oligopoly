"""Learning strategies that the multi-round runner can assign to firms.

The runner calls every strategy as
``next_action(round_num, my_history, rival_histories, bounds, market_params)``
where each history entry is a *per-firm* result (index 0 holds that firm's
price, quantity and profit). The advanced strategies in
:mod:`sim.strategies.advanced_strategies` expect richer inputs (a
``MarketState`` and belief dict), so :class:`AdvancedStrategyAdapter`
builds those from the histories.

``market_params`` may carry ``best_response``: a callable mapping the
rivals' actions (in ``rival_histories`` order) to this firm's profit-
maximising action in the current market. Strategies use it instead of
hard-coded linear-demand formulas when present.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..games.bertrand import BertrandResult
from ..games.cournot import CournotResult
from .advanced_strategies import (
    BehavioralStrategy,
    DeepQLearningStrategy,
    FictitiousPlayStrategy,
    MarketState,
)
from .strategies import QLearning

LEARNING_STRATEGY_TYPES = (
    "fictitious_play",
    "q_learning",
    "deep_q_learning",
    "behavioral",
)

History = Sequence[CournotResult | BertrandResult]


def _action(result: CournotResult | BertrandResult) -> float:
    if isinstance(result, CournotResult):
        return result.quantities[0] if result.quantities else 0.0
    return result.prices[0] if result.prices else 0.0


def _price(result: CournotResult | BertrandResult) -> float:
    if isinstance(result, CournotResult):
        return result.price
    return result.prices[0] if result.prices else 0.0


def _market_state(
    round_num: int, my_history: History, rival_histories: list[History]
) -> MarketState | None:
    """Rebuild the last round's market from per-firm histories."""
    latest = [h[-1] for h in [my_history, *rival_histories] if h]
    if not latest:
        return None
    prices = [_price(r) for r in latest]
    quantities = [r.quantities[0] if r.quantities else 0.0 for r in latest]
    total = sum(quantities)
    shares = (
        [q / total for q in quantities]
        if total > 0
        else [1 / len(latest)] * len(latest)
    )
    # Guard against floating-point drift in MarketState's sum-to-one check
    shares[-1] = 1.0 - sum(shares[:-1])
    return MarketState(
        prices=prices,
        quantities=quantities,
        market_shares=[max(0.0, s) for s in shares],
        total_demand=total,
        round_num=round_num,
    )


class AdvancedStrategyAdapter:
    """Run an advanced strategy through the runner's strategy interface."""

    def __init__(
        self,
        strategy: FictitiousPlayStrategy | DeepQLearningStrategy | BehavioralStrategy,
    ):
        self.strategy = strategy

    def next_action(
        self,
        round_num: int,
        my_history: History,
        rival_histories: list[History],
        bounds: tuple[float, float],
        market_params: dict[str, Any],
    ) -> float:
        state = _market_state(round_num, my_history, rival_histories)
        if state is None:
            # Nothing observed yet: start from the best response to no rivals
            best_response = market_params.get("best_response")
            if best_response is not None:
                return float(best_response([0.0] * len(rival_histories)))
            return (bounds[0] + bounds[1]) / 2
        return self.strategy.next_action(
            round_num,
            state,
            my_history,
            rival_histories,
            {},
            bounds,
            market_params,
        )


def create_learning_strategy(
    strategy_type: str,
    bounds: tuple[float, float],
    reference_action: float,
    seed: int | None = None,
    learning_rate: float | None = None,
    memory_length: int | None = None,
    exploration_rate: float | None = None,
) -> AdvancedStrategyAdapter | QLearning:
    """Create a learning strategy for one firm.

    Args:
        strategy_type: One of :data:`LEARNING_STRATEGY_TYPES`
        bounds: The firm's (min, max) action bounds
        reference_action: A typical action (e.g. the Nash action), used to size
            the tabular Q-learning grid so it is not dominated by loose bounds
        seed: Random seed for the strategy's own RNG
        learning_rate: Learning rate (Q-learning, deep Q-learning, behavioral)
        memory_length: Rounds of rival history remembered (fictitious play)
        exploration_rate: Exploration probability (fictitious play, Q-learning)
    """
    kwargs: dict[str, Any] = {"seed": seed}
    if strategy_type == "fictitious_play":
        if memory_length:
            kwargs["memory_length"] = memory_length
        if exploration_rate is not None:
            kwargs["exploration_rate"] = exploration_rate
        return AdvancedStrategyAdapter(FictitiousPlayStrategy(**kwargs))
    if strategy_type in ("deep_q_learning", "behavioral"):
        if learning_rate:
            kwargs["learning_rate"] = learning_rate
        cls = (
            DeepQLearningStrategy
            if strategy_type == "deep_q_learning"
            else BehavioralStrategy
        )
        return AdvancedStrategyAdapter(cls(**kwargs))
    if strategy_type == "q_learning":
        lo, hi = bounds
        hi = min(hi, max(lo + 1.0, 3 * reference_action))
        epsilon = exploration_rate if exploration_rate is not None else 1.0
        return QLearning(
            min_action=lo,
            max_action=hi,
            step_size=(hi - lo) / 20,
            alpha=learning_rate or 0.1,
            epsilon_0=epsilon,
            epsilon_min=min(0.01, epsilon),
            seed=seed,
        )
    raise ValueError(
        f"Unknown learning strategy '{strategy_type}'; "
        f"expected one of {LEARNING_STRATEGY_TYPES}"
    )
