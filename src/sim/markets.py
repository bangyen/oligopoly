"""Market (demand system) abstraction used by the multi-round runner.

A :class:`Market` bundles everything the runner needs to know about one
demand specification: how a round is played, the one-shot Nash equilibrium,
best responses, sensible action bounds, how firms adapt towards equilibrium
between rounds, and the market-level metrics reported by the API.

Supported markets:

- ``LinearCournotMarket`` / ``LinearBertrandMarket``: the original linear
  (optionally segmented) demand. These delegate to the long-standing engine
  functions so existing runs are unchanged.
- ``IsoelasticCournotMarket`` / ``IsoelasticBertrandMarket``: constant
  elasticity demand Q(P) = (A / P)^e, i.e. P(Q) = A * Q^(-1/e), with e > 1.
- ``CESBertrandMarket``: differentiated-products price competition with nested
  CES demand (elasticity of substitution sigma > 1 within the group, market
  elasticity eta > 1 across it, and per-firm qualities).
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

from sim.games.bertrand import (
    BertrandResult,
    bertrand_segmented_simulation,
    bertrand_simulation,
)
from sim.games.cournot import (
    CournotResult,
    cournot_segmented_simulation,
    cournot_simulation,
)
from sim.models.metrics import (
    calculate_hhi,
    calculate_market_shares_bertrand,
    calculate_market_shares_cournot,
)
from sim.models.models import DemandSegment, SegmentedDemand
from sim.strategies.nash_strategies import (
    adaptive_nash_strategy,
    bertrand_nash_equilibrium,
    cournot_nash_equilibrium,
    validate_market_clearing,
)

RoundResult: TypeAlias = CournotResult | BertrandResult

DEMAND_TYPES = ("linear", "isoelastic", "ces")


def _convergence_factor(round_idx: int, rounds: int) -> float:
    """Same schedule as adaptive_nash_strategy: 0.3 early, 0.1 at the end."""
    return 0.1 + 0.2 * (1 - round_idx / rounds)


def _golden_max(f: Any, lo: float, hi: float, iterations: int = 60) -> float:
    """Maximise a unimodal function on [lo, hi] by golden-section search."""
    ratio = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iterations):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = f(d)
    return (a + b) / 2


def _maximize(f: Any, lo: float, hi: float) -> float:
    """Maximise ``f`` on [lo, hi].

    A coarse grid locates the best region (profit may be discontinuous in
    Bertrand) and a golden-section search refines it.
    """
    if hi <= lo:
        return lo
    grid = [lo + (hi - lo) * k / 400 for k in range(401)]
    values = [f(x) for x in grid]
    k = max(range(len(grid)), key=values.__getitem__)
    left, right = grid[max(0, k - 1)], grid[min(len(grid) - 1, k + 1)]
    refined = _golden_max(f, left, right)
    return float(refined if f(refined) > values[k] else grid[k])


class Market(ABC):
    """A demand system plus the competition model played on it."""

    model: str  # "cournot" or "bertrand"
    demand_type: str

    @abstractmethod
    def play(
        self, actions: list[float], costs: list[float], fixed_costs: list[float]
    ) -> RoundResult:
        """Play one round given each firm's action (quantity or price)."""

    @abstractmethod
    def nash(self, costs: list[float]) -> list[float]:
        """Return the one-shot Nash equilibrium actions for these costs."""

    @abstractmethod
    def action_bounds(self, cost: float, costs: list[float]) -> tuple[float, float]:
        """Return the (min, max) action a firm with ``cost`` may choose."""

    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Return the serialisable demand parameters (persisted with the run)."""

    @abstractmethod
    def scale_demand(self, factor: float) -> None:
        """Grow (factor > 1) or shrink the market, e.g. for market evolution."""

    def validate(self, costs: list[float]) -> None:
        """Raise ValueError if the market cannot be simulated with these costs."""

    def set_qualities(self, qualities: list[float]) -> None:  # noqa: B027
        """Update per-firm product qualities (only differentiated markets use them)."""

    def profit(
        self, firm: int, action: float, actions: list[float], costs: list[float]
    ) -> float:
        """Variable profit of ``firm`` if it plays ``action`` against ``actions``."""
        trial = list(actions)
        trial[firm] = action
        return float(self.play(trial, costs, [0.0] * len(costs)).profits[firm])

    def best_response(
        self, firm: int, actions: list[float], costs: list[float]
    ) -> float:
        """Numerically maximise ``firm``'s profit holding rivals fixed."""
        lo, hi = self.action_bounds(costs[firm], costs)
        return _maximize(lambda x: self.profit(firm, x, actions, costs), lo, hi)

    def collusive_action(
        self, members: list[int], actions: list[float], costs: list[float]
    ) -> float:
        """Common action maximising the members' joint profit.

        Members all play the same action; non-members keep ``actions``.
        """
        zeros = [0.0] * len(costs)

        def joint(x: float) -> float:
            trial = list(actions)
            for m in members:
                trial[m] = x
            profits = self.play(trial, costs, zeros).profits
            return float(sum(profits[m] for m in members))

        bounds = [self.action_bounds(costs[m], costs) for m in members]
        lo = max(b[0] for b in bounds)
        hi = max(lo, min(b[1] for b in bounds))
        return _maximize(joint, lo, hi)

    def initial_actions(self, costs: list[float]) -> list[float]:
        """Nash actions with a small random perturbation (uses ``random``)."""
        actions = []
        for cost, action in zip(costs, self.nash(costs)):
            lo, hi = self.action_bounds(cost, costs)
            jitter = random.uniform(-0.01, 0.01) * action
            actions.append(min(hi, max(lo, action + jitter)))
        return actions

    def adapt(
        self,
        actions: list[float],
        profits: list[float],
        costs: list[float],
        round_idx: int,
        rounds: int,
    ) -> list[float]:
        """Move each firm part of the way towards its Nash action, with noise."""
        factor = _convergence_factor(round_idx, rounds)
        noise_factor = 0.01 * (1 - factor)
        new_actions = []
        for cost, current, target in zip(costs, actions, self.nash(costs)):
            value = (1 - factor) * current + factor * target
            value += random.uniform(-noise_factor, noise_factor) * value
            lo, hi = self.action_bounds(cost, costs)
            new_actions.append(min(hi, max(lo, value)))
        return new_actions

    def clear(self, actions: list[float], costs: list[float]) -> list[float]:
        """Clamp proposed actions to their bounds before the next round."""
        return [
            min(hi, max(lo, a))
            for a, (lo, hi) in zip(
                actions, (self.action_bounds(c, costs) for c in costs)
            )
        ]

    def strategy_params(self) -> dict[str, Any]:
        """Parameters passed to learning strategies as ``market_params``."""
        return {**self.params(), "model_type": self.model, "model": self.model}

    @abstractmethod
    def metrics(
        self, prices: list[float], quantities: list[float]
    ) -> tuple[float, float, float | None]:
        """Return (market_price, HHI, consumer surplus or None) for a round."""


# ---------------------------------------------------------------------------
# Linear demand (original engine)
# ---------------------------------------------------------------------------


def _segments(params: dict[str, Any]) -> SegmentedDemand | None:
    config = params.get("segments")
    if not config:
        return None
    return SegmentedDemand(
        segments=[
            DemandSegment(
                alpha=float(s["alpha"]),
                beta=float(s["beta"]),
                weight=float(s["weight"]),
            )
            for s in config
        ]
    )


def _weighted(params: dict[str, Any]) -> tuple[float, float] | None:
    config = params.get("segments")
    if not config:
        return None
    return (
        sum(s["alpha"] * s["weight"] for s in config),
        sum(s["beta"] * s["weight"] for s in config),
    )


def _hhi_from_shares(shares: list[float]) -> float:
    return calculate_hhi(shares) if shares and sum(shares) > 0 else 0.0


class LinearCournotMarket(Market):
    """Cournot competition with inverse demand P = a - bQ (or segmented)."""

    model = "cournot"
    demand_type = "linear"

    def __init__(self, params: dict[str, Any]):
        self._params = dict(params)
        self._params.setdefault("a", 100.0)
        self._params.setdefault("b", 1.0)

    def _ab(self) -> tuple[float, float]:
        return _weighted(self._params) or (self._params["a"], self._params["b"])

    def play(
        self, actions: list[float], costs: list[float], fixed_costs: list[float]
    ) -> RoundResult:
        segmented = _segments(self._params)
        if segmented:
            return cournot_segmented_simulation(segmented, costs, actions, fixed_costs)
        return cournot_simulation(
            self._params["a"], self._params["b"], costs, actions, fixed_costs
        )

    def nash(self, costs: list[float]) -> list[float]:
        a, b = self._ab()
        return list(cournot_nash_equilibrium(a, b, costs)[0])

    def action_bounds(self, cost: float, costs: list[float]) -> tuple[float, float]:
        a, b = self._ab()
        return (0.1, a / b)

    def best_response(
        self, firm: int, actions: list[float], costs: list[float]
    ) -> float:
        a, b = self._ab()
        rivals = sum(actions) - actions[firm]
        return max(0.0, (a - costs[firm] - b * rivals) / (2 * b))

    def initial_actions(self, costs: list[float]) -> list[float]:
        return [max(0.1, q + random.uniform(-1, 1)) for q in self.nash(costs)]

    def adapt(
        self,
        actions: list[float],
        profits: list[float],
        costs: list[float],
        round_idx: int,
        rounds: int,
    ) -> list[float]:
        return list(
            adaptive_nash_strategy(
                "cournot", actions, profits, costs, self._params, round_idx, rounds
            )
        )

    def clear(self, actions: list[float], costs: list[float]) -> list[float]:
        return list(validate_market_clearing("cournot", actions, costs, self._params))

    def params(self) -> dict[str, Any]:
        return {**self._params, "demand_type": "linear"}

    def scale_demand(self, factor: float) -> None:
        self._params["a"] *= factor
        for segment in self._params.get("segments") or []:
            segment["alpha"] *= factor

    def metrics(
        self, prices: list[float], quantities: list[float]
    ) -> tuple[float, float, float | None]:
        a, b = self._ab()
        price = prices[0] if prices else 0.0
        total = sum(quantities)
        cs = 0.5 * max(0.0, a - price) * total
        return price, _hhi_from_shares(calculate_market_shares_cournot(quantities)), cs


class LinearBertrandMarket(Market):
    """Bertrand competition with demand Q = alpha - beta*p (or segmented).

    By default each firm can serve at most 40% of the market
    (``capacity_constraints``); set ``params["capacity_constraints"] = False``
    for the textbook winner-take-all game.
    """

    model = "bertrand"
    demand_type = "linear"

    def __init__(self, params: dict[str, Any]):
        self._params = dict(params)
        self._params.setdefault("alpha", 100.0)
        self._params.setdefault("beta", 1.0)

    def _ab(self) -> tuple[float, float]:
        return _weighted(self._params) or (
            self._params["alpha"],
            self._params["beta"],
        )

    def play(
        self, actions: list[float], costs: list[float], fixed_costs: list[float]
    ) -> RoundResult:
        segmented = _segments(self._params)
        if segmented:
            return bertrand_segmented_simulation(segmented, costs, actions, fixed_costs)
        return bertrand_simulation(
            self._params["alpha"],
            self._params["beta"],
            costs,
            actions,
            fixed_costs,
            use_capacity_constraints=bool(
                self._params.get("capacity_constraints", True)
            ),
        )

    def nash(self, costs: list[float]) -> list[float]:
        alpha, beta = self._ab()
        return list(bertrand_nash_equilibrium(alpha, beta, costs)[0])

    def action_bounds(self, cost: float, costs: list[float]) -> tuple[float, float]:
        # Nothing sells above the choke price alpha / beta (per segment)
        segments = self._params.get("segments")
        if segments:
            choke = max(s["alpha"] / s["beta"] for s in segments)
        else:
            choke = self._params["alpha"] / self._params["beta"]
        return (cost + 0.1, max(cost + 0.2, choke))

    def initial_actions(self, costs: list[float]) -> list[float]:
        return [
            max(c + 0.1, p + random.uniform(-1, 1))
            for c, p in zip(costs, self.nash(costs))
        ]

    def adapt(
        self,
        actions: list[float],
        profits: list[float],
        costs: list[float],
        round_idx: int,
        rounds: int,
    ) -> list[float]:
        return list(
            adaptive_nash_strategy(
                "bertrand", actions, profits, costs, self._params, round_idx, rounds
            )
        )

    def clear(self, actions: list[float], costs: list[float]) -> list[float]:
        return list(validate_market_clearing("bertrand", actions, costs, self._params))

    def params(self) -> dict[str, Any]:
        return {**self._params, "demand_type": "linear"}

    def scale_demand(self, factor: float) -> None:
        self._params["alpha"] *= factor
        for segment in self._params.get("segments") or []:
            segment["alpha"] *= factor

    def metrics(
        self, prices: list[float], quantities: list[float]
    ) -> tuple[float, float, float | None]:
        alpha, beta = self._ab()
        price = min(prices) if prices else 0.0
        total = sum(quantities)
        # Demand Q = alpha - beta*p has price intercept alpha / beta
        cs = 0.5 * max(0.0, alpha / beta - price) * total
        shares = (
            calculate_market_shares_bertrand(prices, quantities) if total > 0 else []
        )
        return price, _hhi_from_shares(shares), cs


# ---------------------------------------------------------------------------
# Isoelastic demand Q(P) = (A / P)^e
# ---------------------------------------------------------------------------


class _IsoelasticMixin:
    _A: float
    _e: float

    def _setup(self, params: dict[str, Any]) -> None:
        self._A = float(params.get("A", 100.0))
        self._e = float(params.get("elasticity", 2.0))
        if self._A <= 0:
            raise ValueError(f"Isoelastic scale A must be positive, got {self._A}")
        if self._e <= 1:
            raise ValueError(f"Isoelastic elasticity must be > 1, got {self._e}")

    def demand(self, price: float) -> float:
        return (self._A / price) ** self._e if price > 0 else math.inf

    def inverse_demand(self, quantity: float) -> float:
        return self._A * quantity ** (-1 / self._e) if quantity > 0 else math.inf

    def monopoly_price(self, cost: float) -> float:
        return cost * self._e / (self._e - 1)

    def consumer_surplus(self, price: float) -> float:
        """Integral of Q(p) from price to infinity = A^e P^(1-e) / (e - 1)."""
        if price <= 0:
            return 0.0
        return float(self._A**self._e * price ** (1 - self._e) / (self._e - 1))

    def params(self) -> dict[str, Any]:
        return {"demand_type": "isoelastic", "A": self._A, "elasticity": self._e}

    def scale_demand(self, factor: float) -> None:
        # Scaling quantity demanded at every price by `factor`
        self._A *= factor ** (1 / self._e)


class IsoelasticCournotMarket(_IsoelasticMixin, Market):
    """Cournot competition with P(Q) = A * Q^(-1/e).

    Interior Nash equilibrium (textbook): with active set S,
    P* = sum_{j in S} c_j / (|S| - 1/e), share s_i = e * (1 - c_i / P*),
    Q* = (A / P*)^e and q_i = s_i Q*. Firms with s_i <= 0 are inactive.
    """

    model = "cournot"
    demand_type = "isoelastic"

    def __init__(self, params: dict[str, Any]):
        self._setup(params)

    def play(
        self, actions: list[float], costs: list[float], fixed_costs: list[float]
    ) -> RoundResult:
        if any(q < 0 for q in actions):
            raise ValueError("Quantities must be non-negative")
        total = sum(actions)
        price = self.inverse_demand(total) if total > 0 else 0.0
        profits = [
            (price - c) * q - fc for c, q, fc in zip(costs, actions, fixed_costs)
        ]
        return CournotResult(price=price, quantities=list(actions), profits=profits)

    def equilibrium(self, costs: list[float]) -> tuple[list[float], float]:
        """Return (quantities, price) of the Nash equilibrium."""
        active = sorted(range(len(costs)), key=lambda i: costs[i])
        while active:
            price = sum(costs[i] for i in active) / (len(active) - 1 / self._e)
            shares = {i: self._e * (1 - costs[i] / price) for i in active}
            if all(s > 0 for s in shares.values()):
                total = self.demand(price)
                quantities = [shares.get(i, 0.0) * total for i in range(len(costs))]
                return quantities, price
            active = active[:-1]  # drop the highest-cost firm and re-solve
        return [0.0] * len(costs), 0.0

    def nash(self, costs: list[float]) -> list[float]:
        return self.equilibrium(costs)[0]

    def action_bounds(self, cost: float, costs: list[float]) -> tuple[float, float]:
        # Beyond the quantity that drives price to the lowest cost, output
        # can only lose money for everyone.
        return (0.0, self.demand(min(costs)))

    def metrics(
        self, prices: list[float], quantities: list[float]
    ) -> tuple[float, float, float | None]:
        price = prices[0] if prices else 0.0
        return (
            price,
            _hhi_from_shares(calculate_market_shares_cournot(quantities)),
            self.consumer_surplus(price),
        )


class IsoelasticBertrandMarket(_IsoelasticMixin, Market):
    """Homogeneous-good Bertrand competition with Q(P) = (A / P)^e.

    The lowest price takes the market; ties go to the lowest-cost firm(s) and
    are split among equal-cost firms. Equilibrium: if the lowest
    cost is shared, price equals that cost; otherwise the efficient firm prices
    at min(second-lowest cost, its monopoly price c * e / (e - 1)).
    """

    model = "bertrand"
    demand_type = "isoelastic"

    def __init__(self, params: dict[str, Any]):
        self._setup(params)

    def play(
        self, actions: list[float], costs: list[float], fixed_costs: list[float]
    ) -> RoundResult:
        if any(p < 0 for p in actions):
            raise ValueError("Prices must be non-negative")
        low = min(actions)
        tied = [i for i, p in enumerate(actions) if math.isclose(p, low)]
        # Ties go to the lowest-cost firm(s), the textbook convention that makes
        # limit pricing at the rival's cost an exact equilibrium.
        best_cost = min(costs[i] for i in tied)
        winners = [i for i in tied if math.isclose(costs[i], best_cost)]
        total = self.demand(low) if low > 0 else 0.0
        quantities = [0.0] * len(actions)
        for i in winners:
            quantities[i] = total / len(winners)
        profits = [
            (p - c) * q - fc
            for p, c, q, fc in zip(actions, costs, quantities, fixed_costs)
        ]
        return BertrandResult(
            total_demand=total,
            prices=list(actions),
            quantities=quantities,
            profits=profits,
        )

    def nash(self, costs: list[float]) -> list[float]:
        low = min(costs)
        leaders = [i for i, c in enumerate(costs) if math.isclose(c, low)]
        if len(leaders) > 1:
            price = low
        else:
            rivals = [c for i, c in enumerate(costs) if i != leaders[0]]
            price = min([self.monopoly_price(low), *rivals])
        return [price if i in leaders else c for i, c in enumerate(costs)]

    def action_bounds(self, cost: float, costs: list[float]) -> tuple[float, float]:
        return (cost, 2 * self.monopoly_price(max(costs)))

    def metrics(
        self, prices: list[float], quantities: list[float]
    ) -> tuple[float, float, float | None]:
        price = min(prices) if prices else 0.0
        total = sum(quantities)
        shares = (
            calculate_market_shares_bertrand(prices, quantities) if total > 0 else []
        )
        return price, _hhi_from_shares(shares), self.consumer_surplus(price)


# ---------------------------------------------------------------------------
# CES differentiated Bertrand
# ---------------------------------------------------------------------------


class CESBertrandMarket(Market):
    """Differentiated Bertrand competition with nested CES demand.

    Varieties are CES substitutes (elasticity of substitution ``sigma``) with
    qualities theta_i; x_i = p_i / theta_i is the quality-adjusted price and
    P = (sum_j x_j^(1-sigma))^(1/(1-sigma)) the price index. Aggregate demand
    for the group is Q = market_size * P^(-eta) (``market_elasticity`` eta > 1)
    and each variety sells q_i = Q (x_i / P)^(-sigma) / theta_i.

    A firm's perceived elasticity is e_i = sigma - (sigma - eta) s_i with s_i its
    expenditure share, so the Nash price solves p_i = c_i e_i / (e_i - 1); a
    lone firm charges the monopoly price c eta / (eta - 1). Consumer surplus is
    market_size P^(1-eta) / (eta - 1) = total expenditure / (eta - 1).
    """

    model = "bertrand"
    demand_type = "ces"

    def __init__(self, params: dict[str, Any], num_firms: int):
        self._sigma = float(params.get("elasticity", 2.0))
        self._eta = float(params.get("market_elasticity", 2.0))
        self._market_size = float(params.get("market_size", 100.0))
        self._max_markup = float(params.get("max_markup", 10.0))
        qualities = params.get("qualities")
        self._qualities = (
            [float(q) for q in qualities] if qualities else [1.0] * num_firms
        )
        if self._sigma <= 1:
            raise ValueError(
                f"CES elasticity of substitution must be > 1, got {self._sigma}"
            )
        if self._eta <= 1:
            raise ValueError(f"CES market elasticity must be > 1, got {self._eta}")
        if self._market_size <= 0:
            raise ValueError(
                f"CES market size must be positive, got {self._market_size}"
            )
        if len(self._qualities) != num_firms:
            raise ValueError(
                f"CES qualities length ({len(self._qualities)}) must match "
                f"number of firms ({num_firms})"
            )
        if any(q <= 0 for q in self._qualities):
            raise ValueError("CES qualities must be positive")

    def set_qualities(self, qualities: list[float]) -> None:
        self._qualities = list(qualities)

    def _index_and_shares(self, prices: list[float]) -> tuple[float, list[float]]:
        weights = [
            (p / q) ** (1 - self._sigma) for p, q in zip(prices, self._qualities)
        ]
        total = sum(weights)
        return total ** (1 / (1 - self._sigma)), [w / total for w in weights]

    def play(
        self, actions: list[float], costs: list[float], fixed_costs: list[float]
    ) -> RoundResult:
        if any(p <= 0 for p in actions):
            raise ValueError("CES prices must be positive")
        index, shares = self._index_and_shares(actions)
        spend = self._market_size * index ** (1 - self._eta)
        quantities = [spend * s / p for s, p in zip(shares, actions)]
        profits = [
            (p - c) * q - fc
            for p, c, q, fc in zip(actions, costs, quantities, fixed_costs)
        ]
        return BertrandResult(
            total_demand=sum(quantities),
            prices=list(actions),
            quantities=quantities,
            profits=profits,
        )

    def nash(self, costs: list[float]) -> list[float]:
        caps = [c * self._max_markup for c in costs]
        prices = [c * self._sigma / (self._sigma - 1) for c in costs]
        for _ in range(500):
            _, shares = self._index_and_shares(prices)
            updated = []
            for c, s, cap in zip(costs, shares, caps):
                e = self._sigma - (self._sigma - self._eta) * s
                updated.append(min(cap, c * e / (e - 1)))
            done = max(abs(u - p) for u, p in zip(updated, prices)) < 1e-12
            prices = [0.5 * (u + p) for u, p in zip(updated, prices)]
            if done:
                break
        return prices

    def action_bounds(self, cost: float, costs: list[float]) -> tuple[float, float]:
        return (cost, cost * self._max_markup)

    def params(self) -> dict[str, Any]:
        return {
            "demand_type": "ces",
            "elasticity": self._sigma,
            "market_elasticity": self._eta,
            "market_size": self._market_size,
            "max_markup": self._max_markup,
            "qualities": list(self._qualities),
        }

    def scale_demand(self, factor: float) -> None:
        self._market_size *= factor

    def metrics(
        self, prices: list[float], quantities: list[float]
    ) -> tuple[float, float, float | None]:
        total = sum(quantities)
        spend = sum(p * q for p, q in zip(prices, quantities))
        # Quantity-weighted average price; shares by revenue (= expenditure)
        price = spend / total if total else 0.0
        shares = (
            calculate_market_shares_bertrand(prices, quantities) if total > 0 else []
        )
        return price, _hhi_from_shares(shares), spend / (self._eta - 1)


def build_market(model: str, params: dict[str, Any], num_firms: int) -> Market:
    """Create the market for ``model`` from a run's ``params`` dict.

    ``params["demand_type"]`` selects the demand system (default "linear").
    """
    demand_type = params.get("demand_type", "linear")
    if model not in ("cournot", "bertrand"):
        raise ValueError(f"Model must be 'cournot' or 'bertrand', got '{model}'")
    if demand_type == "linear":
        linear = {k: v for k, v in params.items() if k != "demand_type"}
        if model == "cournot":
            return LinearCournotMarket(linear)
        return LinearBertrandMarket(linear)
    if params.get("segments"):
        raise ValueError("Segmented demand is only supported with linear demand")
    if demand_type == "isoelastic":
        if model == "cournot":
            return IsoelasticCournotMarket(params)
        return IsoelasticBertrandMarket(params)
    if demand_type == "ces":
        if model != "bertrand":
            raise ValueError(
                "CES demand models differentiated price competition; use model='bertrand'"
            )
        return CESBertrandMarket(params, num_firms)
    raise ValueError(
        f"Unknown demand_type '{demand_type}'; expected one of {DEMAND_TYPES}"
    )


def market_metrics(
    model: str,
    params: dict[str, Any],
    prices: list[float],
    quantities: list[float],
    round_idx: int | None = None,
) -> tuple[float, float, float | None]:
    """Market price, HHI and consumer surplus for stored round data.

    With market evolution, demand grows by ``growth_rate`` between rounds, so
    ``round_idx`` is used to rebuild the demand curve that round was played on.
    """
    # Metrics only need the demand curve, not per-firm qualities (which can
    # change length as firms enter and exit).
    demand = {k: v for k, v in params.items() if k != "qualities"}
    market = build_market(model, demand, max(1, len(prices)))
    evolution = params.get("market_evolution")
    if evolution is not None and round_idx:
        growth = float(evolution.get("growth_rate", 0.02))
        market.scale_demand((1 + growth) ** round_idx)
    return market.metrics(prices, quantities)
