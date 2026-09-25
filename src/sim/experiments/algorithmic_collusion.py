"""Long-horizon Q-learning experiments on algorithmic collusion.

Implements the design of Calvano, Calzolari, Denicolò and Pastorello (2020),
"Artificial Intelligence, Algorithmic Pricing, and Collusion" (AER), on any
:class:`sim.markets.Market`:

- Each firm chooses from ``grid_size`` actions spanning the one-shot Nash and
  joint-monopoly actions, extended by ``xi`` of that range on both sides.
- The state is every firm's previous action (one-period memory), so a firm
  *can* condition on — and punish — a rival's deviation.
- Tabular Q-learning with learning rate ``alpha``, discount ``delta`` and
  exploration ε_t = exp(-beta t). Q starts at the discounted payoff against a
  uniformly randomising rival.
- Learning stops when every firm's greedy strategy has been unchanged for
  ``convergence_window`` periods (or at ``max_periods``).

After convergence :func:`impulse_response` forces one firm to deviate for one
period and traces how everyone responds. A punishment (the rival's price
falling below its pre-deviation level) followed by a return towards the
pre-deviation prices is the signature of learned, reward-punishment collusion;
static or noisy pricing shows no such response.

Runs never touch the database: one run is millions of periods, so the loop
uses a precomputed profit table instead of the persisted runner.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

from sim.markets import Market


@dataclass
class ExperimentConfig:
    """Hyper-parameters; defaults follow Calvano et al. (2020)."""

    grid_size: int = 15
    xi: float = 0.1
    alpha: float = 0.15
    delta: float = 0.95
    beta: float = 4e-6
    convergence_window: int = 100_000
    max_periods: int = 5_000_000
    seed: int | None = None
    # 1: condition on last period's actions (can punish); 0: memoryless control
    memory: int = 1

    def __post_init__(self) -> None:
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= self.delta < 1:
            raise ValueError("delta must be in [0, 1)")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.memory not in (0, 1):
            raise ValueError("memory must be 0 or 1")


@dataclass
class ExperimentResult:
    """Outcome of one learning run."""

    converged: bool
    periods: int
    grid: list[float]
    nash_action: float
    monopoly_action: float
    nash_profit: float
    monopoly_profit: float
    # Greedy play from the final state until it cycles
    limit_path: list[tuple[int, ...]]
    limit_actions: list[list[float]]
    average_profit: float
    collusion_index: float
    q_tables: list[list[list[float]]] = field(repr=False)
    memory: int = 1


def _action_grid(nash: float, monopoly: float, size: int, xi: float) -> list[float]:
    low, high = min(nash, monopoly), max(nash, monopoly)
    span = high - low
    low, high = low - xi * span, high + xi * span
    return [low + (high - low) * k / (size - 1) for k in range(size)]


class _Game:
    """Precomputed profit table for a symmetric market on an action grid."""

    def __init__(self, market: Market, costs: list[float], config: ExperimentConfig):
        n = len(costs)
        firms = list(range(n))
        nash = market.nash(costs)
        monopoly = market.collusive_action(firms, nash, costs)
        self.nash_action = sum(nash) / n
        self.monopoly_action = monopoly
        zeros = [0.0] * n
        self.nash_profit = sum(market.play(nash, costs, zeros).profits) / n
        self.monopoly_profit = (
            sum(market.play([monopoly] * n, costs, zeros).profits) / n
        )
        lo_bounds = [market.action_bounds(c, costs)[0] for c in costs]
        grid = _action_grid(self.nash_action, monopoly, config.grid_size, config.xi)
        # Keep every grid point feasible for every firm (e.g. price >= cost)
        self.grid = [max(g, max(lo_bounds)) for g in grid]
        self.n = n
        self.m = config.grid_size
        self.states = list(itertools.product(range(self.m), repeat=n))
        self.state_index = {s: i for i, s in enumerate(self.states)}
        # profits[state_index][firm] for the action profile encoded by the state
        self.profits = [
            market.play([self.grid[a] for a in profile], costs, zeros).profits
            for profile in self.states
        ]

    def expected_uniform_profit(self, firm: int, action: int) -> float:
        """Profit of ``action`` against uniformly randomising rivals."""
        total = 0.0
        count = 0
        for profile, profits in zip(self.states, self.profits):
            if profile[firm] == action:
                total += profits[firm]
                count += 1
        return total / count


def _argmax(values: list[float]) -> int:
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def run_experiment(
    market: Market, costs: list[float], config: ExperimentConfig | None = None
) -> ExperimentResult:
    """Train one Q-learner per firm until their strategies stop changing."""
    config = config or ExperimentConfig()
    if len(set(costs)) != 1:
        raise ValueError("Experiments assume symmetric firms (equal costs)")
    game = _Game(market, costs, config)
    rng = random.Random(config.seed)
    n, m = game.n, game.m
    num_states = len(game.states)
    alpha, delta = config.alpha, config.delta

    # Q[firm][state][action], initialised to the discounted uniform payoff
    q = [
        [
            [game.expected_uniform_profit(i, a) / (1 - delta) for a in range(m)]
            for _ in range(num_states)
        ]
        for i in range(n)
    ]
    greedy = [[_argmax(q[i][s]) for s in range(num_states)] for i in range(n)]
    # Memoryless learners observe a single constant state
    remember = config.memory == 1

    state = rng.randrange(num_states) if remember else 0
    decay = math.exp(-config.beta)
    epsilon = 1.0
    stable = 0
    period = 0
    converged = False
    last_profile = state
    while period < config.max_periods:
        period += 1
        epsilon *= decay
        actions = [
            rng.randrange(m) if rng.random() < epsilon else greedy[i][state]
            for i in range(n)
        ]
        profile = game.state_index[tuple(actions)]
        profits = game.profits[profile]
        next_state = profile if remember else 0
        changed = False
        for i in range(n):
            row = q[i][state]
            a = actions[i]
            future = q[i][next_state][greedy[i][next_state]]
            row[a] += alpha * (profits[i] + delta * future - row[a])
            best = greedy[i][state]
            if a == best:
                if row[a] < max(row):
                    greedy[i][state] = _argmax(row)
                    changed = True
            elif row[a] > row[best]:
                greedy[i][state] = a
                changed = True
        stable = 0 if changed else stable + 1
        state = next_state
        last_profile = profile
        if stable >= config.convergence_window:
            converged = True
            break

    path = _limit_path(greedy, game, last_profile if remember else 0, remember)
    limit_profits = [sum(game.profits[s]) / n for s in path]
    average_profit = sum(limit_profits) / len(limit_profits)
    denominator = game.monopoly_profit - game.nash_profit
    return ExperimentResult(
        converged=converged,
        periods=period,
        grid=game.grid,
        nash_action=game.nash_action,
        monopoly_action=game.monopoly_action,
        nash_profit=game.nash_profit,
        monopoly_profit=game.monopoly_profit,
        limit_path=[game.states[s] for s in path],
        limit_actions=[[game.grid[a] for a in game.states[s]] for s in path],
        average_profit=average_profit,
        collusion_index=(average_profit - game.nash_profit) / denominator,
        q_tables=q,
        memory=config.memory,
    )


def _limit_path(
    greedy: list[list[int]], game: _Game, start: int, remember: bool = True
) -> list[int]:
    """Greedy play from ``start`` until a profile repeats; return the cycle.

    Returns action-profile indices.
    """
    seen: dict[int, int] = {}
    order: list[int] = []
    profile = game.state_index[tuple(g[start if remember else 0] for g in greedy)]
    while profile not in seen:
        seen[profile] = len(order)
        order.append(profile)
        profile = game.state_index[tuple(g[profile if remember else 0] for g in greedy)]
    return order[seen[profile] :]


@dataclass
class ImpulseResponse:
    """Actions and profits after a forced one-period deviation.

    Index 0 is the pre-deviation period, index 1 the deviation.
    """

    actions: list[list[float]]
    profits: list[list[float]]
    deviator: int
    pre_deviation_profit: float
    # Deviator's discounted profit over the response vs. staying on the path
    deviation_value: float
    on_path_value: float
    limit_cycle: list[list[float]] = field(default_factory=list)

    @property
    def punished(self) -> bool:
        """Deviator earns less than before in the periods after its deviation."""
        after = [p[self.deviator] for p in self.profits[2:7]]
        return sum(after) / len(after) < self.pre_deviation_profit

    @property
    def deviation_profitable(self) -> bool:
        return self.deviation_value > self.on_path_value

    @property
    def returns_to_path(self) -> bool:
        """Play is back on the pre-deviation limit cycle by the end."""
        return self.actions[-1] in (self.limit_cycle or [self.actions[0]])


def impulse_response(
    market: Market,
    costs: list[float],
    result: ExperimentResult,
    deviator: int = 0,
    periods: int = 25,
    delta: float = 0.95,
) -> ImpulseResponse:
    """Force ``deviator`` to deviate for one period, then let greedy play resume.

    The deviation is the static best response (on the grid) to the rivals'
    actions in the first state of the limit path. Values are discounted sums
    over the response horizon; staying on the path means repeating the limit
    cycle for the same number of periods.
    """
    grid = result.grid
    m, n = len(grid), len(costs)
    states = list(itertools.product(range(m), repeat=n))
    state_index = {s: i for i, s in enumerate(states)}
    greedy = [
        [_argmax(table[s]) for s in range(len(states))] for table in result.q_tables
    ]
    zeros = [0.0] * n

    def outcome(profile: list[int] | tuple[int, ...]) -> list[float]:
        return list(market.play([grid[a] for a in profile], costs, zeros).profits)

    remember = result.memory == 1

    def observe(profile: list[int] | tuple[int, ...]) -> int:
        return state_index[tuple(profile)] if remember else 0

    start = result.limit_path[0]
    rivals_next = [g[observe(start)] for g in greedy]

    def deviation_profit(action: int) -> float:
        profile = list(rivals_next)
        profile[deviator] = action
        return outcome(profile)[deviator]

    profiles: list[list[int]] = [list(start)]
    actions = list(rivals_next)
    actions[deviator] = max(range(m), key=deviation_profit)
    profiles.append(actions)
    for _ in range(periods - 1):
        state = observe(actions)
        actions = [g[state] for g in greedy]
        profiles.append(actions)

    profits = [outcome(p) for p in profiles]
    cycle = [outcome(s) for s in result.limit_path]
    on_path = [cycle[(t + 1) % len(cycle)][deviator] for t in range(periods)]
    deviation = [p[deviator] for p in profits[1:]]
    return ImpulseResponse(
        actions=[[grid[a] for a in p] for p in profiles],
        profits=profits,
        deviator=deviator,
        pre_deviation_profit=profits[0][deviator],
        deviation_value=sum(v * delta**t for t, v in enumerate(deviation)),
        on_path_value=sum(v * delta**t for t, v in enumerate(on_path)),
        limit_cycle=result.limit_actions,
    )
