"""Benchmark learning strategies against equilibrium in every demand system.

Every firm in a symmetric duopoly plays the same learning strategy through the
real multi-round runner. Over the final rounds we measure:

- **Nash gap**: mean |action - Nash action| / Nash action across firms.
- **Collusion index** (Delta): (avg profit - Nash profit) /
  (joint-monopoly profit - Nash profit). 0 means Nash-level profits, 1 means
  the firms jointly act as a monopolist, negative means below Nash.

Usage::

    python -m scripts.learning_benchmark            # full run, prints markdown
    python -m scripts.learning_benchmark --quick    # fewer rounds and seeds
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.markets import build_market
from sim.models.models import Base
from sim.runners.runner import get_run_results, run_game
from sim.strategies.learning import LEARNING_STRATEGY_TYPES

COSTS = [10.0, 10.0]

MARKETS: dict[str, tuple[str, dict[str, Any]]] = {
    "Linear Cournot": ("cournot", {"a": 100.0, "b": 1.0}),
    "Linear Bertrand (winner-take-all)": (
        "bertrand",
        {"alpha": 100.0, "beta": 1.0, "capacity_constraints": False},
    ),
    "Isoelastic Cournot": (
        "cournot",
        {"demand_type": "isoelastic", "A": 100.0, "elasticity": 2.0},
    ),
    "Isoelastic Bertrand": (
        "bertrand",
        {"demand_type": "isoelastic", "A": 100.0, "elasticity": 2.0},
    ),
    "CES Bertrand": (
        "bertrand",
        {
            "demand_type": "ces",
            "elasticity": 3.0,
            "market_elasticity": 2.0,
            "market_size": 5000.0,
        },
    ),
}


@dataclass
class Outcome:
    market: str
    strategy: str
    nash_gap: float
    collusion_index: float
    nash_gap_sd: float
    collusion_index_sd: float


def _benchmarks(model: str, params: dict[str, Any]) -> tuple[list[float], float, float]:
    """Nash actions, and per-firm Nash and joint-monopoly profits."""
    market = build_market(model, dict(params), len(COSTS))
    zeros = [0.0] * len(COSTS)
    nash = market.nash(COSTS)
    nash_profit = statistics.mean(market.play(nash, COSTS, zeros).profits)
    everyone = list(range(len(COSTS)))
    monopoly_action = market.collusive_action(everyone, nash, COSTS)
    monopoly_profit = statistics.mean(
        market.play([monopoly_action] * len(COSTS), COSTS, zeros).profits
    )
    return nash, nash_profit, monopoly_profit


def run_benchmark(rounds: int = 200, seeds: int = 5, window: int = 20) -> list[Outcome]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    outcomes = []
    try:
        for market_name, (model, params) in MARKETS.items():
            nash, nash_profit, monopoly_profit = _benchmarks(model, params)
            for strategy in LEARNING_STRATEGY_TYPES:
                gaps, indices = [], []
                for seed in range(seeds):
                    run_id = run_game(
                        model,
                        rounds,
                        {
                            "params": dict(params),
                            "firms": [{"cost": c} for c in COSTS],
                            "seed": seed,
                            "advanced_strategies": [
                                {"firm_id": i, "strategy_type": strategy}
                                for i in range(len(COSTS))
                            ],
                        },
                        db,
                    )
                    results = get_run_results(run_id, db)["results"]
                    tail = [results[str(r)] for r in range(rounds - window, rounds)]
                    gaps.append(
                        statistics.mean(
                            abs(f["action"] - nash[int(k.split("_")[1])])
                            / nash[int(k.split("_")[1])]
                            for r in tail
                            for k, f in r.items()
                        )
                    )
                    profit = statistics.mean(
                        f["profit"] for r in tail for f in r.values()
                    )
                    indices.append(
                        (profit - nash_profit) / (monopoly_profit - nash_profit)
                    )
                outcomes.append(
                    Outcome(
                        market_name,
                        strategy,
                        statistics.mean(gaps),
                        statistics.mean(indices),
                        statistics.pstdev(gaps),
                        statistics.pstdev(indices),
                    )
                )
    finally:
        db.close()
    return outcomes


def to_markdown(outcomes: list[Outcome], rounds: int, seeds: int, window: int) -> str:
    lines = [
        f"Symmetric duopoly (costs {COSTS}), all firms using the strategy; "
        f"{rounds} rounds, {seeds} seeds, metrics over the last {window} rounds "
        "(mean ± sd across seeds).",
        "",
        "| Market | Strategy | Nash gap | Collusion index Δ |",
        "|--------|----------|---------:|------------------:|",
    ]
    for o in outcomes:
        lines.append(
            f"| {o.market} | `{o.strategy}` | {o.nash_gap:.1%} ± {o.nash_gap_sd:.1%} "
            f"| {o.collusion_index:+.2f} ± {o.collusion_index_sd:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="fewer rounds/seeds")
    args = parser.parse_args()
    rounds, seeds, window = (60, 2, 10) if args.quick else (200, 5, 20)
    print(to_markdown(run_benchmark(rounds, seeds, window), rounds, seeds, window))


if __name__ == "__main__":
    main()
