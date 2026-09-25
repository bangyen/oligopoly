"""Long-horizon Q-learning (Calvano et al., 2020) in every demand system.

For each market, trains a symmetric duopoly of tabular Q-learners with
one-period memory until their strategies converge, then forces a one-period
deviation and checks whether it is punished and unprofitable.

Usage::

    python -m scripts.long_horizon_learning            # 10 seeds, ~4 minutes
    python -m scripts.long_horizon_learning --quick    # 2 seeds, faster decay
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

from scripts.learning_benchmark import COSTS, MARKETS
from sim.experiments.algorithmic_collusion import (
    ExperimentConfig,
    ImpulseResponse,
    impulse_response,
    run_experiment,
)
from sim.markets import build_market


@dataclass
class MarketSummary:
    market: str
    sessions: int
    converged: int
    mean_periods: float
    collusion_index: float
    collusion_index_sd: float
    punished: int
    unprofitable: int
    returned: int
    memoryless_index: float
    memoryless_index_sd: float
    example: ImpulseResponse


def run(seeds: int, config: ExperimentConfig) -> list[MarketSummary]:
    summaries = []
    for name, (model, params) in MARKETS.items():
        results, responses, memoryless = [], [], []
        for seed in range(seeds):
            market = build_market(model, dict(params), len(COSTS))
            session = ExperimentConfig(**{**config.__dict__, "seed": seed})
            result = run_experiment(market, COSTS, session)
            results.append(result)
            responses.append(
                impulse_response(market, COSTS, result, delta=config.delta)
            )
            # Control: the same learners without memory cannot punish
            control = ExperimentConfig(**{**session.__dict__, "memory": 0})
            memoryless.append(run_experiment(market, COSTS, control).collusion_index)
        indices = [r.collusion_index for r in results]
        summaries.append(
            MarketSummary(
                market=name,
                sessions=seeds,
                converged=sum(r.converged for r in results),
                mean_periods=statistics.mean(r.periods for r in results),
                collusion_index=statistics.mean(indices),
                collusion_index_sd=statistics.pstdev(indices),
                punished=sum(ir.punished for ir in responses),
                unprofitable=sum(not ir.deviation_profitable for ir in responses),
                returned=sum(ir.returns_to_path for ir in responses),
                memoryless_index=statistics.mean(memoryless),
                memoryless_index_sd=statistics.pstdev(memoryless),
                example=responses[0],
            )
        )
    return summaries


def to_markdown(summaries: list[MarketSummary], config: ExperimentConfig) -> str:
    lines = [
        f"Symmetric duopoly (costs {COSTS}), {config.grid_size}-point action grid, "
        f"alpha={config.alpha}, delta={config.delta}, beta={config.beta:g}, "
        f"convergence after {config.convergence_window:,} unchanged periods.",
        "",
        "| Market | Converged | Periods (mean) | Δ (memory) | Δ (memoryless control) "
        "| Deviation punished | Deviation unprofitable | Back on path |",
        "|--------|----------:|---------------:|-----------:|-----------------------:"
        "|-------------------:|-----------------------:|-------------:|",
    ]
    for s in summaries:
        n = s.sessions
        lines.append(
            f"| {s.market} | {s.converged}/{n} | {s.mean_periods:,.0f} "
            f"| {s.collusion_index:+.2f} ± {s.collusion_index_sd:.2f} "
            f"| {s.memoryless_index:+.2f} ± {s.memoryless_index_sd:.2f} "
            f"| {s.punished}/{n} | {s.unprofitable}/{n} | {s.returned}/{n} |"
        )
    for s in summaries:
        ir = s.example
        lines += [
            "",
            f"**{s.market}**, seed 0: firm 0 deviates in period 1",
            "",
            "| Period | Firm 0 action | Firm 1 action | Firm 0 profit |",
            "|-------:|--------------:|--------------:|--------------:|",
        ]
        for t, (acts, profs) in enumerate(zip(ir.actions[:11], ir.profits[:11])):
            lines.append(f"| {t} | {acts[0]:.2f} | {acts[1]:.2f} | {profs[0]:.1f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="2 seeds, faster decay")
    parser.add_argument("--seeds", type=int, default=None)
    args = parser.parse_args()
    if args.quick:
        config = ExperimentConfig(beta=1e-4, convergence_window=20_000)
        seeds = args.seeds or 2
    else:
        config = ExperimentConfig()
        seeds = args.seeds or 10
    print(to_markdown(run(seeds, config), config))


if __name__ == "__main__":
    main()
