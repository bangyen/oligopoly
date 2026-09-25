"""Oligopoly simulation: markets, learning firms and algorithmic collusion.

The core API:

- :func:`sim.markets.build_market` — a demand system (linear, isoelastic or CES)
  with its round engine, Nash equilibrium, best responses and metrics.
- :func:`sim.runners.runner.run_game` — multi-round simulations with learning,
  collusive or adaptive-Nash firms, policy shocks and market evolution.
- :mod:`sim.experiments.algorithmic_collusion` — long-horizon Q-learning
  experiments with deviation tests.
"""

from .experiments.algorithmic_collusion import (
    ExperimentConfig,
    impulse_response,
    run_experiment,
)
from .markets import Market, build_market
from .runners.runner import get_run_results, run_game

__all__ = [
    "ExperimentConfig",
    "Market",
    "build_market",
    "get_run_results",
    "impulse_response",
    "run_experiment",
    "run_game",
]
