"""Oligopoly simulation package for market competition modeling.

This package provides tools for simulating oligopoly markets with various
competition models (Cournot, Bertrand), strategies, and advanced features
like collusion dynamics and regulatory interventions.
"""

from .bertrand import BertrandResult, bertrand_simulation
from .collusion import CollusionEventType, CollusionManager, RegulatorState
from .collusion_runner import create_collusion_simulation_config, run_collusion_game
from .collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
)
from .cournot import CournotResult, cournot_simulation

__all__ = [
    "bertrand_simulation",
    "BertrandResult",
    "cournot_simulation",
    "CournotResult",
    "CollusionManager",
    "RegulatorState",
    "CollusionEventType",
    "CartelStrategy",
    "CollusiveStrategy",
    "OpportunisticStrategy",
    "run_collusion_game",
    "create_collusion_simulation_config",
]
