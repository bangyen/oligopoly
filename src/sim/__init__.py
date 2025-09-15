"""Oligopoly simulation package for market competition modeling.

This package provides tools for simulating oligopoly markets with various
competition models (Cournot, Bertrand), strategies, and advanced features
like collusion dynamics and regulatory interventions.
"""

from .collusion import CollusionEventType, CollusionManager, RegulatorState
from .games.bertrand import BertrandResult, bertrand_simulation
from .games.cournot import CournotResult, cournot_simulation
from .runners.collusion_runner import (
    create_collusion_simulation_config,
    run_collusion_game,
)
from .strategies.collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
)

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
