"""Economic validation package for oligopoly simulation.

This package provides comprehensive validation utilities to ensure
economic consistency throughout the simulation pipeline.
"""

from .economic_validation import (
    EconomicValidationError,
    EconomicValidationResult,
    enforce_economic_constraints,
    log_economic_warnings,
    validate_cost_structure,
    validate_demand_parameters,
    validate_market_evolution_config,
    validate_simulation_result,
)
from .simulation_validation import (
    check_economic_plausibility,
    sanitize_simulation_results,
    validate_round_results,
    validate_run_results,
    validate_simulation_config,
)

__all__ = [
    "EconomicValidationError",
    "EconomicValidationResult",
    "enforce_economic_constraints",
    "validate_cost_structure",
    "validate_demand_parameters",
    "validate_market_evolution_config",
    "validate_simulation_result",
    "log_economic_warnings",
    "check_economic_plausibility",
    "sanitize_simulation_results",
    "validate_round_results",
    "validate_run_results",
    "validate_simulation_config",
]
