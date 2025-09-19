"""Economic validation package for oligopoly simulation.

This package provides comprehensive validation utilities to ensure
economic consistency throughout the simulation pipeline.
"""

from .economic_validation import (
    EconomicValidationError,
    enforce_economic_constraints,
    validate_costs,
    validate_demand_parameters,
    validate_firm_viability,
    validate_market_clearing,
    validate_price_quantity_consistency,
    validate_profit_consistency,
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
    "enforce_economic_constraints",
    "validate_costs",
    "validate_demand_parameters",
    "validate_firm_viability",
    "validate_market_clearing",
    "validate_price_quantity_consistency",
    "validate_profit_consistency",
    "validate_simulation_result",
    "check_economic_plausibility",
    "sanitize_simulation_results",
    "validate_round_results",
    "validate_run_results",
    "validate_simulation_config",
]
