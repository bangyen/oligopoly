"""Simulation pipeline validation utilities.

This module provides validation functions for the entire simulation pipeline
to ensure economic consistency across all stages.
"""

from typing import Any

from .economic_validation import (
    EconomicValidationError,
    validate_demand_parameters,
)


def validate_simulation_config(config: dict[str, Any]) -> None:
    """Validate simulation configuration for economic consistency.

    Args:
        config: Simulation configuration dictionary

    Raises:
        EconomicValidationError: If configuration is economically invalid
    """
    # Validate demand parameters
    params = config.get("params", {})
    if "a" in params and "b" in params:
        validate_demand_parameters(
            params["a"], params["b"], params.get("alpha", 0.0), params.get("beta", 0.0)
        )

    # Validate firm configurations
    firms = config.get("firms", [])
    if not firms:
        raise EconomicValidationError("Simulation must have at least one firm")

    # Extract costs from firm configurations
    costs = []
    fixed_costs = []
    for i, firm in enumerate(firms):
        if not isinstance(firm, dict):
            raise EconomicValidationError(f"Firm {i} must be a dictionary")

        cost = firm.get("cost")
        if cost is None:
            raise EconomicValidationError(f"Firm {i} must have a 'cost' field")
        if cost <= 0:
            raise EconomicValidationError(f"Firm {i} cost {cost} must be positive")
        costs.append(cost)

        # Handle fixed costs
        fc = firm.get("fixed_cost", 0.0)
        if fc < 0:
            raise EconomicValidationError(
                f"Firm {i} fixed cost {fc} must be non-negative"
            )
        fixed_costs.append(fc)
