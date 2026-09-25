"""Enhanced economic validation for oligopoly simulation.

This module provides comprehensive validation of economic parameters and results
to ensure realistic market behavior and prevent unrealistic outcomes.
"""


class EconomicValidationError(Exception):
    """Exception raised when economic validation fails."""

    pass


def validate_demand_parameters(a: float, b: float, alpha: float, beta: float) -> None:
    """Validate demand curve parameters for economic realism.

    Args:
        a: Cournot demand intercept (max price)
        b: Cournot demand slope (price sensitivity)
        alpha: Bertrand demand intercept
        beta: Bertrand demand slope

    Raises:
        EconomicValidationError: If parameters are economically invalid
    """
    errors = []

    # Validate Cournot parameters (only if they're being used)
    if a > 0 or b > 0:  # Only validate if at least one parameter is set
        if a <= 0:
            errors.append(f"Cournot demand intercept 'a' must be positive, got {a}")
        if b <= 0:
            errors.append(f"Cournot demand slope 'b' must be positive, got {b}")
        if b > 0 and a > 0 and a / b < 10:  # Market size should be reasonable
            errors.append(f"Market size (a/b) too small: {a / b:.2f}, should be >= 10")

    # Validate Bertrand parameters (only if they're being used)
    if alpha > 0 or beta > 0:  # Only validate if at least one parameter is set
        if alpha <= 0:
            errors.append(
                f"Bertrand demand intercept 'alpha' must be positive, got {alpha}"
            )
        if beta <= 0:
            errors.append(f"Bertrand demand slope 'beta' must be positive, got {beta}")
        if (
            beta > 0 and alpha > 0 and alpha / beta < 10
        ):  # Market size should be reasonable
            errors.append(
                f"Market size (alpha/beta) too small: {alpha / beta:.2f}, should be >= 10"
            )

    if errors:
        raise EconomicValidationError("; ".join(errors))


def validate_cost_structure(
    costs: list[float], fixed_costs: list[float] | None = None
) -> None:
    """Validate firm cost structures for economic realism.

    Args:
        costs: List of marginal costs
        fixed_costs: Optional list of fixed costs

    Raises:
        EconomicValidationError: If cost structure is invalid
    """
    if not costs:
        raise EconomicValidationError("At least one firm must have a cost")

    errors = []

    # Check marginal costs
    for i, cost in enumerate(costs):
        if cost <= 0:
            errors.append(f"Firm {i} marginal cost must be positive, got {cost}")
        if cost > 1000:  # Unrealistically high cost
            errors.append(f"Firm {i} marginal cost {cost} seems unrealistically high")

    # Check cost dispersion
    if len(costs) > 1:
        cost_ratio = max(costs) / min(costs)
        if cost_ratio > 10:  # Too much cost dispersion
            errors.append(
                f"Cost dispersion too high: ratio {cost_ratio:.2f}, should be <= 10"
            )

    # Check fixed costs
    if fixed_costs:
        if len(fixed_costs) != len(costs):
            errors.append(
                f"Fixed costs length {len(fixed_costs)} must match marginal costs length {len(costs)}"
            )
        else:
            for i, fc in enumerate(fixed_costs):
                if fc < 0:
                    errors.append(f"Firm {i} fixed cost must be non-negative, got {fc}")
                if fc > 10000:  # Unrealistically high fixed cost
                    errors.append(
                        f"Firm {i} fixed cost {fc} seems unrealistically high"
                    )

    if errors:
        raise EconomicValidationError("; ".join(errors))
