"""Enhanced economic validation for oligopoly simulation.

This module provides comprehensive validation of economic parameters and results
to ensure realistic market behavior and prevent unrealistic outcomes.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EconomicValidationResult:
    """Result of economic validation with warnings and errors."""

    is_valid: bool
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, float]


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
    costs: List[float], fixed_costs: Optional[List[float]] = None
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


def validate_simulation_result(
    model: str,
    prices: List[float],
    quantities: List[float],
    profits: List[float],
    costs: List[float],
    demand_params: Dict[str, float],
) -> EconomicValidationResult:
    """Validate simulation results for economic realism.

    Args:
        model: 'cournot' or 'bertrand'
        prices: List of prices (or single market price for Cournot)
        quantities: List of quantities
        profits: List of profits
        costs: List of marginal costs
        demand_params: Demand curve parameters

    Returns:
        EconomicValidationResult with validation status and metrics
    """
    warnings_list: List[str] = []
    errors_list: List[str] = []

    # Basic validation
    if not quantities or not profits or not costs:
        errors_list.append("Empty simulation results")
        return EconomicValidationResult(False, warnings_list, errors_list, {})

    if len(quantities) != len(profits) or len(quantities) != len(costs):
        errors_list.append("Mismatched array lengths in simulation results")
        return EconomicValidationResult(False, warnings_list, errors_list, {})

    # Calculate key metrics
    total_quantity = sum(quantities)
    total_profit = sum(profits)
    active_firms = sum(1 for q in quantities if q > 0)

    # Market price (handle both Cournot and Bertrand)
    if model == "cournot":
        market_price = prices[0] if isinstance(prices, list) else prices
    else:
        market_price = min(prices) if prices else 0.0

    # Calculate market shares
    market_shares = [
        q / total_quantity if total_quantity > 0 else 0 for q in quantities
    ]
    hhi = sum(s**2 for s in market_shares)

    # Economic validation checks

    # 1. Check for negative profits (warning, not error)
    negative_profit_firms = sum(1 for p in profits if p < 0)
    if negative_profit_firms > 0:
        warnings_list.append(f"{negative_profit_firms} firms have negative profits")

    # 2. Check for excessive losses
    for i, profit in enumerate(profits):
        if profit < -costs[i] * 0.5:  # Losing more than 50% of marginal cost
            warnings_list.append(f"Firm {i} has excessive losses: {profit:.2f}")

    # 3. Check market concentration with more nuanced thresholds
    if hhi > 0.8:
        warnings_list.append(f"Market highly concentrated: HHI = {hhi:.3f}")
    elif hhi > 0.6:
        warnings_list.append(f"Market moderately concentrated: HHI = {hhi:.3f}")
    elif hhi < 0.1 and len(quantities) > 2:
        warnings_list.append(f"Market unusually competitive: HHI = {hhi:.3f}")
    elif hhi < 0.2 and len(quantities) > 3:
        warnings_list.append(f"Market very competitive: HHI = {hhi:.3f}")

    # Additional check for Bertrand competition - should not be perfect monopoly
    if model == "bertrand" and hhi > 0.95:
        warnings_list.append(
            f"Bertrand competition showing unrealistic monopoly: HHI = {hhi:.3f}"
        )

    # 4. Check price-cost margins with more realistic thresholds
    if model == "cournot":
        for i, (price, cost, qty) in enumerate(
            zip([market_price] * len(costs), costs, quantities)
        ):
            if qty > 0:
                margin = (price - cost) / price if price > 0 else 0
                if margin < 0:
                    errors_list.append(f"Firm {i} selling below marginal cost")
                elif margin > 0.6:  # 60% margin seems excessive for Cournot
                    warnings_list.append(f"Firm {i} has very high margin: {margin:.1%}")
                elif margin > 0.4:  # 40% margin is high but acceptable
                    warnings_list.append(f"Firm {i} has high margin: {margin:.1%}")
    else:  # Bertrand
        for i, (price, cost, qty) in enumerate(zip(prices, costs, quantities)):
            if qty > 0:
                margin = (price - cost) / price if price > 0 else 0
                if margin < 0:
                    errors_list.append(f"Firm {i} selling below marginal cost")
                elif margin > 0.3:  # 30% margin for Bertrand (more competitive)
                    warnings_list.append(f"Firm {i} has high margin: {margin:.1%}")
                elif margin > 0.2:  # 20% margin is high but acceptable for Bertrand
                    warnings_list.append(f"Firm {i} has moderate margin: {margin:.1%}")

    # 5. Check for market failure
    if total_quantity <= 0:
        errors_list.append("No production in market")
    if market_price <= 0:
        errors_list.append("Non-positive market price")

    # 6. Check consumer surplus calculation
    if model == "cournot" and "a" in demand_params:
        cs = 0.5 * (demand_params["a"] - market_price) * total_quantity
        if cs <= 0:
            warnings_list.append(f"Non-positive consumer surplus: {cs:.2f}")
    elif model == "bertrand" and "alpha" in demand_params:
        cs = 0.5 * (demand_params["alpha"] - market_price) * total_quantity
        if cs <= 0:
            warnings_list.append(f"Non-positive consumer surplus: {cs:.2f}")

    # 7. Check for unrealistic outcomes
    if active_firms == 1 and len(quantities) > 1:
        warnings_list.append("Only one firm active despite multiple firms")
    if total_profit < 0 and active_firms > 0:
        warnings_list.append("Negative total industry profit")

    # Calculate efficiency metrics
    metrics: Dict[str, float] = {
        "total_quantity": float(total_quantity),
        "total_profit": float(total_profit),
        "market_price": float(market_price),
        "hhi": float(hhi),
        "active_firms": float(active_firms),
        "avg_margin": float(
            np.mean(
                [
                    (p - c) / p
                    for p, c in zip(
                        prices if model == "bertrand" else [market_price] * len(costs),
                        costs,
                    )
                    if p > 0
                ]
            )
            if costs
            and any(
                p > 0
                for p in (
                    prices if model == "bertrand" else [market_price] * len(costs)
                )
            )
            else 0
        ),
    }

    is_valid = len(errors_list) == 0

    return EconomicValidationResult(is_valid, warnings_list, errors_list, metrics)


def enforce_economic_constraints(
    quantities: List[float],
    costs: List[float],
    market_price: float,
    min_quantity: float = 0.0,
) -> List[float]:
    """Enforce economic constraints on quantities.

    Args:
        quantities: List of quantities to adjust
        costs: List of marginal costs
        market_price: Market price
        min_quantity: Minimum allowed quantity

    Returns:
        Adjusted quantities that satisfy economic constraints
    """
    adjusted_quantities = []

    for i, (qty, cost) in enumerate(zip(quantities, costs)):
        # Firms won't produce if price is below marginal cost
        if market_price < cost:
            adjusted_quantities.append(0.0)
        else:
            # Ensure non-negative quantities
            adjusted_quantities.append(max(min_quantity, qty))

    return adjusted_quantities


def validate_market_evolution_config(config: Dict[str, Any]) -> None:
    """Validate market evolution configuration parameters.

    Args:
        config: Market evolution configuration dictionary

    Raises:
        EconomicValidationError: If configuration is invalid
    """
    errors = []

    # Check growth rate
    growth_rate = config.get("growth_rate", 0.02)
    if not 0 <= growth_rate <= 0.1:  # 0-10% growth per period
        errors.append(f"Growth rate {growth_rate} should be between 0 and 0.1")

    # Check entry cost
    entry_cost = config.get("entry_cost", 100.0)
    if entry_cost <= 0:
        errors.append(f"Entry cost {entry_cost} must be positive")
    if entry_cost > 10000:
        errors.append(f"Entry cost {entry_cost} seems unrealistically high")

    # Check exit threshold
    exit_threshold = config.get("exit_threshold", -50.0)
    if exit_threshold > 0:
        errors.append(f"Exit threshold {exit_threshold} should be negative")
    if exit_threshold < -1000:
        errors.append(f"Exit threshold {exit_threshold} seems unrealistically low")

    # Check innovation rate
    innovation_rate = config.get("innovation_rate", 0.1)
    if not 0 <= innovation_rate <= 1:
        errors.append(f"Innovation rate {innovation_rate} should be between 0 and 1")

    if errors:
        raise EconomicValidationError("; ".join(errors))


def log_economic_warnings(
    result: EconomicValidationResult, logger: Optional[Any] = None
) -> None:
    """Log economic validation warnings and errors.

    Args:
        result: EconomicValidationResult to log
        logger: Optional logger instance
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    for warning in result.warnings:
        logger.warning(f"Economic validation warning: {warning}")

    for error in result.errors:
        logger.error(f"Economic validation error: {error}")

    if not result.is_valid:
        logger.error("Simulation failed economic validation")
    elif result.warnings:
        logger.info("Simulation passed validation with warnings")
    else:
        logger.debug("Simulation passed economic validation")
