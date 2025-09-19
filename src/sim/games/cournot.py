"""Cournot oligopoly simulation implementation.

This module implements the Cournot model of oligopoly competition where firms
simultaneously choose quantities to maximize profits. The simulation computes
market price based on total quantity supplied and calculates individual firm profits.
Supports both single-segment and multi-segment demand models.
"""

from dataclasses import dataclass
from typing import List, Optional

from ..models.models import SegmentedDemand
from ..validation.economic_validation import (
    EconomicValidationError,
    enforce_economic_constraints,
    validate_demand_parameters,
    validate_simulation_result,
)


@dataclass
class CournotResult:
    """Results from a Cournot simulation run.

    Contains the market price, individual firm quantities, and profits
    from a single round of Cournot competition.
    """

    price: float
    quantities: List[float]
    profits: List[float]

    def __repr__(self) -> str:
        """String representation for debugging and output."""
        return f"CournotResult(price={self.price}, quantities={self.quantities}, profits={self.profits})"


def validate_quantities(quantities: List[float]) -> None:
    """Validate that all quantities are non-negative.

    Args:
        quantities: List of firm quantities to validate

    Raises:
        ValueError: If any quantity is negative
    """
    for i, q in enumerate(quantities):
        if q < 0:
            raise ValueError(f"Quantity q_{i} = {q:.1f} must be non-negative")


def cournot_simulation(
    a: float,
    b: float,
    costs: List[float],
    quantities: List[float],
    fixed_costs: Optional[List[float]] = None,
    capacity_limits: Optional[List[float]] = None,
) -> CournotResult:
    """Run a one-round Cournot oligopoly simulation.

    Computes market price based on inverse demand P = max(0, a - b * sum(q_i))
    and calculates individual firm profits π_i = (P - c_i) * q_i - FC_i.

    Firms with costs above the market price will exit (quantity set to 0).
    Production is constrained by capacity limits if provided.

    Args:
        a: Maximum price parameter for demand curve
        b: Price sensitivity parameter for demand curve
        costs: List of marginal costs for each firm
        quantities: List of quantities chosen by each firm
        fixed_costs: Optional list of fixed costs for each firm
        capacity_limits: Optional list of capacity limits for each firm

    Returns:
        CournotResult containing price, quantities, and profits

    Raises:
        ValueError: If quantities are negative or lists have mismatched lengths
        EconomicValidationError: If economic constraints are violated
    """
    # Validate inputs
    validate_quantities(quantities)
    try:
        validate_demand_parameters(a, b, 0.0, 0.0)  # Only validate Cournot params
    except EconomicValidationError as e:
        raise ValueError(str(e))

    if len(costs) != len(quantities):
        raise ValueError(
            f"Costs list length ({len(costs)}) must match quantities list length ({len(quantities)})"
        )

    # Apply capacity constraints if provided
    if capacity_limits:
        if len(capacity_limits) != len(quantities):
            raise ValueError(
                f"Capacity limits length ({len(capacity_limits)}) must match quantities length ({len(quantities)})"
            )
        quantities = [
            min(qty, cap) if cap is not None else qty
            for qty, cap in zip(quantities, capacity_limits)
        ]

    # Calculate market price: P = max(0, a - b * sum(q_i))
    total_quantity = sum(quantities)
    price = max(0.0, a - b * total_quantity)

    # Ensure firms don't produce at losses - exit unprofitable firms
    from ..strategies.nash_strategies import validate_profitable_production

    adjusted_quantities = validate_profitable_production(quantities, costs, price)

    # Recalculate price with adjusted quantities
    adjusted_total_quantity = sum(adjusted_quantities)
    adjusted_price = max(0.0, a - b * adjusted_total_quantity)

    # Calculate profits: π_i = (P - c_i) * q_i - FC_i
    if fixed_costs:
        if len(fixed_costs) != len(adjusted_quantities):
            raise ValueError(
                f"Fixed costs length ({len(fixed_costs)}) must match quantities length ({len(adjusted_quantities)})"
            )
        profits = [
            (adjusted_price - cost) * q - fc
            for cost, q, fc in zip(costs, adjusted_quantities, fixed_costs)
        ]
    else:
        profits = [
            (adjusted_price - cost) * q for cost, q in zip(costs, adjusted_quantities)
        ]

    # Create result
    result = CournotResult(
        price=adjusted_price, quantities=adjusted_quantities, profits=profits
    )

    # Validate economic consistency
    try:
        validate_simulation_result(
            prices=[adjusted_price]
            * len(adjusted_quantities),  # All firms face same price in Cournot
            quantities=adjusted_quantities,
            costs=costs,
            profits=profits,
            market_price=adjusted_price,
            demand_params=(a, b),
            fixed_costs=fixed_costs,
            strict=False,  # Don't raise exceptions, just warn
        )
    except EconomicValidationError:
        # If validation fails, enforce constraints
        enforced_prices, enforced_quantities, enforced_profits = (
            enforce_economic_constraints(
                prices=[adjusted_price] * len(adjusted_quantities),
                quantities=adjusted_quantities,
                costs=costs,
                fixed_costs=fixed_costs,
            )
        )

        # Recalculate market price with enforced quantities
        enforced_total_quantity = sum(enforced_quantities)
        enforced_price = max(0.0, a - b * enforced_total_quantity)

        result = CournotResult(
            price=enforced_price,
            quantities=enforced_quantities,
            profits=enforced_profits,
        )

    return result


def cournot_segmented_simulation(
    segmented_demand: SegmentedDemand,
    costs: List[float],
    quantities: List[float],
    fixed_costs: Optional[List[float]] = None,
) -> CournotResult:
    """Run a one-round Cournot oligopoly simulation with segmented demand.

    Computes market price based on segmented inverse demand where each segment
    contributes to total demand based on its weight. Market price is determined
    by the aggregate demand curve P = max(0, a_eff - b_eff * sum(q_i)) where
    a_eff and b_eff are weighted averages of segment parameters.

    Args:
        segmented_demand: SegmentedDemand object with segment configurations
        costs: List of marginal costs for each firm
        quantities: List of quantities chosen by each firm

    Returns:
        CournotResult containing price, quantities, and profits

    Raises:
        ValueError: If quantities are negative or lists have mismatched lengths
    """
    # Validate inputs
    validate_quantities(quantities)

    if len(costs) != len(quantities):
        raise ValueError(
            f"Costs list length ({len(costs)}) must match quantities list length ({len(quantities)})"
        )

    # Calculate effective demand parameters as weighted averages
    total_quantity = sum(quantities)

    # For segmented demand, we need to find the price that clears the market
    # This requires solving: total_quantity = sum(weight_k * (alpha_k - beta_k * price))
    # Rearranging: total_quantity = sum(weight_k * alpha_k) - price * sum(weight_k * beta_k)
    # So: price = (sum(weight_k * alpha_k) - total_quantity) / sum(weight_k * beta_k)

    weighted_alpha = sum(
        segment.weight * segment.alpha for segment in segmented_demand.segments
    )
    weighted_beta = sum(
        segment.weight * segment.beta for segment in segmented_demand.segments
    )

    if weighted_beta <= 0:
        raise ValueError("Weighted beta parameter must be positive")

    # Calculate market price using effective parameters
    price = max(0.0, (weighted_alpha - total_quantity) / weighted_beta)

    # Ensure firms don't produce at losses - exit unprofitable firms
    from ..strategies.nash_strategies import validate_profitable_production

    adjusted_quantities = validate_profitable_production(quantities, costs, price)

    # Recalculate price with adjusted quantities
    adjusted_total_quantity = sum(adjusted_quantities)
    adjusted_price = max(
        0.0, (weighted_alpha - adjusted_total_quantity) / weighted_beta
    )

    # Calculate profits: π_i = (P - c_i) * q_i - FC_i
    if fixed_costs:
        if len(fixed_costs) != len(adjusted_quantities):
            raise ValueError(
                f"Fixed costs length ({len(fixed_costs)}) must match quantities length ({len(adjusted_quantities)})"
            )
        profits = [
            (adjusted_price - cost) * q - fc
            for cost, q, fc in zip(costs, adjusted_quantities, fixed_costs)
        ]
    else:
        profits = [
            (adjusted_price - cost) * q for cost, q in zip(costs, adjusted_quantities)
        ]

    return CournotResult(
        price=adjusted_price, quantities=adjusted_quantities, profits=profits
    )


def parse_costs(costs_str: str) -> List[float]:
    """Parse comma-separated costs string into list of floats.

    Args:
        costs_str: Comma-separated string of costs (e.g., "10,20,30")

    Returns:
        List of parsed cost values

    Raises:
        ValueError: If parsing fails or costs are invalid
    """
    if not costs_str.strip():
        raise ValueError("Costs list cannot be empty")

    try:
        costs = [float(x.strip()) for x in costs_str.split(",") if x.strip()]
        if not costs:
            raise ValueError("Costs list cannot be empty")
        return costs
    except ValueError as e:
        raise ValueError(f"Invalid costs format '{costs_str}': {e}")


def parse_quantities(quantities_str: str) -> List[float]:
    """Parse comma-separated quantities string into list of floats.

    Args:
        quantities_str: Comma-separated string of quantities (e.g., "10,20,30")

    Returns:
        List of parsed quantity values

    Raises:
        ValueError: If parsing fails or quantities are invalid
    """
    if not quantities_str.strip():
        raise ValueError("Quantities list cannot be empty")

    try:
        quantities = [float(x.strip()) for x in quantities_str.split(",") if x.strip()]
        if not quantities:
            raise ValueError("Quantities list cannot be empty")
        return quantities
    except ValueError as e:
        raise ValueError(f"Invalid quantities format '{quantities_str}': {e}")
