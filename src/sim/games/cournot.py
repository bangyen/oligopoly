"""Cournot oligopoly simulation implementation.

This module implements the Cournot model of oligopoly competition where firms
simultaneously choose quantities to maximize profits. The simulation computes
market price based on total quantity supplied and calculates individual firm profits.
Supports both single-segment and multi-segment demand models.
"""

import logging
from dataclasses import dataclass

from ..models.models import SegmentedDemand
from ..validation.economic_validation import (
    EconomicValidationError,
    validate_cost_structure,
    validate_demand_parameters,
    validate_simulation_result,
)

# Re-export CLI parse helpers from their canonical location.
# Implementation lives in _parsing.py; importing here preserves backward compatibility.
from ._parsing import parse_costs as parse_costs  # noqa: F401
from ._parsing import parse_quantities as parse_quantities  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class CournotResult:
    """Results from a Cournot simulation run.

    Contains the market price, individual firm quantities, and profits
    from a single round of Cournot competition.
    """

    price: float
    quantities: list[float]
    profits: list[float]

    def __repr__(self) -> str:
        """String representation for debugging and output."""
        return f"CournotResult(price={self.price}, quantities={self.quantities}, profits={self.profits})"


def validate_quantities(quantities: list[float]) -> None:
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
    costs: list[float],
    quantities: list[float],
    fixed_costs: list[float] | None = None,
    capacity_limits: list[float] | None = None,
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
        validate_cost_structure(costs, fixed_costs)
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

    # Calculate market price: P = max(0, a - b * sum(q_i)).
    # Every submitted quantity is sold at this price. Firms are not removed
    # after the fact: doing so would let one firm flood the market, push a
    # rival out, and then enjoy the higher recomputed price. Exit decisions
    # belong to the firms' strategies, before quantities are submitted.
    price = max(0.0, a - b * sum(quantities))
    profits = _profits(price, costs, quantities, fixed_costs)
    _log_validation_warnings(price, quantities, profits, costs, {"a": a, "b": b})
    return CournotResult(price=price, quantities=list(quantities), profits=profits)


def _profits(
    price: float,
    costs: list[float],
    quantities: list[float],
    fixed_costs: list[float] | None,
) -> list[float]:
    """Compute π_i = (P - c_i) * q_i - FC_i."""
    if fixed_costs is None:
        fixed_costs = [0.0] * len(costs)
    elif len(fixed_costs) != len(quantities):
        raise ValueError(
            f"Fixed costs length ({len(fixed_costs)}) must match quantities length ({len(quantities)})"
        )
    return [
        (price - cost) * q - fc for cost, q, fc in zip(costs, quantities, fixed_costs)
    ]


def _log_validation_warnings(
    price: float,
    quantities: list[float],
    profits: list[float],
    costs: list[float],
    params: dict[str, float],
) -> None:
    """Log (but never act on) economic-consistency warnings for a round."""
    try:
        validation_result = validate_simulation_result(
            "cournot", [price], quantities, profits, costs, params
        )
        warnings = validation_result.warnings
    except EconomicValidationError as e:
        warnings = [str(e)]
    for warning in warnings:
        logger.warning("Economic validation warning: %s", warning)


def cournot_segmented_simulation(
    segmented_demand: SegmentedDemand,
    costs: list[float],
    quantities: list[float],
    fixed_costs: list[float] | None = None,
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
    try:
        validate_cost_structure(costs, fixed_costs)
    except EconomicValidationError as e:
        raise ValueError(str(e))

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

    # Calculate market price using effective parameters. As in
    # cournot_simulation, all submitted quantities clear at this price.
    price = max(0.0, (weighted_alpha - total_quantity) / weighted_beta)
    profits = _profits(price, costs, quantities, fixed_costs)
    _log_validation_warnings(
        price, quantities, profits, costs, {"a": weighted_alpha, "b": weighted_beta}
    )
    return CournotResult(price=price, quantities=list(quantities), profits=profits)


# End of Cournot implementation
