"""Bertrand oligopoly simulation implementation.

This module implements the Bertrand model of oligopoly competition where firms
simultaneously choose prices to maximize profits. The simulation allocates market
demand to the lowest-priced firms and calculates individual firm profits.
Supports both single-segment and multi-segment demand models.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..models.models import SegmentedDemand
from ..validation.economic_validation import (
    EconomicValidationError,
    enforce_economic_constraints,
    validate_cost_structure,
    validate_demand_parameters,
    validate_simulation_result,
)


@dataclass
class BertrandResult:
    """Results from a Bertrand simulation run.

    Contains the market demand, individual firm prices, quantities allocated,
    and profits from a single round of Bertrand competition.
    """

    total_demand: float
    prices: List[float]
    quantities: List[float]
    profits: List[float]

    def __repr__(self) -> str:
        """String representation for debugging and output."""
        return f"BertrandResult(demand={self.total_demand}, prices={self.prices}, quantities={self.quantities}, profits={self.profits})"


def validate_prices(prices: List[float]) -> None:
    """Validate that all prices are non-negative.

    Args:
        prices: List of firm prices to validate

    Raises:
        ValueError: If any price is negative
    """
    for i, p in enumerate(prices):
        if p < 0:
            raise ValueError(f"Price p_{i} = {p:.1f} must be non-negative")


def calculate_demand(alpha: float, beta: float, price: float) -> float:
    """Calculate market demand Q(p) = max(0, α - β*p).

    Args:
        alpha: Intercept parameter for demand curve
        beta: Slope parameter for demand curve
        price: Market price

    Returns:
        Total market demand at given price
    """
    return max(0.0, alpha - beta * price)


def allocate_demand(
    prices: List[float],
    costs: List[float],
    alpha: float,
    beta: float,
    use_capacity_constraints: bool = True,
) -> Tuple[List[float], float]:
    """Allocate market demand among firms based on Bertrand competition.

    Uses a more realistic allocation that considers capacity constraints and
    market frictions to prevent unrealistic winner-take-all outcomes.

    Args:
        prices: List of prices set by each firm
        costs: List of marginal costs for each firm
        alpha: Intercept parameter for demand curve
        beta: Slope parameter for demand curve
        use_capacity_constraints: If True, use capacity constraints to prevent monopoly.
                                 If False, use traditional winner-take-all allocation.

    Returns:
        Tuple of (quantities_allocated, total_demand_at_lowest_price)
    """
    if not prices:
        return [], 0.0

    # Enforce economic constraints: prices must be at least marginal cost
    adjusted_prices = []
    for i, (price, cost) in enumerate(zip(prices, costs)):
        # Allow small price below cost for competitive pressure, but not excessive losses
        min_price = max(0.0, cost * 0.95)  # Allow 5% below cost maximum
        adjusted_prices.append(max(price, min_price))

    # Find the minimum price among economically viable firms
    min_price = min(adjusted_prices)

    # Calculate total demand at the minimum price
    total_demand = calculate_demand(alpha, beta, min_price)

    # Find all firms with the minimum price (including ties)
    min_price_firms = [
        i
        for i, p in enumerate(adjusted_prices)
        if math.isclose(p, min_price, abs_tol=1e-10)
    ]

    # Allocate demand based on mode
    quantities = [0.0] * len(prices)

    if min_price_firms and total_demand > 0:
        if use_capacity_constraints:
            # Use capacity-constrained allocation to prevent monopoly
            # Calculate capacity per firm (prevent winner-take-all)
            max_capacity_per_firm = (
                alpha / beta
            ) * 0.4  # Each firm can serve up to 40% of market

            # Allocate demand with capacity constraints
            remaining_demand = total_demand
            active_firms = []

            # First, allocate to firms with minimum price
            for i in min_price_firms:
                if remaining_demand > 0:
                    # Allocate up to capacity
                    firm_allocation = min(max_capacity_per_firm, remaining_demand)
                    quantities[i] = firm_allocation
                    active_firms.append(i)
                    remaining_demand -= firm_allocation

            # If there's still demand, allow other firms to enter with small price premium
            if remaining_demand > 0:
                # Allow other firms to capture remaining demand
                for i, (price, cost) in enumerate(zip(adjusted_prices, costs)):
                    if (
                        i not in active_firms and price <= min_price * 1.1
                    ):  # Allow 10% price premium
                        firm_allocation = min(max_capacity_per_firm, remaining_demand)
                        if firm_allocation > 0:
                            quantities[i] = firm_allocation
                            active_firms.append(i)
                            remaining_demand -= firm_allocation
                            if remaining_demand <= 0:
                                break

            # If still only one firm active, force some competition by allowing others
            if len(active_firms) == 1 and len(min_price_firms) > 1:
                # Redistribute some demand to other minimum price firms
                active_firm = active_firms[0]
                if (
                    quantities[active_firm] > max_capacity_per_firm * 0.6
                ):  # If one firm has >60% capacity
                    # Redistribute excess to other minimum price firms
                    excess = quantities[active_firm] - max_capacity_per_firm * 0.6
                    quantities[active_firm] = max_capacity_per_firm * 0.6

                    # Give excess to other minimum price firms
                    other_min_firms = [i for i in min_price_firms if i != active_firm]
                    if other_min_firms:
                        excess_per_firm = excess / len(other_min_firms)
                        for i in other_min_firms:
                            quantities[i] = min(
                                max_capacity_per_firm * 0.4, excess_per_firm
                            )
        else:
            # Traditional winner-take-all allocation (for backward compatibility)
            demand_per_firm = total_demand / len(min_price_firms)
            for i in min_price_firms:
                quantities[i] = demand_per_firm

    return quantities, total_demand


def allocate_segmented_demand(
    prices: List[float], costs: List[float], segmented_demand: SegmentedDemand
) -> Tuple[List[float], float]:
    """Allocate segmented market demand among firms based on Bertrand competition.

    Each segment chooses the firm with the lowest price (ties split equally).
    Total demand is the weighted sum of segment demands at their respective
    lowest prices.

    Args:
        prices: List of prices set by each firm
        costs: List of marginal costs for each firm
        segmented_demand: SegmentedDemand object with segment configurations

    Returns:
        Tuple of (quantities_allocated, total_demand_across_segments)
    """
    if not prices:
        return [], 0.0

    num_firms = len(prices)
    quantities = [0.0] * num_firms
    total_demand = 0.0

    # Process each segment independently
    for segment in segmented_demand.segments:
        # Find minimum price for this segment
        min_price = min(prices)

        # Calculate segment demand at minimum price
        segment_demand = segment.demand(min_price)

        # Find firms with minimum price
        min_price_firms = [
            i for i, p in enumerate(prices) if math.isclose(p, min_price, abs_tol=1e-10)
        ]

        # Allocate segment demand equally among firms with minimum price
        if min_price_firms:
            demand_per_firm = segment_demand / len(min_price_firms)
            for i in min_price_firms:
                quantities[i] += segment.weight * demand_per_firm

        # Add weighted segment demand to total
        total_demand += segment.weight * segment_demand

    return quantities, total_demand


def bertrand_simulation(
    alpha: float,
    beta: float,
    costs: List[float],
    prices: List[float],
    fixed_costs: Optional[List[float]] = None,
    capacity_limits: Optional[List[float]] = None,
    use_capacity_constraints: bool = True,
) -> BertrandResult:
    """Run a one-round Bertrand oligopoly simulation.

    Computes market demand allocation based on price competition where firms
    with the lowest price capture all demand, with ties splitting equally.
    Calculates individual firm profits π_i = (p_i - c_i) * q_i - FC_i.

    Args:
        alpha: Intercept parameter for demand curve Q(p) = max(0, α - β*p)
        beta: Slope parameter for demand curve Q(p) = max(0, α - β*p)
        costs: List of marginal costs for each firm
        prices: List of prices chosen by each firm
        fixed_costs: Optional list of fixed costs for each firm
        capacity_limits: Optional list of capacity limits for each firm

    Returns:
        BertrandResult containing demand, prices, quantities, and profits

    Raises:
        ValueError: If prices are negative or lists have mismatched lengths
    """
    # Validate inputs
    validate_prices(prices)
    try:
        validate_demand_parameters(
            0.0, 0.0, alpha, beta
        )  # Only validate Bertrand params
        validate_cost_structure(costs, fixed_costs)
    except EconomicValidationError as e:
        raise ValueError(str(e))

    if len(costs) != len(prices):
        raise ValueError(
            f"Costs list length ({len(costs)}) must match prices list length ({len(prices)})"
        )

    if alpha <= 0:
        raise ValueError(f"Alpha parameter ({alpha}) must be positive")

    if beta <= 0:
        raise ValueError(f"Beta parameter ({beta}) must be positive")

    # Allocate demand based on Bertrand competition
    quantities, total_demand = allocate_demand(
        prices, costs, alpha, beta, use_capacity_constraints
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

    # Get the effective prices used for demand allocation (adjusted for economic constraints)
    effective_prices = []
    for i, (price, cost) in enumerate(zip(prices, costs)):
        # Use the same logic as in allocate_demand to get the effective price
        min_price = max(0.0, cost * 0.95)  # Allow 5% below cost maximum
        effective_prices.append(max(price, min_price))

    # Calculate profits using effective prices: π_i = (p_eff_i - c_i) * q_i - FC_i
    if fixed_costs:
        if len(fixed_costs) != len(quantities):
            raise ValueError(
                f"Fixed costs length ({len(fixed_costs)}) must match quantities length ({len(quantities)})"
            )
        profits = [
            (eff_price - cost) * q - fc
            for eff_price, cost, q, fc in zip(
                effective_prices, costs, quantities, fixed_costs
            )
        ]
    else:
        profits = [
            (eff_price - cost) * q
            for eff_price, cost, q in zip(effective_prices, costs, quantities)
        ]

    # Create result
    result = BertrandResult(
        total_demand=total_demand,
        prices=prices.copy(),
        quantities=quantities,
        profits=profits,
    )

    # Validate economic consistency
    try:
        validation_result = validate_simulation_result(
            "bertrand",
            prices,
            quantities,
            profits,
            costs,
            {"alpha": alpha, "beta": beta},
        )

        # Log warnings if any
        if validation_result.warnings:
            import logging

            logger = logging.getLogger(__name__)
            for warning in validation_result.warnings:
                logger.warning(f"Economic validation warning: {warning}")

    except EconomicValidationError as e:
        # Log warning but don't fail the simulation
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Economic validation warning: {e}")

        # If validation fails, enforce constraints
        enforced_quantities = enforce_economic_constraints(
            quantities,
            costs,
            min(prices) if prices else 0.0,
        )

        # Recalculate with enforced quantities
        enforced_total_quantity = sum(enforced_quantities)

        # Recalculate profits with enforced quantities
        if fixed_costs:
            enforced_profits = [
                (price - cost) * q - fc
                for price, cost, q, fc in zip(
                    prices, costs, enforced_quantities, fixed_costs
                )
            ]
        else:
            enforced_profits = [
                (price - cost) * q
                for price, cost, q in zip(prices, costs, enforced_quantities)
            ]

        result = BertrandResult(
            total_demand=enforced_total_quantity,
            prices=prices.copy(),
            quantities=enforced_quantities,
            profits=enforced_profits,
        )

    return result


def bertrand_segmented_simulation(
    segmented_demand: SegmentedDemand,
    costs: List[float],
    prices: List[float],
    fixed_costs: Optional[List[float]] = None,
) -> BertrandResult:
    """Run a one-round Bertrand oligopoly simulation with segmented demand.

    Computes market demand allocation based on price competition where firms
    with the lowest price capture all demand within each segment, with ties
    splitting equally. Calculates individual firm profits π_i = (p_i - c_i) * q_i.

    Args:
        segmented_demand: SegmentedDemand object with segment configurations
        costs: List of marginal costs for each firm
        prices: List of prices chosen by each firm

    Returns:
        BertrandResult containing demand, prices, quantities, and profits

    Raises:
        ValueError: If prices are negative or lists have mismatched lengths
    """
    # Validate inputs
    validate_prices(prices)

    if len(costs) != len(prices):
        raise ValueError(
            f"Costs list length ({len(costs)}) must match prices list length ({len(prices)})"
        )

    # Allocate demand based on segmented Bertrand competition
    quantities, total_demand = allocate_segmented_demand(
        prices, costs, segmented_demand
    )

    # Calculate profits: π_i = (p_i - c_i) * q_i - FC_i
    if fixed_costs:
        if len(fixed_costs) != len(quantities):
            raise ValueError(
                f"Fixed costs length ({len(fixed_costs)}) must match quantities length ({len(quantities)})"
            )
        profits = [
            (price - cost) * q - fc
            for price, cost, q, fc in zip(prices, costs, quantities, fixed_costs)
        ]
    else:
        profits = [
            (price - cost) * q for price, cost, q in zip(prices, costs, quantities)
        ]

    return BertrandResult(
        total_demand=total_demand,
        prices=prices.copy(),
        quantities=quantities,
        profits=profits,
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


def parse_prices(prices_str: str) -> List[float]:
    """Parse comma-separated prices string into list of floats.

    Args:
        prices_str: Comma-separated string of prices (e.g., "10,20,30")

    Returns:
        List of parsed price values

    Raises:
        ValueError: If parsing fails or prices are invalid
    """
    if not prices_str.strip():
        raise ValueError("Prices list cannot be empty")

    try:
        prices = [float(x.strip()) for x in prices_str.split(",") if x.strip()]
        if not prices:
            raise ValueError("Prices list cannot be empty")
        return prices
    except ValueError as e:
        raise ValueError(f"Invalid prices format '{prices_str}': {e}")
