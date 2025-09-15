"""Bertrand oligopoly simulation implementation.

This module implements the Bertrand model of oligopoly competition where firms
simultaneously choose prices to maximize profits. The simulation allocates market
demand to the lowest-priced firms and calculates individual firm profits.
Supports both single-segment and multi-segment demand models.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from ..models.models import SegmentedDemand


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
    prices: List[float], costs: List[float], alpha: float, beta: float
) -> Tuple[List[float], float]:
    """Allocate market demand among firms based on Bertrand competition.

    Firms with the lowest price get all demand. If multiple firms tie for
    lowest price, they split the demand equally. Market demand is calculated
    at the lowest price.

    Args:
        prices: List of prices set by each firm
        costs: List of marginal costs for each firm
        alpha: Intercept parameter for demand curve
        beta: Slope parameter for demand curve

    Returns:
        Tuple of (quantities_allocated, total_demand_at_lowest_price)
    """
    if not prices:
        return [], 0.0

    # Find the minimum price
    min_price = min(prices)

    # Calculate total demand at the minimum price
    total_demand = calculate_demand(alpha, beta, min_price)

    # Find all firms with the minimum price
    min_price_firms = [
        i for i, p in enumerate(prices) if math.isclose(p, min_price, abs_tol=1e-10)
    ]

    # Allocate demand equally among firms with minimum price
    quantities = [0.0] * len(prices)
    if min_price_firms:
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
    alpha: float, beta: float, costs: List[float], prices: List[float]
) -> BertrandResult:
    """Run a one-round Bertrand oligopoly simulation.

    Computes market demand allocation based on price competition where firms
    with the lowest price capture all demand, with ties splitting equally.
    Calculates individual firm profits π_i = (p_i - c_i) * q_i.

    Args:
        alpha: Intercept parameter for demand curve Q(p) = max(0, α - β*p)
        beta: Slope parameter for demand curve Q(p) = max(0, α - β*p)
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

    if alpha <= 0:
        raise ValueError(f"Alpha parameter ({alpha}) must be positive")

    if beta <= 0:
        raise ValueError(f"Beta parameter ({beta}) must be positive")

    # Allocate demand based on Bertrand competition
    quantities, total_demand = allocate_demand(prices, costs, alpha, beta)

    # Calculate profits: π_i = (p_i - c_i) * q_i
    profits = [(price - cost) * q for price, cost, q in zip(prices, costs, quantities)]

    return BertrandResult(
        total_demand=total_demand,
        prices=prices.copy(),
        quantities=quantities,
        profits=profits,
    )


def bertrand_segmented_simulation(
    segmented_demand: SegmentedDemand, costs: List[float], prices: List[float]
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

    # Calculate profits: π_i = (p_i - c_i) * q_i
    profits = [(price - cost) * q for price, cost, q in zip(prices, costs, quantities)]

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
