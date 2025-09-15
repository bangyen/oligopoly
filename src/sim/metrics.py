"""Economic metrics for oligopoly market analysis.

This module provides calculations for key economic indicators including
Herfindahl-Hirschman Index (HHI) for market concentration and consumer
surplus for welfare analysis in oligopoly simulations.
"""

import math
from typing import List, Tuple


def calculate_hhi(market_shares: List[float]) -> float:
    """Calculate Herfindahl-Hirschman Index (HHI) for market concentration.

    HHI measures market concentration by summing the squares of market shares.
    Higher values indicate more concentrated markets. Perfect competition
    approaches 0, while monopoly equals 1.0 (or 10,000 in percentage terms).

    Args:
        market_shares: List of market shares (as proportions, not percentages)

    Returns:
        HHI value (sum of squared shares)

    Raises:
        ValueError: If shares are negative or don't sum to approximately 1
    """
    if not market_shares:
        raise ValueError("Market shares list cannot be empty")

    # Validate shares are non-negative
    for i, share in enumerate(market_shares):
        if share < 0:
            raise ValueError(f"Market share {i} = {share:.3f} must be non-negative")

    # Validate shares sum to approximately 1 (allowing for small floating point errors)
    total_share = sum(market_shares)
    if not math.isclose(total_share, 1.0, abs_tol=1e-6):
        raise ValueError(f"Market shares must sum to 1.0, got {total_share:.6f}")

    # Calculate HHI: sum of squared shares
    hhi = sum(share**2 for share in market_shares)
    return hhi


def calculate_market_shares_cournot(quantities: List[float]) -> List[float]:
    """Calculate market shares from quantities in Cournot competition.

    In Cournot competition, market share is quantity share: q_i / sum(q_j).

    Args:
        quantities: List of quantities produced by each firm

    Returns:
        List of market shares (proportions)

    Raises:
        ValueError: If quantities are negative or total is zero
    """
    if not quantities:
        raise ValueError("Quantities list cannot be empty")

    # Validate quantities are non-negative
    for i, qty in enumerate(quantities):
        if qty < 0:
            raise ValueError(f"Quantity {i} = {qty:.3f} must be non-negative")

    total_qty = sum(quantities)
    if total_qty == 0:
        raise ValueError("Total quantity cannot be zero")

    # Calculate shares
    shares = [qty / total_qty for qty in quantities]
    return shares


def calculate_market_shares_bertrand(
    prices: List[float], quantities: List[float]
) -> List[float]:
    """Calculate market shares from revenue in Bertrand competition.

    In Bertrand competition, market share is revenue share: (p_i * q_i) / sum(p_j * q_j).

    Args:
        prices: List of prices set by each firm
        quantities: List of quantities sold by each firm

    Returns:
        List of market shares (proportions)

    Raises:
        ValueError: If prices/quantities are negative or total revenue is zero
    """
    if not prices or not quantities:
        raise ValueError("Prices and quantities lists cannot be empty")

    if len(prices) != len(quantities):
        raise ValueError(
            f"Prices ({len(prices)}) and quantities ({len(quantities)}) must have same length"
        )

    # Validate prices and quantities are non-negative
    for i, (price, qty) in enumerate(zip(prices, quantities)):
        if price < 0:
            raise ValueError(f"Price {i} = {price:.3f} must be non-negative")
        if qty < 0:
            raise ValueError(f"Quantity {i} = {qty:.3f} must be non-negative")

    # Calculate revenues
    revenues = [price * qty for price, qty in zip(prices, quantities)]
    total_revenue = sum(revenues)

    if total_revenue == 0:
        raise ValueError("Total revenue cannot be zero")

    # Calculate shares
    shares = [revenue / total_revenue for revenue in revenues]
    return shares


def calculate_consumer_surplus(
    price_intercept: float, market_price: float, market_quantity: float
) -> float:
    """Calculate consumer surplus for linear demand curve.

    Consumer surplus is the area under the demand curve above the market price.
    For linear demand P(Q) = a - b*Q, CS = 0.5 * (a - P_market) * Q_market.

    Args:
        price_intercept: Maximum price (a) when quantity is zero
        market_price: Current market price
        market_quantity: Current market quantity

    Returns:
        Consumer surplus value

    Raises:
        ValueError: If inputs are negative or price exceeds intercept
    """
    if price_intercept <= 0:
        raise ValueError(f"Price intercept {price_intercept:.3f} must be positive")

    if market_price < 0:
        raise ValueError(f"Market price {market_price:.3f} must be non-negative")

    if market_quantity < 0:
        raise ValueError(f"Market quantity {market_quantity:.3f} must be non-negative")

    if market_price > price_intercept:
        raise ValueError(
            f"Market price {market_price:.3f} cannot exceed intercept {price_intercept:.3f}"
        )

    # CS = 0.5 * (P_intercept - P_market) * Q_market
    consumer_surplus = 0.5 * (price_intercept - market_price) * market_quantity
    return consumer_surplus


def calculate_round_metrics_cournot(
    quantities: List[float], market_price: float, demand_a: float
) -> Tuple[float, float]:
    """Calculate HHI and consumer surplus for a Cournot round.

    Args:
        quantities: List of quantities produced by each firm
        market_price: Market price for this round
        demand_a: Price intercept parameter from demand curve

    Returns:
        Tuple of (HHI, consumer_surplus)
    """
    # Calculate market shares from quantities
    shares = calculate_market_shares_cournot(quantities)

    # Calculate HHI
    hhi = calculate_hhi(shares)

    # Calculate consumer surplus
    total_quantity = sum(quantities)
    cs = calculate_consumer_surplus(demand_a, market_price, total_quantity)

    return hhi, cs


def calculate_round_metrics_bertrand(
    prices: List[float],
    quantities: List[float],
    total_demand: float,
    demand_alpha: float,
) -> Tuple[float, float]:
    """Calculate HHI and consumer surplus for a Bertrand round.

    Args:
        prices: List of prices set by each firm
        quantities: List of quantities sold by each firm
        total_demand: Total market demand
        demand_alpha: Demand intercept parameter

    Returns:
        Tuple of (HHI, consumer_surplus)
    """
    # Calculate market shares from revenue
    shares = calculate_market_shares_bertrand(prices, quantities)

    # Calculate HHI
    hhi = calculate_hhi(shares)

    # Calculate consumer surplus
    # In Bertrand, market price is the minimum price
    market_price = min(prices) if prices else 0.0
    cs = calculate_consumer_surplus(demand_alpha, market_price, total_demand)

    return hhi, cs
