"""Economic metrics for oligopoly market analysis.

This module provides calculations for key economic indicators including
Herfindahl-Hirschman Index (HHI) for market concentration and consumer
surplus for welfare analysis in oligopoly simulations.
"""

import math


def calculate_hhi(market_shares: list[float]) -> float:
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


def calculate_market_shares_cournot(quantities: list[float]) -> list[float]:
    """Calculate market shares from quantities in Cournot competition.

    In Cournot competition, market share is quantity share: q_i / sum(q_j).
    If total quantity is zero (all firms exited), returns equal shares.

    Args:
        quantities: List of quantities produced by each firm

    Returns:
        List of market shares (proportions)

    Raises:
        ValueError: If quantities are negative
    """
    if not quantities:
        raise ValueError("Quantities list cannot be empty")

    # Validate quantities are non-negative
    for i, qty in enumerate(quantities):
        if qty < 0:
            raise ValueError(f"Quantity {i} = {qty:.3f} must be non-negative")

    total_qty = sum(quantities)
    if total_qty == 0:
        # All firms have zero quantity - return equal shares
        return [1.0 / len(quantities)] * len(quantities)

    # Calculate shares
    shares = [qty / total_qty for qty in quantities]
    return shares


def calculate_market_shares_bertrand(
    prices: list[float], quantities: list[float]
) -> list[float]:
    """Calculate market shares from revenue in Bertrand competition.

    In Bertrand competition, market share is revenue share: (p_i * q_i) / sum(p_j * q_j).
    If total revenue is zero (all firms have zero sales), returns equal shares.

    Args:
        prices: List of prices set by each firm
        quantities: List of quantities sold by each firm

    Returns:
        List of market shares (proportions)

    Raises:
        ValueError: If prices/quantities are negative
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
        # All firms have zero revenue - return equal shares
        return [1.0 / len(prices)] * len(prices)

    # Calculate shares
    shares = [revenue / total_revenue for revenue in revenues]
    return shares
