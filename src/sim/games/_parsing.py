"""CLI string-parsing helpers for oligopoly simulation inputs.

These utilities convert comma-separated CLI strings into typed Python lists.
They belong here — in a dedicated adapter module — rather than in the core
game simulation modules (cournot.py / bertrand.py).
"""

from typing import List


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
