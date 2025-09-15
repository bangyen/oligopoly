"""Cournot oligopoly simulation implementation.

This module implements the Cournot model of oligopoly competition where firms
simultaneously choose quantities to maximize profits. The simulation computes
market price based on total quantity supplied and calculates individual firm profits.
"""

from dataclasses import dataclass
from typing import List


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
    a: float, b: float, costs: List[float], quantities: List[float]
) -> CournotResult:
    """Run a one-round Cournot oligopoly simulation.

    Computes market price based on inverse demand P = max(0, a - b * sum(q_i))
    and calculates individual firm profits π_i = (P - c_i) * q_i.

    Args:
        a: Maximum price parameter for demand curve
        b: Price sensitivity parameter for demand curve
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

    # Calculate market price: P = max(0, a - b * sum(q_i))
    total_quantity = sum(quantities)
    price = max(0.0, a - b * total_quantity)

    # Calculate profits: π_i = (P - c_i) * q_i
    profits = [(price - cost) * q for cost, q in zip(costs, quantities)]

    return CournotResult(price=price, quantities=quantities.copy(), profits=profits)


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
