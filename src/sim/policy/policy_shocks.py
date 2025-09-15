"""Policy shock implementation for oligopoly simulations.

This module implements policy interventions that can be applied during simulation
rounds, including taxes, subsidies, and price caps. These shocks modify firm
behavior and market outcomes according to economic policy interventions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Union

from ..games.bertrand import BertrandResult
from ..games.cournot import CournotResult


class PolicyType(str, Enum):
    """Types of policy interventions that can be applied."""

    TAX = "tax"
    SUBSIDY = "subsidy"
    PRICE_CAP = "price_cap"


@dataclass
class PolicyEvent:
    """A policy intervention event to be applied at a specific round.

    Represents a policy shock that affects firm profits or market prices
    during a specific round of the simulation.
    """

    round_idx: int
    policy_type: PolicyType
    value: float

    def __post_init__(self) -> None:
        """Validate policy event parameters after initialization."""
        if self.round_idx < 0:
            raise ValueError(f"Round index must be non-negative, got {self.round_idx}")

        if self.value < 0:
            raise ValueError(f"Policy value must be non-negative, got {self.value}")

        if self.policy_type == PolicyType.TAX and self.value >= 1.0:
            raise ValueError(f"Tax rate must be less than 1.0, got {self.value}")


def apply_tax_shock(profits: List[float], tax_rate: float) -> List[float]:
    """Apply a profit tax to all firms.

    Reduces each firm's profit by the tax rate: profit_t = profit_base * (1 - τ)

    Args:
        profits: List of base profits for each firm
        tax_rate: Tax rate (0 ≤ τ < 1)

    Returns:
        List of profits after tax application

    Raises:
        ValueError: If tax_rate is invalid
    """
    if tax_rate < 0 or tax_rate >= 1.0:
        raise ValueError(f"Tax rate must be in [0, 1), got {tax_rate}")

    return [profit * (1.0 - tax_rate) for profit in profits]


def apply_subsidy_shock(
    profits: List[float], quantities: List[float], subsidy_per_unit: float
) -> List[float]:
    """Apply a per-unit subsidy to all firms.

    Increases each firm's profit by the subsidy amount: profit_t = profit_base + σ * qty

    Args:
        profits: List of base profits for each firm
        quantities: List of quantities produced by each firm
        subsidy_per_unit: Subsidy amount per unit produced

    Returns:
        List of profits after subsidy application

    Raises:
        ValueError: If subsidy_per_unit is negative or lists have mismatched lengths
    """
    if subsidy_per_unit < 0:
        raise ValueError(
            f"Subsidy per unit must be non-negative, got {subsidy_per_unit}"
        )

    if len(profits) != len(quantities):
        raise ValueError(
            f"Profits list length ({len(profits)}) must match quantities list length ({len(quantities)})"
        )

    return [profit + subsidy_per_unit * qty for profit, qty in zip(profits, quantities)]


def apply_price_cap_shock(
    result: Union[CournotResult, BertrandResult],
    price_cap: float,
    costs: List[float],
) -> Union[CournotResult, BertrandResult]:
    """Apply a price cap to market prices.

    If the unconstrained market price exceeds the cap, sets price to cap
    and recalculates profits accordingly.

    Args:
        result: Original simulation result (Cournot or Bertrand)
        price_cap: Maximum allowed price
        costs: List of marginal costs for each firm

    Returns:
        Modified result with price cap applied

    Raises:
        ValueError: If price_cap is negative
    """
    if price_cap < 0:
        raise ValueError(f"Price cap must be non-negative, got {price_cap}")

    if isinstance(result, CournotResult):
        # For Cournot: cap the market price and recalculate profits
        capped_price = min(result.price, price_cap)
        new_profits = [
            (capped_price - cost) * qty for cost, qty in zip(costs, result.quantities)
        ]

        return CournotResult(
            price=capped_price, quantities=result.quantities.copy(), profits=new_profits
        )

    elif isinstance(result, BertrandResult):
        # For Bertrand: cap individual firm prices and recalculate
        capped_prices = [min(price, price_cap) for price in result.prices]

        # Recalculate quantities and profits with capped prices

        # We need the market parameters - this is a limitation of current design
        # For now, we'll assume the capped prices are applied and recalculate
        new_profits = [
            (price - cost) * qty
            for price, cost, qty in zip(capped_prices, costs, result.quantities)
        ]

        return BertrandResult(
            total_demand=result.total_demand,
            prices=capped_prices,
            quantities=result.quantities.copy(),
            profits=new_profits,
        )

    else:
        raise ValueError(f"Unsupported result type: {type(result)}")


def apply_policy_shock(
    result: Union[CournotResult, BertrandResult],
    event: PolicyEvent,
    costs: List[float],
) -> Union[CournotResult, BertrandResult]:
    """Apply a policy shock to simulation results.

    Applies the specified policy intervention to modify firm profits or market prices
    according to the policy type and value.

    Args:
        result: Original simulation result
        event: Policy event to apply
        costs: List of marginal costs for each firm

    Returns:
        Modified result with policy shock applied

    Raises:
        ValueError: If policy event is invalid
    """
    if event.policy_type == PolicyType.TAX:
        new_profits = apply_tax_shock(result.profits, event.value)
        if isinstance(result, CournotResult):
            return CournotResult(
                price=result.price,
                quantities=result.quantities.copy(),
                profits=new_profits,
            )
        else:  # BertrandResult
            return BertrandResult(
                total_demand=result.total_demand,
                prices=result.prices.copy(),
                quantities=result.quantities.copy(),
                profits=new_profits,
            )

    elif event.policy_type == PolicyType.SUBSIDY:
        new_profits = apply_subsidy_shock(
            result.profits, result.quantities, event.value
        )
        if isinstance(result, CournotResult):
            return CournotResult(
                price=result.price,
                quantities=result.quantities.copy(),
                profits=new_profits,
            )
        else:  # BertrandResult
            return BertrandResult(
                total_demand=result.total_demand,
                prices=result.prices.copy(),
                quantities=result.quantities.copy(),
                profits=new_profits,
            )

    elif event.policy_type == PolicyType.PRICE_CAP:
        return apply_price_cap_shock(result, event.value, costs)

    else:
        raise ValueError(f"Unsupported policy type: {event.policy_type}")


def validate_policy_events(events: List[PolicyEvent], total_rounds: int) -> None:
    """Validate a list of policy events.

    Ensures all events have valid round indices and policy parameters.

    Args:
        events: List of policy events to validate
        total_rounds: Total number of rounds in the simulation

    Raises:
        ValueError: If any event is invalid
    """
    for i, event in enumerate(events):
        if event.round_idx >= total_rounds:
            raise ValueError(
                f"Event {i}: Round index {event.round_idx} must be less than total rounds {total_rounds}"
            )
