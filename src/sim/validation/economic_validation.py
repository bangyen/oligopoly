"""Economic validation utilities for oligopoly simulation.

This module provides validation functions to ensure economic consistency
in simulation results, preventing impossible economic outcomes.
"""

import math
from typing import List, Optional, Tuple


class EconomicValidationError(Exception):
    """Exception raised when economic validation fails."""

    pass


def validate_demand_parameters(a: float, b: float, alpha: float, beta: float) -> None:
    """Validate demand curve parameters for economic consistency.

    Args:
        a: Cournot demand intercept (P = a - b*Q)
        b: Cournot demand slope (P = a - b*Q)
        alpha: Bertrand demand intercept (Q = alpha - beta*P)
        beta: Bertrand demand slope (Q = alpha - beta*P)

    Raises:
        EconomicValidationError: If parameters are economically invalid
    """
    # Only validate parameters that are actually used (non-zero)
    if a > 0:  # Cournot parameters
        if a <= 0:
            raise EconomicValidationError(
                f"Demand intercept 'a' must be positive, got {a}"
            )
        if b <= 0:
            raise EconomicValidationError(f"Demand slope 'b' must be positive, got {b}")

    if alpha > 0:  # Bertrand parameters
        if alpha <= 0:
            raise EconomicValidationError(
                f"Demand intercept 'alpha' must be positive, got {alpha}"
            )
        if beta <= 0:
            raise EconomicValidationError(
                f"Demand slope 'beta' must be positive, got {beta}"
            )


def validate_costs(costs: List[float], min_cost: float = 0.0) -> None:
    """Validate firm cost parameters.

    Args:
        costs: List of marginal costs for each firm
        min_cost: Minimum allowed marginal cost

    Raises:
        EconomicValidationError: If costs are invalid
    """
    if not costs:
        raise EconomicValidationError("Costs list cannot be empty")

    for i, cost in enumerate(costs):
        if cost < min_cost:
            raise EconomicValidationError(
                f"Firm {i} marginal cost {cost} is below minimum {min_cost}"
            )


def validate_price_quantity_consistency(
    prices: List[float], quantities: List[float], min_price: float = 0.0
) -> None:
    """Validate that price-quantity relationships are economically consistent.

    Args:
        prices: List of prices for each firm
        quantities: List of quantities for each firm
        min_price: Minimum viable price

    Raises:
        EconomicValidationError: If price-quantity relationships are invalid
    """
    if len(prices) != len(quantities):
        raise EconomicValidationError(
            f"Prices and quantities lists must have same length: {len(prices)} vs {len(quantities)}"
        )

    for i, (price, qty) in enumerate(zip(prices, quantities)):
        # If quantity > 0, price must be >= min_price
        if qty > 0 and price < min_price:
            raise EconomicValidationError(
                f"Firm {i}: quantity {qty} > 0 but price {price} < minimum {min_price}"
            )

        # If price = 0, quantity should be 0 (no free production)
        if math.isclose(price, 0.0, abs_tol=1e-10) and qty > 0:
            raise EconomicValidationError(
                f"Firm {i}: price is 0 but quantity {qty} > 0 (economically impossible)"
            )


def validate_profit_consistency(
    prices: List[float],
    quantities: List[float],
    costs: List[float],
    profits: List[float],
    fixed_costs: Optional[List[float]] = None,
    tolerance: float = 1e-6,
) -> None:
    """Validate that profits are consistent with prices, quantities, and costs.

    Args:
        prices: List of prices for each firm
        quantities: List of quantities for each firm
        costs: List of marginal costs for each firm
        profits: List of profits for each firm
        fixed_costs: Optional list of fixed costs for each firm
        tolerance: Numerical tolerance for profit calculations

    Raises:
        EconomicValidationError: If profits are inconsistent
    """
    if fixed_costs is None:
        fixed_costs = [0.0] * len(prices)

    for i, (price, qty, cost, profit, fc) in enumerate(
        zip(prices, quantities, costs, profits, fixed_costs)
    ):
        # Calculate expected profit: π = (P - c) * q - FC
        expected_profit = (price - cost) * qty - fc

        if not math.isclose(profit, expected_profit, abs_tol=tolerance):
            raise EconomicValidationError(
                f"Firm {i}: profit {profit} doesn't match expected {(price - cost) * qty - fc}"
            )


def validate_market_clearing(
    total_quantity: float,
    market_price: float,
    demand_params: Tuple[float, float],
    tolerance: float = 1e-6,
    model: str = "cournot",
) -> None:
    """Validate that market clears (supply = demand).

    Args:
        total_quantity: Total quantity supplied
        market_price: Market clearing price
        demand_params: Tuple of (a, b) for Cournot or (alpha, beta) for Bertrand
        tolerance: Numerical tolerance for market clearing
        model: Model type ("cournot" or "bertrand")

    Raises:
        EconomicValidationError: If market doesn't clear
    """
    a, b = demand_params

    if model == "cournot":
        # Cournot: P = a - b*Q
        expected_price = max(0.0, a - b * total_quantity)
    else:  # bertrand
        # Bertrand: Q = alpha - beta*P, so P = (alpha - Q) / beta
        alpha, beta = a, b
        if beta > 0:
            expected_price = max(0.0, (alpha - total_quantity) / beta)
        else:
            expected_price = 0.0

    if not math.isclose(market_price, expected_price, abs_tol=tolerance):
        raise EconomicValidationError(
            f"Market price {market_price} doesn't match demand curve: expected {expected_price}"
        )


def validate_firm_viability(
    costs: List[float],
    prices: List[float],
    quantities: List[float],
    profits: List[float],
    fixed_costs: Optional[List[float]] = None,
    min_profit_threshold: float = 0.0,
) -> List[bool]:
    """Validate which firms should remain viable in the market.

    Args:
        costs: List of marginal costs for each firm
        prices: List of prices for each firm
        quantities: List of quantities for each firm
        profits: List of profits for each firm
        fixed_costs: Optional list of fixed costs for each firm
        min_profit_threshold: Minimum profit threshold for viability

    Returns:
        List of booleans indicating which firms should remain viable

    Raises:
        EconomicValidationError: If firms are producing when they shouldn't
    """
    if fixed_costs is None:
        fixed_costs = [0.0] * len(costs)

    viable_firms = []
    for i, (cost, price, qty, profit, fc) in enumerate(
        zip(costs, prices, quantities, profits, fixed_costs)
    ):
        # Firm should exit if:
        # 1. Price < marginal cost (can't cover variable costs)
        # 2. Profit < -fixed_cost (can't cover fixed costs)
        should_exit = (price < cost) or (profit < -fc + min_profit_threshold)
        viable_firms.append(not should_exit)

        # If firm should exit but is still producing, that's an error
        if should_exit and qty > 0:
            raise EconomicValidationError(
                f"Firm {i} should exit (price {price} < cost {cost} or profit {profit} < threshold) "
                f"but is still producing quantity {qty}"
            )

    return viable_firms


def enforce_economic_constraints(
    prices: List[float],
    quantities: List[float],
    costs: List[float],
    fixed_costs: Optional[List[float]] = None,
    min_price: float = 0.01,  # Minimum viable price
) -> Tuple[List[float], List[float], List[float]]:
    """Enforce economic constraints on prices, quantities, and profits.

    Args:
        prices: List of prices for each firm
        quantities: List of quantities for each firm
        costs: List of marginal costs for each firm
        fixed_costs: Optional list of fixed costs for each firm
        min_price: Minimum viable price

    Returns:
        Tuple of (adjusted_prices, adjusted_quantities, adjusted_profits)
    """
    if fixed_costs is None:
        fixed_costs = [0.0] * len(prices)

    adjusted_prices = []
    adjusted_quantities = []
    adjusted_profits = []

    for i, (price, qty, cost, fc) in enumerate(
        zip(prices, quantities, costs, fixed_costs)
    ):
        # If price is too low, force exit
        if price < min_price:
            adjusted_prices.append(0.0)
            adjusted_quantities.append(0.0)
            adjusted_profits.append(-fc)  # Only pay fixed costs
        # If price < marginal cost, force exit
        elif price < cost:
            adjusted_prices.append(0.0)
            adjusted_quantities.append(0.0)
            adjusted_profits.append(-fc)  # Only pay fixed costs
        else:
            # Firm can operate
            adjusted_prices.append(price)
            adjusted_quantities.append(qty)
            adjusted_profits.append((price - cost) * qty - fc)

    return adjusted_prices, adjusted_quantities, adjusted_profits


def validate_simulation_result(
    prices: List[float],
    quantities: List[float],
    costs: List[float],
    profits: List[float],
    market_price: float,
    demand_params: Tuple[float, float],
    fixed_costs: Optional[List[float]] = None,
    strict: bool = True,
    model: str = "cournot",
) -> None:
    """Comprehensive validation of simulation results.

    Args:
        prices: List of prices for each firm
        quantities: List of quantities for each firm
        costs: List of marginal costs for each firm
        profits: List of profits for each firm
        market_price: Market clearing price
        demand_params: Tuple of (a, b) for demand curve
        fixed_costs: Optional list of fixed costs for each firm
        strict: If True, raise exceptions on violations; if False, only warn
        model: Model type ("cournot" or "bertrand")

    Raises:
        EconomicValidationError: If validation fails and strict=True
    """
    try:
        # Validate basic parameters
        validate_costs(costs)
        validate_price_quantity_consistency(prices, quantities)
        validate_profit_consistency(prices, quantities, costs, profits, fixed_costs)

        # Validate market clearing
        total_quantity = sum(quantities)
        validate_market_clearing(
            total_quantity, market_price, demand_params, model=model
        )

        # Validate firm viability
        validate_firm_viability(costs, prices, quantities, profits, fixed_costs)

    except EconomicValidationError as e:
        if strict:
            raise
        else:
            print(f"Economic validation warning: {e}")
