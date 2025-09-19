"""Simulation pipeline validation utilities.

This module provides validation functions for the entire simulation pipeline
to ensure economic consistency across all stages.
"""

import math
from typing import Any, Dict, List, Tuple

from .economic_validation import (
    EconomicValidationError,
    validate_demand_parameters,
)


def validate_simulation_config(config: Dict[str, Any]) -> None:
    """Validate simulation configuration for economic consistency.

    Args:
        config: Simulation configuration dictionary

    Raises:
        EconomicValidationError: If configuration is economically invalid
    """
    # Validate demand parameters
    params = config.get("params", {})
    if "a" in params and "b" in params:
        validate_demand_parameters(
            params["a"], params["b"], params.get("alpha", 0.0), params.get("beta", 0.0)
        )

    # Validate firm configurations
    firms = config.get("firms", [])
    if not firms:
        raise EconomicValidationError("Simulation must have at least one firm")

    # Extract costs from firm configurations
    costs = []
    fixed_costs = []
    for i, firm in enumerate(firms):
        if not isinstance(firm, dict):
            raise EconomicValidationError(f"Firm {i} must be a dictionary")

        cost = firm.get("cost")
        if cost is None:
            raise EconomicValidationError(f"Firm {i} must have a 'cost' field")
        if cost <= 0:
            raise EconomicValidationError(f"Firm {i} cost {cost} must be positive")
        costs.append(cost)

        # Handle fixed costs
        fc = firm.get("fixed_cost", 0.0)
        if fc < 0:
            raise EconomicValidationError(
                f"Firm {i} fixed cost {fc} must be non-negative"
            )
        fixed_costs.append(fc)


def validate_round_results(
    round_results: List[Dict[str, Any]], model: str, demand_params: Tuple[float, float]
) -> None:
    """Validate results from a single simulation round.

    Args:
        round_results: List of firm results for the round
        model: Simulation model ("cournot" or "bertrand")
        demand_params: Demand curve parameters

    Raises:
        EconomicValidationError: If round results are invalid
    """
    if not round_results:
        raise EconomicValidationError("Round must have at least one firm result")

    prices = [r["price"] for r in round_results]
    quantities = [r["quantity"] for r in round_results]
    costs = [r["cost"] for r in round_results]
    profits = [r["profit"] for r in round_results]

    # Validate basic consistency
    if len(set(len(x) for x in [prices, quantities, costs, profits])) != 1:
        raise EconomicValidationError("All result arrays must have same length")

    # Validate economic relationships
    for i, (price, qty, cost, profit) in enumerate(
        zip(prices, quantities, costs, profits)
    ):
        # Price-quantity consistency
        if qty > 0 and price <= 0:
            raise EconomicValidationError(
                f"Firm {i}: quantity {qty} > 0 but price {price} <= 0"
            )

        # Profit calculation consistency
        expected_profit = (price - cost) * qty
        if not math.isclose(profit, expected_profit, abs_tol=1e-6):
            raise EconomicValidationError(
                f"Firm {i}: profit {profit} doesn't match expected {expected_profit}"
            )

    # Model-specific validation
    if model == "cournot":
        # All firms should face the same market price
        if len(set(prices)) > 1:
            raise EconomicValidationError(
                "Cournot model: all firms must face same market price"
            )
    elif model == "bertrand":
        # Market price should be the minimum price
        market_price = min(prices)
        if not all(
            math.isclose(p, market_price, abs_tol=1e-6) or qty == 0
            for p, qty in zip(prices, quantities)
        ):
            raise EconomicValidationError(
                "Bertrand model: only lowest-price firms should have positive quantities"
            )


def validate_run_results(
    run_results: Dict[str, Any], model: str, demand_params: Tuple[float, float]
) -> None:
    """Validate results from an entire simulation run.

    Args:
        run_results: Complete run results dictionary
        model: Simulation model ("cournot" or "bertrand")
        demand_params: Demand curve parameters

    Raises:
        EconomicValidationError: If run results are invalid
    """
    rounds_data = run_results.get("results", {})

    if not rounds_data:
        raise EconomicValidationError("Run must have at least one round")

    # Validate each round
    for round_idx, round_data in rounds_data.items():
        firm_results = []
        for firm_id, firm_data in round_data.items():
            firm_results.append(
                {
                    "price": firm_data["price"],
                    "quantity": firm_data["quantity"],
                    "cost": firm_data.get("cost", 0.0),  # Assume cost from context
                    "profit": firm_data["profit"],
                }
            )

        validate_round_results(firm_results, model, demand_params)

    # Validate run-level consistency
    num_firms = len(list(rounds_data.values())[0])
    for round_data in rounds_data.values():
        if len(round_data) != num_firms:
            raise EconomicValidationError(
                f"Number of firms inconsistent across rounds: {len(round_data)} vs {num_firms}"
            )


def sanitize_simulation_results(
    results: Dict[str, Any], model: str, min_price: float = 0.01
) -> Dict[str, Any]:
    """Sanitize simulation results to ensure economic consistency.

    Args:
        results: Raw simulation results
        model: Simulation model ("cournot" or "bertrand")
        min_price: Minimum viable price

    Returns:
        Sanitized results with economic constraints enforced
    """
    sanitized = results.copy()
    rounds_data = sanitized.get("results", {})

    for round_idx, round_data in rounds_data.items():
        for firm_id, firm_data in round_data.items():
            price = firm_data["price"]
            qty = firm_data["quantity"]

            # Enforce minimum price constraint
            if qty > 0 and price < min_price:
                # Force firm to exit
                round_data[firm_id] = {
                    "price": 0.0,
                    "quantity": 0.0,
                    "profit": -firm_data.get("fixed_cost", 0.0),
                }
            # Ensure zero quantity when price is zero
            elif price == 0.0 and qty > 0:
                round_data[firm_id] = {
                    "price": 0.0,
                    "quantity": 0.0,
                    "profit": -firm_data.get("fixed_cost", 0.0),
                }

    return sanitized


def check_economic_plausibility(
    results: Dict[str, Any],
    model: str,
    demand_params: Tuple[float, float],
    tolerance: float = 1e-6,
) -> List[str]:
    """Check economic plausibility of simulation results.

    Args:
        results: Simulation results
        model: Simulation model
        demand_params: Demand curve parameters
        tolerance: Numerical tolerance

    Returns:
        List of warnings about economic implausibility
    """
    warnings = []
    rounds_data = results.get("results", {})

    for round_idx, round_data in rounds_data.items():
        prices = [firm_data["price"] for firm_data in round_data.values()]
        quantities = [firm_data["quantity"] for firm_data in round_data.values()]
        profits = [firm_data["profit"] for firm_data in round_data.values()]

        # Check for zero prices with positive quantities
        for firm_id, (price, qty) in enumerate(zip(prices, quantities)):
            if math.isclose(price, 0.0, abs_tol=tolerance) and qty > 0:
                warnings.append(
                    f"Round {round_idx}, Firm {firm_id}: zero price with positive quantity {qty}"
                )

        # Check for extremely negative profits
        for firm_id, profit in enumerate(profits):
            if profit < -1000:  # Arbitrary threshold
                warnings.append(
                    f"Round {round_idx}, Firm {firm_id}: extremely negative profit {profit}"
                )

        # Check for unrealistic price-quantity combinations
        total_qty = sum(quantities)
        if model == "cournot" and total_qty > 0:
            market_price = prices[0]  # All firms face same price in Cournot
            a, b = demand_params
            expected_price = max(0.0, a - b * total_qty)
            if not math.isclose(market_price, expected_price, abs_tol=tolerance):
                warnings.append(
                    f"Round {round_idx}: market price {market_price} doesn't match demand curve (expected {expected_price})"
                )

    return warnings
