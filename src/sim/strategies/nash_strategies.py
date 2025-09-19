"""Nash equilibrium strategies for oligopoly market simulation.

This module implements Nash equilibrium strategies for both Cournot and Bertrand
models, providing economically sound firm behavior that converges to theoretical
equilibrium outcomes.
"""

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.sim.models.models import SegmentedDemand


def should_firm_exit(
    price: float, marginal_cost: float, min_profit_threshold: float = 0.0
) -> bool:
    """Determine if firm should exit when price <= marginal cost.

    Args:
        price: Market price
        marginal_cost: Firm's marginal cost
        min_profit_threshold: Minimum profit threshold for staying in market

    Returns:
        True if firm should exit, False otherwise
    """
    return price <= marginal_cost + min_profit_threshold


def validate_profitable_production(
    quantities: List[float], costs: List[float], price: float
) -> List[float]:
    """Ensure firms don't produce when price < marginal cost.

    Args:
        quantities: Current firm quantities
        costs: Firm marginal costs
        price: Market price

    Returns:
        Adjusted quantities with unprofitable firms set to zero
    """
    adjusted_quantities = []
    for qty, cost in zip(quantities, costs):
        if should_firm_exit(price, cost):
            adjusted_quantities.append(0.0)  # Exit market
        else:
            adjusted_quantities.append(qty)
    return adjusted_quantities


def cournot_nash_equilibrium(
    a: float, b: float, costs: List[float], fixed_costs: Optional[List[float]] = None
) -> Tuple[List[float], float, List[float]]:
    """Calculate Cournot Nash equilibrium quantities, price, and profits.

    For n firms with costs c_i and demand P = a - b*Q, the Nash equilibrium
    quantities are: q_i* = (a - (n+1)*c_i + sum(c_j for j≠i)) / (b*(n+1))

    Firms with costs above the equilibrium price will exit the market.
    Fixed costs are included in profit calculations but not in quantity optimization.

    Args:
        a: Demand intercept parameter
        b: Demand slope parameter
        costs: List of marginal costs for each firm
        fixed_costs: Optional list of fixed costs for each firm

    Returns:
        Tuple of (equilibrium_quantities, equilibrium_price, equilibrium_profits)
    """
    n = len(costs)
    if n == 0:
        return [], 0.0, []

    # Default fixed costs to zero if not provided
    if fixed_costs is None:
        fixed_costs = [0.0] * n
    elif len(fixed_costs) != n:
        raise ValueError(
            f"Fixed costs length ({len(fixed_costs)}) must match costs length ({n})"
        )

    # Start with all firms and iteratively remove unprofitable ones
    active_firms = list(range(n))
    quantities = [0.0] * n
    price = 0.0

    # Iterate until no more firms exit
    max_iterations = n  # Prevent infinite loops
    for iteration in range(max_iterations):
        if not active_firms:
            break

        # Calculate Nash equilibrium for active firms only
        active_costs = [costs[i] for i in active_firms]
        active_total_cost = sum(active_costs)
        active_n = len(active_firms)

        # Calculate quantities for active firms
        active_quantities = []
        for i, cost_i in enumerate(active_costs):
            # q_i* = (a - (n+1)*c_i + sum(c_j for j≠i)) / (b*(n+1))
            q_i = (a - (active_n + 1) * cost_i + active_total_cost) / (
                b * (active_n + 1)
            )
            active_quantities.append(max(0.0, q_i))

        # Calculate equilibrium price
        total_quantity = sum(active_quantities)
        price = max(0.0, a - b * total_quantity)

        # Check which firms should exit (considering both marginal cost and fixed cost)
        firms_to_remove = []
        for i, firm_idx in enumerate(active_firms):
            # Firm exits if price <= marginal cost OR if profit < 0 (including fixed costs)
            if should_firm_exit(price, costs[firm_idx]):
                firms_to_remove.append(i)
                quantities[firm_idx] = 0.0
            else:
                quantities[firm_idx] = active_quantities[i]
                # Check if firm can cover fixed costs
                profit_without_fixed = (price - costs[firm_idx]) * active_quantities[i]
                if profit_without_fixed < fixed_costs[firm_idx]:
                    firms_to_remove.append(i)
                    quantities[firm_idx] = 0.0

        # If no firms can be profitable, break to avoid infinite loop
        if len(active_firms) == len(firms_to_remove):
            break

        # Remove unprofitable firms
        for i in reversed(firms_to_remove):  # Reverse to maintain indices
            active_firms.pop(i)

        # If no firms exit, we've reached equilibrium
        if not firms_to_remove:
            break

    # Calculate final profits including fixed costs
    profits = [
        (price - cost) * q - fc for cost, q, fc in zip(costs, quantities, fixed_costs)
    ]

    return quantities, price, profits


def bertrand_nash_equilibrium(
    alpha: float, beta: float, costs: List[float]
) -> Tuple[List[float], List[float], List[float], float]:
    """Calculate Bertrand Nash equilibrium prices, quantities, and profits.

    In Bertrand competition with homogeneous products, the Nash equilibrium
    is for all firms to price at the marginal cost of the second-lowest cost firm,
    with only the lowest-cost firm(s) producing.

    Args:
        alpha: Demand intercept parameter
        beta: Demand slope parameter
        costs: List of marginal costs for each firm

    Returns:
        Tuple of (equilibrium_prices, equilibrium_quantities, equilibrium_profits, market_price)
    """
    n = len(costs)
    if n == 0:
        return [], [], [], 0.0

    # Sort costs to find the lowest and second-lowest
    sorted_costs = sorted(costs)
    min_cost = sorted_costs[0]
    second_min_cost = sorted_costs[1] if n > 1 else min_cost

    # In Nash equilibrium, price equals second-lowest cost (or min cost if only one firm)
    equilibrium_price = second_min_cost

    # Calculate market demand at equilibrium price
    total_demand = max(0.0, alpha - beta * equilibrium_price)

    # Allocate quantities: only firms with min cost produce
    prices = [equilibrium_price] * n
    quantities = [0.0] * n
    profits = [0.0] * n

    # Find all firms with minimum cost
    min_cost_firms = [i for i, cost in enumerate(costs) if cost == min_cost]

    if min_cost_firms and total_demand > 0:
        # Split demand equally among minimum cost firms
        qty_per_firm = total_demand / len(min_cost_firms)
        for i in min_cost_firms:
            quantities[i] = qty_per_firm
            profits[i] = (equilibrium_price - min_cost) * qty_per_firm

    return prices, quantities, profits, equilibrium_price


def cournot_best_response(
    a: float, b: float, my_cost: float, rival_quantities: List[float]
) -> float:
    """Calculate Cournot best response quantity for a firm.

    Given rival quantities, the best response is:
    q_i = (a - c_i - b * sum(q_j for j≠i)) / (2*b)

    Args:
        a: Demand intercept parameter
        b: Demand slope parameter
        my_cost: This firm's marginal cost
        rival_quantities: Quantities chosen by rival firms

    Returns:
        Best response quantity
    """
    total_rival_quantity = sum(rival_quantities)
    best_response = (a - my_cost - b * total_rival_quantity) / (2 * b)
    return max(0.0, best_response)


def bertrand_best_response(
    alpha: float, beta: float, my_cost: float, rival_prices: List[float]
) -> float:
    """Calculate Bertrand best response price for a firm.

    In Bertrand competition, the best response is to undercut the lowest rival price
    by a small amount, but not below marginal cost.

    Args:
        alpha: Demand intercept parameter
        beta: Demand slope parameter
        my_cost: This firm's marginal cost
        rival_prices: Prices chosen by rival firms

    Returns:
        Best response price
    """
    if not rival_prices:
        # If no rivals, price at monopoly level
        monopoly_price = (alpha + beta * my_cost) / (2 * beta)
        return max(my_cost, monopoly_price)

    min_rival_price = min(rival_prices)

    # Best response: undercut by small amount, but not below cost
    epsilon = 0.01  # Small undercutting amount
    best_response = max(my_cost, min_rival_price - epsilon)

    return best_response


def adaptive_nash_strategy(
    model: str,
    current_actions: List[float],
    profits: List[float],
    costs: List[float],
    params: Dict[str, Any],
    round_idx: int,
    max_rounds: int,
) -> List[float]:
    """Adaptive strategy that converges to Nash equilibrium.

    This strategy gradually moves firms toward their Nash equilibrium actions
    while maintaining some exploration in early rounds and enforcing economic bounds.

    Args:
        model: "cournot" or "bertrand"
        current_actions: Current firm actions
        profits: Current firm profits
        costs: Firm marginal costs
        params: Market parameters
        round_idx: Current round index
        max_rounds: Total number of rounds

    Returns:
        New actions for next round
    """
    # Calculate convergence factor (starts at 0.3, decreases to 0.1)
    # Reduced initial convergence to prevent overshooting
    convergence_factor = 0.1 + 0.2 * (1 - round_idx / max_rounds)

    if model == "cournot":
        # Check if segmented demand
        segments_config = params.get("segments")
        if segments_config:
            # For segmented demand, calculate Nash equilibrium using effective parameters
            weighted_alpha = sum(
                segment["weight"] * segment["alpha"] for segment in segments_config
            )
            weighted_beta = sum(
                segment["weight"] * segment["beta"] for segment in segments_config
            )
            nash_quantities, _, _ = cournot_nash_equilibrium(
                weighted_alpha, weighted_beta, costs
            )
            a = weighted_alpha
            b = weighted_beta
        else:
            # Standard demand
            a = params.get("a", 100.0)
            b = params.get("b", 1.0)
            nash_quantities, _, _ = cournot_nash_equilibrium(a, b, costs)

        # Calculate reasonable bounds based on market size
        max_individual_qty = a / b / len(costs)  # Equal share of total market
        min_qty = 0.1

        # Move toward Nash equilibrium
        new_actions = []
        for i, (current_qty, nash_qty) in enumerate(
            zip(current_actions, nash_quantities)
        ):
            # Weighted average between current and Nash equilibrium
            new_qty = (
                1 - convergence_factor
            ) * current_qty + convergence_factor * nash_qty

            # Add small random perturbation for exploration (reduced noise)
            noise_factor = params.get("noise_factor", 0.01) * (1 - convergence_factor)
            noise = random.uniform(-noise_factor, noise_factor) * new_qty
            new_qty = new_qty + noise

            # Enforce bounds
            new_qty = max(min_qty, min(max_individual_qty, new_qty))

            new_actions.append(new_qty)

    else:  # bertrand
        alpha = params.get("alpha", 100.0)
        beta = params.get("beta", 1.0)

        # Calculate Nash equilibrium prices
        nash_prices, _, _, _ = bertrand_nash_equilibrium(alpha, beta, costs)

        # Move toward Nash equilibrium
        new_actions = []
        for i, (current_price, nash_price) in enumerate(
            zip(current_actions, nash_prices)
        ):
            # Weighted average between current and Nash equilibrium
            new_price = (
                1 - convergence_factor
            ) * current_price + convergence_factor * nash_price

            # Add small random perturbation for exploration (reduced noise)
            noise_factor = params.get("noise_factor", 0.01) * (1 - convergence_factor)
            noise = random.uniform(-noise_factor, noise_factor) * new_price
            new_price = new_price + noise

            # Enforce bounds: price must be above marginal cost
            new_price = max(costs[i] + 0.1, new_price)

            new_actions.append(new_price)

    return new_actions


def cournot_segmented_nash_equilibrium(
    segmented_demand: "SegmentedDemand", costs: List[float]
) -> Tuple[List[float], float, List[float]]:
    """Calculate Cournot Nash equilibrium for segmented demand.

    For segmented demand, we need to solve the Nash equilibrium using
    the effective demand parameters derived from the segments.

    Args:
        segmented_demand: SegmentedDemand object with segment configurations
        costs: List of marginal costs for each firm

    Returns:
        Tuple of (equilibrium_quantities, equilibrium_price, equilibrium_profits)
    """
    # Calculate effective demand parameters
    weighted_alpha = sum(
        segment.weight * segment.alpha for segment in segmented_demand.segments
    )
    weighted_beta = sum(
        segment.weight * segment.beta for segment in segmented_demand.segments
    )

    # Use the effective parameters to calculate Nash equilibrium
    return cournot_nash_equilibrium(weighted_alpha, weighted_beta, costs)


def validate_market_clearing(
    model: str, actions: List[float], costs: List[float], params: Dict[str, Any]
) -> List[float]:
    """Validate and adjust actions to ensure market clearing conditions.

    Ensures that:
    - Cournot: Total quantity doesn't exceed demand at zero price AND firms don't produce at losses
    - Bertrand: Prices are above marginal costs

    Args:
        model: "cournot" or "bertrand"
        actions: Firm actions to validate
        costs: Firm marginal costs
        params: Market parameters

    Returns:
        Validated and adjusted actions
    """
    if model == "cournot":
        # Check if segmented demand
        segments_config = params.get("segments")
        if segments_config:
            # For segmented demand, calculate effective parameters
            weighted_alpha = sum(
                segment["weight"] * segment["alpha"] for segment in segments_config
            )
            weighted_beta = sum(
                segment["weight"] * segment["beta"] for segment in segments_config
            )
            max_total_qty = weighted_alpha / weighted_beta
            a = weighted_alpha
            b = weighted_beta
        else:
            # Standard demand
            a = params.get("a", 100.0)
            b = params.get("b", 1.0)
            max_total_qty = a / b

        total_qty = sum(actions)

        if total_qty > max_total_qty:
            # Scale down all quantities proportionally
            scale_factor = max_total_qty / total_qty
            actions = [qty * scale_factor for qty in actions]

        # Calculate market price and ensure profitable production
        total_quantity = sum(actions)
        price = max(0.0, a - b * total_quantity)

        # Ensure minimum viable price (prevent zero prices)
        min_viable_price = min(costs) + 0.1  # Small margin above lowest cost
        if price < min_viable_price and total_quantity > 0:
            # Adjust quantities to achieve minimum viable price
            target_quantity = (a - min_viable_price) / b
            if target_quantity > 0:
                scale_factor = target_quantity / total_quantity
                actions = [qty * scale_factor for qty in actions]
                price = min_viable_price

        # Remove unprofitable firms
        actions = validate_profitable_production(actions, costs, price)

    else:  # bertrand
        # Ensure all prices are above marginal costs
        actions = [max(cost + 0.1, price) for price, cost in zip(actions, costs)]

    return actions


def validate_economic_parameters(
    model: str, params: Dict[str, Any], costs: List[float]
) -> None:
    """Validate that economic parameters create a viable market.

    Args:
        model: "cournot" or "bertrand"
        params: Market parameters
        costs: Firm marginal costs

    Raises:
        ValueError: If parameters create economically impossible conditions
    """
    if model == "cournot":
        # Check if segmented demand
        segments_config = params.get("segments")
        if segments_config:
            # For segmented demand, calculate effective parameters
            weighted_alpha = sum(
                segment["weight"] * segment["alpha"] for segment in segments_config
            )
            weighted_beta = sum(
                segment["weight"] * segment["beta"] for segment in segments_config
            )
            a = weighted_alpha
            b = weighted_beta
        else:
            a = params.get("a", 100.0)
            b = params.get("b", 1.0)

        # Check if any firm can be profitable
        min_cost = min(costs)
        if min_cost >= a:
            raise ValueError(
                f"All firm costs ({costs}) exceed demand intercept ({a}). "
                "No firm can be profitable in this market."
            )

        # Check if demand is too flat (low b)
        if b < 0.1:
            raise ValueError(
                f"Demand slope ({b}) is too flat. This creates unrealistic market conditions."
            )

    else:  # bertrand
        alpha = params.get("alpha", 100.0)
        beta = params.get("beta", 1.0)

        # Check if any firm can be profitable
        min_cost = min(costs)
        if min_cost >= alpha:
            raise ValueError(
                f"All firm costs ({costs}) exceed demand intercept ({alpha}). "
                "No firm can be profitable in this market."
            )

        # Check if demand is too flat (low beta)
        if beta < 0.1:
            raise ValueError(
                f"Demand slope ({beta}) is too flat. This creates unrealistic market conditions."
            )
