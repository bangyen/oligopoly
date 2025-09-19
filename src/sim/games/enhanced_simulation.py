"""Enhanced oligopoly simulation with advanced economic features.

This module provides enhanced simulation functions that support:
- Capacity constraints
- Fixed costs
- Economies of scale
- Non-linear demand functions
- Enhanced profit calculations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..models.models import CostStructure, IsoelasticDemand
from .bertrand import BertrandResult, bertrand_simulation
from .cournot import CournotResult, cournot_simulation


@dataclass
class EnhancedSimulationConfig:
    """Configuration for enhanced simulation with advanced economic features."""

    demand_type: str = "linear"  # "linear" or "isoelastic"
    demand_params: Dict[str, Any] = field(
        default_factory=dict
    )  # Parameters for demand function
    cost_structures: List[CostStructure] = field(
        default_factory=list
    )  # Enhanced cost structures
    capacity_limits: Optional[List[float]] = None
    fixed_costs: Optional[List[float]] = None


def enhanced_cournot_simulation(
    config: EnhancedSimulationConfig, quantities: List[float]
) -> CournotResult:
    """Enhanced Cournot simulation with advanced economic features.

    Args:
        config: Enhanced simulation configuration
        quantities: List of quantities chosen by each firm

    Returns:
        CournotResult with enhanced profit calculations
    """
    if config.demand_type == "linear":
        a = config.demand_params.get("a", 100.0)
        b = config.demand_params.get("b", 1.0)

        # Extract cost information
        costs = [cs.marginal_cost for cs in config.cost_structures]
        fixed_costs = [cs.fixed_cost for cs in config.cost_structures]
        capacity_limits = [cs.capacity_limit for cs in config.cost_structures]

        # Filter out None values for capacity limits
        capacity_limits_filtered: Optional[List[float]] = [
            cap for cap in capacity_limits if cap is not None
        ]
        if capacity_limits_filtered is not None and len(
            capacity_limits_filtered
        ) != len(capacity_limits):
            # If any capacity limits are None, don't pass the parameter
            capacity_limits_filtered = None

        return cournot_simulation(
            a=a,
            b=b,
            costs=costs,
            quantities=quantities,
            fixed_costs=fixed_costs,
            capacity_limits=capacity_limits_filtered,
        )

    elif config.demand_type == "isoelastic":
        a_param = config.demand_params.get("A", 100.0)
        elasticity = config.demand_params.get("elasticity", 2.0)

        # Create isoelastic demand
        demand = IsoelasticDemand(A=a_param, elasticity=elasticity)

        # Calculate market price
        total_quantity = sum(quantities)
        price = demand.price(total_quantity)

        # Apply capacity constraints
        if config.capacity_limits:
            quantities = [
                min(qty, cap) for qty, cap in zip(quantities, config.capacity_limits)
            ]

        # Calculate profits with enhanced cost structure
        profits = []
        for i, (qty, cost_struct) in enumerate(zip(quantities, config.cost_structures)):
            if qty > 0:
                # Calculate profit: π = P * q - total_cost(q)
                profit = price * qty - cost_struct.total_cost(qty)
            else:
                # If not producing, only pay fixed costs
                profit = -cost_struct.fixed_cost
            profits.append(profit)

        return CournotResult(price=price, quantities=quantities, profits=profits)

    else:
        raise ValueError(f"Unsupported demand type: {config.demand_type}")


def enhanced_bertrand_simulation(
    config: EnhancedSimulationConfig, prices: List[float]
) -> BertrandResult:
    """Enhanced Bertrand simulation with advanced economic features.

    Args:
        config: Enhanced simulation configuration
        prices: List of prices chosen by each firm

    Returns:
        BertrandResult with enhanced profit calculations
    """
    if config.demand_type == "linear":
        alpha = config.demand_params.get("alpha", 100.0)
        beta = config.demand_params.get("beta", 1.0)

        # Extract cost information
        costs = [cs.marginal_cost for cs in config.cost_structures]
        fixed_costs = [cs.fixed_cost for cs in config.cost_structures]
        capacity_limits = [cs.capacity_limit for cs in config.cost_structures]

        # Filter out None values for capacity limits
        capacity_limits_filtered: Optional[List[float]] = [
            cap for cap in capacity_limits if cap is not None
        ]
        if capacity_limits_filtered is not None and len(
            capacity_limits_filtered
        ) != len(capacity_limits):
            # If any capacity limits are None, don't pass the parameter
            capacity_limits_filtered = None

        return bertrand_simulation(
            alpha=alpha,
            beta=beta,
            costs=costs,
            prices=prices,
            fixed_costs=fixed_costs,
            capacity_limits=capacity_limits_filtered,
        )

    elif config.demand_type == "isoelastic":
        a_param = config.demand_params.get("A", 100.0)
        elasticity = config.demand_params.get("elasticity", 2.0)

        # Find the lowest price (Bertrand competition)
        min_price = min(prices)
        winning_firms = [i for i, p in enumerate(prices) if p == min_price]

        # Calculate total demand at the winning price
        # For isoelastic demand: Q = A * P^(-ε)
        if min_price > 0:
            total_demand = a_param * (min_price ** (-elasticity))
        else:
            total_demand = 0.0

        # Allocate demand among winning firms
        quantities = [0.0] * len(prices)
        if total_demand > 0 and winning_firms:
            qty_per_firm = total_demand / len(winning_firms)
            for i in winning_firms:
                quantities[i] = qty_per_firm

        # Apply capacity constraints
        if config.capacity_limits:
            quantities = [
                min(qty, cap) for qty, cap in zip(quantities, config.capacity_limits)
            ]

        # Calculate profits with enhanced cost structure
        profits = []
        for i, (price, qty, cost_struct) in enumerate(
            zip(prices, quantities, config.cost_structures)
        ):
            if qty > 0:
                # Calculate profit: π = P * q - total_cost(q)
                profit = price * qty - cost_struct.total_cost(qty)
            else:
                # If not producing, only pay fixed costs
                profit = -cost_struct.fixed_cost
            profits.append(profit)

        return BertrandResult(
            total_demand=sum(quantities),
            prices=prices.copy(),
            quantities=quantities,
            profits=profits,
        )

    else:
        raise ValueError(f"Unsupported demand type: {config.demand_type}")


def validate_enhanced_economic_parameters(config: EnhancedSimulationConfig) -> None:
    """Validate enhanced economic parameters for economic realism.

    Args:
        config: Enhanced simulation configuration

    Raises:
        ValueError: If parameters create economically impossible conditions
    """
    if not config.cost_structures:
        raise ValueError("Cost structures must be provided")

    # Validate cost structures
    for i, cost_struct in enumerate(config.cost_structures):
        if cost_struct.marginal_cost <= 0:
            raise ValueError(
                f"Firm {i} marginal cost must be positive, got {cost_struct.marginal_cost}"
            )
        if cost_struct.fixed_cost < 0:
            raise ValueError(
                f"Firm {i} fixed cost must be non-negative, got {cost_struct.fixed_cost}"
            )
        if cost_struct.capacity_limit is not None and cost_struct.capacity_limit <= 0:
            raise ValueError(
                f"Firm {i} capacity limit must be positive, got {cost_struct.capacity_limit}"
            )

    # Validate demand parameters
    if config.demand_type == "linear":
        a = config.demand_params.get("a", 100.0)
        b = config.demand_params.get("b", 1.0)

        if b <= 0:
            raise ValueError(f"Demand slope must be positive, got {b}")

        # Check if any firm can be profitable
        min_marginal_cost = min(cs.marginal_cost for cs in config.cost_structures)
        if min_marginal_cost >= a:
            raise ValueError(
                f"All firm marginal costs exceed demand intercept ({a}). "
                "No firm can be profitable in this market."
            )

    elif config.demand_type == "isoelastic":
        a_param = config.demand_params.get("A", 100.0)
        elasticity = config.demand_params.get("elasticity", 2.0)

        if a_param <= 0:
            raise ValueError(f"Demand scale parameter must be positive, got {a_param}")
        if elasticity <= 1.0:
            raise ValueError(
                f"Elasticity must be > 1 for economic realism, got {elasticity}"
            )

    else:
        raise ValueError(f"Unsupported demand type: {config.demand_type}")
