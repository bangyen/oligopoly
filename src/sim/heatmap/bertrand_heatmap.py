"""Bertrand heatmap computation for price strategy spaces.

This module computes 2D profit and market share surfaces by sweeping over price grids
for two firms while holding other firms' prices fixed at specified values.
"""

from typing import List, Tuple

import numpy as np

from ..games.bertrand import bertrand_segmented_simulation, bertrand_simulation
from ..models.models import SegmentedDemand


def compute_bertrand_heatmap(
    alpha: float,
    beta: float,
    costs: List[float],
    firm_i: int,
    firm_j: int,
    p_i_grid: List[float],
    p_j_grid: List[float],
    other_prices: List[float],
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """Compute profit and market share heatmaps for Bertrand competition between two firms.

    Sweeps over price grids for firms i and j while holding other firms' prices
    fixed. Computes profit surface and market share surface for firm i across all combinations.

    Args:
        alpha: Intercept parameter for demand curve Q(p) = max(0, α - β*p)
        beta: Slope parameter for demand curve
        costs: List of marginal costs for each firm
        firm_i: Index of firm to compute surfaces for
        firm_j: Index of second firm in the heatmap
        p_i_grid: Grid of prices for firm i
        p_j_grid: Grid of prices for firm j
        other_prices: Fixed prices for all other firms

    Returns:
        Tuple of (profit_matrix, market_share_matrix, p_i_grid, p_j_grid) where:
        - profit_matrix[i,j] is the profit of firm_i when firm_i chooses p_i_grid[i] and firm_j chooses p_j_grid[j]
        - market_share_matrix[i,j] is the market share of firm_i in the same scenario

    Raises:
        ValueError: If firm indices are invalid or grid sizes don't match
    """
    # Validate inputs
    if firm_i < 0 or firm_i >= len(costs):
        raise ValueError(
            f"400: firm_i ({firm_i}) must be less than number of firms ({len(costs)})"
        )
    if firm_j < 0 or firm_j >= len(costs):
        raise ValueError(
            f"400: firm_j ({firm_j}) must be less than number of firms ({len(costs)})"
        )
    if firm_i == firm_j:
        raise ValueError("400: firm_i and firm_j must be different")

    if len(other_prices) != len(costs) - 2:
        raise ValueError(
            f"400: other_prices length ({len(other_prices)}) must equal "
            f"number of firms - 2 ({len(costs) - 2})"
        )

    # Create profit and market share matrices
    profit_matrix = np.zeros((len(p_i_grid), len(p_j_grid)))
    market_share_matrix = np.zeros((len(p_i_grid), len(p_j_grid)))

    # Sweep over price combinations
    for i, p_i in enumerate(p_i_grid):
        for j, p_j in enumerate(p_j_grid):
            # Construct full price vector
            prices = [0.0] * len(costs)

            # Insert prices for firms i and j
            prices[firm_i] = p_i
            prices[firm_j] = p_j

            # Insert fixed prices for other firms
            other_idx = 0
            for k in range(len(costs)):
                if k != firm_i and k != firm_j:
                    prices[k] = other_prices[other_idx]
                    other_idx += 1

            # Run Bertrand simulation
            result = bertrand_simulation(alpha, beta, costs, prices)

            # Store profit and market share for firm i
            profit_matrix[i, j] = result.profits[firm_i]

            # Calculate market share as quantity / total_demand
            if result.total_demand > 0:
                market_share_matrix[i, j] = (
                    result.quantities[firm_i] / result.total_demand
                )
            else:
                market_share_matrix[i, j] = 0.0

    return profit_matrix, market_share_matrix, p_i_grid, p_j_grid


def compute_bertrand_segmented_heatmap(
    segmented_demand: SegmentedDemand,
    costs: List[float],
    firm_i: int,
    firm_j: int,
    p_i_grid: List[float],
    p_j_grid: List[float],
    other_prices: List[float],
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """Compute profit and market share heatmaps for Bertrand competition with segmented demand.

    Sweeps over price grids for firms i and j while holding other firms' prices
    fixed. Computes profit surface and market share surface for firm i across all combinations.

    Args:
        segmented_demand: SegmentedDemand object with segment configurations
        costs: List of marginal costs for each firm
        firm_i: Index of firm to compute surfaces for
        firm_j: Index of second firm in the heatmap
        p_i_grid: Grid of prices for firm i
        p_j_grid: Grid of prices for firm j
        other_prices: Fixed prices for all other firms

    Returns:
        Tuple of (profit_matrix, market_share_matrix, p_i_grid, p_j_grid) where:
        - profit_matrix[i,j] is the profit of firm_i when firm_i chooses p_i_grid[i] and firm_j chooses p_j_grid[j]
        - market_share_matrix[i,j] is the market share of firm_i in the same scenario

    Raises:
        ValueError: If firm indices are invalid or grid sizes don't match
    """
    # Validate inputs
    if firm_i < 0 or firm_i >= len(costs):
        raise ValueError(
            f"400: firm_i ({firm_i}) must be less than number of firms ({len(costs)})"
        )
    if firm_j < 0 or firm_j >= len(costs):
        raise ValueError(
            f"400: firm_j ({firm_j}) must be less than number of firms ({len(costs)})"
        )
    if firm_i == firm_j:
        raise ValueError("400: firm_i and firm_j must be different")

    if len(other_prices) != len(costs) - 2:
        raise ValueError(
            f"400: other_prices length ({len(other_prices)}) must equal "
            f"number of firms - 2 ({len(costs) - 2})"
        )

    # Create profit and market share matrices
    profit_matrix = np.zeros((len(p_i_grid), len(p_j_grid)))
    market_share_matrix = np.zeros((len(p_i_grid), len(p_j_grid)))

    # Sweep over price combinations
    for i, p_i in enumerate(p_i_grid):
        for j, p_j in enumerate(p_j_grid):
            # Construct full price vector
            prices = [0.0] * len(costs)

            # Insert prices for firms i and j
            prices[firm_i] = p_i
            prices[firm_j] = p_j

            # Insert fixed prices for other firms
            other_idx = 0
            for k in range(len(costs)):
                if k != firm_i and k != firm_j:
                    prices[k] = other_prices[other_idx]
                    other_idx += 1

            # Run Bertrand segmented simulation
            result = bertrand_segmented_simulation(segmented_demand, costs, prices)

            # Store profit and market share for firm i
            profit_matrix[i, j] = result.profits[firm_i]

            # Calculate market share as quantity / total_demand
            if result.total_demand > 0:
                market_share_matrix[i, j] = (
                    result.quantities[firm_i] / result.total_demand
                )
            else:
                market_share_matrix[i, j] = 0.0

    return profit_matrix, market_share_matrix, p_i_grid, p_j_grid


def create_price_grid(min_p: float, max_p: float, num_points: int) -> List[float]:
    """Create a linear grid of price values.

    Args:
        min_p: Minimum price value
        max_p: Maximum price value
        num_points: Number of grid points

    Returns:
        List of price values

    Raises:
        ValueError: If min_p >= max_p or num_points <= 0
    """
    if min_p >= max_p:
        raise ValueError(f"min_p ({min_p}) must be less than max_p ({max_p})")
    if num_points <= 0:
        raise ValueError(f"num_points ({num_points}) must be positive")

    return list(np.linspace(min_p, max_p, num_points))
