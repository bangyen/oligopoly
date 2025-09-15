"""Cournot heatmap computation for quantity strategy spaces.

This module computes 2D profit surfaces by sweeping over quantity grids for two firms
while holding other firms' quantities fixed at specified values.
"""

from typing import List, Tuple

import numpy as np

from ..games.cournot import cournot_segmented_simulation, cournot_simulation
from ..models.models import SegmentedDemand


def compute_cournot_heatmap(
    a: float,
    b: float,
    costs: List[float],
    firm_i: int,
    firm_j: int,
    q_i_grid: List[float],
    q_j_grid: List[float],
    other_quantities: List[float],
) -> Tuple[np.ndarray, List[float], List[float]]:
    """Compute profit heatmap for Cournot competition between two firms.

    Sweeps over quantity grids for firms i and j while holding other firms' quantities
    fixed. Computes profit surface for firm i across all combinations.

    Args:
        a: Maximum price parameter for demand curve P = max(0, a - b*Q)
        b: Price sensitivity parameter for demand curve
        costs: List of marginal costs for each firm
        firm_i: Index of firm to compute profit surface for
        firm_j: Index of second firm in the heatmap
        q_i_grid: Grid of quantities for firm i
        q_j_grid: Grid of quantities for firm j
        other_quantities: Fixed quantities for all other firms

    Returns:
        Tuple of (profit_matrix, q_i_grid, q_j_grid) where profit_matrix[i,j]
        is the profit of firm_i when firm_i chooses q_i_grid[i] and firm_j chooses q_j_grid[j]

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

    if len(other_quantities) != len(costs) - 2:
        raise ValueError(
            f"400: other_quantities length ({len(other_quantities)}) must equal "
            f"number of firms - 2 ({len(costs) - 2})"
        )

    # Create profit matrix
    profit_matrix = np.zeros((len(q_i_grid), len(q_j_grid)))

    # Sweep over quantity combinations
    for i, q_i in enumerate(q_i_grid):
        for j, q_j in enumerate(q_j_grid):
            # Construct full quantity vector
            quantities = [0.0] * len(costs)

            # Insert quantities for firms i and j
            quantities[firm_i] = q_i
            quantities[firm_j] = q_j

            # Insert fixed quantities for other firms
            other_idx = 0
            for k in range(len(costs)):
                if k != firm_i and k != firm_j:
                    quantities[k] = other_quantities[other_idx]
                    other_idx += 1

            # Run Cournot simulation
            result = cournot_simulation(a, b, costs, quantities)

            # Store profit for firm i
            profit_matrix[i, j] = result.profits[firm_i]

    return profit_matrix, q_i_grid, q_j_grid


def compute_cournot_segmented_heatmap(
    segmented_demand: SegmentedDemand,
    costs: List[float],
    firm_i: int,
    firm_j: int,
    q_i_grid: List[float],
    q_j_grid: List[float],
    other_quantities: List[float],
) -> Tuple[np.ndarray, List[float], List[float]]:
    """Compute profit heatmap for Cournot competition with segmented demand.

    Sweeps over quantity grids for firms i and j while holding other firms' quantities
    fixed. Computes profit surface for firm i across all combinations using segmented demand.

    Args:
        segmented_demand: SegmentedDemand object with segment configurations
        costs: List of marginal costs for each firm
        firm_i: Index of firm to compute profit surface for
        firm_j: Index of second firm in the heatmap
        q_i_grid: Grid of quantities for firm i
        q_j_grid: Grid of quantities for firm j
        other_quantities: Fixed quantities for all other firms

    Returns:
        Tuple of (profit_matrix, q_i_grid, q_j_grid) where profit_matrix[i,j]
        is the profit of firm_i when firm_i chooses q_i_grid[i] and firm_j chooses q_j_grid[j]

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

    if len(other_quantities) != len(costs) - 2:
        raise ValueError(
            f"400: other_quantities length ({len(other_quantities)}) must equal "
            f"number of firms - 2 ({len(costs) - 2})"
        )

    # Create profit matrix
    profit_matrix = np.zeros((len(q_i_grid), len(q_j_grid)))

    # Sweep over quantity combinations
    for i, q_i in enumerate(q_i_grid):
        for j, q_j in enumerate(q_j_grid):
            # Construct full quantity vector
            quantities = [0.0] * len(costs)

            # Insert quantities for firms i and j
            quantities[firm_i] = q_i
            quantities[firm_j] = q_j

            # Insert fixed quantities for other firms
            other_idx = 0
            for k in range(len(costs)):
                if k != firm_i and k != firm_j:
                    quantities[k] = other_quantities[other_idx]
                    other_idx += 1

            # Run Cournot segmented simulation
            result = cournot_segmented_simulation(segmented_demand, costs, quantities)

            # Store profit for firm i
            profit_matrix[i, j] = result.profits[firm_i]

    return profit_matrix, q_i_grid, q_j_grid


def create_quantity_grid(min_q: float, max_q: float, num_points: int) -> List[float]:
    """Create a linear grid of quantity values.

    Args:
        min_q: Minimum quantity value
        max_q: Maximum quantity value
        num_points: Number of grid points

    Returns:
        List of quantity values

    Raises:
        ValueError: If min_q >= max_q or num_points <= 0
    """
    if min_q >= max_q:
        raise ValueError(f"min_q ({min_q}) must be less than max_q ({max_q})")
    if num_points <= 0:
        raise ValueError(f"num_points ({num_points}) must be positive")

    return list(np.linspace(min_q, max_q, num_points))
