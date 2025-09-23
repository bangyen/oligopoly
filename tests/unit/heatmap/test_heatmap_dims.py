"""Tests for heatmap dimension validation.

This module tests that heatmap computations return arrays with correct dimensions
for different grid sizes and configurations.
"""

import numpy as np

from src.sim.heatmap.bertrand_heatmap import compute_bertrand_heatmap, create_price_grid
from src.sim.heatmap.cournot_heatmap import (
    compute_cournot_heatmap,
    create_quantity_grid,
)


class TestHeatmapDimensions:
    """Test heatmap dimension validation for different grid sizes."""

    def test_cournot_heatmap_dims_small_grid(self):
        """Test Cournot heatmap dimensions for small grid (5x5)."""
        # Setup
        a, b = 100.0, 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 20.0, 5)
        q_j_grid = create_quantity_grid(0.0, 20.0, 5)
        other_quantities = [10.0]

        # Compute heatmap
        profit_matrix, returned_q_i, returned_q_j = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Validate dimensions
        assert profit_matrix.shape == (
            5,
            5,
        ), f"Expected (5, 5), got {profit_matrix.shape}"
        assert len(returned_q_i) == 5, f"Expected 5, got {len(returned_q_i)}"
        assert len(returned_q_j) == 5, f"Expected 5, got {len(returned_q_j)}"

        # Validate grid values match
        assert np.allclose(returned_q_i, q_i_grid), "q_i_grid values don't match"
        assert np.allclose(returned_q_j, q_j_grid), "q_j_grid values don't match"

    def test_cournot_heatmap_dims_large_grid(self):
        """Test Cournot heatmap dimensions for large grid (20x20)."""
        # Setup
        a, b = 100.0, 1.0
        costs = [10.0, 15.0, 20.0, 25.0]
        firm_i, firm_j = 0, 2
        q_i_grid = create_quantity_grid(0.0, 30.0, 20)
        q_j_grid = create_quantity_grid(0.0, 30.0, 20)
        other_quantities = [15.0, 20.0]

        # Compute heatmap
        profit_matrix, returned_q_i, returned_q_j = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Validate dimensions
        assert profit_matrix.shape == (
            20,
            20,
        ), f"Expected (20, 20), got {profit_matrix.shape}"
        assert len(returned_q_i) == 20, f"Expected 20, got {len(returned_q_i)}"
        assert len(returned_q_j) == 20, f"Expected 20, got {len(returned_q_j)}"

    def test_bertrand_heatmap_dims_small_grid(self):
        """Test Bertrand heatmap dimensions for small grid (7x7)."""
        # Setup
        alpha, beta = 100.0, 1.0
        costs = [10.0, 15.0]
        firm_i, firm_j = 0, 1
        p_i_grid = create_price_grid(0.0, 50.0, 7)
        p_j_grid = create_price_grid(0.0, 50.0, 7)
        other_prices = []

        # Compute heatmap
        profit_matrix, market_share_matrix, returned_p_i, returned_p_j = (
            compute_bertrand_heatmap(
                alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_prices
            )
        )

        # Validate dimensions
        assert profit_matrix.shape == (
            7,
            7,
        ), f"Expected (7, 7), got {profit_matrix.shape}"
        assert market_share_matrix.shape == (
            7,
            7,
        ), f"Expected (7, 7), got {market_share_matrix.shape}"
        assert len(returned_p_i) == 7, f"Expected 7, got {len(returned_p_i)}"
        assert len(returned_p_j) == 7, f"Expected 7, got {len(returned_p_j)}"

    def test_bertrand_heatmap_dims_large_grid(self):
        """Test Bertrand heatmap dimensions for large grid (25x25)."""
        # Setup
        alpha, beta = 100.0, 1.0
        costs = [10.0, 15.0, 20.0, 25.0, 30.0]
        firm_i, firm_j = 1, 3
        p_i_grid = create_price_grid(0.0, 80.0, 25)
        p_j_grid = create_price_grid(0.0, 80.0, 25)
        other_prices = [20.0, 25.0, 30.0]

        # Compute heatmap
        profit_matrix, market_share_matrix, returned_p_i, returned_p_j = (
            compute_bertrand_heatmap(
                alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_prices
            )
        )

        # Validate dimensions
        assert profit_matrix.shape == (
            25,
            25,
        ), f"Expected (25, 25), got {profit_matrix.shape}"
        assert market_share_matrix.shape == (
            25,
            25,
        ), f"Expected (25, 25), got {market_share_matrix.shape}"
        assert len(returned_p_i) == 25, f"Expected 25, got {len(returned_p_i)}"
        assert len(returned_p_j) == 25, f"Expected 25, got {len(returned_p_j)}"

    def test_different_grid_sizes(self):
        """Test heatmap dimensions when grid sizes differ (M != N)."""
        # Setup Cournot with different grid sizes
        a, b = 100.0, 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 20.0, 8)  # 8 points
        q_j_grid = create_quantity_grid(0.0, 20.0, 12)  # 12 points
        other_quantities = [10.0]

        # Compute heatmap
        profit_matrix, returned_q_i, returned_q_j = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Validate dimensions
        assert profit_matrix.shape == (
            8,
            12,
        ), f"Expected (8, 12), got {profit_matrix.shape}"
        assert len(returned_q_i) == 8, f"Expected 8, got {len(returned_q_i)}"
        assert len(returned_q_j) == 12, f"Expected 12, got {len(returned_q_j)}"

    def test_edge_case_minimal_grid(self):
        """Test heatmap dimensions for minimal grid (2x2)."""
        # Setup
        a, b = 100.0, 1.0
        costs = [10.0, 15.0]
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 10.0, 2)
        q_j_grid = create_quantity_grid(0.0, 10.0, 2)
        other_quantities = []

        # Compute heatmap
        profit_matrix, returned_q_i, returned_q_j = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Validate dimensions
        assert profit_matrix.shape == (
            2,
            2,
        ), f"Expected (2, 2), got {profit_matrix.shape}"
        assert len(returned_q_i) == 2, f"Expected 2, got {len(returned_q_i)}"
        assert len(returned_q_j) == 2, f"Expected 2, got {len(returned_q_j)}"

    def test_grid_creation_dimensions(self):
        """Test that grid creation functions return correct dimensions."""
        # Test quantity grid
        q_grid = create_quantity_grid(0.0, 20.0, 10)
        assert len(q_grid) == 10, f"Expected 10, got {len(q_grid)}"
        assert q_grid[0] == 0.0, f"Expected 0.0, got {q_grid[0]}"
        assert q_grid[-1] == 20.0, f"Expected 20.0, got {q_grid[-1]}"

        # Test price grid
        p_grid = create_price_grid(0.0, 50.0, 15)
        assert len(p_grid) == 15, f"Expected 15, got {len(p_grid)}"
        assert p_grid[0] == 0.0, f"Expected 0.0, got {p_grid[0]}"
        assert p_grid[-1] == 50.0, f"Expected 50.0, got {p_grid[-1]}"

    def test_profit_matrix_values_valid(self):
        """Test that profit matrix contains valid (finite) values."""
        # Setup
        a, b = 100.0, 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 20.0, 5)
        q_j_grid = create_quantity_grid(0.0, 20.0, 5)
        other_quantities = [10.0]

        # Compute heatmap
        profit_matrix, _, _ = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Validate values
        assert np.all(np.isfinite(profit_matrix)), (
            "Profit matrix contains non-finite values"
        )
        assert np.all(profit_matrix >= 0), "Profit matrix contains negative values"

        # Check that matrix is not all zeros (should have some positive profits)
        assert np.any(profit_matrix > 0), "Profit matrix is all zeros"
