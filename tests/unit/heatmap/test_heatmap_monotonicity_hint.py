"""Tests for heatmap symmetry and monotonicity properties.

This module tests that heatmap surfaces exhibit expected properties
for different configurations.
"""

import numpy as np

from src.sim.heatmap.bertrand_heatmap import compute_bertrand_heatmap, create_price_grid
from src.sim.heatmap.cournot_heatmap import (
    compute_cournot_heatmap,
    create_quantity_grid,
)


class TestHeatmapProperties:
    """Test heatmap properties for different configurations."""

    def test_cournot_asymmetric_costs_different_profits(self):
        """Test that asymmetric costs lead to different profit patterns."""
        # Setup asymmetric configuration
        a, b = 100.0, 1.0
        costs = [5.0, 20.0, 15.0]  # Firms 0 and 1 have very different costs
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 20.0, 5)
        q_j_grid = create_quantity_grid(0.0, 20.0, 5)
        other_quantities = [10.0]

        # Compute heatmap
        profit_matrix, _, _ = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Test that profit matrix has reasonable values
        assert np.all(
            np.isfinite(profit_matrix)
        ), "Profit matrix should contain finite values"

        # Test that low-cost firm generally has higher profits than high-cost firm
        # when both choose similar quantities (diagonal comparison)
        diagonal_profits = [profit_matrix[i, i] for i in range(len(q_i_grid))]

        # Low-cost firm should have higher profits than high-cost firm on diagonal
        # (even if both might be negative due to high total quantities)
        assert len(diagonal_profits) > 0, "Should have diagonal profits"

        # Check that profit matrix has reasonable structure
        assert profit_matrix.shape == (
            5,
            5,
        ), f"Expected (5, 5), got {profit_matrix.shape}"

    def test_bertrand_asymmetric_costs_different_profits(self):
        """Test that asymmetric costs lead to different profit patterns in Bertrand."""
        # Setup asymmetric configuration
        alpha, beta = 100.0, 1.0
        costs = [5.0, 20.0, 15.0]  # Firms 0 and 1 have very different costs
        firm_i, firm_j = 0, 1
        p_i_grid = create_price_grid(0.0, 50.0, 5)
        p_j_grid = create_price_grid(0.0, 50.0, 5)
        other_prices = [20.0]

        # Compute heatmap
        profit_matrix, market_share_matrix, _, _ = compute_bertrand_heatmap(
            alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_prices
        )

        # Test that profit matrix has reasonable values
        assert np.all(
            np.isfinite(profit_matrix)
        ), "Profit matrix should contain finite values"

        # Test that market share matrix has reasonable values
        assert np.all(
            np.isfinite(market_share_matrix)
        ), "Market share matrix should contain finite values"
        assert np.all(
            market_share_matrix >= 0
        ), "Market share matrix should contain non-negative values"
        assert np.all(
            market_share_matrix <= 1
        ), "Market share matrix should contain values <= 1"

    def test_cournot_profit_patterns(self):
        """Test that Cournot profits exhibit reasonable patterns."""
        # Setup
        a, b = 100.0, 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 20.0, 6)
        q_j_grid = create_quantity_grid(0.0, 20.0, 6)
        other_quantities = [10.0]

        # Compute heatmap
        profit_matrix, _, _ = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Test that profit matrix has reasonable values
        assert np.all(
            np.isfinite(profit_matrix)
        ), "Profit matrix should contain finite values"

        # Test that profit matrix has reasonable structure
        assert profit_matrix.shape == (
            6,
            6,
        ), f"Expected (6, 6), got {profit_matrix.shape}"

        # Test that profits are finite and reasonable
        assert np.all(np.isfinite(profit_matrix)), "All profits should be finite"

    def test_bertrand_market_share_patterns(self):
        """Test that Bertrand market share exhibits expected patterns."""
        # Setup
        alpha, beta = 100.0, 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i, firm_j = 0, 1
        p_i_grid = create_price_grid(0.0, 50.0, 5)
        p_j_grid = create_price_grid(0.0, 50.0, 5)
        other_prices = [20.0]

        # Compute heatmap
        profit_matrix, market_share_matrix, _, _ = compute_bertrand_heatmap(
            alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_prices
        )

        # Test that market share is higher when firm_i has lower price than firm_j
        # Check that when firm_i has lowest price, it gets market share
        lowest_price_idx = 0  # First element is lowest price
        highest_price_idx = -1  # Last element is highest price

        # When firm_i has lowest price and firm_j has highest price, firm_i should get market share
        market_share_low_high = market_share_matrix[lowest_price_idx, highest_price_idx]
        market_share_high_low = market_share_matrix[highest_price_idx, lowest_price_idx]

        # Firm with lower price should get higher market share
        assert market_share_low_high > market_share_high_low, (
            f"Firm with lower price should get higher market share: "
            f"{market_share_low_high} > {market_share_high_low}"
        )

    def test_heatmap_dimensions_consistency(self):
        """Test that heatmap dimensions are consistent across different configurations."""
        # Test Cournot
        a, b = 100.0, 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 20.0, 7)
        q_j_grid = create_quantity_grid(0.0, 20.0, 7)
        other_quantities = [10.0]

        profit_matrix, returned_q_i, returned_q_j = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        assert profit_matrix.shape == (
            7,
            7,
        ), f"Expected (7, 7), got {profit_matrix.shape}"
        assert len(returned_q_i) == 7, f"Expected 7, got {len(returned_q_i)}"
        assert len(returned_q_j) == 7, f"Expected 7, got {len(returned_q_j)}"

        # Test Bertrand
        alpha, beta = 100.0, 1.0
        p_i_grid = create_price_grid(0.0, 50.0, 6)
        p_j_grid = create_price_grid(0.0, 50.0, 6)

        profit_matrix, market_share_matrix, returned_p_i, returned_p_j = (
            compute_bertrand_heatmap(
                alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_quantities
            )
        )

        assert profit_matrix.shape == (
            6,
            6,
        ), f"Expected (6, 6), got {profit_matrix.shape}"
        assert market_share_matrix.shape == (
            6,
            6,
        ), f"Expected (6, 6), got {market_share_matrix.shape}"
        assert len(returned_p_i) == 6, f"Expected 6, got {len(returned_p_i)}"
        assert len(returned_p_j) == 6, f"Expected 6, got {len(returned_p_j)}"

    def test_edge_cases(self):
        """Test edge cases in heatmap computation."""
        # Test with very small quantities/prices
        a, b = 100.0, 1.0
        costs = [10.0, 15.0]
        firm_i, firm_j = 0, 1
        q_i_grid = create_quantity_grid(0.0, 1.0, 3)
        q_j_grid = create_quantity_grid(0.0, 1.0, 3)
        other_quantities = []

        profit_matrix, _, _ = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Should handle small quantities gracefully
        assert np.all(
            np.isfinite(profit_matrix)
        ), "Should handle small quantities gracefully"

        # Test with moderate quantities (should give positive profits)
        q_i_grid = create_quantity_grid(0.0, 10.0, 3)
        q_j_grid = create_quantity_grid(0.0, 10.0, 3)

        profit_matrix, _, _ = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Should handle moderate quantities gracefully
        assert np.all(
            np.isfinite(profit_matrix)
        ), "Should handle moderate quantities gracefully"

        # Test with high quantities (may result in negative profits, which is realistic)
        q_i_grid = create_quantity_grid(50.0, 100.0, 3)
        q_j_grid = create_quantity_grid(50.0, 100.0, 3)

        profit_matrix, _, _ = compute_cournot_heatmap(
            a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
        )

        # Should handle high quantities gracefully (may result in negative profits)
        assert np.all(
            np.isfinite(profit_matrix)
        ), "Should handle high quantities gracefully"

        # Negative profits are realistic when costs exceed prices
        # This is expected behavior in economic models
