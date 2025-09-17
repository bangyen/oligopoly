"""Tests for Bertrand heatmap computation module."""

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.sim.heatmap.bertrand_heatmap import (
    compute_bertrand_heatmap,
    compute_bertrand_segmented_heatmap,
    create_price_grid,
)
from src.sim.models.models import DemandSegment, SegmentedDemand


class TestCreatePriceGrid:
    """Test the create_price_grid function."""

    def test_create_price_grid_basic(self) -> None:
        """Test basic price grid creation."""
        grid = create_price_grid(0.0, 10.0, 5)

        expected = [0.0, 2.5, 5.0, 7.5, 10.0]
        assert len(grid) == 5
        for i, (actual, exp) in enumerate(zip(grid, expected)):
            assert math.isclose(
                actual, exp, abs_tol=1e-10
            ), f"Index {i}: {actual} != {exp}"

    def test_create_price_grid_single_point(self) -> None:
        """Test price grid with single point."""
        # For single point, min and max should be different
        grid = create_price_grid(5.0, 5.1, 1)

        assert len(grid) == 1
        assert math.isclose(grid[0], 5.0, abs_tol=1e-10)

    def test_create_price_grid_negative_values(self) -> None:
        """Test price grid with negative values."""
        grid = create_price_grid(-10.0, 10.0, 3)

        expected = [-10.0, 0.0, 10.0]
        assert len(grid) == 3
        for actual, exp in zip(grid, expected):
            assert math.isclose(actual, exp, abs_tol=1e-10)

    def test_create_price_grid_invalid_min_max(self) -> None:
        """Test that min_p >= max_p raises ValueError."""
        with pytest.raises(
            ValueError, match="min_p \\(5\\.0\\) must be less than max_p \\(5\\.0\\)"
        ):
            create_price_grid(5.0, 5.0, 3)

        with pytest.raises(
            ValueError, match="min_p \\(10\\.0\\) must be less than max_p \\(5\\.0\\)"
        ):
            create_price_grid(10.0, 5.0, 3)

    def test_create_price_grid_invalid_num_points(self) -> None:
        """Test that num_points <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_points \\(0\\) must be positive"):
            create_price_grid(0.0, 10.0, 0)

        with pytest.raises(ValueError, match="num_points \\(-1\\) must be positive"):
            create_price_grid(0.0, 10.0, -1)


class TestComputeBertrandHeatmap:
    """Test the compute_bertrand_heatmap function."""

    def test_compute_bertrand_heatmap_basic(self) -> None:
        """Test basic Bertrand heatmap computation."""
        alpha = 100.0
        beta = 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i = 0
        firm_j = 1
        p_i_grid = [12.0, 14.0, 16.0]
        p_j_grid = [13.0, 15.0, 17.0]
        other_prices: list[float] = [18.0]  # For firm 2

        with patch("src.sim.heatmap.bertrand_heatmap.bertrand_simulation") as mock_sim:
            # Mock simulation result
            mock_result = Mock()
            mock_result.profits = [5.0, 3.0, 1.0]
            mock_result.quantities = [2.0, 1.5, 0.5]
            mock_result.total_demand = 4.0
            mock_sim.return_value = mock_result

            profit_matrix, market_share_matrix, p_i_out, p_j_out = (
                compute_bertrand_heatmap(
                    alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_prices
                )
            )

            # Check output shapes
            assert profit_matrix.shape == (3, 3)
            assert market_share_matrix.shape == (3, 3)
            assert p_i_out == p_i_grid
            assert p_j_out == p_j_grid

            # Check that simulation was called for each combination
            assert mock_sim.call_count == 9  # 3x3 grid

            # Check that all profit values are the same (mocked)
            assert np.all(profit_matrix == 5.0)
            # Check market share calculation: 2.0 / 4.0 = 0.5
            assert np.all(market_share_matrix == 0.5)

    def test_compute_bertrand_heatmap_zero_demand(self) -> None:
        """Test heatmap computation when total demand is zero."""
        alpha = 100.0
        beta = 1.0
        costs = [10.0, 15.0]
        firm_i = 0
        firm_j = 1
        p_i_grid = [12.0]
        p_j_grid = [13.0]
        other_prices: list[float] = []

        with patch("src.sim.heatmap.bertrand_heatmap.bertrand_simulation") as mock_sim:
            # Mock simulation result with zero demand
            mock_result = Mock()
            mock_result.profits = [0.0, 0.0]
            mock_result.quantities = [0.0, 0.0]
            mock_result.total_demand = 0.0
            mock_sim.return_value = mock_result

            profit_matrix, market_share_matrix, _, _ = compute_bertrand_heatmap(
                alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_prices
            )

            # Market share should be 0 when total demand is 0
            assert market_share_matrix[0, 0] == 0.0

    def test_compute_bertrand_heatmap_invalid_firm_i(self) -> None:
        """Test that invalid firm_i raises ValueError."""
        costs = [10.0, 15.0]

        with pytest.raises(
            ValueError,
            match="firm_i \\(-1\\) must be less than number of firms \\(2\\)",
        ):
            compute_bertrand_heatmap(100.0, 1.0, costs, -1, 1, [12.0], [13.0], [])

        with pytest.raises(
            ValueError, match="firm_i \\(2\\) must be less than number of firms \\(2\\)"
        ):
            compute_bertrand_heatmap(100.0, 1.0, costs, 2, 1, [12.0], [13.0], [])

    def test_compute_bertrand_heatmap_invalid_firm_j(self) -> None:
        """Test that invalid firm_j raises ValueError."""
        costs = [10.0, 15.0]

        with pytest.raises(
            ValueError,
            match="firm_j \\(-1\\) must be less than number of firms \\(2\\)",
        ):
            compute_bertrand_heatmap(100.0, 1.0, costs, 0, -1, [12.0], [13.0], [])

        with pytest.raises(
            ValueError, match="firm_j \\(2\\) must be less than number of firms \\(2\\)"
        ):
            compute_bertrand_heatmap(100.0, 1.0, costs, 0, 2, [12.0], [13.0], [])

    def test_compute_bertrand_heatmap_same_firms(self) -> None:
        """Test that firm_i == firm_j raises ValueError."""
        costs = [10.0, 15.0]

        with pytest.raises(ValueError, match="firm_i and firm_j must be different"):
            compute_bertrand_heatmap(100.0, 1.0, costs, 0, 0, [12.0], [13.0], [])

    def test_compute_bertrand_heatmap_invalid_other_prices_length(self) -> None:
        """Test that wrong other_prices length raises ValueError."""
        costs = [10.0, 15.0, 20.0, 25.0]  # 4 firms

        with pytest.raises(
            ValueError,
            match="other_prices length \\(1\\) must equal number of firms - 2 \\(2\\)",
        ):
            compute_bertrand_heatmap(
                100.0,
                1.0,
                costs,
                0,
                1,
                [12.0],
                [13.0],
                [18.0],  # Should have 2 other prices
            )

    def test_compute_bertrand_heatmap_price_construction(self) -> None:
        """Test that price vector is constructed correctly."""
        alpha = 100.0
        beta = 1.0
        costs = [10.0, 15.0, 20.0, 25.0]  # 4 firms
        firm_i = 1
        firm_j = 3
        p_i_grid = [12.0]
        p_j_grid = [13.0]
        other_prices: list[float] = [18.0, 22.0]  # For firms 0 and 2

        with patch("src.sim.heatmap.bertrand_heatmap.bertrand_simulation") as mock_sim:
            mock_result = Mock()
            mock_result.profits = [0.0, 0.0, 0.0, 0.0]
            mock_result.quantities = [0.0, 0.0, 0.0, 0.0]
            mock_result.total_demand = 0.0
            mock_sim.return_value = mock_result

            compute_bertrand_heatmap(
                alpha, beta, costs, firm_i, firm_j, p_i_grid, p_j_grid, other_prices
            )

            # Check that simulation was called with correct price vector
            call_args = mock_sim.call_args[0]
            prices = call_args[3]  # prices is the 4th argument

            # Expected: [18.0, 12.0, 22.0, 13.0] (firms 0, 1, 2, 3)
            expected_prices = [18.0, 12.0, 22.0, 13.0]
            assert prices == expected_prices


class TestComputeBertrandSegmentedHeatmap:
    """Test the compute_bertrand_segmented_heatmap function."""

    def test_compute_bertrand_segmented_heatmap_basic(self) -> None:
        """Test basic segmented Bertrand heatmap computation."""
        # Create a simple segmented demand
        segmented_demand = SegmentedDemand(
            segments=[
                DemandSegment(alpha=50.0, beta=1.0, weight=0.5),
                DemandSegment(alpha=100.0, beta=1.0, weight=0.5),
            ]
        )
        costs = [10.0, 15.0, 20.0]
        firm_i = 0
        firm_j = 1
        p_i_grid = [12.0, 14.0]
        p_j_grid = [13.0, 15.0]
        other_prices = [18.0]

        with patch(
            "src.sim.heatmap.bertrand_heatmap.bertrand_segmented_simulation"
        ) as mock_sim:
            # Mock simulation result
            mock_result = Mock()
            mock_result.profits = [5.0, 3.0, 1.0]
            mock_result.quantities = [2.0, 1.5, 0.5]
            mock_result.total_demand = 4.0
            mock_sim.return_value = mock_result

            profit_matrix, market_share_matrix, p_i_out, p_j_out = (
                compute_bertrand_segmented_heatmap(
                    segmented_demand,
                    costs,
                    firm_i,
                    firm_j,
                    p_i_grid,
                    p_j_grid,
                    other_prices,
                )
            )

            # Check output shapes
            assert profit_matrix.shape == (2, 2)
            assert market_share_matrix.shape == (2, 2)
            assert p_i_out == p_i_grid
            assert p_j_out == p_j_grid

            # Check that simulation was called for each combination
            assert mock_sim.call_count == 4  # 2x2 grid

            # Check that all profit values are the same (mocked)
            assert np.all(profit_matrix == 5.0)
            # Check market share calculation: 2.0 / 4.0 = 0.5
            assert np.all(market_share_matrix == 0.5)

    def test_compute_bertrand_segmented_heatmap_validation(self) -> None:
        """Test that segmented heatmap has same validation as regular heatmap."""
        segmented_demand = SegmentedDemand(
            segments=[DemandSegment(alpha=50.0, beta=1.0, weight=1.0)]
        )
        costs = [10.0, 15.0]

        # Test invalid firm_i
        with pytest.raises(
            ValueError,
            match="firm_i \\(-1\\) must be less than number of firms \\(2\\)",
        ):
            compute_bertrand_segmented_heatmap(
                segmented_demand, costs, -1, 1, [12.0], [13.0], []
            )

        # Test invalid firm_j
        with pytest.raises(
            ValueError, match="firm_j \\(2\\) must be less than number of firms \\(2\\)"
        ):
            compute_bertrand_segmented_heatmap(
                segmented_demand, costs, 0, 2, [12.0], [13.0], []
            )

        # Test same firms
        with pytest.raises(ValueError, match="firm_i and firm_j must be different"):
            compute_bertrand_segmented_heatmap(
                segmented_demand, costs, 0, 0, [12.0], [13.0], []
            )

    def test_compute_bertrand_segmented_heatmap_zero_demand(self) -> None:
        """Test segmented heatmap computation when total demand is zero."""
        segmented_demand = SegmentedDemand(
            segments=[DemandSegment(alpha=50.0, beta=1.0, weight=1.0)]
        )
        costs = [10.0, 15.0]
        firm_i = 0
        firm_j = 1
        p_i_grid = [12.0]
        p_j_grid = [13.0]
        other_prices: list[float] = []

        with patch(
            "src.sim.heatmap.bertrand_heatmap.bertrand_segmented_simulation"
        ) as mock_sim:
            # Mock simulation result with zero demand
            mock_result = Mock()
            mock_result.profits = [0.0, 0.0]
            mock_result.quantities = [0.0, 0.0]
            mock_result.total_demand = 0.0
            mock_sim.return_value = mock_result

            profit_matrix, market_share_matrix, _, _ = (
                compute_bertrand_segmented_heatmap(
                    segmented_demand,
                    costs,
                    firm_i,
                    firm_j,
                    p_i_grid,
                    p_j_grid,
                    other_prices,
                )
            )

            # Market share should be 0 when total demand is 0
            assert market_share_matrix[0, 0] == 0.0
