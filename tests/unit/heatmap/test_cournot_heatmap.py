"""Tests for Cournot heatmap computation module."""

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.sim.heatmap.cournot_heatmap import (
    compute_cournot_heatmap,
    compute_cournot_segmented_heatmap,
    create_quantity_grid,
)
from src.sim.models.models import DemandSegment, SegmentedDemand


class TestCreateQuantityGrid:
    """Test the create_quantity_grid function."""

    def test_create_quantity_grid_basic(self) -> None:
        """Test basic quantity grid creation."""
        grid = create_quantity_grid(0.0, 10.0, 5)

        expected = [0.0, 2.5, 5.0, 7.5, 10.0]
        assert len(grid) == 5
        for i, (actual, exp) in enumerate(zip(grid, expected)):
            assert math.isclose(actual, exp, abs_tol=1e-10), (
                f"Index {i}: {actual} != {exp}"
            )

    def test_create_quantity_grid_single_point(self) -> None:
        """Test quantity grid with single point."""
        # For single point, min and max should be different
        grid = create_quantity_grid(5.0, 5.1, 1)

        assert len(grid) == 1
        assert math.isclose(grid[0], 5.0, abs_tol=1e-10)

    def test_create_quantity_grid_negative_values(self) -> None:
        """Test quantity grid with negative values."""
        grid = create_quantity_grid(-10.0, 10.0, 3)

        expected = [-10.0, 0.0, 10.0]
        assert len(grid) == 3
        for actual, exp in zip(grid, expected):
            assert math.isclose(actual, exp, abs_tol=1e-10)

    def test_create_quantity_grid_invalid_min_max(self) -> None:
        """Test that min_q >= max_q raises ValueError."""
        with pytest.raises(
            ValueError, match="min_q \\(5\\.0\\) must be less than max_q \\(5\\.0\\)"
        ):
            create_quantity_grid(5.0, 5.0, 3)

        with pytest.raises(
            ValueError, match="min_q \\(10\\.0\\) must be less than max_q \\(5\\.0\\)"
        ):
            create_quantity_grid(10.0, 5.0, 3)

    def test_create_quantity_grid_invalid_num_points(self) -> None:
        """Test that num_points <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="num_points \\(0\\) must be positive"):
            create_quantity_grid(0.0, 10.0, 0)

        with pytest.raises(ValueError, match="num_points \\(-1\\) must be positive"):
            create_quantity_grid(0.0, 10.0, -1)


class TestComputeCournotHeatmap:
    """Test the compute_cournot_heatmap function."""

    def test_compute_cournot_heatmap_basic(self) -> None:
        """Test basic Cournot heatmap computation."""
        a = 100.0
        b = 1.0
        costs = [10.0, 15.0, 20.0]
        firm_i = 0
        firm_j = 1
        q_i_grid = [5.0, 10.0, 15.0]
        q_j_grid = [6.0, 12.0, 18.0]
        other_quantities = [8.0]  # For firm 2

        with patch("src.sim.heatmap.cournot_heatmap.cournot_simulation") as mock_sim:
            # Mock simulation result
            mock_result = Mock()
            mock_result.profits = [25.0, 15.0, 5.0]
            mock_sim.return_value = mock_result

            profit_matrix, q_i_out, q_j_out = compute_cournot_heatmap(
                a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
            )

            # Check output shapes
            assert profit_matrix.shape == (3, 3)
            assert q_i_out == q_i_grid
            assert q_j_out == q_j_grid

            # Check that simulation was called for each combination
            assert mock_sim.call_count == 9  # 3x3 grid

            # Check that all profit values are the same (mocked)
            assert np.all(profit_matrix == 25.0)

    def test_compute_cournot_heatmap_invalid_firm_i(self) -> None:
        """Test that invalid firm_i raises ValueError."""
        costs = [10.0, 15.0]

        with pytest.raises(
            ValueError,
            match="firm_i \\(-1\\) must be less than number of firms \\(2\\)",
        ):
            compute_cournot_heatmap(100.0, 1.0, costs, -1, 1, [5.0], [6.0], [])

        with pytest.raises(
            ValueError, match="firm_i \\(2\\) must be less than number of firms \\(2\\)"
        ):
            compute_cournot_heatmap(100.0, 1.0, costs, 2, 1, [5.0], [6.0], [])

    def test_compute_cournot_heatmap_invalid_firm_j(self) -> None:
        """Test that invalid firm_j raises ValueError."""
        costs = [10.0, 15.0]

        with pytest.raises(
            ValueError,
            match="firm_j \\(-1\\) must be less than number of firms \\(2\\)",
        ):
            compute_cournot_heatmap(100.0, 1.0, costs, 0, -1, [5.0], [6.0], [])

        with pytest.raises(
            ValueError, match="firm_j \\(2\\) must be less than number of firms \\(2\\)"
        ):
            compute_cournot_heatmap(100.0, 1.0, costs, 0, 2, [5.0], [6.0], [])

    def test_compute_cournot_heatmap_same_firms(self) -> None:
        """Test that firm_i == firm_j raises ValueError."""
        costs = [10.0, 15.0]

        with pytest.raises(ValueError, match="firm_i and firm_j must be different"):
            compute_cournot_heatmap(100.0, 1.0, costs, 0, 0, [5.0], [6.0], [])

    def test_compute_cournot_heatmap_invalid_other_quantities_length(self) -> None:
        """Test that wrong other_quantities length raises ValueError."""
        costs = [10.0, 15.0, 20.0, 25.0]  # 4 firms

        with pytest.raises(
            ValueError,
            match="other_quantities length \\(1\\) must equal number of firms - 2 \\(2\\)",
        ):
            compute_cournot_heatmap(
                100.0,
                1.0,
                costs,
                0,
                1,
                [5.0],
                [6.0],
                [8.0],  # Should have 2 other quantities
            )

    def test_compute_cournot_heatmap_quantity_construction(self) -> None:
        """Test that quantity vector is constructed correctly."""
        a = 100.0
        b = 1.0
        costs = [10.0, 15.0, 20.0, 25.0]  # 4 firms
        firm_i = 1
        firm_j = 3
        q_i_grid = [5.0]
        q_j_grid = [6.0]
        other_quantities = [8.0, 10.0]  # For firms 0 and 2

        with patch("src.sim.heatmap.cournot_heatmap.cournot_simulation") as mock_sim:
            mock_result = Mock()
            mock_result.profits = [0.0, 0.0, 0.0, 0.0]
            mock_sim.return_value = mock_result

            compute_cournot_heatmap(
                a, b, costs, firm_i, firm_j, q_i_grid, q_j_grid, other_quantities
            )

            # Check that simulation was called with correct quantity vector
            call_args = mock_sim.call_args[0]
            quantities = call_args[3]  # quantities is the 4th argument

            # Expected: [8.0, 5.0, 10.0, 6.0] (firms 0, 1, 2, 3)
            expected_quantities = [8.0, 5.0, 10.0, 6.0]
            assert quantities == expected_quantities


class TestComputeCournotSegmentedHeatmap:
    """Test the compute_cournot_segmented_heatmap function."""

    def test_compute_cournot_segmented_heatmap_basic(self) -> None:
        """Test basic segmented Cournot heatmap computation."""
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
        q_i_grid = [5.0, 10.0]
        q_j_grid = [6.0, 12.0]
        other_quantities = [8.0]

        with patch(
            "src.sim.heatmap.cournot_heatmap.cournot_segmented_simulation"
        ) as mock_sim:
            # Mock simulation result
            mock_result = Mock()
            mock_result.profits = [25.0, 15.0, 5.0]
            mock_sim.return_value = mock_result

            profit_matrix, q_i_out, q_j_out = compute_cournot_segmented_heatmap(
                segmented_demand,
                costs,
                firm_i,
                firm_j,
                q_i_grid,
                q_j_grid,
                other_quantities,
            )

            # Check output shapes
            assert profit_matrix.shape == (2, 2)
            assert q_i_out == q_i_grid
            assert q_j_out == q_j_grid

            # Check that simulation was called for each combination
            assert mock_sim.call_count == 4  # 2x2 grid

            # Check that all profit values are the same (mocked)
            assert np.all(profit_matrix == 25.0)

    def test_compute_cournot_segmented_heatmap_validation(self) -> None:
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
            compute_cournot_segmented_heatmap(
                segmented_demand, costs, -1, 1, [5.0], [6.0], []
            )

        # Test invalid firm_j
        with pytest.raises(
            ValueError, match="firm_j \\(2\\) must be less than number of firms \\(2\\)"
        ):
            compute_cournot_segmented_heatmap(
                segmented_demand, costs, 0, 2, [5.0], [6.0], []
            )

        # Test same firms
        with pytest.raises(ValueError, match="firm_i and firm_j must be different"):
            compute_cournot_segmented_heatmap(
                segmented_demand, costs, 0, 0, [5.0], [6.0], []
            )

    def test_compute_cournot_segmented_heatmap_quantity_construction(self) -> None:
        """Test that segmented heatmap constructs quantity vector correctly."""
        segmented_demand = SegmentedDemand(
            segments=[DemandSegment(alpha=50.0, beta=1.0, weight=1.0)]
        )
        costs = [10.0, 15.0, 20.0, 25.0]  # 4 firms
        firm_i = 1
        firm_j = 3
        q_i_grid = [5.0]
        q_j_grid = [6.0]
        other_quantities = [8.0, 10.0]  # For firms 0 and 2

        with patch(
            "src.sim.heatmap.cournot_heatmap.cournot_segmented_simulation"
        ) as mock_sim:
            mock_result = Mock()
            mock_result.profits = [0.0, 0.0, 0.0, 0.0]
            mock_sim.return_value = mock_result

            compute_cournot_segmented_heatmap(
                segmented_demand,
                costs,
                firm_i,
                firm_j,
                q_i_grid,
                q_j_grid,
                other_quantities,
            )

            # Check that simulation was called with correct quantity vector
            call_args = mock_sim.call_args[0]
            quantities = call_args[2]  # quantities is the 3rd argument

            # Expected: [8.0, 5.0, 10.0, 6.0] (firms 0, 1, 2, 3)
            expected_quantities = [8.0, 5.0, 10.0, 6.0]
            assert quantities == expected_quantities
