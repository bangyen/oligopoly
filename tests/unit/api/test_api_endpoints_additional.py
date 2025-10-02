"""Additional tests for api_endpoints.py to improve coverage.

This module tests additional edge cases, error handling, and validation
scenarios in the API endpoints module.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.sim.api_endpoints import (
    generate_heatmap,
    get_run_metrics,
    get_run_replay,
    get_statistics,
    health_check,
)


class TestGetRunMetricsAdditional:
    """Test additional scenarios for get_run_metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_run_metrics_with_results(self):
        """Test get_run_metrics with actual results data."""
        mock_db = Mock(spec=Session)

        # Mock run object
        mock_run = Mock()
        mock_run.id = "test-run"
        mock_run.model = "cournot"
        mock_run.rounds = 2
        mock_run.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_run.updated_at = datetime(2023, 1, 1, 13, 0, 0)

        # Mock query result for Run
        mock_run_query = Mock()
        mock_run_query.filter.return_value.first.return_value = mock_run

        # Mock results with actual data
        mock_results = {
            "rounds_data": [
                {"round": 0, "price": 50.0, "total_qty": 35.0, "total_profit": 1400.0},
                {"round": 1, "price": 51.0, "total_qty": 33.0, "total_profit": 1325.0},
            ],
            "firms_data": [
                {
                    "quantities": [20.0, 19.0],
                    "profits": [800.0, 779.0],
                    "actions": [20.0, 19.0],
                },
                {
                    "quantities": [15.0, 14.0],
                    "profits": [600.0, 546.0],
                    "actions": [15.0, 14.0],
                },
            ],
        }

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            else:  # Result model
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        with patch("src.sim.runners.runner.get_run_results") as mock_get_run_results:
            with patch("src.sim.models.metrics.calculate_hhi") as mock_calculate_hhi:
                mock_calculate_hhi.return_value = 0.5  # Mock HHI calculation
                mock_get_run_results.return_value = mock_results

                result = await get_run_metrics("test-run", mock_db)

            # Check result
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["round"] == 0
            assert result[0]["total_quantity"] == 35.0
            assert result[0]["total_profit"] == 1400.0
            assert result[0]["num_firms"] == 2

    @pytest.mark.asyncio
    async def test_get_run_metrics_results_not_found(self):
        """Test get_run_metrics when results are not found."""
        mock_db = Mock(spec=Session)

        # Mock run object
        mock_run = Mock()
        mock_run.id = "test-run"
        mock_run.model = "cournot"
        mock_run.rounds = 100
        mock_run.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_run.updated_at = datetime(2023, 1, 1, 13, 0, 0)

        # Mock query result for Run
        mock_run_query = Mock()
        mock_run_query.filter.return_value.first.return_value = mock_run

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            else:  # Result model
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        with patch("src.sim.runners.runner.get_run_results") as mock_get_run_results:
            mock_get_run_results.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_run_metrics("test-run", mock_db)

            assert exc_info.value.status_code == 404
            assert "Run results not found" in str(exc_info.value.detail)


class TestGetRunReplayAdditional:
    """Test additional scenarios for get_run_replay endpoint."""

    @pytest.mark.asyncio
    async def test_get_run_replay_with_results(self):
        """Test get_run_replay with actual results data."""
        mock_db = Mock(spec=Session)

        # Mock run object
        mock_run = Mock()
        mock_run.id = "test-run"
        mock_run.model = "cournot"
        mock_run.rounds = 2
        mock_run.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_run.updated_at = datetime(2023, 1, 1, 13, 0, 0)

        # Mock query result for Run
        mock_run_query = Mock()
        mock_run_query.filter.return_value.first.return_value = mock_run

        # Mock results with actual data
        mock_results = {
            "rounds_data": [
                {"round": 0, "price": 50.0, "total_qty": 35.0, "total_profit": 1400.0},
                {"round": 1, "price": 51.0, "total_qty": 33.0, "total_profit": 1325.0},
            ],
            "firms_data": [
                {
                    "quantities": [20.0, 19.0],
                    "profits": [800.0, 779.0],
                    "actions": [20.0, 19.0],
                },
                {
                    "quantities": [15.0, 14.0],
                    "profits": [600.0, 546.0],
                    "actions": [15.0, 14.0],
                },
            ],
        }

        # Mock events
        mock_event = Mock()
        mock_event.event_type = "collusion"
        mock_event.round_idx = 0
        mock_event.firm_id = 0
        mock_event.value = 100.0
        mock_event.description = "Test event"

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            elif model.__name__ == "Event":
                mock_event_query = Mock()
                mock_event_query.filter.return_value.all.return_value = [mock_event]
                return mock_event_query
            else:  # Result model
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        with patch("src.sim.runners.runner.get_run_results") as mock_get_run_results:
            mock_get_run_results.return_value = mock_results

            result = await get_run_replay("test-run", mock_db)

            # Check result
            assert isinstance(result, dict)
            assert "run_id" in result
            assert "model" in result
            assert "rounds" in result
            assert "frames" in result
            assert len(result["frames"]) == 2
            assert result["frames"][0]["round"] == 0
            assert result["frames"][0]["price"] == 50.0
            assert len(result["frames"][0]["firms"]) == 2
            assert len(result["frames"][0]["events"]) == 1

    @pytest.mark.asyncio
    async def test_get_run_replay_results_not_found(self):
        """Test get_run_replay when results are not found."""
        mock_db = Mock(spec=Session)

        # Mock run object
        mock_run = Mock()
        mock_run.id = "test-run"
        mock_run.model = "cournot"
        mock_run.rounds = 100
        mock_run.created_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_run.updated_at = datetime(2023, 1, 1, 13, 0, 0)

        # Mock query result for Run
        mock_run_query = Mock()
        mock_run_query.filter.return_value.first.return_value = mock_run

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            else:  # Result model
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        with patch("src.sim.runners.runner.get_run_results") as mock_get_run_results:
            mock_get_run_results.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                await get_run_replay("test-run", mock_db)

            assert exc_info.value.status_code == 404
            assert "Run results not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_run_replay_database_error(self):
        """Test get_run_replay with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await get_run_replay("test-run", mock_db)

        assert exc_info.value.status_code == 500
        assert "Failed to get run replay" in str(exc_info.value.detail)


class TestGetStatisticsAdditional:
    """Test additional scenarios for get_statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self):
        """Test get_statistics with actual data."""
        mock_db = Mock(spec=Session)

        # Mock query results
        mock_run_query = Mock()
        mock_run_query.count.return_value = 5

        mock_event_query = Mock()
        mock_event_query.count.return_value = 25

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            elif model.__name__ == "Event":
                return mock_event_query
            else:
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        with patch("src.sim.api_endpoints.get_metrics_summary") as mock_get_metrics:
            mock_get_metrics.return_value = {
                "total_requests": 100,
                "successful_requests": 95,
                "failed_requests": 5,
                "success_rate": 0.95,
            }

            result = await get_statistics(mock_db)

            # Check result
            assert isinstance(result, dict)
            assert "total_runs" in result
            assert "total_events" in result
            assert "metrics" in result
            assert result["total_runs"] == 5
            assert result["total_events"] == 25
            assert result["metrics"]["total_requests"] == 100

    @pytest.mark.asyncio
    async def test_get_statistics_database_error(self):
        """Test get_statistics with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await get_statistics(mock_db)

        assert exc_info.value.status_code == 500
        assert "Failed to get statistics" in str(exc_info.value.detail)


class TestHealthCheckAdditional:
    """Test additional scenarios for health_check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test health_check with successful health status."""
        mock_db = Mock(spec=Session)

        # Mock health status
        mock_health = Mock()
        mock_health.status = "healthy"
        mock_health.timestamp = datetime(2023, 1, 1, 12, 0, 0)
        mock_health.uptime_seconds = 3600
        mock_health.version = "1.0.0"
        mock_health.checks = {"database": "ok", "memory": "ok"}

        with patch("src.sim.monitoring.get_health_status") as mock_get_health:
            mock_get_health.return_value = mock_health

            result = await health_check(mock_db)

            # Check result
            assert isinstance(result, dict)
            assert "status" in result
            assert "timestamp" in result
            assert "uptime_seconds" in result
            assert "version" in result
            assert "checks" in result
            assert result["status"] == "healthy"
            assert result["uptime_seconds"] == 3600
            assert result["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health_check with health check failure."""
        mock_db = Mock(spec=Session)

        with patch("src.sim.monitoring.get_health_status") as mock_get_health:
            mock_get_health.side_effect = Exception("Health check failed")

            result = await health_check(mock_db)

            # Check result - should return unhealthy status
            assert isinstance(result, dict)
            assert "status" in result
            assert "error" in result
            assert "timestamp" in result
            assert result["status"] == "unhealthy"
            assert "Health check failed" in result["error"]


class TestGenerateHeatmapAdditional:
    """Test additional scenarios for generate_heatmap endpoint."""

    @pytest.mark.asyncio
    async def test_generate_heatmap_cournot_success(self):
        """Test generate_heatmap for Cournot model successfully."""
        mock_db = Mock(spec=Session)

        with patch(
            "src.sim.heatmap.cournot_heatmap.compute_cournot_heatmap"
        ) as mock_compute:
            import numpy as np

            mock_compute.return_value = (
                np.array([[100.0, 90.0], [90.0, 80.0]]),  # profit_surface
                np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # action_i_grid
                np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # action_j_grid
            )

            result = await generate_heatmap(
                model="cournot",
                firm_i=0,
                firm_j=1,
                costs=[10.0, 12.0],
                grid_size=20,
                params={"a": 100.0, "b": 1.0},
                db=mock_db,
            )

            # Check result
            assert isinstance(result, dict)
            assert "profit_surface" in result
            assert "market_share_surface" in result
            assert "action_i_grid" in result
            assert "action_j_grid" in result
            assert "model" in result
            assert "firm_i" in result
            assert "firm_j" in result
            assert result["model"] == "cournot"
            assert result["firm_i"] == 0
            assert result["firm_j"] == 1
            assert (
                result["market_share_surface"] is None
            )  # Cournot doesn't have market share

    @pytest.mark.asyncio
    async def test_generate_heatmap_bertrand_success(self):
        """Test generate_heatmap for Bertrand model successfully."""
        mock_db = Mock(spec=Session)

        with patch(
            "src.sim.heatmap.bertrand_heatmap.compute_bertrand_heatmap"
        ) as mock_compute:
            import numpy as np

            mock_compute.return_value = (
                np.array([[100.0, 90.0], [90.0, 80.0]]),  # profit_surface
                np.array([[0.5, 0.4], [0.4, 0.3]]),  # market_share_surface
                np.array([20.0, 30.0, 40.0, 50.0, 60.0]),  # action_i_grid
                np.array([20.0, 30.0, 40.0, 50.0, 60.0]),  # action_j_grid
            )

            result = await generate_heatmap(
                model="bertrand",
                firm_i=0,
                firm_j=1,
                costs=[10.0, 12.0],
                grid_size=20,
                params={"alpha": 200.0, "beta": 2.0},
                db=mock_db,
            )

            # Check result
            assert isinstance(result, dict)
            assert "profit_surface" in result
            assert "market_share_surface" in result
            assert "action_i_grid" in result
            assert "action_j_grid" in result
            assert "model" in result
            assert "firm_i" in result
            assert "firm_j" in result
            assert result["model"] == "bertrand"
            assert result["firm_i"] == 0
            assert result["firm_j"] == 1
            assert (
                result["market_share_surface"] is not None
            )  # Bertrand has market share

    @pytest.mark.asyncio
    async def test_generate_heatmap_insufficient_firms(self):
        """Test generate_heatmap with insufficient firms."""
        mock_db = Mock(spec=Session)

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap(
                model="cournot",
                firm_i=0,
                firm_j=2,  # firm_j=2 but only 2 firms (indices 0, 1)
                costs=[10.0, 12.0],
                grid_size=20,
                params={"a": 100.0, "b": 1.0},
                db=mock_db,
            )

        assert exc_info.value.status_code == 500
        assert "Not enough firms specified" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_small_grid_size(self):
        """Test generate_heatmap with grid size too small."""
        mock_db = Mock(spec=Session)

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap(
                model="cournot",
                firm_i=0,
                firm_j=1,
                costs=[10.0, 12.0],
                grid_size=4,  # Too small
                params={"a": 100.0, "b": 1.0},
                db=mock_db,
            )

        assert exc_info.value.status_code == 422
        assert "Grid size must be at least 5" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_missing_cournot_params(self):
        """Test generate_heatmap with missing Cournot parameters."""
        mock_db = Mock(spec=Session)

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap(
                model="cournot",
                firm_i=0,
                firm_j=1,
                costs=[10.0, 12.0],
                grid_size=20,
                params={"a": 100.0},  # Missing 'b'
                db=mock_db,
            )

        assert exc_info.value.status_code == 500
        assert "Missing required Cournot parameters" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_missing_bertrand_params(self):
        """Test generate_heatmap with missing Bertrand parameters."""
        mock_db = Mock(spec=Session)

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap(
                model="bertrand",
                firm_i=0,
                firm_j=1,
                costs=[10.0, 12.0],
                grid_size=20,
                params={"alpha": 200.0},  # Missing 'beta'
                db=mock_db,
            )

        assert exc_info.value.status_code == 500
        assert "Missing required Bertrand parameters" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_invalid_model(self):
        """Test generate_heatmap with invalid model."""
        mock_db = Mock(spec=Session)

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap(
                model="invalid",
                firm_i=0,
                firm_j=1,
                costs=[10.0, 12.0],
                grid_size=20,
                params={"a": 100.0, "b": 1.0},
                db=mock_db,
            )

        assert exc_info.value.status_code == 422
        assert "Model must be 'cournot' or 'bertrand'" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_computation_error(self):
        """Test generate_heatmap with computation error."""
        mock_db = Mock(spec=Session)

        with patch(
            "src.sim.heatmap.cournot_heatmap.compute_cournot_heatmap"
        ) as mock_compute:
            mock_compute.side_effect = Exception("Computation failed")

            with pytest.raises(HTTPException) as exc_info:
                await generate_heatmap(
                    model="cournot",
                    firm_i=0,
                    firm_j=1,
                    costs=[10.0, 12.0],
                    grid_size=20,
                    params={"a": 100.0, "b": 1.0},
                    db=mock_db,
                )

            assert exc_info.value.status_code == 500
            assert "Failed to generate heatmap" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_with_none_params(self):
        """Test generate_heatmap with None params."""
        mock_db = Mock(spec=Session)

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap(
                model="cournot",
                firm_i=0,
                firm_j=1,
                costs=[10.0, 12.0],
                grid_size=20,
                params=None,  # None params
                db=mock_db,
            )

        assert exc_info.value.status_code == 500
        assert "Missing required Cournot parameters: a, b" in str(exc_info.value.detail)
