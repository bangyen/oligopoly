"""Additional tests for API endpoints module to improve coverage."""

from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.sim.api_endpoints import (
    generate_heatmap,
    get_run_replay,
    get_statistics,
    health_check,
)


class TestGetRunReplay:
    """Test the get_run_replay endpoint."""

    @pytest.mark.asyncio
    async def test_get_run_replay_success(self) -> None:
        """Test successful retrieval of run replay."""
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

        # Mock query result for Event (empty list to avoid iteration issues)
        mock_event_query = Mock()
        mock_event_query.filter.return_value.order_by.return_value.all.return_value = []

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            else:  # Event model
                return mock_event_query

        mock_db.query.side_effect = mock_query_side_effect

        result = await get_run_replay("test-run", mock_db)

        # Check result
        assert isinstance(result, dict)
        assert "run_id" in result
        assert "model" in result
        assert "rounds" in result
        assert "frames" in result

    @pytest.mark.asyncio
    async def test_get_run_replay_not_found(self) -> None:
        """Test get_run_replay when run is not found."""
        mock_db = Mock(spec=Session)

        # Mock query result returning None
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db.query.return_value = mock_query

        with pytest.raises(HTTPException) as exc_info:
            await get_run_replay("nonexistent-run", mock_db)

        assert exc_info.value.status_code == 404
        assert "Run not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_run_replay_database_error(self) -> None:
        """Test get_run_replay with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await get_run_replay("test-run", mock_db)

        assert exc_info.value.status_code == 500
        assert "Failed to get run replay" in str(exc_info.value.detail)


class TestGetStatistics:
    """Test the get_statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_statistics_success(self) -> None:
        """Test successful retrieval of statistics."""
        mock_db = Mock(spec=Session)

        # Mock query results
        mock_run_query = Mock()
        mock_run_query.count.return_value = 10

        mock_event_query = Mock()
        mock_event_query.count.return_value = 1000

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            else:  # Event model
                return mock_event_query

        mock_db.query.side_effect = mock_query_side_effect

        result = await get_statistics(mock_db)

        # Check result
        assert isinstance(result, dict)
        assert "total_runs" in result
        assert "total_events" in result
        assert result["total_runs"] == 10
        assert result["total_events"] == 1000

    @pytest.mark.asyncio
    async def test_get_statistics_database_error(self) -> None:
        """Test get_statistics with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await get_statistics(mock_db)

        assert exc_info.value.status_code == 500
        assert "Failed to get statistics" in str(exc_info.value.detail)


class TestHealthCheck:
    """Test the health_check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        """Test successful health check."""
        mock_db = Mock(spec=Session)

        # Mock successful database connection
        mock_db.execute.return_value = None

        result = await health_check(mock_db)

        # Check result
        assert isinstance(result, dict)
        assert "status" in result
        assert "checks" in result
        assert result["status"] == "healthy"
        assert "database" in result["checks"]

    @pytest.mark.asyncio
    async def test_health_check_database_error(self) -> None:
        """Test health check with database error."""
        mock_db = Mock(spec=Session)
        mock_db.execute.side_effect = Exception("Database connection failed")

        result = await health_check(mock_db)

        # Check result
        assert isinstance(result, dict)
        assert "status" in result
        assert "checks" in result
        assert result["status"] == "unhealthy"
        assert "database" in result["checks"]


class TestGenerateHeatmap:
    """Test the generate_heatmap endpoint."""

    @pytest.mark.asyncio
    async def test_generate_heatmap_missing_parameters(self) -> None:
        """Test generate_heatmap with missing parameters."""
        mock_db = Mock(spec=Session)

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap("cournot", 0, 1, [10.0, 15.0], 20, None, mock_db)

        assert exc_info.value.status_code == 500
        assert "Missing required Cournot parameters" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_not_found(self) -> None:
        """Test generate_heatmap when run is not found."""
        mock_db = Mock(spec=Session)

        # Mock query result returning None
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db.query.return_value = mock_query

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap("cournot", 0, 1, [10.0, 15.0], 20, None, mock_db)

        assert exc_info.value.status_code == 500
        assert "Missing required Cournot parameters" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_generate_heatmap_database_error(self) -> None:
        """Test generate_heatmap with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await generate_heatmap("cournot", 0, 1, [10.0, 15.0], 20, None, mock_db)

        assert exc_info.value.status_code == 500
        assert "Missing required Cournot parameters" in str(exc_info.value.detail)
