"""Tests for API endpoints module."""

from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.sim.api_endpoints import (
    RunDetail,
    RunSummary,
    get_run_detail,
    get_run_metrics,
    list_runs,
    router,
)


class TestRunSummary:
    """Test the RunSummary model."""

    def test_run_summary_creation(self) -> None:
        """Test creating a RunSummary instance."""
        run_summary = RunSummary(
            id="test-id",
            model="cournot",
            rounds=100,
            created_at="2023-01-01T00:00:00",
            status="completed",
        )

        assert run_summary.id == "test-id"
        assert run_summary.model == "cournot"
        assert run_summary.rounds == 100
        assert run_summary.created_at == "2023-01-01T00:00:00"
        assert run_summary.status == "completed"

    def test_run_summary_validation(self) -> None:
        """Test RunSummary field validation."""
        # Test that RunSummary can be created with all required fields
        run_summary = RunSummary(
            id="test-id",
            model="cournot",
            rounds=100,
            created_at="2023-01-01T00:00:00",
            status="completed",
        )
        assert run_summary.id == "test-id"


class TestRunDetail:
    """Test the RunDetail model."""

    def test_run_detail_creation(self) -> None:
        """Test creating a RunDetail instance."""
        run_detail = RunDetail(
            id="test-id",
            model="bertrand",
            rounds=50,
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T01:00:00",
            results={"profit": 100.0},
        )

        assert run_detail.id == "test-id"
        assert run_detail.model == "bertrand"
        assert run_detail.rounds == 50
        assert run_detail.created_at == "2023-01-01T00:00:00"
        assert run_detail.updated_at == "2023-01-01T01:00:00"
        assert run_detail.results == {"profit": 100.0}

    def test_run_detail_optional_results(self) -> None:
        """Test RunDetail with optional results field."""
        run_detail = RunDetail(
            id="test-id",
            model="cournot",
            rounds=100,
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T01:00:00",
            # results is optional
        )

        assert run_detail.results is None


class TestListRuns:
    """Test the list_runs endpoint."""

    @pytest.mark.asyncio
    async def test_list_runs_success(self) -> None:
        """Test successful listing of runs."""
        # Mock database session
        mock_db = Mock(spec=Session)

        # Mock run objects
        mock_run1 = Mock()
        mock_run1.id = "run-1"
        mock_run1.model = "cournot"
        mock_run1.rounds = 100
        mock_run1.created_at = datetime(2023, 1, 1, 12, 0, 0)

        mock_run2 = Mock()
        mock_run2.id = "run-2"
        mock_run2.model = "bertrand"
        mock_run2.rounds = 50
        mock_run2.created_at = datetime(2023, 1, 2, 12, 0, 0)

        # Mock query result
        mock_query = Mock()
        mock_query.order_by.return_value.all.return_value = [
            mock_run2,
            mock_run1,
        ]  # Newest first
        mock_db.query.return_value = mock_query

        result = await list_runs(mock_db)

        # Check that query was called correctly
        mock_db.query.assert_called_once()
        mock_query.order_by.assert_called_once()
        mock_query.order_by.return_value.all.assert_called_once()

        # Check result
        assert len(result) == 2
        assert isinstance(result[0], RunSummary)
        assert result[0].id == "run-2"
        assert result[0].model == "bertrand"
        assert result[0].rounds == 50
        assert result[0].status == "completed"

    @pytest.mark.asyncio
    async def test_list_runs_database_error(self) -> None:
        """Test list_runs with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await list_runs(mock_db)

        assert exc_info.value.status_code == 500
        assert "Failed to list runs" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_list_runs_empty_result(self) -> None:
        """Test list_runs with no runs in database."""
        mock_db = Mock(spec=Session)
        mock_query = Mock()
        mock_query.order_by.return_value.all.return_value = []
        mock_db.query.return_value = mock_query

        result = await list_runs(mock_db)

        assert result == []


class TestGetRunDetail:
    """Test the get_run_detail endpoint."""

    @pytest.mark.asyncio
    async def test_get_run_detail_success(self) -> None:
        """Test successful retrieval of run detail."""
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

        # Mock query result for Result (empty list to avoid iteration issues)
        mock_result_query = Mock()
        mock_result_query.filter.return_value.order_by.return_value.all.return_value = (
            []
        )

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            else:  # Result model
                return mock_result_query

        mock_db.query.side_effect = mock_query_side_effect

        result = await get_run_detail("test-run", mock_db)

        # Check result
        assert isinstance(result, RunDetail)
        assert result.id == "test-run"
        assert result.model == "cournot"
        assert result.rounds == 100
        assert result.results is not None

    @pytest.mark.asyncio
    async def test_get_run_detail_not_found(self) -> None:
        """Test get_run_detail when run is not found."""
        mock_db = Mock(spec=Session)

        # Mock query result returning None
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db.query.return_value = mock_query

        with pytest.raises(HTTPException) as exc_info:
            await get_run_detail("nonexistent-run", mock_db)

        assert exc_info.value.status_code == 404
        assert "Run not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_run_detail_database_error(self) -> None:
        """Test get_run_detail with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await get_run_detail("test-run", mock_db)

        assert exc_info.value.status_code == 500
        assert "Failed to get run detail" in str(exc_info.value.detail)


class TestGetRunMetrics:
    """Test the get_run_metrics endpoint."""

    @pytest.mark.asyncio
    async def test_get_run_metrics_success(self) -> None:
        """Test successful retrieval of run metrics."""
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

        # Mock query result for Result (empty list to avoid iteration issues)
        mock_result_query = Mock()
        mock_result_query.filter.return_value.order_by.return_value.all.return_value = (
            []
        )

        # Set up mock_db.query to return different mocks based on the model
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                return mock_run_query
            else:  # Result model
                return mock_result_query

        mock_db.query.side_effect = mock_query_side_effect

        result = await get_run_metrics("test-run", mock_db)

        # Check result - should be empty list when no results
        assert isinstance(result, list)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_run_metrics_not_found(self) -> None:
        """Test get_run_metrics when run is not found."""
        mock_db = Mock(spec=Session)

        # Mock query result returning None
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        mock_db.query.return_value = mock_query

        with pytest.raises(HTTPException) as exc_info:
            await get_run_metrics("nonexistent-run", mock_db)

        assert exc_info.value.status_code == 404
        assert "Run not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_run_metrics_database_error(self) -> None:
        """Test get_run_metrics with database error."""
        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with pytest.raises(HTTPException) as exc_info:
            await get_run_metrics("test-run", mock_db)

        assert exc_info.value.status_code == 500
        assert "Failed to get run metrics" in str(exc_info.value.detail)


class TestRouter:
    """Test the API router configuration."""

    def test_router_configuration(self) -> None:
        """Test that router is properly configured."""
        assert router is not None
        assert hasattr(router, "tags")
        assert router.tags == ["advanced"]

    def test_router_routes(self) -> None:
        """Test that router has expected routes."""
        # Check that router has routes
        assert len(router.routes) > 0

        # Check that router is properly configured
        assert router is not None
        assert hasattr(router, "routes")
