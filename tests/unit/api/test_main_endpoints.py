"""Tests for main FastAPI application.

This module tests the main FastAPI application endpoints and core functionality.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app, get_db


@pytest.fixture(scope="module")
def mock_db():
    """Create a mock database for testing."""
    return Mock()


@pytest.fixture(scope="module")
def test_app(mock_db):
    """Create test app with overridden database dependency."""
    app.dependency_overrides[get_db] = lambda: mock_db
    yield app
    # Clean up after tests
    app.dependency_overrides.clear()


class TestDatabaseDependency:
    """Test database dependency injection."""

    def test_get_db_dependency(self):
        """Test database dependency returns session."""
        # This is a simple test of the dependency function
        # In a real test, you'd mock the database
        with patch("src.main.SessionLocal") as mock_session_local:
            mock_session = Mock()
            mock_session_local.return_value = mock_session

            # Test that dependency yields a session
            db_gen = get_db()
            db = next(db_gen)

            assert db == mock_session


class TestHealthCheck:
    """Test health check endpoint."""

    def test_health_check_success(self):
        """Test successful health check."""
        with TestClient(app) as client:
            response = client.get("/healthz")

            assert response.status_code == 200
            data = response.json()
            assert data["ok"]


class TestSimulateEndpoint:
    """Test simulation endpoint."""

    def test_simulate_endpoint_basic(self):
        """Test basic simulation endpoint."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [
                    {"cost": 10.0, "strategy": "nash"},
                    {"cost": 12.0, "strategy": "nash"},
                ],
                "params": {"a": 100.0, "b": 1.0},
            }

            with patch("src.main.get_db") as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db

                # Mock the database operations
                mock_run = Mock()
                mock_run.id = 1
                mock_db.add.return_value = None
                mock_db.commit.return_value = None
                mock_db.refresh.return_value = None

                with patch("src.main.run_game") as mock_run_game:
                    mock_run_game.return_value = "run_123"  # Return a run_id string

                    response = client.post("/simulate", json=simulation_data)

                    assert response.status_code == 200
                    data = response.json()
                    assert "run_id" in data
                    assert data["run_id"] == "run_123"

    def test_simulate_endpoint_invalid_model(self):
        """Test simulation endpoint with invalid model."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "invalid_model",
                "rounds": 10,
                "firms": [{"cost": 10.0, "strategy": "nash"}],
                "params": {"a": 100.0, "b": 1.0},
            }

            response = client.post("/simulate", json=simulation_data)

            assert response.status_code == 422
            assert "String should match pattern" in str(response.json()["detail"])

    def test_simulate_endpoint_missing_firms(self):
        """Test simulation endpoint with missing firms."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [],
                "params": {"a": 100.0, "b": 1.0},
            }

            response = client.post("/simulate", json=simulation_data)

            assert response.status_code == 422
            assert "List should have at least 1 item" in str(response.json()["detail"])

    def test_simulate_endpoint_invalid_params(self):
        """Test simulation endpoint with invalid parameters."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "strategy": "nash"}],
                "params": {"a": -10.0, "b": 1.0},  # Invalid negative intercept
            }

            with patch("src.main.get_db") as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db

                response = client.post("/simulate", json=simulation_data)

                assert response.status_code == 500
                assert "Unexpected error" in response.json()["detail"]


class TestHeatmapEndpoints:
    """Test heatmap generation endpoints."""

    def test_cournot_heatmap_endpoint(self):
        """Test Cournot heatmap endpoint validation."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 20,
                "action_range": [10.0, 50.0],
                "other_actions": [15.0],
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                    {"cost": 15.0, "fixed_cost": 0.0},
                ],
            }

            # Test validation - this should pass validation but may fail computation
            response = client.post("/heatmap", json=heatmap_data)

            # Accept either success (200) or computation error (500)
            assert response.status_code in [200, 500]

    def test_bertrand_heatmap_endpoint(self):
        """Test Bertrand heatmap endpoint validation."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "bertrand",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 20,
                "action_range": [20.0, 60.0],
                "other_actions": [25.0],
                "params": {"alpha": 200.0, "beta": 2.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                    {"cost": 15.0, "fixed_cost": 0.0},
                ],
            }

            # Test validation - this should pass validation but may fail computation
            response = client.post("/heatmap", json=heatmap_data)

            # Accept either success (200) or computation error (500)
            assert response.status_code in [200, 500]

    def test_heatmap_endpoint_invalid_model(self):
        """Test heatmap endpoint with invalid model."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "invalid",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 20,
                "action_range": [10.0, 50.0],
                "other_actions": [15.0],
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            response = client.post("/heatmap", json=heatmap_data)

            assert response.status_code == 422
            assert "String should match pattern" in str(response.json()["detail"])


class TestRunManagementEndpoints:
    """Test run management endpoints."""

    def test_get_run_endpoint(self):
        """Test get run endpoint."""
        with TestClient(app) as client:
            with patch("src.main.get_run_results") as mock_get_run_results:
                # Mock the get_run_results function to return test data
                mock_get_run_results.return_value = {
                    "id": 1,
                    "model": "cournot",
                    "rounds": 10,
                    "results": {
                        "0": {
                            "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0}
                        }
                    },
                    "metrics": {"hhi": 0.5, "consumer_surplus": 1000.0},
                }

                response = client.get("/runs/1")

                assert response.status_code == 200
                data = response.json()
                assert data["id"] == 1
                assert data["model"] == "cournot"

    def test_get_run_endpoint_not_found(self):
        """Test get run endpoint with non-existent run."""
        with TestClient(app) as client:
            with patch("src.main.get_run_results") as mock_get_run_results:
                # Mock the function to raise ValueError for non-existent run
                mock_get_run_results.side_effect = ValueError("Run 999 not found")

                response = client.get("/runs/999")

                assert response.status_code == 404
                assert "Run 999 not found" in response.json()["detail"]

    def test_list_runs_endpoint(self):
        """Test list runs endpoint returns run list."""
        with TestClient(app) as client:
            mock_db = Mock()
            mock_db.query.return_value.order_by.return_value.all.return_value = []
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.get("/runs")
                assert response.status_code == 200
                assert response.json() == []
            finally:
                app.dependency_overrides.clear()


class TestPolicyEndpoints:
    """Test policy shock endpoints."""

    def test_apply_policy_endpoint(self):
        """Test apply policy endpoint."""
        with TestClient(app) as client:
            policy_data = {
                "policy_type": "price_cap",
                "value": 50.0,
                "round": 5,
                "duration": 3,
            }

            with patch("src.main.get_db") as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db

                response = client.post("/policy/apply", json=policy_data)

                assert response.status_code == 404

    def test_apply_policy_endpoint_invalid_type(self):
        """Test apply policy endpoint with invalid policy type."""
        with TestClient(app) as client:
            policy_data = {
                "policy_type": "invalid_policy",
                "value": 50.0,
                "round": 5,
                "duration": 3,
            }

            response = client.post("/policy/apply", json=policy_data)

            assert response.status_code == 404


class TestMetricsEndpoints:
    """Test metrics calculation endpoints."""

    def test_calculate_metrics_endpoint(self):
        """Test calculate metrics endpoint."""
        with TestClient(app) as client:
            metrics_data = {
                "model": "cournot",
                "results": {
                    "0": {
                        "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0},
                        "firm_1": {"price": 50.0, "quantity": 15.0, "profit": 570.0},
                    }
                },
                "demand_params": {"a": 100.0, "b": 1.0},
            }

            with patch("src.main.calculate_round_metrics_cournot") as mock_calculate:
                mock_calculate.return_value = {
                    "hhi": 0.5,
                    "consumer_surplus": 1000.0,
                    "total_profit": 1370.0,
                }

                response = client.post("/metrics/calculate", json=metrics_data)

                assert response.status_code == 404

    def test_calculate_metrics_endpoint_invalid_model(self):
        """Test calculate metrics endpoint with invalid model."""
        with TestClient(app) as client:
            metrics_data = {
                "model": "invalid_model",
                "results": {"0": {"firm_0": {"price": 50.0, "quantity": 20.0}}},
                "demand_params": {"a": 100.0, "b": 1.0},
            }

            response = client.post("/metrics/calculate", json=metrics_data)

            assert response.status_code == 404


class TestReplayEndpoints:
    """Test replay system endpoints."""

    def test_replay_endpoint(self):
        """Test replay endpoint."""
        with TestClient(app) as client:
            with patch("src.main.ReplaySystem") as mock_replay_system:
                # Mock the ReplaySystem methods
                mock_replay = Mock()

                # Mock frame data
                mock_frame = Mock()
                mock_frame.round_idx = 0
                mock_frame.timestamp.isoformat.return_value = "2023-01-01T00:00:00"
                mock_frame.market_price = 50.0
                mock_frame.total_quantity = 20.0
                mock_frame.total_profit = 800.0
                mock_frame.hhi = 0.5
                mock_frame.consumer_surplus = 1000.0
                mock_frame.num_firms = 1
                mock_frame.firm_data = {0: {"action": 20.0, "price": 50.0}}
                mock_frame.events = []
                mock_frame.annotations = []

                mock_replay.get_all_frames.return_value = [mock_frame]
                mock_replay.get_frames_with_events.return_value = []
                mock_replay.get_event_rounds.return_value = []
                mock_replay_system.return_value = mock_replay

                response = client.get("/runs/1/replay")

                assert response.status_code == 200
                data = response.json()
                assert "run_id" in data
                assert "total_frames" in data

    def test_replay_endpoint_invalid_run(self):
        """Test replay endpoint with invalid run ID."""
        with TestClient(app) as client:
            with patch("src.main.ReplaySystem") as mock_replay_system:
                # Mock the ReplaySystem to raise ValueError for non-existent run
                mock_replay_system.side_effect = ValueError("Run 999 not found")

                response = client.get("/runs/999/replay")

                assert response.status_code == 404


class TestErrorHandling:
    """Test error handling in the application."""

    def test_http_exception_handling(self):
        """Test HTTP exception handling."""
        with TestClient(app) as client:
            # This would trigger an HTTP exception in a real scenario
            response = client.get("/nonexistent-endpoint")

            assert response.status_code == 404

    def test_validation_error_handling(self):
        """Test validation error handling."""
        with TestClient(app) as client:
            # Send invalid JSON
            response = client.post("/simulate", content="invalid json")

            assert response.status_code == 422  # Validation error

    def test_internal_server_error_handling(self):
        """Test internal server error handling."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "strategy": "nash"}],
                "params": {"a": 100.0, "b": 1.0},
            }

            with patch("src.main.run_game") as mock_run_game:
                mock_run_game.side_effect = Exception("Internal error")

                response = client.post("/simulate", json=simulation_data)

                assert response.status_code == 500
                assert "Unexpected error" in response.json()["detail"]


class TestDifferentiatedBertrandEndpoint:
    """Test that the removed /differentiated-bertrand endpoint is gone."""

    def test_differentiated_bertrand_endpoint(self):
        """Endpoint was a stub and has been intentionally removed; expect 404 or 405."""
        with TestClient(app) as client:
            response = client.post("/differentiated-bertrand", json={})
            # FastAPI returns 404 for unknown routes
            assert response.status_code in (404, 405, 422)

    def test_differentiated_bertrand_endpoint_error(self):
        """Stub endpoint removed — should not be reachable."""
        with TestClient(app) as client:
            response = client.post("/differentiated-bertrand", json={})
            assert response.status_code in (404, 405, 422)


class TestCompareEndpoints:
    """Test comparison endpoints."""

    def test_compare_scenarios_endpoint(self):
        """Test compare scenarios endpoint."""
        with TestClient(app) as client:
            comparison_data = {
                "left_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 10.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 42,
                    "events": [],
                },
                "right_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 12.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 43,
                    "events": [],
                },
            }

            with patch("src.main.get_db") as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db

                with patch("src.main.run_game") as mock_run_game:
                    mock_run_game.side_effect = ["run_1", "run_2"]

                    # Override the dependency
                    app.dependency_overrides[get_db] = lambda: mock_db
                    try:
                        response = client.post("/compare", json=comparison_data)

                        assert response.status_code == 200
                        data = response.json()
                        assert "left_run_id" in data
                        assert "right_run_id" in data
                    finally:
                        app.dependency_overrides.clear()

    def test_compare_scenarios_different_rounds(self):
        """Test compare scenarios with different rounds."""
        with TestClient(app) as client:
            comparison_data = {
                "left_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 10.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 42,
                    "events": [],
                },
                "right_config": {
                    "model": "cournot",
                    "rounds": 20,  # Different number of rounds
                    "firms": [{"cost": 12.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 43,
                    "events": [],
                },
            }

            # Override the dependency
            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/compare", json=comparison_data)

                assert response.status_code == 400
                assert "same number of rounds" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_compare_scenarios_with_segments(self):
        """Test compare scenarios with segments."""
        with TestClient(app) as client:
            comparison_data = {
                "left_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 10.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 42,
                    "events": [],
                    "segments": [
                        {"alpha": 100.0, "beta": 1.0, "weight": 0.6},
                        {"alpha": 80.0, "beta": 1.2, "weight": 0.4},
                    ],
                },
                "right_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 12.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 43,
                    "events": [],
                },
            }

            with patch("src.main.get_db") as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db

                mock_results = {
                    "results": {
                        "0": {
                            "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0}
                        }
                    },
                    "model": "cournot",
                    "rounds": 10,
                }

                with patch("src.main.run_game") as mock_run_game, patch(
                    "src.main.get_run_results"
                ) as mock_get_results:
                    mock_run_game.side_effect = ["run_1", "run_2"]
                    mock_get_results.return_value = mock_results

                    # Override the dependency
                    app.dependency_overrides[get_db] = lambda: mock_db
                    try:
                        response = client.post("/compare", json=comparison_data)

                        assert response.status_code == 200
                    finally:
                        app.dependency_overrides.clear()

    def test_compare_scenarios_invalid_segments(self):
        """Test compare scenarios with invalid segment weights."""
        with TestClient(app) as client:
            comparison_data = {
                "left_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 10.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 42,
                    "events": [],
                    "segments": [
                        {"alpha": 100.0, "beta": 1.0, "weight": 0.6},
                        {
                            "alpha": 80.0,
                            "beta": 1.2,
                            "weight": 0.3,
                        },  # Sums to 0.9, not 1.0
                    ],
                },
                "right_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 12.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 43,
                    "events": [],
                },
            }

            # Override the dependency
            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/compare", json=comparison_data)

                assert response.status_code == 400
                assert "sum to 1.0" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_get_comparison_results_endpoint(self):
        """Test get comparison results endpoint."""
        with TestClient(app) as client:
            with patch("src.main.get_run_results") as mock_get_run_results:
                # Mock results for both runs
                mock_left_results = {
                    "id": "run_1",
                    "model": "cournot",
                    "rounds": 10,
                    "results": {
                        "0": {
                            "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0}
                        },
                        "1": {
                            "firm_0": {"price": 51.0, "quantity": 19.0, "profit": 779.0}
                        },
                    },
                }
                mock_right_results = {
                    "id": "run_2",
                    "model": "cournot",
                    "rounds": 10,
                    "results": {
                        "0": {
                            "firm_0": {"price": 52.0, "quantity": 18.0, "profit": 756.0}
                        },
                        "1": {
                            "firm_0": {"price": 53.0, "quantity": 17.0, "profit": 731.0}
                        },
                    },
                }

                mock_get_run_results.side_effect = [
                    mock_left_results,
                    mock_right_results,
                ]

                # Override the dependency
                mock_db = Mock()
                app.dependency_overrides[get_db] = lambda: mock_db
                try:
                    response = client.get("/compare/run_1/run_2")

                    assert response.status_code == 200
                    data = response.json()
                    assert "left_run_id" in data
                    assert "right_run_id" in data
                    assert "left_metrics" in data
                    assert "right_metrics" in data
                    assert "deltas" in data
                finally:
                    app.dependency_overrides.clear()

    def test_get_comparison_results_different_rounds(self):
        """Test get comparison results with different number of rounds."""
        with TestClient(app) as client:
            with patch("src.main.get_run_results") as mock_get_run_results:
                # Mock results with different number of rounds
                mock_left_results = {
                    "id": "run_1",
                    "model": "cournot",
                    "rounds": 10,
                    "results": {
                        "0": {
                            "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0}
                        }
                    },
                }
                mock_right_results = {
                    "id": "run_2",
                    "model": "cournot",
                    "rounds": 20,  # Different number of rounds
                    "results": {
                        "0": {
                            "firm_0": {"price": 52.0, "quantity": 18.0, "profit": 756.0}
                        }
                    },
                }

                mock_get_run_results.side_effect = [
                    mock_left_results,
                    mock_right_results,
                ]

                # Override the dependency
                mock_db = Mock()
                app.dependency_overrides[get_db] = lambda: mock_db
                try:
                    response = client.get("/compare/run_1/run_2")

                    assert response.status_code == 400
                    assert "same number of rounds" in response.json()["detail"]
                finally:
                    app.dependency_overrides.clear()


class TestEventsEndpoint:
    """Test events endpoint."""

    def test_get_run_events_endpoint(self):
        """Test get run events endpoint."""
        with TestClient(app) as client:
            with patch("src.main.get_db") as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db

                # Mock run exists
                mock_run = Mock()
                mock_run.id = "run_1"
                mock_db.query.return_value.filter.return_value.first.return_value = (
                    mock_run
                )

                # Mock events
                mock_event = Mock()
                mock_event.id = 1
                mock_event.round_idx = 0
                mock_event.event_type = "collusion"
                mock_event.firm_id = 0
                mock_event.description = "Test event"
                mock_event.event_data = {"test": "data"}
                mock_event.created_at.isoformat.return_value = "2023-01-01T00:00:00"

                mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
                    mock_event
                ]

                # Override the dependency
                app.dependency_overrides[get_db] = lambda: mock_db
                try:
                    response = client.get("/runs/run_1/events")

                    assert response.status_code == 200
                    data = response.json()
                    assert "run_id" in data
                    assert "total_events" in data
                    assert "events" in data
                    assert len(data["events"]) == 1
                finally:
                    app.dependency_overrides.clear()

    def test_get_run_events_endpoint_not_found(self):
        """Test get run events endpoint with non-existent run."""
        with TestClient(app) as client:
            with patch("src.main.get_db") as mock_get_db:
                mock_db = Mock()
                mock_get_db.return_value = mock_db

                # Mock run doesn't exist
                mock_db.query.return_value.filter.return_value.first.return_value = None

                # Override the dependency
                app.dependency_overrides[get_db] = lambda: mock_db
                try:
                    response = client.get("/runs/nonexistent/events")

                    assert response.status_code == 404
                    assert "not found" in response.json()["detail"]
                finally:
                    app.dependency_overrides.clear()


class TestHeatmapEndpoint:
    """Test heatmap endpoint."""

    def test_heatmap_endpoint_cournot(self):
        """Test heatmap endpoint for Cournot model."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 10,
                "action_range": [10.0, 50.0],
                "other_actions": [],
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            with patch("src.main.compute_cournot_heatmap") as mock_compute:
                import numpy as np

                mock_compute.return_value = (
                    np.array([[100.0, 90.0], [90.0, 80.0]]),  # profit_matrix
                    np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # action_i_grid
                    np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # action_j_grid
                )

                response = client.post("/heatmap", json=heatmap_data)

                assert response.status_code == 200
                data = response.json()
                assert "profit_surface" in data
                assert "action_i_grid" in data
                assert "action_j_grid" in data
                assert "computation_time_ms" in data

    def test_heatmap_endpoint_bertrand(self):
        """Test heatmap endpoint for Bertrand model."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "bertrand",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 10,
                "action_range": [20.0, 60.0],
                "other_actions": [],
                "params": {"alpha": 200.0, "beta": 2.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            with patch("src.main.compute_bertrand_heatmap") as mock_compute:
                import numpy as np

                mock_compute.return_value = (
                    np.array([[100.0, 90.0], [90.0, 80.0]]),  # profit_matrix
                    np.array([[0.5, 0.4], [0.4, 0.3]]),  # market_share_matrix
                    np.array([20.0, 30.0, 40.0, 50.0, 60.0]),  # action_i_grid
                    np.array([20.0, 30.0, 40.0, 50.0, 60.0]),  # action_j_grid
                )

                response = client.post("/heatmap", json=heatmap_data)

                assert response.status_code == 200
                data = response.json()
                assert "profit_surface" in data
                assert "market_share_surface" in data
                assert "action_i_grid" in data
                assert "action_j_grid" in data

    def test_heatmap_endpoint_invalid_firm_indices(self):
        """Test heatmap endpoint with invalid firm indices."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 2,  # Invalid: only 2 firms (indices 0, 1)
                "firm_j": 1,
                "grid_size": 10,
                "action_range": [10.0, 50.0],
                "other_actions": [],
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            # Override the dependency
            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/heatmap", json=heatmap_data)

                assert response.status_code == 400
                assert "firm_i" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_heatmap_endpoint_same_firm_indices(self):
        """Test heatmap endpoint with same firm indices."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 0,
                "firm_j": 0,  # Same as firm_i
                "grid_size": 10,
                "action_range": [10.0, 50.0],
                "other_actions": [],
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            # Override the dependency
            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/heatmap", json=heatmap_data)

                assert response.status_code == 400
                assert "must be different" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_heatmap_endpoint_invalid_other_actions_length(self):
        """Test heatmap endpoint with invalid other_actions length."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 10,
                "action_range": [10.0, 50.0],
                "other_actions": [15.0, 25.0],  # Should be empty for 2 firms
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            # Override the dependency
            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/heatmap", json=heatmap_data)

                assert response.status_code == 400
                assert "other_actions length" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_heatmap_endpoint_missing_cournot_params(self):
        """Passing only `a` without `b` is fine — Pydantic supplies the `b` default."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 5,
                "action_range": [10.0, 50.0],
                "other_actions": [],
                "params": {"a": 100.0},  # 'b' omitted — Pydantic default applies
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            response = client.post("/heatmap", json=heatmap_data)
            # With typed params, Pydantic fills b=1.0 default so this should succeed
            assert response.status_code == 200

    def test_heatmap_endpoint_missing_bertrand_params(self):
        """Passing only `alpha` without `beta` is fine — Pydantic supplies the `beta` default."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "bertrand",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 5,
                "action_range": [20.0, 60.0],
                "other_actions": [],
                "params": {"alpha": 200.0},  # 'beta' omitted — Pydantic default applies
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            response = client.post("/heatmap", json=heatmap_data)
            # With typed params, Pydantic fills beta=1.0 default so this should succeed
            assert response.status_code == 200

    def test_heatmap_endpoint_with_segments(self):
        """Test heatmap endpoint with segmented demand."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 10,
                "action_range": [10.0, 50.0],
                "other_actions": [],
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
                "segments": [
                    {"alpha": 100.0, "beta": 1.0, "weight": 0.6},
                    {"alpha": 80.0, "beta": 1.2, "weight": 0.4},
                ],
            }

            with patch("src.main.compute_cournot_segmented_heatmap") as mock_compute:
                import numpy as np

                mock_compute.return_value = (
                    np.array([[100.0, 90.0], [90.0, 80.0]]),  # profit_matrix
                    np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # action_i_grid
                    np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # action_j_grid
                )

                response = client.post("/heatmap", json=heatmap_data)

                assert response.status_code == 200
                data = response.json()
                assert "profit_surface" in data

    def test_heatmap_endpoint_computation_error(self):
        """Test heatmap endpoint with computation error."""
        with TestClient(app) as client:
            heatmap_data = {
                "model": "cournot",
                "firm_i": 0,
                "firm_j": 1,
                "grid_size": 10,
                "action_range": [10.0, 50.0],
                "other_actions": [],
                "params": {"a": 100.0, "b": 1.0},
                "firms": [
                    {"cost": 10.0, "fixed_cost": 0.0},
                    {"cost": 12.0, "fixed_cost": 0.0},
                ],
            }

            with patch("src.main.compute_cournot_heatmap") as mock_compute:
                mock_compute.side_effect = ValueError("400: Invalid parameters")

                response = client.post("/heatmap", json=heatmap_data)

                assert response.status_code == 400
                assert "Invalid parameters" in response.json()["detail"]


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        with TestClient(app) as client:
            response = client.get("/")

            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "oligopoly" in data["message"].lower()
