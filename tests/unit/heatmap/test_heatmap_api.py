"""Tests for heatmap API endpoint performance and functionality.

This module tests that the /heatmap API endpoint responds within time budgets
and returns correct data structures for different grid sizes.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.main import app


class TestHeatmapAPI:
    """Test heatmap API endpoint performance and functionality."""

    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        return TestClient(app)

    def test_heatmap_api_small_grid_performance(self):
        """Test API response time for small grid (<2s for 5x5)."""
        client = TestClient(app)

        # Setup small grid request
        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 5,
            "action_range": [0.0, 20.0],
            "other_actions": [10.0],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        }

        # Measure response time
        start_time = time.time()
        response = client.post("/heatmap", json=request_data)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000

        # Validate response
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        # Validate response time
        assert (
            response_time_ms < 2000
        ), f"Response time {response_time_ms:.1f}ms exceeds 2s budget"

        # Validate response structure
        data = response.json()
        assert "profit_surface" in data
        assert "action_i_grid" in data
        assert "action_j_grid" in data
        assert "computation_time_ms" in data
        assert "model" in data
        assert "firm_i" in data
        assert "firm_j" in data

        # Validate dimensions
        assert (
            len(data["profit_surface"]) == 5
        ), f"Expected 5 rows, got {len(data['profit_surface'])}"
        assert (
            len(data["profit_surface"][0]) == 5
        ), f"Expected 5 cols, got {len(data['profit_surface'][0])}"
        assert (
            len(data["action_i_grid"]) == 5
        ), f"Expected 5 grid points, got {len(data['action_i_grid'])}"
        assert (
            len(data["action_j_grid"]) == 5
        ), f"Expected 5 grid points, got {len(data['action_j_grid'])}"

    def test_heatmap_api_medium_grid_performance(self):
        """Test API response time for medium grid (<2s for 10x10)."""
        client = TestClient(app)

        # Setup medium grid request
        request_data = {
            "model": "bertrand",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 10,
            "action_range": [0.0, 50.0],
            "other_actions": [20.0],
            "params": {"alpha": 100.0, "beta": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        }

        # Measure response time
        start_time = time.time()
        response = client.post("/heatmap", json=request_data)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000

        # Validate response
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        # Validate response time
        assert (
            response_time_ms < 2000
        ), f"Response time {response_time_ms:.1f}ms exceeds 2s budget"

        # Validate response structure
        data = response.json()
        assert "profit_surface" in data
        assert "market_share_surface" in data  # Bertrand should have market share
        assert "action_i_grid" in data
        assert "action_j_grid" in data

        # Validate dimensions
        assert (
            len(data["profit_surface"]) == 10
        ), f"Expected 10 rows, got {len(data['profit_surface'])}"
        assert (
            len(data["profit_surface"][0]) == 10
        ), f"Expected 10 cols, got {len(data['profit_surface'][0])}"
        assert (
            len(data["market_share_surface"]) == 10
        ), f"Expected 10 rows, got {len(data['market_share_surface'])}"

    def test_heatmap_api_large_grid_performance(self):
        """Test API response time for large grid (<2s for 15x15)."""
        client = TestClient(app)

        # Setup large grid request
        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 2,
            "grid_size": 15,
            "action_range": [0.0, 30.0],
            "other_actions": [15.0, 20.0],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}, {"cost": 25.0}],
        }

        # Measure response time
        start_time = time.time()
        response = client.post("/heatmap", json=request_data)
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000

        # Validate response
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        # Validate response time
        assert (
            response_time_ms < 2000
        ), f"Response time {response_time_ms:.1f}ms exceeds 2s budget"

        # Validate dimensions
        data = response.json()
        assert (
            len(data["profit_surface"]) == 15
        ), f"Expected 15 rows, got {len(data['profit_surface'])}"
        assert (
            len(data["profit_surface"][0]) == 15
        ), f"Expected 15 cols, got {len(data['profit_surface'][0])}"

    def test_heatmap_api_validation_errors(self):
        """Test API validation error handling."""
        client = TestClient(app)

        # Test invalid model
        request_data = {
            "model": "invalid_model",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 5,
            "action_range": [0.0, 20.0],
            "other_actions": [],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 422
        ), f"Expected 422 for invalid model, got {response.status_code}"

        # Test invalid firm indices
        request_data = {
            "model": "cournot",
            "firm_i": 5,  # Invalid firm index
            "firm_j": 1,
            "grid_size": 5,
            "action_range": [0.0, 20.0],
            "other_actions": [],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 400
        ), f"Expected 400 for invalid firm index, got {response.status_code}"

        # Test missing parameters
        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 5,
            "action_range": [0.0, 20.0],
            "other_actions": [],
            "params": {"a": 100.0},  # Missing 'b' parameter
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 400
        ), f"Expected 400 for missing parameters, got {response.status_code}"

    def test_heatmap_api_bertrand_market_share(self):
        """Test that Bertrand API returns market share surface."""
        client = TestClient(app)

        request_data = {
            "model": "bertrand",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 8,
            "action_range": [0.0, 50.0],
            "other_actions": [20.0],
            "params": {"alpha": 100.0, "beta": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        assert (
            "market_share_surface" in data
        ), "Bertrand response should include market share surface"
        assert (
            data["market_share_surface"] is not None
        ), "Market share surface should not be None"

        # Validate market share surface dimensions
        assert (
            len(data["market_share_surface"]) == 8
        ), f"Expected 8 rows, got {len(data['market_share_surface'])}"
        assert (
            len(data["market_share_surface"][0]) == 8
        ), f"Expected 8 cols, got {len(data['market_share_surface'][0])}"

        # Validate market share values are between 0 and 1
        for row in data["market_share_surface"]:
            for value in row:
                assert 0.0 <= value <= 1.0, f"Market share value {value} not in [0,1]"

    def test_heatmap_api_cournot_no_market_share(self):
        """Test that Cournot API does not return market share surface."""
        client = TestClient(app)

        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 8,
            "action_range": [0.0, 30.0],
            "other_actions": [15.0],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        assert (
            "market_share_surface" in data
        ), "Response should include market_share_surface field"
        assert (
            data["market_share_surface"] is None
        ), "Cournot response should have None market share surface"

    def test_heatmap_api_computation_time_field(self):
        """Test that API returns computation time field."""
        client = TestClient(app)

        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 10,
            "action_range": [0.0, 25.0],
            "other_actions": [12.0],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        assert (
            "computation_time_ms" in data
        ), "Response should include computation_time_ms field"
        assert isinstance(
            data["computation_time_ms"], (int, float)
        ), "Computation time should be numeric"
        assert (
            data["computation_time_ms"] >= 0
        ), "Computation time should be non-negative"

    def test_heatmap_api_grid_size_limits(self):
        """Test API behavior with grid size limits."""
        client = TestClient(app)

        # Test minimum grid size
        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 5,  # Minimum allowed
            "action_range": [0.0, 20.0],
            "other_actions": [],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 200
        ), f"Expected 200 for minimum grid size, got {response.status_code}"

        # Test maximum grid size
        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 50,  # Maximum allowed
            "action_range": [0.0, 20.0],
            "other_actions": [],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 200
        ), f"Expected 200 for maximum grid size, got {response.status_code}"

        # Test invalid grid size (too small)
        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 4,  # Too small
            "action_range": [0.0, 20.0],
            "other_actions": [],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 422
        ), f"Expected 422 for too small grid size, got {response.status_code}"

    def test_heatmap_api_profit_values_validity(self):
        """Test that API returns valid profit values."""
        client = TestClient(app)

        request_data = {
            "model": "cournot",
            "firm_i": 0,
            "firm_j": 1,
            "grid_size": 6,
            "action_range": [0.0, 25.0],
            "other_actions": [12.0],
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        }

        response = client.post("/heatmap", json=request_data)
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        profit_surface = data["profit_surface"]

        # Validate profit values
        for row in profit_surface:
            for profit in row:
                assert isinstance(
                    profit, (int, float)
                ), f"Profit value {profit} should be numeric"
                assert profit >= 0, f"Profit value {profit} should be non-negative"
                assert not (
                    profit != profit
                ), f"Profit value {profit} should not be NaN"  # NaN check
