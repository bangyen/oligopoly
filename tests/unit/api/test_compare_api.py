"""Test suite for API comparison endpoints.

This module tests the POST /compare and GET /compare/{left_run_id}/{right_run_id} endpoints
to ensure proper functionality, validation, and alignment of results.
"""

import atexit
import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from main import app, get_db
from sim.models.models import Base

# Create temporary database file
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
temp_file.close()

# Test database setup - use temporary database
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_file.name}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def cleanup_temp_db():
    """Clean up temporary database file."""
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


atexit.register(cleanup_temp_db)


def override_get_db() -> Generator[Session, None, None]:
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)
        yield db
    finally:
        db.close()


# Override the dependency
app.dependency_overrides[get_db] = override_get_db

# Create test client
client = TestClient(app)


@pytest.fixture
def setup_database() -> Generator[None, None, None]:
    """Set up test database."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_compare_scenarios_success(setup_database: None) -> None:
    """Test successful comparison of two scenarios."""
    # Define two different scenarios
    left_config = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
        "seed": 42,
    }

    right_config = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 12}, {"cost": 18}],  # Different costs
        "seed": 42,
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    # Test POST /compare
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    assert "left_run_id" in data
    assert "right_run_id" in data
    assert data["left_run_id"] != data["right_run_id"]

    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Test GET /compare/{left_run_id}/{right_run_id}
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    assert comparison_data["left_run_id"] == left_run_id
    assert comparison_data["right_run_id"] == right_run_id
    assert comparison_data["rounds"] == 5

    # Check that metrics arrays exist and have correct structure
    assert "left_metrics" in comparison_data
    assert "right_metrics" in comparison_data
    assert "deltas" in comparison_data

    # Check that all expected metrics are present
    expected_metrics = [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]
    for metric in expected_metrics:
        assert metric in comparison_data["left_metrics"]
        assert metric in comparison_data["right_metrics"]
        assert metric in comparison_data["deltas"]

        # Check that arrays have the same length (5 rounds)
        assert len(comparison_data["left_metrics"][metric]) == 5
        assert len(comparison_data["right_metrics"][metric]) == 5
        assert len(comparison_data["deltas"][metric]) == 5


def test_compare_scenarios_different_rounds(setup_database: None) -> None:
    """Test comparison fails when scenarios have different number of rounds."""
    left_config = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
    }

    right_config = {
        "model": "cournot",
        "rounds": 3,  # Different number of rounds
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 12}, {"cost": 18}],
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    response = client.post("/compare", json=comparison_request)
    assert response.status_code in [
        400,
        500,
    ]  # Could be either validation error or runtime error
    assert "same number of rounds" in response.json()["detail"]


def test_compare_scenarios_invalid_config(setup_database: None) -> None:
    """Test comparison fails with invalid configuration."""
    left_config = {
        "model": "invalid_model",  # Invalid model
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
    }

    right_config = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 12}, {"cost": 18}],
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    response = client.post("/compare", json=comparison_request)
    assert response.status_code in [
        400,
        422,
    ]  # Could be validation error or unprocessable entity


def test_get_comparison_results_nonexistent_runs(setup_database: None) -> None:
    """Test getting comparison results for non-existent runs."""
    response = client.get("/compare/nonexistent1/nonexistent2")
    assert response.status_code == 404


def test_get_comparison_results_different_rounds(setup_database: None) -> None:
    """Test getting comparison results for runs with different rounds."""
    # Create two runs with different numbers of rounds
    left_config = {
        "model": "cournot",
        "rounds": 3,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}],
    }

    right_config = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 12}],
    }

    # Create runs individually
    left_response = client.post("/simulate", json=left_config)
    right_response = client.post("/simulate", json=right_config)

    assert left_response.status_code == 200
    assert right_response.status_code == 200

    left_run_id = left_response.json()["run_id"]
    right_run_id = right_response.json()["run_id"]

    # Try to get comparison results
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code in [
        400,
        500,
    ]  # Could be validation error or runtime error
    assert "same number of rounds" in response.json()["detail"]


def test_compare_scenarios_bertrand(setup_database: None) -> None:
    """Test comparison with Bertrand model."""
    left_config = {
        "model": "bertrand",
        "rounds": 4,
        "params": {"alpha": 100, "beta": 1},
        "firms": [{"cost": 20}, {"cost": 25}],
        "seed": 123,
    }

    right_config = {
        "model": "bertrand",
        "rounds": 4,
        "params": {"alpha": 100, "beta": 1},
        "firms": [{"cost": 22}, {"cost": 28}],  # Different costs
        "seed": 123,
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    # Test POST /compare
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Test GET /compare/{left_run_id}/{right_run_id}
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    assert comparison_data["rounds"] == 4

    # Check that all metrics arrays have correct length
    for metric in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        assert len(comparison_data["left_metrics"][metric]) == 4
        assert len(comparison_data["right_metrics"][metric]) == 4
        assert len(comparison_data["deltas"][metric]) == 4


def test_compare_scenarios_with_events(setup_database: None) -> None:
    """Test comparison with policy events."""
    left_config = {
        "model": "cournot",
        "rounds": 6,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
        "events": [
            {"round_idx": 2, "policy_type": "tax", "value": 0.1},
        ],
    }

    right_config = {
        "model": "cournot",
        "rounds": 6,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 12}, {"cost": 18}],
        "events": [
            {"round_idx": 3, "policy_type": "subsidy", "value": 0.05},
        ],
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Test GET /compare/{left_run_id}/{right_run_id}
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    assert comparison_data["rounds"] == 6

    # Check that all metrics arrays have correct length
    for metric in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        assert len(comparison_data["left_metrics"][metric]) == 6
        assert len(comparison_data["right_metrics"][metric]) == 6
        assert len(comparison_data["deltas"][metric]) == 6
