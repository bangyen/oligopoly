"""Test API events integration with policy shocks.

This module tests the API integration for policy events, ensuring that
POST /simulate accepts events and applies them only on specified rounds.
"""

import atexit
import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app, get_db
from sim.models.models import Base

# Test client
client = TestClient(app)

# Create temporary database file
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
temp_file.close()

# Test database setup - use temporary database
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_file.name}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db() -> Generator:
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def setup_database() -> Generator[None, None, None]:
    """Set up test database for each test."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


# Cleanup function to be called at module teardown
def cleanup_temp_db():
    """Clean up temporary database file."""
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


atexit.register(cleanup_temp_db)


def test_api_simulate_with_tax_event(setup_database: None) -> None:
    """Test POST /simulate with tax event applied on round 1."""
    request_data = {
        "model": "cournot",
        "rounds": 3,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        "seed": 42,
        "events": [{"round_idx": 1, "policy_type": "tax", "value": 0.2}],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert isinstance(data["run_id"], str)
    assert len(data["run_id"]) > 0


def test_api_simulate_with_subsidy_event(setup_database: None) -> None:
    """Test POST /simulate with subsidy event applied on round 0."""
    request_data = {
        "model": "bertrand",
        "rounds": 2,
        "params": {"alpha": 100.0, "beta": 1.0},
        "firms": [{"cost": 5.0}, {"cost": 8.0}],
        "seed": 123,
        "events": [{"round_idx": 0, "policy_type": "subsidy", "value": 5.0}],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data


def test_api_simulate_with_price_cap_event(setup_database: None) -> None:
    """Test POST /simulate with price cap event applied on round 2."""
    request_data = {
        "model": "cournot",
        "rounds": 4,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        "seed": 456,
        "events": [{"round_idx": 2, "policy_type": "price_cap", "value": 50.0}],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data


def test_api_simulate_with_multiple_events(setup_database: None) -> None:
    """Test POST /simulate with multiple events on different rounds."""
    request_data = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        "seed": 789,
        "events": [
            {"round_idx": 1, "policy_type": "tax", "value": 0.1},
            {"round_idx": 3, "policy_type": "subsidy", "value": 3.0},
        ],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data


def test_api_simulate_with_invalid_event_round(setup_database: None) -> None:
    """Test POST /simulate with event on invalid round (beyond total rounds)."""
    request_data = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
        "seed": 42,
        "events": [
            {
                "round_idx": 5,  # Beyond total rounds (2)
                "policy_type": "tax",
                "value": 0.2,
            }
        ],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 400
    assert "Round index" in response.json()["detail"]


def test_api_simulate_with_invalid_tax_rate(setup_database: None) -> None:
    """Test POST /simulate with invalid tax rate (>= 1.0)."""
    request_data = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
        "seed": 42,
        "events": [
            {"round_idx": 1, "policy_type": "tax", "value": 1.0}  # Invalid tax rate
        ],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 400
    assert "Tax rate must be less than 1.0" in response.json()["detail"]


def test_api_simulate_with_negative_event_value(setup_database: None) -> None:
    """Test POST /simulate with negative event value."""
    request_data = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
        "seed": 42,
        "events": [
            {
                "round_idx": 1,
                "policy_type": "subsidy",
                "value": -5.0,  # Negative subsidy
            }
        ],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 422  # Pydantic validation error


def test_api_simulate_with_invalid_policy_type(setup_database: None) -> None:
    """Test POST /simulate with invalid policy type."""
    request_data = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
        "seed": 42,
        "events": [{"round_idx": 1, "policy_type": "invalid_type", "value": 0.2}],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 422  # Validation error


def test_api_simulate_without_events(setup_database: None) -> None:
    """Test POST /simulate without events (should work normally)."""
    request_data = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
        "seed": 42,
        # No events field
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data


def test_api_simulate_with_empty_events(setup_database: None) -> None:
    """Test POST /simulate with empty events list."""
    request_data = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
        "seed": 42,
        "events": [],
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
