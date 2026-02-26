"""Test suite for run retrieval functionality.

This module tests the GET /runs/{id} endpoint to ensure
proper data retrieval and validation.
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

from src.main import app, get_db
from src.sim.models.models import Base

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


# Cleanup function to be called at module teardown
def cleanup_temp_db():
    """Clean up temporary database file."""
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


atexit.register(cleanup_temp_db)


@pytest.fixture(scope="function")
def setup_database() -> Generator[None, None, None]:
    """Set up test database for each test."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_get_run_valid_id(setup_database: None) -> None:
    """Test GET /runs/{id} returns arrays of equal length with valid data."""
    # First create a simulation
    request_data = {
        "model": "cournot",
        "rounds": 3,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        "seed": 42,
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Now get the run results
    get_response = client.get(f"/runs/{run_id}")
    assert get_response.status_code == 200
    data = get_response.json()

    # Verify structure
    assert "run_id" in data
    assert "model" in data
    assert "rounds" in data
    assert "created_at" in data
    assert "results" in data
    assert "metrics" in data

    # Verify rounds length
    assert len(data["results"]) == data["rounds"]
    assert len(data["metrics"]) == data["rounds"]

    # Verify all price, qty, profit are finite and qty, price >= 0 in results
    for ridx in data["results"]:
        round_firms = data["results"][ridx]
        for fid in round_firms:
            firm_data = round_firms[fid]
            assert isinstance(firm_data["price"], (int, float))
            assert firm_data["price"] >= 0
            assert isinstance(firm_data["quantity"], (int, float))
            assert firm_data["quantity"] >= 0
            assert isinstance(firm_data["profit"], (int, float))
            assert isinstance(firm_data["action"], (int, float))
            assert firm_data["action"] >= 0

    # Verify metrics structure
    for ridx in data["metrics"]:
        m = data["metrics"][ridx]
        assert "hhi" in m
        assert "consumer_surplus" in m
        assert "market_price" in m
        assert m["market_price"] >= 0


def test_get_run_invalid_id(setup_database: None) -> None:
    """Test GET /runs/{id} returns 404 for non-existent run_id."""
    response = client.get("/runs/non-existent-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_run_bertrand_model(setup_database: None) -> None:
    """Test GET /runs/{id} works for Bertrand model."""
    # Create Bertrand simulation
    request_data = {
        "model": "bertrand",
        "rounds": 2,
        "params": {"alpha": 100.0, "beta": 1.0},
        "firms": [{"cost": 5.0}, {"cost": 8.0}],
        "seed": 42,
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Get results
    get_response = client.get(f"/runs/{run_id}")
    assert get_response.status_code == 200
    data = get_response.json()

    # Verify Bertrand-specific structure
    assert data["model"] == "bertrand"
    assert len(data["results"]) == 2

    # Verify quantities are non-negative
    for ridx in data["results"]:
        for fid in data["results"][ridx]:
            assert data["results"][ridx][fid]["quantity"] >= 0
            assert data["results"][ridx][fid]["price"] >= 0


def test_get_run_large_simulation(setup_database: None) -> None:
    """Test GET /runs/{id} works with larger simulations."""
    request_data = {
        "model": "cournot",
        "rounds": 10,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}, {"cost": 25.0}],
        "seed": 42,
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Get results
    get_response = client.get(f"/runs/{run_id}")
    assert get_response.status_code == 200
    data = get_response.json()

    # Verify structure
    assert data["rounds"] == 10
    assert len(data["results"]) == 10

    # Verify all firms have data
    for ridx in data["results"]:
        assert len(data["results"][ridx]) == 4


def test_get_run_data_types(setup_database: None) -> None:
    """Test GET /runs/{id} returns proper data types."""
    request_data = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
        "seed": 42,
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    get_response = client.get(f"/runs/{run_id}")
    assert get_response.status_code == 200
    data = get_response.json()

    # Verify data types
    assert isinstance(data["run_id"], str)
    assert isinstance(data["model"], str)
    assert isinstance(data["rounds"], int)
    assert isinstance(data["created_at"], str)
    assert isinstance(data["results"], dict)
    assert isinstance(data["metrics"], dict)

    # Verify nested data types
    for ridx in data["results"]:
        round_firms = data["results"][ridx]
        for fid in round_firms:
            firm_data = round_firms[fid]
            assert isinstance(firm_data["price"], (int, float))
            assert isinstance(firm_data["quantity"], (int, float))
            assert isinstance(firm_data["profit"], (int, float))
            assert isinstance(firm_data["action"], (int, float))

    for ridx in data["metrics"]:
        m = data["metrics"][ridx]
        assert isinstance(m["hhi"], (int, float))
        assert isinstance(m["consumer_surplus"], (int, float))
        assert isinstance(m["market_price"], (int, float))
