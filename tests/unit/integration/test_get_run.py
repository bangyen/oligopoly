"""Test suite for run retrieval functionality.

This module tests the GET /runs/{id} endpoint to ensure
proper data retrieval and validation.
"""

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from main import app, get_db
from sim.models.models import Base

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/test_get_run.db"
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
    assert "rounds_data" in data
    assert "firms_data" in data

    # Verify arrays have equal length
    assert len(data["rounds_data"]) == data["rounds"]
    assert len(data["firms_data"]) == 2

    for firm_data in data["firms_data"]:
        assert len(firm_data["actions"]) == data["rounds"]
        assert len(firm_data["quantities"]) == data["rounds"]
        assert len(firm_data["profits"]) == data["rounds"]

    # Verify all price, qty, profit are finite and qty, price >= 0
    for round_data in data["rounds_data"]:
        assert isinstance(round_data["price"], (int, float))
        assert round_data["price"] >= 0
        assert isinstance(round_data["total_qty"], (int, float))
        assert round_data["total_qty"] >= 0
        assert isinstance(round_data["total_profit"], (int, float))

    for firm_data in data["firms_data"]:
        for action in firm_data["actions"]:
            assert isinstance(action, (int, float))
            assert action >= 0
        for qty in firm_data["quantities"]:
            assert isinstance(qty, (int, float))
            assert qty >= 0
        for profit in firm_data["profits"]:
            assert isinstance(profit, (int, float))


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
    assert len(data["rounds_data"]) == 2
    assert len(data["firms_data"]) == 2

    # Verify all prices are non-negative
    for round_data in data["rounds_data"]:
        assert round_data["price"] >= 0

    # Verify all quantities are non-negative
    for firm_data in data["firms_data"]:
        for qty in firm_data["quantities"]:
            assert qty >= 0


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
    assert len(data["rounds_data"]) == 10
    assert len(data["firms_data"]) == 4

    # Verify all firms have data for all rounds
    for firm_data in data["firms_data"]:
        assert len(firm_data["actions"]) == 10
        assert len(firm_data["quantities"]) == 10
        assert len(firm_data["profits"]) == 10


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
    assert isinstance(data["rounds_data"], list)
    assert isinstance(data["firms_data"], list)

    # Verify nested data types
    for round_data in data["rounds_data"]:
        assert isinstance(round_data["round"], int)
        assert isinstance(round_data["price"], (int, float))
        assert isinstance(round_data["total_qty"], (int, float))
        assert isinstance(round_data["total_profit"], (int, float))

    for firm_data in data["firms_data"]:
        assert isinstance(firm_data["firm_id"], int)
        assert isinstance(firm_data["actions"], list)
        assert isinstance(firm_data["quantities"], list)
        assert isinstance(firm_data["profits"], list)
