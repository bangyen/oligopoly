"""Test suite for API simulation endpoints.

This module tests the POST /simulate and GET /runs/{id} endpoints
to ensure proper functionality, validation, and persistence.
"""

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from main import app, get_db
from sim.models import Base

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
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


def test_api_simulate_cournot(setup_database: None) -> None:
    """Test POST /simulate returns 200 and run_id for Cournot model."""
    request_data = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        "seed": 42,
    }

    response = client.post("/simulate", json=request_data)

    if response.status_code != 200:
        print(f"Error response: {response.status_code}")
        print(f"Error details: {response.text}")

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert isinstance(data["run_id"], str)
    assert len(data["run_id"]) > 0


def test_api_simulate_bertrand(setup_database: None) -> None:
    """Test POST /simulate returns 200 and run_id for Bertrand model."""
    request_data = {
        "model": "bertrand",
        "rounds": 3,
        "params": {"alpha": 100.0, "beta": 1.0},
        "firms": [{"cost": 5.0}, {"cost": 8.0}],
        "seed": 123,
    }

    response = client.post("/simulate", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert isinstance(data["run_id"], str)
    assert len(data["run_id"]) > 0


def test_api_simulate_invalid_model(setup_database: None) -> None:
    """Test POST /simulate returns 422 for invalid model."""
    request_data = {
        "model": "invalid_model",
        "rounds": 5,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 422


def test_api_simulate_invalid_rounds(setup_database: None) -> None:
    """Test POST /simulate returns 422 for invalid rounds."""
    request_data = {
        "model": "cournot",
        "rounds": 0,  # Invalid: must be > 0
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 422


def test_api_simulate_no_firms(setup_database: None) -> None:
    """Test POST /simulate returns 422 for empty firms list."""
    request_data = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [],  # Invalid: must have at least one firm
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 422


def test_persistence_counts(setup_database: None) -> None:
    """Test DB has exactly `rounds` rows for given run_id; results rows == rounds * num_firms."""
    request_data = {
        "model": "cournot",
        "rounds": 4,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        "seed": 42,
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    # Get the run results to verify persistence
    get_response = client.get(f"/runs/{run_id}")
    assert get_response.status_code == 200
    data = get_response.json()

    # Verify rounds count
    assert data["rounds"] == 4
    assert len(data["rounds_data"]) == 4

    # Verify firms count
    assert len(data["firms_data"]) == 3

    # Verify each firm has results for all rounds
    for firm_data in data["firms_data"]:
        assert len(firm_data["actions"]) == 4
        assert len(firm_data["quantities"]) == 4
        assert len(firm_data["profits"]) == 4


def test_get_run_valid_id(setup_database: None) -> None:
    """Test GET /runs/{id} returns arrays of equal length with valid data."""
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


def test_idempotency_different_seeds(setup_database: None) -> None:
    """Test same run config with different seeds -> different run_ids."""
    base_config = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
    }

    # Run with seed 1
    config1 = {**base_config, "seed": 1}
    response1 = client.post("/simulate", json=config1)
    assert response1.status_code == 200
    run_id1 = response1.json()["run_id"]

    # Run with seed 2
    config2 = {**base_config, "seed": 2}
    response2 = client.post("/simulate", json=config2)
    assert response2.status_code == 200
    run_id2 = response2.json()["run_id"]

    # Should have different run_ids
    assert run_id1 != run_id2


def test_idempotency_same_seed(setup_database: None) -> None:
    """Test same run config with same seed -> deterministic results (optional)."""
    config = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        "seed": 42,
    }

    # Run twice with same seed
    response1 = client.post("/simulate", json=config)
    assert response1.status_code == 200
    run_id1 = response1.json()["run_id"]

    response2 = client.post("/simulate", json=config)
    assert response2.status_code == 200
    run_id2 = response2.json()["run_id"]

    # Should have different run_ids (different runs)
    assert run_id1 != run_id2

    # But results should be deterministic
    data1 = client.get(f"/runs/{run_id1}").json()
    data2 = client.get(f"/runs/{run_id2}").json()

    # Compare first round results (should be identical with same seed)
    assert data1["firms_data"][0]["actions"][0] == data2["firms_data"][0]["actions"][0]
    assert data1["firms_data"][1]["actions"][0] == data2["firms_data"][1]["actions"][0]


def test_bertrand_simulation_results(setup_database: None) -> None:
    """Test Bertrand simulation produces valid results."""
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
