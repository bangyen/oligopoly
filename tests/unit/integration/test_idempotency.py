"""Test suite for idempotency and deterministic behavior.

This module tests that simulations with the same configuration
and seed produce deterministic results, while different seeds
produce different run_ids.
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
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/test_idempotency.db"
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


def test_idempotency_different_seeds(setup_database: None) -> None:
    """Test same run config with different seeds -> different run_ids."""
    base_config = {
        "model": "cournot",
        "rounds": 3,
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

    # Results should also be different due to different seeds
    data1 = client.get(f"/runs/{run_id1}").json()
    data2 = client.get(f"/runs/{run_id2}").json()

    # Compare first round actions (should be different with different seeds)
    assert data1["firms_data"][0]["actions"][0] != data2["firms_data"][0]["actions"][0]


def test_idempotency_same_seed(setup_database: None) -> None:
    """Test same run config with same seed -> deterministic results (optional)."""
    config = {
        "model": "cournot",
        "rounds": 3,
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

    # Compare all rounds
    for round_idx in range(3):
        assert (
            data1["firms_data"][0]["actions"][round_idx]
            == data2["firms_data"][0]["actions"][round_idx]
        )
        assert (
            data1["firms_data"][1]["actions"][round_idx]
            == data2["firms_data"][1]["actions"][round_idx]
        )
        assert (
            data1["rounds_data"][round_idx]["price"]
            == data2["rounds_data"][round_idx]["price"]
        )


def test_idempotency_no_seed(setup_database: None) -> None:
    """Test same run config without seed -> different results each time."""
    config = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        # No seed specified
    }

    # Run twice without seed
    response1 = client.post("/simulate", json=config)
    assert response1.status_code == 200
    run_id1 = response1.json()["run_id"]

    response2 = client.post("/simulate", json=config)
    assert response2.status_code == 200
    run_id2 = response2.json()["run_id"]

    # Should have different run_ids
    assert run_id1 != run_id2

    # Results should be different (random behavior)
    data1 = client.get(f"/runs/{run_id1}").json()
    data2 = client.get(f"/runs/{run_id2}").json()

    # First round actions should be different (random initialization)
    assert data1["firms_data"][0]["actions"][0] != data2["firms_data"][0]["actions"][0]


def test_idempotency_bertrand_model(setup_database: None) -> None:
    """Test idempotency with Bertrand model."""
    config = {
        "model": "bertrand",
        "rounds": 2,
        "params": {"alpha": 100.0, "beta": 1.0},
        "firms": [{"cost": 5.0}, {"cost": 8.0}],
        "seed": 123,
    }

    # Run twice with same seed
    response1 = client.post("/simulate", json=config)
    assert response1.status_code == 200
    run_id1 = response1.json()["run_id"]

    response2 = client.post("/simulate", json=config)
    assert response2.status_code == 200
    run_id2 = response2.json()["run_id"]

    # Should have different run_ids
    assert run_id1 != run_id2

    # But results should be deterministic
    data1 = client.get(f"/runs/{run_id1}").json()
    data2 = client.get(f"/runs/{run_id2}").json()

    # Compare first round results
    assert data1["firms_data"][0]["actions"][0] == data2["firms_data"][0]["actions"][0]
    assert data1["firms_data"][1]["actions"][0] == data2["firms_data"][1]["actions"][0]


def test_idempotency_different_configs(setup_database: None) -> None:
    """Test different configs produce different results."""
    config1 = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        "seed": 42,
    }

    config2 = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 120.0, "b": 1.0},  # Different demand parameter
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        "seed": 42,
    }

    response1 = client.post("/simulate", json=config1)
    assert response1.status_code == 200
    run_id1 = response1.json()["run_id"]

    response2 = client.post("/simulate", json=config2)
    assert response2.status_code == 200
    run_id2 = response2.json()["run_id"]

    # Should have different run_ids
    assert run_id1 != run_id2

    # Results should be different due to different parameters
    data1 = client.get(f"/runs/{run_id1}").json()
    data2 = client.get(f"/runs/{run_id2}").json()

    # Prices should be different due to different demand parameters
    assert data1["rounds_data"][0]["price"] != data2["rounds_data"][0]["price"]


def test_idempotency_multiple_firms(setup_database: None) -> None:
    """Test idempotency with multiple firms."""
    config = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}, {"cost": 25.0}],
        "seed": 42,
    }

    # Run twice with same seed
    response1 = client.post("/simulate", json=config)
    assert response1.status_code == 200
    run_id1 = response1.json()["run_id"]

    response2 = client.post("/simulate", json=config)
    assert response2.status_code == 200
    run_id2 = response2.json()["run_id"]

    # Should have different run_ids
    assert run_id1 != run_id2

    # But results should be deterministic for all firms
    data1 = client.get(f"/runs/{run_id1}").json()
    data2 = client.get(f"/runs/{run_id2}").json()

    # Compare all firms' first round actions
    for firm_idx in range(4):
        assert (
            data1["firms_data"][firm_idx]["actions"][0]
            == data2["firms_data"][firm_idx]["actions"][0]
        )
