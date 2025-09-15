"""Test suite for comparison delta calculations.

This module tests that delta calculations (right - left) are computed correctly
with proper precision and handling of edge cases.
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


def test_delta_calculation_precision(setup_database: None) -> None:
    """Test that delta calculations maintain proper precision (±1e-6)."""
    # Create scenarios with known differences
    left_config = {
        "model": "cournot",
        "rounds": 3,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10.0}, {"cost": 15.0}],
        "seed": 42,  # Same seed for reproducibility
    }

    right_config = {
        "model": "cournot",
        "rounds": 3,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 12.0}, {"cost": 18.0}],  # +2 and +3 cost differences
        "seed": 42,  # Same seed for reproducibility
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    # Run comparison
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Get comparison results
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    left_metrics = comparison_data["left_metrics"]
    right_metrics = comparison_data["right_metrics"]
    deltas = comparison_data["deltas"]

    # Test delta calculation precision for each metric
    for metric_name in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        left_values = left_metrics[metric_name]
        right_values = right_metrics[metric_name]
        delta_values = deltas[metric_name]

        assert len(left_values) == len(right_values) == len(delta_values)

        for i in range(len(delta_values)):
            expected_delta = right_values[i] - left_values[i]
            actual_delta = delta_values[i]

            # Check precision within ±1e-6
            assert abs(actual_delta - expected_delta) < 1e-6, (
                f"Delta calculation error for {metric_name}[{i}]: "
                f"expected {expected_delta}, got {actual_delta}"
            )


def test_delta_calculation_zero_difference(setup_database: None) -> None:
    """Test delta calculation when scenarios are identical."""
    # Create identical scenarios
    config = {
        "model": "cournot",
        "rounds": 4,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
        "seed": 123,
    }

    comparison_request = {
        "left_config": config,
        "right_config": config,
    }

    # Run comparison
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Get comparison results
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    deltas = comparison_data["deltas"]

    # All deltas should be approximately zero
    for metric_name in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        delta_values = deltas[metric_name]
        for delta in delta_values:
            assert (
                abs(delta) < 1e-6
            ), f"Expected zero delta for {metric_name}, got {delta}"


def test_delta_calculation_bertrand_model(setup_database: None) -> None:
    """Test delta calculation with Bertrand model."""
    left_config = {
        "model": "bertrand",
        "rounds": 3,
        "params": {"alpha": 100, "beta": 1},
        "firms": [{"cost": 20}, {"cost": 25}],
        "seed": 456,
    }

    right_config = {
        "model": "bertrand",
        "rounds": 3,
        "params": {"alpha": 100, "beta": 1},
        "firms": [{"cost": 22}, {"cost": 28}],  # Different costs
        "seed": 456,
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    # Run comparison
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Get comparison results
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    left_metrics = comparison_data["left_metrics"]
    right_metrics = comparison_data["right_metrics"]
    deltas = comparison_data["deltas"]

    # Test delta calculation precision for Bertrand model
    for metric_name in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        left_values = left_metrics[metric_name]
        right_values = right_metrics[metric_name]
        delta_values = deltas[metric_name]

        for i in range(len(delta_values)):
            expected_delta = right_values[i] - left_values[i]
            actual_delta = delta_values[i]

            assert abs(actual_delta - expected_delta) < 1e-6, (
                f"Bertrand delta calculation error for {metric_name}[{i}]: "
                f"expected {expected_delta}, got {actual_delta}"
            )


def test_delta_calculation_with_policy_events(setup_database: None) -> None:
    """Test delta calculation with policy events affecting outcomes."""
    left_config = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
        "events": [
            {"round_idx": 2, "policy_type": "tax", "value": 0.1},
        ],
        "seed": 789,
    }

    right_config = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
        "events": [
            {"round_idx": 2, "policy_type": "subsidy", "value": 0.05},
        ],
        "seed": 789,
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    # Run comparison
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Get comparison results
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    left_metrics = comparison_data["left_metrics"]
    right_metrics = comparison_data["right_metrics"]
    deltas = comparison_data["deltas"]

    # Test delta calculation precision with policy events
    for metric_name in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        left_values = left_metrics[metric_name]
        right_values = right_metrics[metric_name]
        delta_values = deltas[metric_name]

        for i in range(len(delta_values)):
            expected_delta = right_values[i] - left_values[i]
            actual_delta = delta_values[i]

            assert abs(actual_delta - expected_delta) < 1e-6, (
                f"Policy event delta calculation error for {metric_name}[{i}]: "
                f"expected {expected_delta}, got {actual_delta}"
            )


def test_delta_calculation_edge_cases(setup_database: None) -> None:
    """Test delta calculation with edge cases (zero values, negative values)."""
    # Create scenarios that might produce edge cases
    left_config = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 50, "b": 2},  # Smaller market
        "firms": [{"cost": 5}, {"cost": 8}],
        "seed": 999,
    }

    right_config = {
        "model": "cournot",
        "rounds": 2,
        "params": {"a": 50, "b": 2},
        "firms": [{"cost": 6}, {"cost": 9}],  # Slightly higher costs
        "seed": 999,
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    # Run comparison
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Get comparison results
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    left_metrics = comparison_data["left_metrics"]
    right_metrics = comparison_data["right_metrics"]
    deltas = comparison_data["deltas"]

    # Test delta calculation precision for edge cases
    for metric_name in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        left_values = left_metrics[metric_name]
        right_values = right_metrics[metric_name]
        delta_values = deltas[metric_name]

        for i in range(len(delta_values)):
            expected_delta = right_values[i] - left_values[i]
            actual_delta = delta_values[i]

            assert abs(actual_delta - expected_delta) < 1e-6, (
                f"Edge case delta calculation error for {metric_name}[{i}]: "
                f"expected {expected_delta}, got {actual_delta}"
            )


def test_delta_calculation_array_lengths(setup_database: None) -> None:
    """Test that delta arrays have correct lengths matching input arrays."""
    left_config = {
        "model": "cournot",
        "rounds": 6,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 10}, {"cost": 15}],
        "seed": 111,
    }

    right_config = {
        "model": "cournot",
        "rounds": 6,
        "params": {"a": 100, "b": 1},
        "firms": [{"cost": 12}, {"cost": 18}],
        "seed": 111,
    }

    comparison_request = {
        "left_config": left_config,
        "right_config": right_config,
    }

    # Run comparison
    response = client.post("/compare", json=comparison_request)
    assert response.status_code == 200

    data = response.json()
    left_run_id = data["left_run_id"]
    right_run_id = data["right_run_id"]

    # Get comparison results
    response = client.get(f"/compare/{left_run_id}/{right_run_id}")
    assert response.status_code == 200

    comparison_data = response.json()
    left_metrics = comparison_data["left_metrics"]
    right_metrics = comparison_data["right_metrics"]
    deltas = comparison_data["deltas"]

    # Test that all arrays have the same length
    for metric_name in [
        "market_price",
        "total_quantity",
        "total_profit",
        "hhi",
        "consumer_surplus",
    ]:
        left_length = len(left_metrics[metric_name])
        right_length = len(right_metrics[metric_name])
        delta_length = len(deltas[metric_name])

        assert left_length == right_length == delta_length == 6, (
            f"Array length mismatch for {metric_name}: "
            f"left={left_length}, right={right_length}, delta={delta_length}"
        )
