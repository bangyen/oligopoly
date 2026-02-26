"""Test suite for API simulation endpoints.

This module tests the POST /simulate and GET /runs/{id} endpoints
to ensure proper functionality, validation, and persistence.
"""

from fastapi.testclient import TestClient

from src.main import app, get_db
from tests.utils import override_get_db_for_testing

# Override the database dependency for testing
override_get_db_for_testing(app, get_db)

# Create test client
client = TestClient(app)


def test_api_simulate_cournot() -> None:
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


def test_api_simulate_bertrand() -> None:
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


def test_api_simulate_invalid_model() -> None:
    """Test POST /simulate returns 422 for invalid model."""
    request_data = {
        "model": "invalid_model",
        "rounds": 5,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 422


def test_api_simulate_invalid_rounds() -> None:
    """Test POST /simulate returns 422 for invalid rounds."""
    request_data = {
        "model": "cournot",
        "rounds": 0,  # Invalid: must be > 0
        "params": {"a": 100.0, "b": 1.0},
        "firms": [{"cost": 10.0}],
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 422


def test_api_simulate_no_firms() -> None:
    """Test POST /simulate returns 422 for empty firms list."""
    request_data = {
        "model": "cournot",
        "rounds": 5,
        "params": {"a": 100.0, "b": 1.0},
        "firms": [],  # Invalid: must have at least one firm
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 422


def test_persistence_counts() -> None:
    """Test DB has exactly `rounds` entries for given run_id; results nested by round then firm."""
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
    results = data["results"]
    assert len(results) == 4  # 4 rounds

    # Verify 3 firms per round
    for round_idx, round_firms in results.items():
        assert len(round_firms) == 3


def test_get_run_valid_id() -> None:
    """Test GET /runs/{id} returns canonical nested-dict format with valid data."""
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
    assert "results" in data
    assert "params" in data

    results = data["results"]
    assert len(results) == data["rounds"]

    for round_idx, round_firms in results.items():
        assert len(round_firms) == 2  # 2 firms
        for firm_id, firm_data in round_firms.items():
            assert isinstance(firm_data["action"], (int, float))
            assert isinstance(firm_data["price"], (int, float))
            assert firm_data["price"] >= 0
            assert isinstance(firm_data["quantity"], (int, float))
            assert firm_data["quantity"] >= 0
            assert isinstance(firm_data["profit"], (int, float))


def test_get_run_invalid_id() -> None:
    """Test GET /runs/{id} returns 404 for non-existent run_id."""
    response = client.get("/runs/non-existent-id")
    assert response.status_code == 404


def test_idempotency_different_seeds() -> None:
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


def test_idempotency_same_seed() -> None:
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
    assert (
        data1["results"]["0"]["firm_0"]["action"]
        == data2["results"]["0"]["firm_0"]["action"]
    )
    assert (
        data1["results"]["0"]["firm_1"]["action"]
        == data2["results"]["0"]["firm_1"]["action"]
    )


def test_bertrand_simulation_results() -> None:
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
    results = data["results"]
    assert len(results) == 2  # 2 rounds
    for round_idx, round_firms in results.items():
        assert len(round_firms) == 2  # 2 firms
        for firm_id, firm_data in round_firms.items():
            assert firm_data["price"] >= 0
            assert firm_data["quantity"] >= 0
