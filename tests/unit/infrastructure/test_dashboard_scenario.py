"""Tests for the dashboard's Scenario Lab endpoint and Nash metrics."""

import pytest
from fastapi.testclient import TestClient

from dashboard.main import _session_factory, app
from sim.models.models import Run


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _scenario(client, **body):
    request = {
        "model": "cournot",
        "rounds": 10,
        "firms": [{"cost": 10.0}, {"cost": 14.0}],
        "seed": 1,
    }
    request.update(body)
    return client.post("/api/scenario", json=request)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"model": "bertrand", "capacity_constraints": False},
        {"demand_type": "isoelastic", "params": {"A": 100.0, "elasticity": 2.0}},
        {"model": "bertrand", "enhanced_demand": {"demand_type": "ces"}},
        {
            "advanced_strategies": [
                {"firm_id": 0, "strategy_type": "cartel"},
                {"firm_id": 1, "strategy_type": "collusive"},
            ]
        },
        {"market_evolution": {"entry_cost": 5.0}},
    ],
)
def test_scenario_runs_every_option(client, body) -> None:
    response = _scenario(client, **body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["run"]["results"]) == 10
    assert set(data["run"]["metrics"]["9"]) >= {
        "market_price",
        "hhi",
        "consumer_surplus",
    }
    assert isinstance(data["events"], list)


def test_scenario_reports_cartel_events(client) -> None:
    data = _scenario(
        client,
        advanced_strategies=[
            {"firm_id": 0, "strategy_type": "cartel"},
            {"firm_id": 1, "strategy_type": "cartel"},
        ],
    ).json()
    assert any(e["event_type"] == "cartel_formed" for e in data["events"])


def test_scenario_validation_errors_pass_through(client) -> None:
    response = _scenario(client, enhanced_demand={"demand_type": "ces"})
    assert response.status_code == 400
    assert "bertrand" in response.json()["detail"]
    assert _scenario(client, rounds=0).status_code == 422


def test_scenario_runs_are_not_retained(client) -> None:
    _scenario(client)
    db = _session_factory()
    try:
        assert db.query(Run).count() == 0
    finally:
        db.close()
