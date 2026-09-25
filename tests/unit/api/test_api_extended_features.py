"""API tests for non-linear demand, learning strategies and market evolution."""

import pytest
from fastapi.testclient import TestClient

from sim.api import app, get_db
from tests.utils import override_get_db_for_testing


@pytest.fixture
def client():
    saved = dict(app.dependency_overrides)
    override_get_db_for_testing(app, get_db)
    yield TestClient(app)
    app.dependency_overrides.clear()
    app.dependency_overrides.update(saved)


def _simulate(client, **overrides):
    body = {
        "model": "cournot",
        "rounds": 8,
        "firms": [{"cost": 10.0}, {"cost": 12.0}],
        "seed": 1,
        **overrides,
    }
    return client.post("/simulate", json=body)


def _run(client, **overrides):
    response = _simulate(client, **overrides)
    assert response.status_code == 200, response.text
    run = client.get(f"/runs/{response.json()['run_id']}")
    assert run.status_code == 200, run.text
    return run.json()


class TestIsoelasticDemand:
    @pytest.mark.parametrize("model", ["cournot", "bertrand"])
    def test_simulate_and_fetch(self, client, model) -> None:
        data = _run(
            client,
            model=model,
            demand_type="isoelastic",
            params={"A": 100.0, "elasticity": 2.0},
        )
        assert data["params"]["demand_type"] == "isoelastic"
        metrics = data["metrics"]["7"]
        assert metrics["consumer_surplus"] > 0
        # Consumer surplus = A^e P^(1-e) / (e - 1) at the market price
        assert metrics["consumer_surplus"] == pytest.approx(
            100.0**2 / metrics["market_price"]
        )

    def test_defaults_when_params_omitted(self, client) -> None:
        data = _run(client, demand_type="isoelastic")
        assert data["params"]["A"] == 100.0
        assert data["params"]["elasticity"] == 2.0

    @pytest.mark.parametrize(
        ("overrides", "detail"),
        [
            (
                {"demand_type": "isoelastic", "params": {"a": 100.0, "b": 1.0}},
                "IsoelasticParams",
            ),
            ({"params": {"A": 100.0, "elasticity": 2.0}}, "demand_type='isoelastic'"),
            (
                {
                    "demand_type": "isoelastic",
                    "segments": [{"alpha": 100.0, "beta": 1.0, "weight": 1.0}],
                },
                "linear demand",
            ),
        ],
    )
    def test_invalid(self, client, overrides, detail) -> None:
        response = _simulate(client, **overrides)
        assert response.status_code == 400
        assert detail in response.json()["detail"]

    def test_elasticity_must_exceed_one(self, client) -> None:
        response = _simulate(
            client, demand_type="isoelastic", params={"A": 100.0, "elasticity": 1.0}
        )
        assert response.status_code == 422


class TestCESDemand:
    def test_single_firm_prices_at_monopoly_markup(self, client) -> None:
        data = _run(
            client,
            model="bertrand",
            rounds=40,
            firms=[{"cost": 10.0}],
            enhanced_demand={"demand_type": "ces", "market_elasticity": 2.0},
        )
        # c * eta / (eta - 1) = 20
        assert data["results"]["39"]["firm_0"]["price"] == pytest.approx(20.0, rel=0.02)

    def test_simulate_and_fetch(self, client) -> None:
        data = _run(
            client,
            model="bertrand",
            firms=[{"cost": 10.0}, {"cost": 12.0}, {"cost": 11.0}],
            enhanced_demand={
                "demand_type": "ces",
                "elasticity": 3.0,
                "market_size": 300.0,
                "qualities": [1.0, 1.2, 0.9],
            },
        )
        assert data["params"]["demand_type"] == "ces"
        assert data["params"]["qualities"] == [1.0, 1.2, 0.9]
        # Consumer surplus = total spending / (market_elasticity - 1)
        last = data["results"]["7"].values()
        spend = sum(f["price"] * f["quantity"] for f in last)
        assert data["metrics"]["7"]["consumer_surplus"] == pytest.approx(spend)
        # Differentiated products: all three firms sell every round
        assert all(f["quantity"] > 0 for f in data["results"]["7"].values())

    def test_linear_enhanced_demand_is_default(self, client) -> None:
        data = _run(client, model="bertrand", enhanced_demand={"demand_type": "linear"})
        assert "demand_type" not in data["params"]

    @pytest.mark.parametrize(
        ("overrides", "detail"),
        [
            ({"model": "cournot"}, "model='bertrand'"),
            ({"params": {"alpha": 100.0, "beta": 1.0}}, "omit params"),
            ({"demand_type": "isoelastic"}, "either"),
            (
                {"enhanced_demand": {"demand_type": "ces", "qualities": [1.0]}},
                "qualities",
            ),
        ],
    )
    def test_invalid(self, client, overrides, detail) -> None:
        body = {"model": "bertrand", "enhanced_demand": {"demand_type": "ces"}}
        body.update(overrides)
        response = _simulate(client, **body)
        assert response.status_code == 400
        assert detail in response.json()["detail"]


class TestAdvancedStrategies:
    @pytest.mark.parametrize(
        "strategy_type",
        ["fictitious_play", "q_learning", "deep_q_learning", "behavioral"],
    )
    def test_each_strategy(self, client, strategy_type) -> None:
        data = _run(
            client,
            advanced_strategies=[
                {"firm_id": 0, "strategy_type": strategy_type, "learning_rate": 0.2}
            ],
        )
        assert data["params"]["advanced_strategies"][0]["strategy_type"] == (
            strategy_type
        )
        assert len(data["results"]) == 8

    @pytest.mark.parametrize(
        ("strategies", "status", "detail"),
        [
            ([{"firm_id": 2, "strategy_type": "q_learning"}], 400, "out of range"),
            (
                [
                    {"firm_id": 0, "strategy_type": "q_learning"},
                    {"firm_id": 0, "strategy_type": "behavioral"},
                ],
                400,
                "more than one",
            ),
            ([{"firm_id": 0, "strategy_type": "genetic"}], 422, None),
        ],
    )
    def test_invalid(self, client, strategies, status, detail) -> None:
        response = _simulate(client, advanced_strategies=strategies)
        assert response.status_code == status
        if detail:
            assert detail in response.json()["detail"]


class TestCapacityConstraints:
    def test_winner_take_all(self, client) -> None:
        """Without capacity limits the lowest price serves the whole market."""
        data = _run(
            client,
            model="bertrand",
            rounds=30,
            capacity_constraints=False,
            firms=[{"cost": 10.0}, {"cost": 20.0}],
        )
        assert data["params"]["capacity_constraints"] is False
        for round_idx in range(20, 30):
            firms = data["results"][str(round_idx)].values()
            assert sum(f["quantity"] > 0 for f in firms) == 1
            # Price is driven to the rival's cost (limit pricing)
            assert data["metrics"][str(round_idx)]["market_price"] == pytest.approx(
                20.0, rel=0.02
            )

    def test_capacity_constrained_default_shares_market(self, client) -> None:
        data = _run(
            client, model="bertrand", rounds=30, firms=[{"cost": 10.0}, {"cost": 20.0}]
        )
        last = data["results"]["29"]
        assert last["firm_1"]["quantity"] > 0.0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"model": "cournot"},
            {"demand_type": "isoelastic"},
            {"segments": [{"alpha": 100.0, "beta": 1.0, "weight": 1.0}]},
        ],
    )
    def test_invalid(self, client, overrides) -> None:
        body = {"model": "bertrand", "capacity_constraints": False, **overrides}
        response = _simulate(client, **body)
        assert response.status_code == 400
        assert "capacity_constraints" in response.json()["detail"]


class TestCollusionStrategies:
    @pytest.mark.parametrize("strategy_type", ["cartel", "collusive", "opportunistic"])
    def test_cartel_on_isoelastic_market(self, client, strategy_type) -> None:
        response = _simulate(
            client,
            rounds=10,
            demand_type="isoelastic",
            advanced_strategies=[
                {"firm_id": 0, "strategy_type": "cartel"},
                {"firm_id": 1, "strategy_type": strategy_type},
            ],
        )
        assert response.status_code == 200, response.text
        run_id = response.json()["run_id"]
        events = client.get(f"/runs/{run_id}/events").json()["events"]
        assert any(e["event_type"] == "cartel_formed" for e in events)


class TestMarketEvolution:
    def test_entry_is_recorded(self, client) -> None:
        response = _simulate(
            client,
            rounds=40,
            params={"a": 200.0, "b": 1.0},
            seed=9,
            market_evolution={
                "growth_rate": 0.05,
                "entry_cost": 10.0,
                "innovation_rate": 0.5,
            },
        )
        assert response.status_code == 200, response.text
        run_id = response.json()["run_id"]
        data = client.get(f"/runs/{run_id}").json()
        assert max(m["num_firms"] for m in data["metrics"].values()) > 2
        events = client.get(f"/runs/{run_id}/events").json()["events"]
        assert any(e["event_type"] == "firm_entry" for e in events)
        replay = client.get(f"/runs/{run_id}/replay")
        assert replay.status_code == 200, replay.text

    def test_metrics_follow_demand_growth(self, client) -> None:
        """Surplus uses the grown demand curve, not the initial one."""
        data = _run(
            client,
            rounds=10,
            demand_type="isoelastic",
            market_evolution={
                "growth_rate": 0.1,
                "entry_cost": 1e9,
                "innovation_rate": 0.0,
            },
        )
        for round_idx in ("0", "9"):
            firms = data["results"][round_idx].values()
            price = next(iter(firms))["price"]
            total = sum(f["quantity"] for f in firms)
            # CS = integral of Q(p) above P = P * Q / (e - 1) for isoelastic demand
            assert data["metrics"][round_idx]["consumer_surplus"] == pytest.approx(
                price * total / (2.0 - 1)
            )

    def test_defaults(self, client) -> None:
        data = _run(client, market_evolution={})
        assert data["params"]["market_evolution"]["entry_cost"] == 100.0

    def test_invalid(self, client) -> None:
        response = _simulate(client, market_evolution={"innovation_rate": 2.0})
        assert response.status_code == 422


def test_compare_with_new_features(client) -> None:
    scenario = {
        "model": "bertrand",
        "rounds": 5,
        "firms": [{"cost": 10.0}, {"cost": 12.0}],
        "seed": 3,
    }
    response = client.post(
        "/compare",
        json={
            "left_config": scenario,
            "right_config": {
                **scenario,
                "enhanced_demand": {"demand_type": "ces", "elasticity": 3.0},
                "advanced_strategies": [
                    {"firm_id": 1, "strategy_type": "fictitious_play"}
                ],
            },
        },
    )
    assert response.status_code == 200, response.text
    ids = response.json()
    result = client.get(f"/compare/{ids['left_run_id']}/{ids['right_run_id']}")
    assert result.status_code == 200, result.text
    body = result.json()
    for side in ("left_metrics", "right_metrics", "deltas"):
        assert all(v is not None for v in body[side]["consumer_surplus"])


def test_compare_reports_scenario_in_errors(client) -> None:
    scenario = {"model": "cournot", "rounds": 5, "firms": [{"cost": 10.0}]}
    response = client.post(
        "/compare",
        json={
            "left_config": scenario,
            "right_config": {**scenario, "enhanced_demand": {"demand_type": "ces"}},
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Right scenario:")
