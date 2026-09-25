"""Runner integration tests for non-linear demand, learning strategies and
market evolution."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.markets import CESBertrandMarket, IsoelasticCournotMarket
from sim.models.models import Base, Event, Run
from sim.runners.runner import get_run_results, run_game
from sim.strategies.learning import LEARNING_STRATEGY_TYPES


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _last_round(db, run_id):
    results = get_run_results(run_id, db)["results"]
    return results[str(max(int(k) for k in results))]


class TestNonLinearDemand:
    def test_isoelastic_cournot_converges_to_nash(self, db) -> None:
        costs = [10.0, 12.0]
        params = {"demand_type": "isoelastic", "A": 100.0, "elasticity": 2.0}
        run_id = run_game(
            "cournot",
            60,
            {"params": params, "firms": [{"cost": c} for c in costs], "seed": 1},
            db,
        )
        expected, price = IsoelasticCournotMarket(params).equilibrium(costs)
        last = _last_round(db, run_id)
        assert [last[f"firm_{i}"]["quantity"] for i in range(2)] == pytest.approx(
            expected, rel=0.02
        )
        assert last["firm_0"]["price"] == pytest.approx(price, rel=0.02)
        run = db.query(Run).filter(Run.id == run_id).one()
        assert run.params["demand_type"] == "isoelastic"

    def test_isoelastic_bertrand_limit_prices(self, db) -> None:
        run_id = run_game(
            "bertrand",
            60,
            {
                "params": {"demand_type": "isoelastic", "A": 100.0, "elasticity": 2.0},
                "firms": [{"cost": 10.0}, {"cost": 15.0}],
                "seed": 3,
            },
            db,
        )
        last = _last_round(db, run_id)
        # Undercutting drives the market price to the rival's cost (limit price)
        market_price = min(f["price"] for f in last.values())
        assert market_price == pytest.approx(15.0, rel=0.02)
        total = sum(f["quantity"] for f in last.values())
        assert total == pytest.approx((100.0 / market_price) ** 2)

    def test_ces_bertrand_converges_to_nash(self, db) -> None:
        costs = [10.0, 12.0, 14.0]
        params = {"demand_type": "ces", "elasticity": 3.0, "market_size": 300.0}
        run_id = run_game(
            "bertrand",
            60,
            {"params": params, "firms": [{"cost": c} for c in costs], "seed": 2},
            db,
        )
        expected = CESBertrandMarket(params, 3).nash(costs)
        last = _last_round(db, run_id)
        assert [last[f"firm_{i}"]["price"] for i in range(3)] == pytest.approx(
            expected, rel=0.05
        )
        # Differentiated products: every firm sells something
        assert all(last[f"firm_{i}"]["quantity"] > 0 for i in range(3))

    @pytest.mark.parametrize(
        ("model", "params", "firms", "message"),
        [
            ("cournot", {"demand_type": "ces"}, 2, "bertrand"),
            ("bertrand", {"demand_type": "ces", "market_elasticity": 1.0}, 2, "> 1"),
            ("cournot", {"demand_type": "isoelastic", "elasticity": 0.5}, 2, "> 1"),
            ("cournot", {"demand_type": "logit"}, 2, "Unknown demand_type"),
        ],
    )
    def test_invalid_specs(self, db, model, params, firms, message) -> None:
        with pytest.raises(ValueError, match=message):
            run_game(
                model, 5, {"params": params, "firms": [{"cost": 10.0}] * firms}, db
            )


class TestLearningStrategies:
    @pytest.mark.parametrize("model", ["cournot", "bertrand"])
    @pytest.mark.parametrize("strategy_type", LEARNING_STRATEGY_TYPES)
    def test_each_strategy_runs(self, db, model, strategy_type) -> None:
        params = {"a": 100.0, "b": 1.0} if model == "cournot" else {}
        run_id = run_game(
            model,
            15,
            {
                "params": params,
                "firms": [{"cost": 10.0}, {"cost": 12.0}],
                "seed": 7,
                "advanced_strategies": [{"firm_id": 1, "strategy_type": strategy_type}],
            },
            db,
        )
        results = get_run_results(run_id, db)["results"]
        assert len(results) == 15
        run = db.query(Run).filter(Run.id == run_id).one()
        assert run.params["advanced_strategies"][0]["strategy_type"] == strategy_type

    def test_fictitious_play_reaches_cournot_nash(self, db) -> None:
        """Both firms best-responding to beliefs settle at the Nash quantities."""
        run_id = run_game(
            "cournot",
            40,
            {
                "params": {"a": 100.0, "b": 1.0},
                "firms": [{"cost": 10.0}, {"cost": 10.0}],
                "seed": 11,
                "advanced_strategies": [
                    {
                        "firm_id": i,
                        "strategy_type": "fictitious_play",
                        "exploration_rate": 0.0,
                    }
                    for i in range(2)
                ],
            },
            db,
        )
        last = _last_round(db, run_id)
        assert last["firm_0"]["quantity"] == pytest.approx(30.0, rel=0.05)
        assert last["firm_1"]["quantity"] == pytest.approx(30.0, rel=0.05)

    def test_learning_strategy_on_nonlinear_market(self, db) -> None:
        run_id = run_game(
            "bertrand",
            20,
            {
                "params": {"demand_type": "ces", "elasticity": 3.0},
                "firms": [{"cost": 10.0}, {"cost": 10.0}],
                "seed": 5,
                "advanced_strategies": [
                    {"firm_id": 0, "strategy_type": "fictitious_play"}
                ],
            },
            db,
        )
        assert len(get_run_results(run_id, db)["results"]) == 20

    def test_seeded_runs_are_reproducible(self, db) -> None:
        config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 12.0}],
            "seed": 42,
            "advanced_strategies": [
                {"firm_id": 0, "strategy_type": "deep_q_learning"},
                {"firm_id": 1, "strategy_type": "q_learning"},
            ],
        }
        first = get_run_results(run_game("cournot", 10, config, db), db)["results"]
        second = get_run_results(run_game("cournot", 10, config, db), db)["results"]
        assert first == second

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ({"firm_id": 5, "strategy_type": "q_learning"}, "out of range"),
            ({"firm_id": 0, "strategy_type": "genetic"}, "Unknown learning"),
        ],
    )
    def test_invalid_specs(self, db, spec, message) -> None:
        with pytest.raises((ValueError, RuntimeError), match=message):
            run_game(
                "cournot",
                5,
                {"firms": [{"cost": 10.0}] * 2, "advanced_strategies": [spec]},
                db,
            )


class TestMarketEvolution:
    def _run(self, db, evolution, rounds=40, model="cournot", params=None):
        return run_game(
            model,
            rounds,
            {
                "params": params or {"a": 200.0, "b": 1.0},
                "firms": [{"cost": 10.0}, {"cost": 12.0}],
                "seed": 9,
                "market_evolution": evolution,
            },
            db,
        )

    def test_profitable_market_attracts_entry(self, db) -> None:
        run_id = self._run(
            db, {"growth_rate": 0.05, "entry_cost": 10.0, "innovation_rate": 0.5}
        )
        results = get_run_results(run_id, db)["results"]
        firm_counts = [len(results[k]) for k in sorted(results, key=int)]
        assert firm_counts[0] == 2
        assert max(firm_counts) > 2
        entries = (
            db.query(Event)
            .filter(Event.run_id == run_id, Event.event_type == "firm_entry")
            .all()
        )
        assert len(entries) == max(firm_counts) - 2 + sum(
            1
            for e in db.query(Event).filter(
                Event.run_id == run_id, Event.event_type == "firm_exit"
            )
        )

    def test_innovation_events_are_firm_level(self, db) -> None:
        run_id = self._run(
            db, {"growth_rate": 0.0, "entry_cost": 1e9, "innovation_rate": 1.0}
        )
        innovations = (
            db.query(Event)
            .filter(Event.run_id == run_id, Event.event_type == "innovation")
            .all()
        )
        assert innovations
        for event in innovations:
            data = event.event_data
            assert data["new_cost"] < data["old_cost"] * 0.97

    def test_growth_scales_demand(self, db) -> None:
        run_id = self._run(
            db,
            {"growth_rate": 0.1, "entry_cost": 1e9, "innovation_rate": 0.0},
            rounds=10,
        )
        results = get_run_results(run_id, db)["results"]
        first = sum(f["quantity"] for f in results["0"].values())
        last = sum(f["quantity"] for f in results["9"].values())
        assert last > first * 1.5  # demand grew ~2.36x over 9 rounds

    def test_evolution_with_nonlinear_demand(self, db) -> None:
        run_id = self._run(
            db,
            {"growth_rate": 0.02, "entry_cost": 1.0, "innovation_rate": 0.3},
            model="bertrand",
            params={"demand_type": "ces", "elasticity": 3.0, "market_size": 5000.0},
        )
        assert get_run_results(run_id, db)["results"]

    def test_cartel_survives_evolution(self, db) -> None:
        run_id = run_game(
            "cournot",
            30,
            {
                "params": {"a": 200.0, "b": 1.0},
                "firms": [{"cost": 10.0, "strategy_type": "cartel"}] * 2,
                "seed": 4,
                "market_evolution": {"entry_cost": 10.0, "growth_rate": 0.02},
            },
            db,
        )
        assert get_run_results(run_id, db)["results"]


def _events(db, run_id, event_type):
    return (
        db.query(Event)
        .filter(Event.run_id == run_id, Event.event_type == event_type)
        .all()
    )


class TestCollusion:
    def _cartel_run(self, db, model, params, rounds=20, strategy="cartel", n=2):
        return run_game(
            model,
            rounds,
            {
                "params": params,
                "firms": [{"cost": 10.0}] * n,
                "seed": 1,
                "advanced_strategies": [
                    {"firm_id": i, "strategy_type": strategy} for i in range(n)
                ],
            },
            db,
        )

    def test_linear_cournot_cartel_plays_joint_monopoly(self, db) -> None:
        """Joint monopoly with a=100, b=1, c=10: Q=45 split, price 55."""
        run_id = self._cartel_run(db, "cournot", {"a": 100.0, "b": 1.0})
        last = _last_round(db, run_id)
        assert last["firm_0"]["quantity"] == pytest.approx(22.5, rel=1e-3)
        assert last["firm_1"]["quantity"] == pytest.approx(22.5, rel=1e-3)
        assert last["firm_0"]["price"] == pytest.approx(55.0, rel=1e-3)
        (formed,) = _events(db, run_id, "cartel_formed")
        assert formed.event_data["collusive_quantity"] == pytest.approx(22.5, rel=1e-3)

    def test_isoelastic_cournot_cartel(self, db) -> None:
        """Monopoly price c e/(e-1) = 20, so Q = (100/20)^2 = 25 split evenly."""
        run_id = self._cartel_run(
            db, "cournot", {"demand_type": "isoelastic", "A": 100.0, "elasticity": 2.0}
        )
        last = _last_round(db, run_id)
        assert last["firm_0"]["price"] == pytest.approx(20.0, rel=1e-3)
        assert last["firm_0"]["quantity"] == pytest.approx(12.5, rel=1e-3)

    def test_ces_bertrand_cartel_prices_above_nash(self, db) -> None:
        params = {"demand_type": "ces", "elasticity": 3.0, "market_size": 300.0}
        run_id = self._cartel_run(db, "bertrand", params, n=3)
        nash = CESBertrandMarket(params, 3).nash([10.0] * 3)[0]
        last = _last_round(db, run_id)
        assert all(f["price"] > nash * 1.2 for f in last.values())

    def test_defection_breaks_and_reforms_cartel(self, db) -> None:
        run_id = run_game(
            "cournot",
            60,
            {
                "params": {"a": 100.0, "b": 1.0},
                "firms": [{"cost": 10.0}] * 2,
                "seed": 2,
                "advanced_strategies": [
                    {"firm_id": 0, "strategy_type": "cartel"},
                    {"firm_id": 1, "strategy_type": "collusive"},
                ],
            },
            db,
        )
        formed = _events(db, run_id, "cartel_formed")
        defections = [
            e for e in _events(db, run_id, "firm_defected") if e.firm_id is not None
        ]
        assert defections, "the collusive firm should defect at some point"
        assert all(e.firm_id == 1 for e in defections)
        assert len(formed) >= 2, "cartel should re-form after breaking down"

    def test_single_colluder_forms_no_cartel(self, db) -> None:
        run_id = run_game(
            "cournot",
            10,
            {
                "params": {"a": 100.0, "b": 1.0},
                "firms": [{"cost": 10.0}] * 2,
                "advanced_strategies": [{"firm_id": 0, "strategy_type": "cartel"}],
            },
            db,
        )
        assert not _events(db, run_id, "cartel_formed")
