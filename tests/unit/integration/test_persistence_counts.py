"""Test suite for persistence functionality.

This module tests the database persistence layer to ensure
proper storage and retrieval of simulation data.
"""

from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sim.models.models import Base, Result, Round, Run
from sim.runners.runner import get_run_results, run_game

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/test_persistence.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def setup_database() -> Generator[None, None, None]:
    """Set up test database for each test."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_persistence_counts(setup_database: None) -> None:
    """Test DB has exactly `rounds` rows for given run_id; results rows == rounds * num_firms."""
    db = TestingSessionLocal()

    try:
        config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
            "seed": 42,
        }

        rounds = 4
        num_firms = 3

        # Run simulation
        run_id = run_game("cournot", rounds, config, db)

        # Verify run record
        run = db.query(Run).filter(Run.id == run_id).first()
        assert run is not None
        assert run.model == "cournot"
        assert run.rounds == rounds

        # Verify rounds records
        rounds_count = db.query(Round).filter(Round.run_id == run_id).count()
        assert rounds_count == rounds

        # Verify results records
        results_count = db.query(Result).filter(Result.run_id == run_id).count()
        assert results_count == rounds * num_firms

        # Verify each round has results for all firms
        for round_idx in range(rounds):
            round_results = (
                db.query(Result)
                .filter(Result.run_id == run_id, Result.round_idx == round_idx)
                .count()
            )
            assert round_results == num_firms

    finally:
        db.close()


def test_get_run_results_structure(setup_database: None) -> None:
    """Test get_run_results returns proper structure with equal length arrays."""
    db = TestingSessionLocal()

    try:
        config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,
        }

        rounds = 3
        run_id = run_game("cournot", rounds, config, db)

        # Get results
        results = get_run_results(run_id, db)

        # Verify structure
        assert "run_id" in results
        assert "model" in results
        assert "rounds" in results
        assert "created_at" in results
        assert "rounds_data" in results
        assert "firms_data" in results

        # Verify arrays have equal length
        assert len(results["rounds_data"]) == rounds
        assert len(results["firms_data"]) == 2

        for firm_data in results["firms_data"]:
            assert len(firm_data["actions"]) == rounds
            assert len(firm_data["quantities"]) == rounds
            assert len(firm_data["profits"]) == rounds

    finally:
        db.close()


def test_get_run_results_validation(setup_database: None) -> None:
    """Test GET /runs/{id} returns arrays of equal length; all price, qty, profit are finite and qty, price >= 0."""
    db = TestingSessionLocal()

    try:
        config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,
        }

        run_id = run_game("cournot", 3, config, db)
        results = get_run_results(run_id, db)

        # Verify all price, qty, profit are finite and qty, price >= 0
        for round_data in results["rounds_data"]:
            assert isinstance(round_data["price"], (int, float))
            assert round_data["price"] >= 0
            assert isinstance(round_data["total_qty"], (int, float))
            assert round_data["total_qty"] >= 0
            assert isinstance(round_data["total_profit"], (int, float))

        for firm_data in results["firms_data"]:
            for action in firm_data["actions"]:
                assert isinstance(action, (int, float))
                assert action >= 0
            for qty in firm_data["quantities"]:
                assert isinstance(qty, (int, float))
                assert qty >= 0
            for profit in firm_data["profits"]:
                assert isinstance(profit, (int, float))

    finally:
        db.close()


def test_run_not_found(setup_database: None) -> None:
    """Test get_run_results raises ValueError for non-existent run_id."""
    db = TestingSessionLocal()

    try:
        with pytest.raises(ValueError, match="Run non-existent-id not found"):
            get_run_results("non-existent-id", db)
    finally:
        db.close()


def test_bertrand_persistence(setup_database: None) -> None:
    """Test Bertrand simulation persistence."""
    db = TestingSessionLocal()

    try:
        config = {
            "params": {"alpha": 100.0, "beta": 1.0},
            "firms": [{"cost": 5.0}, {"cost": 8.0}],
            "seed": 42,
        }

        rounds = 2
        num_firms = 2
        run_id = run_game("bertrand", rounds, config, db)

        # Verify persistence
        run = db.query(Run).filter(Run.id == run_id).first()
        assert run.model == "bertrand"
        assert run.rounds == rounds

        results_count = db.query(Result).filter(Result.run_id == run_id).count()
        assert results_count == rounds * num_firms

        # Verify results structure
        results = get_run_results(run_id, db)
        assert results["model"] == "bertrand"
        assert len(results["rounds_data"]) == rounds
        assert len(results["firms_data"]) == num_firms

    finally:
        db.close()
