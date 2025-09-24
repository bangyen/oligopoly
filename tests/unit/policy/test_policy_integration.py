"""Integration test demonstrating policy shocks in action.

This test shows how policy shocks affect simulation outcomes by comparing
results with and without policy interventions.
"""

import atexit
import math
import os
import tempfile
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.sim.models.models import Base
from src.sim.policy.policy_shocks import PolicyEvent, PolicyType
from src.sim.runners.runner import get_run_results, run_game

# Create temporary database file
temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
temp_file.close()

# Test database setup - use temporary database
SQLALCHEMY_DATABASE_URL = f"sqlite:///{temp_file.name}"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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


def test_tax_policy_integration(setup_database: None) -> None:
    """Test that tax policy reduces profits by the expected amount."""
    db = TestingSessionLocal()

    try:
        # Run simulation without tax
        config_no_tax = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,
            "events": [],
        }

        run_id_no_tax = run_game("cournot", 3, config_no_tax, db)
        results_no_tax = get_run_results(run_id_no_tax, db)

        # Run simulation with 20% tax on round 1
        config_with_tax = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,  # Same seed for reproducibility
            "events": [PolicyEvent(round_idx=1, policy_type=PolicyType.TAX, value=0.2)],
        }

        run_id_with_tax = run_game("cournot", 3, config_with_tax, db)
        results_with_tax = get_run_results(run_id_with_tax, db)

        # Compare results
        rounds_data_no_tax = results_no_tax["rounds_data"]
        rounds_data_with_tax = results_with_tax["rounds_data"]

        # Round 0 should be identical (no tax applied)
        round_0_no_tax = rounds_data_no_tax[0]
        round_0_with_tax = rounds_data_with_tax[0]

        assert math.isclose(
            round_0_no_tax["total_profit"],
            round_0_with_tax["total_profit"],
            abs_tol=1e-6,
        ), f"Round 0 total profits should be identical, got {round_0_no_tax['total_profit']} vs {round_0_with_tax['total_profit']}"

        # Round 1 should have 20% tax applied
        round_1_no_tax = rounds_data_no_tax[1]
        round_1_with_tax = rounds_data_with_tax[1]

        expected_taxed_profit = round_1_no_tax["total_profit"] * 0.8  # 20% tax
        assert math.isclose(
            round_1_with_tax["total_profit"], expected_taxed_profit, abs_tol=1e-6
        ), f"Round 1: expected {expected_taxed_profit}, got {round_1_with_tax['total_profit']}"

        # Round 2 should be identical again (no tax applied)
        round_2_no_tax = rounds_data_no_tax[2]
        round_2_with_tax = rounds_data_with_tax[2]

        assert math.isclose(
            round_2_no_tax["total_profit"],
            round_2_with_tax["total_profit"],
            abs_tol=1e-6,
        ), f"Round 2 total profits should be identical, got {round_2_no_tax['total_profit']} vs {round_2_with_tax['total_profit']}"

    finally:
        db.close()


def test_subsidy_policy_integration(setup_database: None) -> None:
    """Test that subsidy policy increases profits by the expected amount."""
    db = TestingSessionLocal()

    try:
        # Run simulation without subsidy
        config_no_subsidy = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,
            "events": [],
        }

        run_id_no_subsidy = run_game("cournot", 2, config_no_subsidy, db)
        results_no_subsidy = get_run_results(run_id_no_subsidy, db)

        # Run simulation with $5 per-unit subsidy on round 0
        config_with_subsidy = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,  # Same seed for reproducibility
            "events": [
                PolicyEvent(round_idx=0, policy_type=PolicyType.SUBSIDY, value=5.0)
            ],
        }

        run_id_with_subsidy = run_game("cournot", 2, config_with_subsidy, db)
        results_with_subsidy = get_run_results(run_id_with_subsidy, db)

        # Compare results
        rounds_data_no_subsidy = results_no_subsidy["rounds_data"]
        rounds_data_with_subsidy = results_with_subsidy["rounds_data"]

        # Round 0 should have subsidy applied
        round_0_no_subsidy = rounds_data_no_subsidy[0]
        round_0_with_subsidy = rounds_data_with_subsidy[0]

        # Calculate expected subsidy effect
        total_qty = round_0_no_subsidy["total_qty"]
        expected_subsidized_profit = (
            round_0_no_subsidy["total_profit"] + 5.0 * total_qty
        )

        assert math.isclose(
            round_0_with_subsidy["total_profit"],
            expected_subsidized_profit,
            abs_tol=1e-6,
        ), f"Round 0: expected {expected_subsidized_profit}, got {round_0_with_subsidy['total_profit']}"

        # Round 1 should be identical (no subsidy applied)
        round_1_no_subsidy = rounds_data_no_subsidy[1]
        round_1_with_subsidy = rounds_data_with_subsidy[1]

        assert math.isclose(
            round_1_no_subsidy["total_profit"],
            round_1_with_subsidy["total_profit"],
            abs_tol=1e-6,
        ), f"Round 1 total profits should be identical, got {round_1_no_subsidy['total_profit']} vs {round_1_with_subsidy['total_profit']}"

    finally:
        db.close()


def test_price_cap_policy_integration(setup_database: None) -> None:
    """Test that price cap policy limits prices and recalculates profits."""
    db = TestingSessionLocal()

    try:
        # Run simulation without price cap
        config_no_cap = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,
            "events": [],
        }

        run_id_no_cap = run_game("cournot", 2, config_no_cap, db)
        results_no_cap = get_run_results(run_id_no_cap, db)

        # Run simulation with price cap of $50 on round 0
        config_with_cap = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "seed": 42,  # Same seed for reproducibility
            "events": [
                PolicyEvent(round_idx=0, policy_type=PolicyType.PRICE_CAP, value=50.0)
            ],
        }

        run_id_with_cap = run_game("cournot", 2, config_with_cap, db)
        results_with_cap = get_run_results(run_id_with_cap, db)

        # Compare results
        rounds_data_no_cap = results_no_cap["rounds_data"]
        rounds_data_with_cap = results_with_cap["rounds_data"]

        # Round 0 should have price cap applied if needed
        round_0_no_cap = rounds_data_no_cap[0]
        round_0_with_cap = rounds_data_with_cap[0]

        # Check that price is capped
        price_no_cap = round_0_no_cap["price"]
        price_with_cap = round_0_with_cap["price"]

        assert (
            price_with_cap <= 50.0
        ), f"Price should be capped at 50.0, got {price_with_cap}"

        if price_no_cap > 50.0:
            # If original price exceeded cap, profits should be recalculated
            # We can't easily calculate expected profits without knowing individual firm quantities
            # So we just verify that the price is capped and profits are different
            assert (
                price_with_cap == 50.0
            ), f"Price should be exactly 50.0 when capped, got {price_with_cap}"
            assert not math.isclose(
                round_0_no_cap["total_profit"],
                round_0_with_cap["total_profit"],
                abs_tol=1e-6,
            ), "Profits should be different when price cap is applied"

        # Round 1 should be identical (no cap applied)
        round_1_no_cap = rounds_data_no_cap[1]
        round_1_with_cap = rounds_data_with_cap[1]

        assert math.isclose(
            round_1_no_cap["total_profit"],
            round_1_with_cap["total_profit"],
            abs_tol=1e-6,
        ), f"Round 1 total profits should be identical, got {round_1_no_cap['total_profit']} vs {round_1_with_cap['total_profit']}"

    finally:
        db.close()
