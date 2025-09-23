"""Tests for experiment runner metrics calculation.

This module tests that the experiment runner correctly calculates and exports
summary metrics including average price, HHI, consumer surplus, and profits.
"""

import csv
import math
import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.sim.experiments.runner import ExperimentConfig, ExperimentRunner
from src.sim.models.models import Base


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()

    engine = create_engine(f"sqlite:///{temp_file.name}")
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    Base.metadata.create_all(bind=engine)

    db = session_local()
    try:
        yield db
    finally:
        db.close()
        os.unlink(temp_file.name)


@pytest.fixture
def temp_artifacts_dir():
    """Create a temporary artifacts directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil

    shutil.rmtree(temp_dir)


def test_csv_contains_required_metrics_columns(temp_db, temp_artifacts_dir):
    """Test that CSV contains all required metrics columns."""
    config = ExperimentConfig(
        config_id="metrics_test",
        model="cournot",
        rounds=5,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
    )

    runner = ExperimentRunner(temp_artifacts_dir)
    experiments = [config]
    seeds_per_config = 1

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read CSV headers
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

    # Check required metrics columns
    required_metrics = [
        "avg_price",  # Average market price across rounds
        "avg_hhi",  # Average Herfindahl-Hirschman Index
        "avg_cs",  # Average consumer surplus
        "total_profit",  # Total profit across all firms and rounds
        "mean_profit_per_firm",  # Average profit per firm
    ]

    for metric in required_metrics:
        assert metric in headers, f"Missing required metric column: {metric}"

    # Check firm-specific profit columns
    firm_profit_columns = [
        col for col in headers if col.startswith("firm_") and col.endswith("_profit")
    ]
    assert len(firm_profit_columns) == 3  # Should have 3 firms

    # Check that firm columns are numbered correctly
    expected_firm_cols = ["firm_0_profit", "firm_1_profit", "firm_2_profit"]
    for col in expected_firm_cols:
        assert col in headers, f"Missing firm profit column: {col}"


def test_metrics_values_are_reasonable(temp_db, temp_artifacts_dir):
    """Test that calculated metrics have reasonable values."""
    config = ExperimentConfig(
        config_id="reasonable_test",
        model="cournot",
        rounds=10,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
    )

    runner = ExperimentRunner(temp_artifacts_dir)
    experiments = [config]
    seeds_per_config = 1

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read results
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    assert len(results) == 1
    result = results[0]

    # Check that metrics are reasonable
    avg_price = float(result["avg_price"])
    avg_hhi = float(result["avg_hhi"])
    avg_cs = float(result["avg_cs"])
    total_profit = float(result["total_profit"])
    mean_profit_per_firm = float(result["mean_profit_per_firm"])

    # Price should be positive and reasonable for demand curve a=100, b=1
    assert avg_price > 0, f"Average price should be positive, got {avg_price}"
    assert avg_price < 100, (
        f"Average price should be less than demand intercept, got {avg_price}"
    )

    # HHI should be between 0 and 1 (or 0 and 10000 in percentage terms)
    assert 0 <= avg_hhi <= 1, f"HHI should be between 0 and 1, got {avg_hhi}"

    # Consumer surplus should be non-negative
    assert avg_cs >= 0, f"Consumer surplus should be non-negative, got {avg_cs}"

    # Profits should be reasonable (could be negative if costs are high)
    assert isinstance(total_profit, (int, float)), (
        f"Total profit should be numeric, got {total_profit}"
    )
    assert isinstance(mean_profit_per_firm, (int, float)), (
        f"Mean profit per firm should be numeric, got {mean_profit_per_firm}"
    )

    # Mean profit per firm should equal total profit / number of firms
    expected_mean = total_profit / 2  # 2 firms
    assert math.isclose(mean_profit_per_firm, expected_mean, abs_tol=1e-6), (
        f"Mean profit per firm should equal total_profit/num_firms: {mean_profit_per_firm} vs {expected_mean}"
    )


def test_bertrand_vs_cournot_metrics_differ(temp_db, temp_artifacts_dir):
    """Test that Bertrand and Cournot models produce different metrics."""
    cournot_config = ExperimentConfig(
        config_id="cournot_test",
        model="cournot",
        rounds=5,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
    )

    bertrand_config = ExperimentConfig(
        config_id="bertrand_test",
        model="bertrand",
        rounds=5,
        params={"alpha": 100.0, "beta": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
    )

    runner = ExperimentRunner(temp_artifacts_dir)
    experiments = [cournot_config, bertrand_config]
    seeds_per_config = 1

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read results
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    assert len(results) == 2

    # Find results for each model
    cournot_result = next(r for r in results if r["model"] == "cournot")
    bertrand_result = next(r for r in results if r["model"] == "bertrand")

    # Compare key metrics - they should be different
    cournot_price = float(cournot_result["avg_price"])
    bertrand_price = float(bertrand_result["avg_price"])

    cournot_hhi = float(cournot_result["avg_hhi"])
    bertrand_hhi = float(bertrand_result["avg_hhi"])

    # Prices should be different (Bertrand typically has lower prices)
    assert not math.isclose(cournot_price, bertrand_price, abs_tol=1e-6), (
        f"Cournot and Bertrand prices should differ: {cournot_price} vs {bertrand_price}"
    )

    # HHI values should also differ
    assert not math.isclose(cournot_hhi, bertrand_hhi, abs_tol=1e-6), (
        f"Cournot and Bertrand HHI should differ: {cournot_hhi} vs {bertrand_hhi}"
    )


def test_firm_specific_profits_sum_to_total(temp_db, temp_artifacts_dir):
    """Test that individual firm profits sum to total profit."""
    config = ExperimentConfig(
        config_id="profit_sum_test",
        model="cournot",
        rounds=5,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
    )

    runner = ExperimentRunner(temp_artifacts_dir)
    experiments = [config]
    seeds_per_config = 1

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read results
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    assert len(results) == 1
    result = results[0]

    # Get individual firm profits
    firm_profits = []
    for i in range(3):  # 3 firms
        firm_profit = float(result[f"firm_{i}_profit"])
        firm_profits.append(firm_profit)

    # Sum of individual profits should equal total profit
    sum_firm_profits = sum(firm_profits)
    total_profit = float(result["total_profit"])

    assert math.isclose(sum_firm_profits, total_profit, abs_tol=1e-6), (
        f"Sum of firm profits ({sum_firm_profits}) should equal total profit ({total_profit})"
    )


def test_segmented_demand_metrics(temp_db, temp_artifacts_dir):
    """Test metrics calculation with segmented demand."""
    config = ExperimentConfig(
        config_id="segmented_test",
        model="cournot",
        rounds=5,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
        segments=[
            {"alpha": 200.0, "beta": 1.0, "weight": 0.6},
            {"alpha": 150.0, "beta": 0.8, "weight": 0.4},
        ],
    )

    runner = ExperimentRunner(temp_artifacts_dir)
    experiments = [config]
    seeds_per_config = 1

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read results
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    assert len(results) == 1
    result = results[0]

    # Metrics should be calculated correctly for segmented demand
    avg_price = float(result["avg_price"])
    avg_hhi = float(result["avg_hhi"])
    avg_cs = float(result["avg_cs"])

    # All metrics should be reasonable
    assert avg_price > 0, (
        f"Average price should be positive for segmented demand, got {avg_price}"
    )
    assert 0 <= avg_hhi <= 1, (
        f"HHI should be between 0 and 1 for segmented demand, got {avg_hhi}"
    )
    assert avg_cs >= 0, (
        f"Consumer surplus should be non-negative for segmented demand, got {avg_cs}"
    )


def test_policy_events_affect_metrics(temp_db, temp_artifacts_dir):
    """Test that policy events affect the calculated metrics."""
    # Baseline config without policies
    baseline_config = ExperimentConfig(
        config_id="baseline",
        model="cournot",
        rounds=5,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
    )

    # Config with tax policy
    tax_config = ExperimentConfig(
        config_id="with_tax",
        model="cournot",
        rounds=5,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
        policies=[{"round_idx": 2, "policy_type": "TAX", "value": 0.1}],
    )

    runner = ExperimentRunner(temp_artifacts_dir)
    experiments = [baseline_config, tax_config]
    seeds_per_config = 1

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read results
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    assert len(results) == 2

    # Find results for each config
    baseline_result = next(r for r in results if r["config_id"] == "baseline")
    tax_result = next(r for r in results if r["config_id"] == "with_tax")

    # Tax should reduce profits
    baseline_profit = float(baseline_result["total_profit"])
    tax_profit = float(tax_result["total_profit"])

    assert tax_profit < baseline_profit, (
        f"Tax policy should reduce profits: baseline {baseline_profit} vs taxed {tax_profit}"
    )

    # Prices might also be affected, but the adaptive strategy might make them similar
    baseline_price = float(baseline_result["avg_price"])
    tax_price = float(tax_result["avg_price"])

    # Just check that both are reasonable (not zero)
    assert baseline_price > 0, f"Baseline price should be positive: {baseline_price}"
    assert tax_price > 0, f"Tax price should be positive: {tax_price}"
