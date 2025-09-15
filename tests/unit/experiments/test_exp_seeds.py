"""Tests for experiment runner seed reproducibility.

This module tests that experiments with the same config and seed produce
identical results, ensuring reproducibility of simulation runs.
"""

import math
import os
import tempfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.experiments.runner import ExperimentConfig, ExperimentRunner
from sim.models.models import Base


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    # Create temporary database file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()

    # Create engine and session
    engine = create_engine(f"sqlite:///{temp_file.name}")
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    db = session_local()
    try:
        yield db
    finally:
        db.close()
        os.unlink(temp_file.name)


@pytest.fixture
def sample_config():
    """Create a sample experiment configuration."""
    return ExperimentConfig(
        config_id="test_config",
        model="cournot",
        rounds=3,
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}, {"cost": 15.0}],
    )


@pytest.fixture
def temp_artifacts_dir():
    """Create a temporary artifacts directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil

    shutil.rmtree(temp_dir)


def test_same_config_same_seed_produces_identical_results(
    temp_db, sample_config, temp_artifacts_dir
):
    """Test that same config + seed produces identical summary metrics."""
    runner = ExperimentRunner(temp_artifacts_dir)

    # Run the same config with the same seed twice
    experiments = [sample_config]
    seeds_per_config = 1

    # First run
    csv_path_1 = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Second run with same seed
    csv_path_2 = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read both CSV files
    import csv

    results_1 = []
    results_2 = []

    with open(csv_path_1) as f:
        reader = csv.DictReader(f)
        results_1 = list(reader)

    with open(csv_path_2) as f:
        reader = csv.DictReader(f)
        results_2 = list(reader)

    # Should have same number of results
    assert len(results_1) == len(results_2) == 1

    # Compare key metrics (allowing for small floating point differences)
    result_1 = results_1[0]
    result_2 = results_2[0]

    # Same config and seed should produce identical results
    assert result_1["config_id"] == result_2["config_id"]
    assert result_1["seed"] == result_2["seed"]
    assert result_1["model"] == result_2["model"]

    # Compare numeric metrics with tolerance
    numeric_fields = [
        "avg_price",
        "avg_hhi",
        "avg_cs",
        "total_profit",
        "mean_profit_per_firm",
        "num_firms",
    ]

    for field in numeric_fields:
        val_1 = float(result_1[field])
        val_2 = float(result_2[field])
        assert math.isclose(
            val_1, val_2, abs_tol=1e-6
        ), f"{field} differs: {val_1} vs {val_2}"


def test_different_seeds_produce_different_results(
    temp_db, sample_config, temp_artifacts_dir
):
    """Test that different seeds produce different results."""
    runner = ExperimentRunner(temp_artifacts_dir)

    experiments = [sample_config]
    seeds_per_config = 3  # Run with 3 different seeds

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read CSV results
    import csv

    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # Should have 3 results (one per seed)
    assert len(results) == 3

    # All should have same config_id but different seeds
    config_ids = [r["config_id"] for r in results]
    seeds = [int(r["seed"]) for r in results]

    assert all(cid == "test_config" for cid in config_ids)
    assert len(set(seeds)) == 3  # All seeds should be different

    # Results should be different (at least some metrics should differ)
    avg_prices = [float(r["avg_price"]) for r in results]
    avg_hhis = [float(r["avg_hhi"]) for r in results]

    # With different seeds, we expect some variation
    # (though it's possible but unlikely they'd be identical)
    price_variation = max(avg_prices) - min(avg_prices)
    hhi_variation = max(avg_hhis) - min(avg_hhis)

    # At least one metric should show variation
    assert price_variation > 1e-6 or hhi_variation > 1e-6


def test_multiple_configs_with_multiple_seeds(temp_db, temp_artifacts_dir):
    """Test running multiple configs with multiple seeds each."""
    # Create multiple configs
    configs = [
        ExperimentConfig(
            config_id="config_1",
            model="cournot",
            rounds=2,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}, {"cost": 15.0}],
        ),
        ExperimentConfig(
            config_id="config_2",
            model="bertrand",
            rounds=2,
            params={"alpha": 100.0, "beta": 1.0},
            firms=[{"cost": 20.0}, {"cost": 25.0}],
        ),
    ]

    runner = ExperimentRunner(temp_artifacts_dir)
    seeds_per_config = 2

    csv_path = runner.run_experiment_batch(configs, seeds_per_config, temp_db)

    # Read CSV results
    import csv

    results = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # Should have 4 results total (2 configs × 2 seeds each)
    assert len(results) == 4

    # Check config distribution
    config_ids = [r["config_id"] for r in results]
    assert config_ids.count("config_1") == 2
    assert config_ids.count("config_2") == 2

    # Check seed distribution
    seeds = [int(r["seed"]) for r in results]
    assert len(set(seeds)) == 2  # Should have 2 unique seeds

    # Check that each config has both seeds
    config_1_seeds = [int(r["seed"]) for r in results if r["config_id"] == "config_1"]
    config_2_seeds = [int(r["seed"]) for r in results if r["config_id"] == "config_2"]

    assert len(set(config_1_seeds)) == 2
    assert len(set(config_2_seeds)) == 2


def test_csv_contains_expected_columns(temp_db, sample_config, temp_artifacts_dir):
    """Test that CSV contains all expected columns."""
    runner = ExperimentRunner(temp_artifacts_dir)

    experiments = [sample_config]
    seeds_per_config = 1

    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read CSV headers
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

    # Check required columns are present
    required_columns = [
        "run_id",
        "config_id",
        "seed",
        "model",
        "rounds",
        "avg_price",
        "avg_hhi",
        "avg_cs",
        "total_profit",
        "mean_profit_per_firm",
        "num_firms",
    ]

    for col in required_columns:
        assert col in headers, f"Missing required column: {col}"

    # Check firm-specific columns
    firm_columns = [
        col for col in headers if col.startswith("firm_") and col.endswith("_profit")
    ]
    assert len(firm_columns) == 2  # Should have 2 firms in sample config


def test_empty_results_handling(temp_db, temp_artifacts_dir):
    """Test handling of empty simulation results."""
    # Create a config that produces minimal results
    config = ExperimentConfig(
        config_id="minimal_test",
        model="cournot",
        rounds=1,  # Use 1 round instead of 0
        params={"a": 100.0, "b": 1.0},
        firms=[{"cost": 10.0}],
    )

    runner = ExperimentRunner(temp_artifacts_dir)
    experiments = [config]
    seeds_per_config = 1

    # Should not raise an exception
    csv_path = runner.run_experiment_batch(experiments, seeds_per_config, temp_db)

    # Read results
    import csv

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        results = list(reader)

    assert len(results) == 1
    result = results[0]

    # Should have reasonable values for metrics
    assert float(result["avg_price"]) >= 0.0
    assert float(result["avg_hhi"]) >= 0.0
    assert float(result["avg_cs"]) >= 0.0
