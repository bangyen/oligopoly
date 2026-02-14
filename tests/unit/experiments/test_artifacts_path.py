"""Tests for experiment runner artifacts and file handling.

This module tests that the experiment runner correctly creates CSV files
in the artifacts directory with proper headers and content.
"""

import csv
import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from src.sim.experiments.runner import ExperimentConfig, ExperimentRunner
from src.sim.models.models import Base


@pytest.fixture
def temp_db_url():
    """Create a temporary database for testing and return its URL."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_file.close()

    db_url = f"sqlite:///{temp_file.name}"
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)

    try:
        yield db_url
    finally:
        os.unlink(temp_file.name)


def test_artifacts_directory_creation():
    """Test that artifacts directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"

        # Directory shouldn't exist yet
        assert not artifacts_dir.exists()

        # Create runner - should create directory
        ExperimentRunner(str(artifacts_dir))

        # Directory should now exist
        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()


def test_csv_file_creation_and_naming(temp_db_url):
    """Test that CSV files are created with proper naming convention."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        runner = ExperimentRunner(str(artifacts_dir))

        config = ExperimentConfig(
            config_id="naming_test",
            model="cournot",
            rounds=3,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}],
        )

        experiments = [config]
        seeds_per_config = 1

        csv_path = runner.run_experiment_batch(
            experiments, seeds_per_config, temp_db_url
        )

        # Check file exists
        assert Path(csv_path).exists()

        # Check naming convention: exp_<timestamp>.csv
        filename = Path(csv_path).name
        assert filename.startswith("exp_")
        assert filename.endswith(".csv")

        # Should be in artifacts directory
        assert Path(csv_path).parent == artifacts_dir


def test_csv_file_is_non_empty(temp_db_url):
    """Test that generated CSV file is not empty."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        runner = ExperimentRunner(str(artifacts_dir))

        config = ExperimentConfig(
            config_id="non_empty_test",
            model="cournot",
            rounds=3,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}, {"cost": 15.0}],
        )

        experiments = [config]
        seeds_per_config = 2  # Multiple seeds to ensure content

        csv_path = runner.run_experiment_batch(
            experiments, seeds_per_config, temp_db_url
        )

        # Check file size
        file_size = Path(csv_path).stat().st_size
        assert file_size > 0, f"CSV file should not be empty, got size {file_size}"

        # Check content
        with open(csv_path) as f:
            content = f.read()
            assert len(content) > 0, "CSV file should have content"

            # Should have headers
            assert "run_id" in content
            assert "config_id" in content
            assert "avg_price" in content


def test_csv_headers_match_schema(temp_db_url):
    """Test that CSV headers match the expected schema."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        runner = ExperimentRunner(str(artifacts_dir))

        config = ExperimentConfig(
            config_id="schema_test",
            model="cournot",
            rounds=3,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}, {"cost": 15.0}, {"cost": 20.0}],
        )

        experiments = [config]
        seeds_per_config = 1

        csv_path = runner.run_experiment_batch(
            experiments, seeds_per_config, temp_db_url
        )

        # Read headers
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        # Expected schema
        expected_headers = [
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
            "firm_0_profit",
            "firm_1_profit",
            "firm_2_profit",
            "cartel_duration",
            "total_defections",
            "firm_0_strategy",
            "firm_1_strategy",
            "firm_2_strategy",
            "firm_0_defections",
            "firm_1_defections",
            "firm_2_defections",
        ]

        # Check all expected headers are present
        for header in expected_headers:
            assert header in headers, f"Missing expected header: {header}"

        # Check no unexpected headers
        unexpected_headers = set(headers) - set(expected_headers)
        assert (
            len(unexpected_headers) == 0
        ), f"Unexpected headers found: {unexpected_headers}"


def test_csv_data_types_are_correct(temp_db_url):
    """Test that CSV data has correct types."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        runner = ExperimentRunner(str(artifacts_dir))

        config = ExperimentConfig(
            config_id="data_types_test",
            model="cournot",
            rounds=3,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}, {"cost": 15.0}],
        )

        experiments = [config]
        seeds_per_config = 1

        csv_path = runner.run_experiment_batch(
            experiments, seeds_per_config, temp_db_url
        )

        # Read data
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        # Check string fields
        assert isinstance(row["run_id"], str)
        assert isinstance(row["config_id"], str)
        assert isinstance(row["model"], str)

        # Check numeric fields can be converted to numbers
        numeric_fields = [
            "seed",
            "rounds",
            "avg_price",
            "avg_hhi",
            "avg_cs",
            "total_profit",
            "mean_profit_per_firm",
            "num_firms",
            "firm_0_profit",
            "firm_1_profit",
        ]

        for field in numeric_fields:
            try:
                float(row[field])
            except ValueError:
                pytest.fail(f"Field {field} should be numeric, got: {row[field]}")


def test_multiple_experiments_in_single_csv(temp_db_url):
    """Test that multiple experiments are written to the same CSV file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        runner = ExperimentRunner(str(artifacts_dir))

        configs = [
            ExperimentConfig(
                config_id="config_1",
                model="cournot",
                rounds=2,
                params={"a": 100.0, "b": 1.0},
                firms=[{"cost": 10.0}],
            ),
            ExperimentConfig(
                config_id="config_2",
                model="bertrand",
                rounds=2,
                params={"alpha": 100.0, "beta": 1.0},
                firms=[{"cost": 15.0}],
            ),
        ]

        seeds_per_config = 2

        csv_path = runner.run_experiment_batch(configs, seeds_per_config, temp_db_url)

        # Read data
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Should have 4 rows total (2 configs × 2 seeds each)
        assert len(rows) == 4

        # Check config distribution
        config_ids = [row["config_id"] for row in rows]
        assert config_ids.count("config_1") == 2
        assert config_ids.count("config_2") == 2

        # Check model distribution
        models = [row["model"] for row in rows]
        assert models.count("cournot") == 2
        assert models.count("bertrand") == 2


def test_csv_file_permissions_and_accessibility(temp_db_url):
    """Test that CSV file has proper permissions and is accessible."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        runner = ExperimentRunner(str(artifacts_dir))

        config = ExperimentConfig(
            config_id="permissions_test",
            model="cournot",
            rounds=2,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}],
        )

        experiments = [config]
        seeds_per_config = 1

        csv_path = runner.run_experiment_batch(
            experiments, seeds_per_config, temp_db_url
        )

        # Check file exists and is readable
        csv_file = Path(csv_path)
        assert csv_file.exists()
        assert csv_file.is_file()

        # Check file is readable
        assert os.access(csv_path, os.R_OK), "CSV file should be readable"

        # Check file can be opened and read
        with open(csv_path) as f:
            content = f.read()
            assert len(content) > 0

        # Check file can be opened with csv module
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0


def test_csv_handles_empty_experiments(temp_db_url):
    """Test CSV creation with empty experiment list."""
    with tempfile.TemporaryDirectory() as temp_dir:
        artifacts_dir = Path(temp_dir) / "artifacts"
        runner = ExperimentRunner(str(artifacts_dir))

        # Empty experiments list
        experiments = []
        seeds_per_config = 1

        csv_path = runner.run_experiment_batch(
            experiments, seeds_per_config, temp_db_url
        )

        # File should still be created
        assert Path(csv_path).exists()

        # Should have headers but no data rows
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        assert headers is not None
        assert len(rows) == 0  # No data rows
