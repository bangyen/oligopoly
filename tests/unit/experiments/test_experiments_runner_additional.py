"""Additional tests for experiments/runner.py to improve coverage.

This module tests additional edge cases, error handling, and validation
scenarios in the experiment runner implementation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

from src.sim.experiments.runner import (
    ExperimentConfig,
    ExperimentRunner,
    run_experiment_batch_from_file,
)


class TestExperimentConfig:
    """Test the ExperimentConfig class."""

    def test_experiment_config_creation(self):
        """Test creating an ExperimentConfig instance."""
        config = ExperimentConfig(
            config_id="test_config",
            model="cournot",
            rounds=10,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}, {"cost": 12.0}],
            segments=[{"alpha": 100.0, "beta": 1.0, "weight": 1.0}],
            policies=[{"round_idx": 5, "policy_type": "tax", "value": 5.0}],
        )
        assert config.config_id == "test_config"
        assert config.model == "cournot"
        assert config.rounds == 10
        assert config.params == {"a": 100.0, "b": 1.0}
        assert config.firms == [{"cost": 10.0}, {"cost": 12.0}]
        assert config.segments == [{"alpha": 100.0, "beta": 1.0, "weight": 1.0}]
        assert config.policies == [{"round_idx": 5, "policy_type": "tax", "value": 5.0}]

    def test_experiment_config_defaults(self):
        """Test ExperimentConfig with default values."""
        config = ExperimentConfig(
            config_id="test_config",
            model="bertrand",
            rounds=5,
            params={"alpha": 200.0, "beta": 2.0},
            firms=[{"cost": 15.0}],
        )
        assert config.segments == []
        assert config.policies == []

    def test_to_simulation_config_basic(self):
        """Test to_simulation_config with basic configuration."""
        config = ExperimentConfig(
            config_id="test_config",
            model="cournot",
            rounds=10,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}, {"cost": 12.0}],
        )

        sim_config = config.to_simulation_config(seed=42)

        assert sim_config["params"] == {"a": 100.0, "b": 1.0}
        assert sim_config["firms"] == [{"cost": 10.0}, {"cost": 12.0}]
        assert sim_config["seed"] == 42
        assert sim_config["events"] == []

    def test_to_simulation_config_with_segments(self):
        """Test to_simulation_config with segments."""
        config = ExperimentConfig(
            config_id="test_config",
            model="cournot",
            rounds=10,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}],
            segments=[{"alpha": 100.0, "beta": 1.0, "weight": 0.6}],
        )

        sim_config = config.to_simulation_config(seed=42)

        assert "segments" in sim_config["params"]
        assert sim_config["params"]["segments"] == [
            {"alpha": 100.0, "beta": 1.0, "weight": 0.6}
        ]

    def test_to_simulation_config_with_policies(self):
        """Test to_simulation_config with policies."""
        config = ExperimentConfig(
            config_id="test_config",
            model="cournot",
            rounds=10,
            params={"a": 100.0, "b": 1.0},
            firms=[{"cost": 10.0}],
            policies=[
                {"round_idx": 5, "policy_type": "tax", "value": 0.1},
                {"round_idx": 8, "policy_type": "subsidy", "value": 2.0},
            ],
        )

        sim_config = config.to_simulation_config(seed=42)

        assert len(sim_config["events"]) == 2
        assert sim_config["events"][0].round_idx == 5
        assert sim_config["events"][0].policy_type.value == "tax"
        assert sim_config["events"][0].value == 0.1
        assert sim_config["events"][1].round_idx == 8
        assert sim_config["events"][1].policy_type.value == "subsidy"
        assert sim_config["events"][1].value == 2.0


class TestExperimentRunner:
    """Test the ExperimentRunner class."""

    def test_experiment_runner_creation(self):
        """Test creating an ExperimentRunner instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)
            assert runner.artifacts_dir == Path(temp_dir)
            assert runner.artifacts_dir.exists()

    def test_experiment_runner_creation_existing_dir(self):
        """Test creating an ExperimentRunner with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the directory first
            Path(temp_dir).mkdir(exist_ok=True)
            runner = ExperimentRunner(artifacts_dir=temp_dir)
            assert runner.artifacts_dir == Path(temp_dir)
            assert runner.artifacts_dir.exists()

    def test_load_experiments_valid_file(self):
        """Test load_experiments with valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = [
                {
                    "config_id": "test_config",
                    "model": "cournot",
                    "rounds": 10,
                    "params": {"a": 100.0, "b": 1.0},
                    "firms": [{"cost": 10.0}],
                }
            ]
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = ExperimentRunner()
            experiments = runner.load_experiments(config_path)

            assert len(experiments) == 1
            assert experiments[0].config_id == "test_config"
            assert experiments[0].model == "cournot"
            assert experiments[0].rounds == 10
        finally:
            Path(config_path).unlink()

    def test_load_experiments_file_not_found(self):
        """Test load_experiments with non-existent file."""
        runner = ExperimentRunner()

        with pytest.raises(FileNotFoundError) as exc_info:
            runner.load_experiments("nonexistent.json")
        assert "Experiment config file not found" in str(exc_info.value)

    def test_load_experiments_invalid_json(self):
        """Test load_experiments with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            config_path = f.name

        try:
            runner = ExperimentRunner()

            with pytest.raises(ValueError) as exc_info:
                runner.load_experiments(config_path)
            assert "Invalid JSON in config file" in str(exc_info.value)
        finally:
            Path(config_path).unlink()

    def test_load_experiments_not_list(self):
        """Test load_experiments with JSON that's not a list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {"not": "a list"}
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = ExperimentRunner()

            with pytest.raises(ValueError) as exc_info:
                runner.load_experiments(config_path)
            assert "Config file must contain a list" in str(exc_info.value)
        finally:
            Path(config_path).unlink()

    def test_load_experiments_missing_required_field(self):
        """Test load_experiments with missing required field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = [
                {
                    "config_id": "test_config",
                    # Missing "model" field
                    "rounds": 10,
                    "params": {"a": 100.0, "b": 1.0},
                    "firms": [{"cost": 10.0}],
                }
            ]
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = ExperimentRunner()

            with pytest.raises(ValueError) as exc_info:
                runner.load_experiments(config_path)
            assert "Missing required field in config 0" in str(exc_info.value)
        finally:
            Path(config_path).unlink()

    def test_load_experiments_with_default_config_id(self):
        """Test load_experiments with default config_id."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = [
                {
                    # No config_id provided
                    "model": "cournot",
                    "rounds": 10,
                    "params": {"a": 100.0, "b": 1.0},
                    "firms": [{"cost": 10.0}],
                }
            ]
            json.dump(config_data, f)
            config_path = f.name

        try:
            runner = ExperimentRunner()
            experiments = runner.load_experiments(config_path)

            assert len(experiments) == 1
            assert experiments[0].config_id == "config_0"  # Default config_id
        finally:
            Path(config_path).unlink()

    def test_run_experiment_batch_success(self):
        """Test run_experiment_batch with successful execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            # Create test experiment configs
            experiments = [
                ExperimentConfig(
                    config_id="test_config_1",
                    model="cournot",
                    rounds=2,
                    params={"a": 100.0, "b": 1.0},
                    firms=[{"cost": 10.0}],
                ),
                ExperimentConfig(
                    config_id="test_config_2",
                    model="bertrand",
                    rounds=2,
                    params={"alpha": 200.0, "beta": 2.0},
                    firms=[{"cost": 12.0}],
                ),
            ]

            Mock(spec=Session)

            with patch("src.sim.experiments.runner.run_game") as mock_run_game:
                with patch(
                    "src.sim.experiments.runner.get_run_results"
                ) as mock_get_results:
                    # Mock run_game to return run IDs
                    mock_run_game.side_effect = ["run_1", "run_2", "run_3", "run_4"]

                    # Mock get_run_results to return test data in canonical format
                    mock_get_results.return_value = {
                        "results": {
                            "0": {
                                "firm_0": {
                                    "price": 50.0,
                                    "quantity": 20.0,
                                    "profit": 800.0,
                                    "action": 20.0,
                                }
                            },
                            "1": {
                                "firm_0": {
                                    "price": 51.0,
                                    "quantity": 19.0,
                                    "profit": 779.0,
                                    "action": 19.0,
                                }
                            },
                        }
                    }

                    csv_path = runner.run_experiment_batch(
                        experiments,
                        seeds_per_config=2,
                        db_url="sqlite:///:memory:",
                    )

                    # Check that CSV file was created
                    assert Path(csv_path).exists()

                    # Check that run_game was called the expected number of times
                    assert mock_run_game.call_count == 4  # 2 configs * 2 seeds

    def test_run_experiment_batch_simulation_failure(self):
        """Test run_experiment_batch with simulation failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            experiments = [
                ExperimentConfig(
                    config_id="test_config",
                    model="cournot",
                    rounds=2,
                    params={"a": 100.0, "b": 1.0},
                    firms=[{"cost": 10.0}],
                )
            ]

            Mock(spec=Session)

            with patch("src.sim.experiments.runner.run_game") as mock_run_game:
                mock_run_game.side_effect = Exception("Simulation failed")

                with pytest.raises(RuntimeError) as exc_info:
                    runner.run_experiment_batch(
                        experiments,
                        seeds_per_config=1,
                        db_url="sqlite:///:memory:",
                    )
                assert "Failed to run config test_config seed 0" in str(exc_info.value)

    def test_calculate_summary_metrics_empty_results(self):
        """Test _calculate_summary_metrics with empty results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            exp_config = ExperimentConfig(
                config_id="test_config",
                model="cournot",
                rounds=10,
                params={"a": 100.0, "b": 1.0},
                firms=[{"cost": 10.0}, {"cost": 12.0}],
            )

            run_results = {"results": {}}

            metrics = runner._calculate_summary_metrics(run_results, exp_config)

            assert metrics["avg_price"] == 0.0
            assert metrics["avg_hhi"] == 0.0
            assert metrics["avg_cs"] == 0.0
            assert metrics["total_profit"] == 0.0
            assert metrics["mean_profit_per_firm"] == 0.0
            assert metrics["num_firms"] == 2
            assert "firm_0_profit" in metrics
            assert "firm_1_profit" in metrics

    def test_calculate_summary_metrics_cournot(self):
        """Test _calculate_summary_metrics with Cournot model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            exp_config = ExperimentConfig(
                config_id="test_config",
                model="cournot",
                rounds=2,
                params={"a": 100.0, "b": 1.0},
                firms=[{"cost": 10.0}, {"cost": 12.0}],
            )

            run_results = {
                "results": {
                    "0": {
                        "firm_0": {
                            "price": 50.0,
                            "quantity": 20.0,
                            "profit": 800.0,
                            "action": 20.0,
                        },
                        "firm_1": {
                            "price": 50.0,
                            "quantity": 15.0,
                            "profit": 600.0,
                            "action": 15.0,
                        },
                    },
                    "1": {
                        "firm_0": {
                            "price": 51.0,
                            "quantity": 19.0,
                            "profit": 779.0,
                            "action": 19.0,
                        },
                        "firm_1": {
                            "price": 51.0,
                            "quantity": 14.0,
                            "profit": 546.0,
                            "action": 14.0,
                        },
                    },
                },
            }

            metrics = runner._calculate_summary_metrics(run_results, exp_config)

            assert metrics["avg_price"] == 50.5  # (50.0 + 51.0) / 2
            assert metrics["total_profit"] == 2725.0  # 1400.0 + 1325.0
            assert metrics["mean_profit_per_firm"] == 1362.5  # 2725.0 / 2
            assert metrics["num_firms"] == 2
            assert "avg_hhi" in metrics
            assert "avg_cs" in metrics

    def test_calculate_summary_metrics_bertrand(self):
        """Test _calculate_summary_metrics with Bertrand model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            exp_config = ExperimentConfig(
                config_id="test_config",
                model="bertrand",
                rounds=2,
                params={"alpha": 200.0, "beta": 2.0},
                firms=[{"cost": 10.0}, {"cost": 12.0}],
            )

            run_results = {
                "results": {
                    "0": {
                        "firm_0": {
                            "price": 50.0,
                            "quantity": 20.0,
                            "profit": 800.0,
                            "action": 45.0,
                        },
                        "firm_1": {
                            "price": 47.0,
                            "quantity": 15.0,
                            "profit": 600.0,
                            "action": 47.0,
                        },
                    },
                    "1": {
                        "firm_0": {
                            "price": 51.0,
                            "quantity": 19.0,
                            "profit": 779.0,
                            "action": 46.0,
                        },
                        "firm_1": {
                            "price": 48.0,
                            "quantity": 14.0,
                            "profit": 546.0,
                            "action": 48.0,
                        },
                    },
                },
            }

            metrics = runner._calculate_summary_metrics(run_results, exp_config)

            assert metrics["avg_price"] == 50.5
            assert metrics["total_profit"] == 2725.0
            assert metrics["mean_profit_per_firm"] == 1362.5
            assert metrics["num_firms"] == 2
            assert "avg_hhi" in metrics
            assert "avg_cs" in metrics

    def test_calculate_summary_metrics_with_segments(self):
        """Test _calculate_summary_metrics with segmented demand."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            exp_config = ExperimentConfig(
                config_id="test_config",
                model="cournot",
                rounds=2,
                params={"a": 100.0, "b": 1.0},
                firms=[{"cost": 10.0}],
                segments=[
                    {"alpha": 100.0, "beta": 1.0, "weight": 0.6},
                    {"alpha": 80.0, "beta": 1.2, "weight": 0.4},
                ],
            )

            run_results = {
                "results": {
                    "0": {
                        "firm_0": {
                            "price": 50.0,
                            "quantity": 35.0,
                            "profit": 1400.0,
                            "action": 35.0,
                        },
                    },
                    "1": {
                        "firm_0": {
                            "price": 51.0,
                            "quantity": 33.0,
                            "profit": 1325.0,
                            "action": 33.0,
                        },
                    },
                },
            }

            metrics = runner._calculate_summary_metrics(run_results, exp_config)

            assert metrics["avg_price"] == 50.5
            assert metrics["total_profit"] == 2725.0
            assert "avg_hhi" in metrics
            assert "avg_cs" in metrics

    def test_calculate_cs_cournot(self):
        """Test _calculate_cs_cournot method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            # Test normal case
            cs = runner._calculate_cs_cournot(
                demand_a=100.0, market_price=50.0, total_qty=20.0
            )
            expected_cs = 0.5 * (100.0 - 50.0) * 20.0  # 500.0
            assert cs == expected_cs

            # Test price >= demand_a
            cs = runner._calculate_cs_cournot(
                demand_a=100.0, market_price=100.0, total_qty=20.0
            )
            assert cs == 0.0

            # Test total_qty <= 0
            cs = runner._calculate_cs_cournot(
                demand_a=100.0, market_price=50.0, total_qty=0.0
            )
            assert cs == 0.0

    def test_calculate_cs_bertrand(self):
        """Test _calculate_cs_bertrand method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            # Test normal case
            cs = runner._calculate_cs_bertrand(
                demand_alpha=200.0, market_price=100.0, total_qty=30.0
            )
            expected_cs = 0.5 * (200.0 - 100.0) * 30.0  # 1500.0
            assert cs == expected_cs

            # Test price >= demand_alpha
            cs = runner._calculate_cs_bertrand(
                demand_alpha=200.0, market_price=200.0, total_qty=30.0
            )
            assert cs == 0.0

            # Test total_qty <= 0
            cs = runner._calculate_cs_bertrand(
                demand_alpha=200.0, market_price=100.0, total_qty=0.0
            )
            assert cs == 0.0

    def test_write_csv(self):
        """Test _write_csv method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            headers = ["run_id", "config_id", "seed", "model", "rounds", "avg_price"]
            results = [
                {
                    "run_id": "run_1",
                    "config_id": "config_1",
                    "seed": 0,
                    "model": "cournot",
                    "rounds": 10,
                    "avg_price": 50.0,
                },
                {
                    "run_id": "run_2",
                    "config_id": "config_1",
                    "seed": 1,
                    "model": "cournot",
                    "rounds": 10,
                    "avg_price": 51.0,
                },
            ]

            csv_path = Path(temp_dir) / "test.csv"
            runner._write_csv(csv_path, headers, results)

            # Check that file was created
            assert csv_path.exists()

            # Check file contents
            with open(csv_path) as f:
                content = f.read()
                assert "run_id,config_id,seed,model,rounds,avg_price" in content
                assert "run_1,config_1,0,cournot,10,50.0" in content
                assert "run_2,config_1,1,cournot,10,51.0" in content

    def test_write_csv_missing_headers(self):
        """Test _write_csv with missing headers in results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentRunner(artifacts_dir=temp_dir)

            headers = ["run_id", "config_id", "seed", "model", "rounds", "avg_price"]
            results = [
                {
                    "run_id": "run_1",
                    "config_id": "config_1",
                    "seed": 0,
                    "model": "cournot",
                    "rounds": 10,
                    # Missing "avg_price"
                }
            ]

            csv_path = Path(temp_dir) / "test.csv"
            runner._write_csv(csv_path, headers, results)

            # Check that file was created
            assert csv_path.exists()

            # Check file contents - missing header should be filled with 0.0
            with open(csv_path) as f:
                content = f.read()
                assert "run_1,config_1,0,cournot,10,0.0" in content


class TestRunExperimentBatchFromFile:
    """Test the run_experiment_batch_from_file function."""

    def test_run_experiment_batch_from_file_success(self):
        """Test run_experiment_batch_from_file with successful execution."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = [
                {
                    "config_id": "test_config",
                    "model": "cournot",
                    "rounds": 2,
                    "params": {"a": 100.0, "b": 1.0},
                    "firms": [{"cost": 10.0}],
                }
            ]
            json.dump(config_data, f)
            config_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with patch("src.sim.experiments.runner.run_game") as mock_run_game:
                    with patch(
                        "src.sim.experiments.runner.get_run_results"
                    ) as mock_get_results:
                        # Mock run_game to return run IDs
                        mock_run_game.return_value = "run_1"

                        # Mock get_run_results to return test data in canonical format
                        mock_get_results.return_value = {
                            "results": {
                                "0": {
                                    "firm_0": {
                                        "price": 50.0,
                                        "quantity": 20.0,
                                        "profit": 800.0,
                                        "action": 20.0,
                                    }
                                },
                                "1": {
                                    "firm_0": {
                                        "price": 51.0,
                                        "quantity": 19.0,
                                        "profit": 779.0,
                                        "action": 19.0,
                                    }
                                },
                            }
                        }

                        csv_path = run_experiment_batch_from_file(
                            config_path=config_path,
                            seeds_per_config=1,
                            db_url="sqlite:///:memory:",
                            artifacts_dir=temp_dir,
                        )

                        # Check that CSV file was created
                        assert Path(csv_path).exists()
        finally:
            Path(config_path).unlink()

    def test_run_experiment_batch_from_file_file_not_found(self):
        """Test run_experiment_batch_from_file with non-existent file."""
        Mock(spec=Session)

        with pytest.raises(FileNotFoundError) as exc_info:
            run_experiment_batch_from_file(
                config_path="nonexistent.json",
                seeds_per_config=1,
                db_url="sqlite:///:memory:",
            )
        assert "Experiment config file not found" in str(exc_info.value)
