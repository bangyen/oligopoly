"""Experiment runner for batch simulations.

This module provides functionality for running multiple simulation configurations
with different seeds and exporting summary metrics to CSV files.
"""

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from tqdm import tqdm

from src.sim.models.metrics import (
    calculate_hhi,
    calculate_market_shares_bertrand,
    calculate_market_shares_cournot,
)
from src.sim.policy.policy_shocks import PolicyEvent, PolicyType
from src.sim.runners.runner import get_run_results, run_game

def _simulation_worker(args):
    """Top-level worker function for multiprocessing (must be picklable)."""
    exp_config, seed, db_url, metrics_calc_func = args
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        config = exp_config.to_simulation_config(seed)
        run_id = run_game(exp_config.model, exp_config.rounds, config, db)
        run_results = get_run_results(run_id, db)
        metrics = metrics_calc_func(run_results, exp_config)
        return {
            "run_id": run_id,
            "config_id": exp_config.config_id,
            "seed": seed,
            "model": exp_config.model,
            "rounds": exp_config.rounds,
            **metrics,
        }
    finally:
        db.close()


class ExperimentConfig:
    """Configuration for a single experiment run.

    Represents one simulation configuration that will be run with multiple seeds.
    """

    def __init__(
        self,
        config_id: str,
        model: str,
        rounds: int,
        params: Dict[str, Any],
        firms: List[Dict[str, Any]],
        segments: Optional[List[Dict[str, Any]]] = None,
        policies: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize experiment configuration.

        Args:
            config_id: Unique identifier for this configuration
            model: Simulation model ("cournot" or "bertrand")
            rounds: Number of simulation rounds
            params: Market parameters
            firms: List of firm configurations
            segments: Optional segmented demand configuration
            policies: Optional policy events configuration
        """
        self.config_id = config_id
        self.model = model
        self.rounds = rounds
        self.params = params
        self.firms = firms
        self.segments = segments or []
        self.policies = policies or []

    def to_simulation_config(self, seed: int) -> Dict[str, Any]:
        """Convert to simulation configuration with seed.

        Args:
            seed: Random seed for this run

        Returns:
            Configuration dictionary for run_game
        """
        config = {
            "params": self.params.copy(),
            "firms": self.firms.copy(),
            "seed": seed,
            "events": [],
        }

        if self.segments:
            params = config["params"]
            assert isinstance(params, dict)
            params["segments"] = self.segments.copy()

        # Convert policy dictionaries to PolicyEvent objects
        if self.policies:
            policy_events = []
            for policy_dict in self.policies:
                policy_event = PolicyEvent(
                    round_idx=policy_dict["round_idx"],
                    policy_type=PolicyType(policy_dict["policy_type"].lower()),
                    value=policy_dict["value"],
                )
                policy_events.append(policy_event)
            config["events"] = policy_events

        return config


class ExperimentRunner:
    """Runs batch simulations and exports results to CSV.

    This class manages running multiple experiment configurations with different
    seeds and exporting summary metrics to CSV files.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        """Initialize experiment runner.

        Args:
            artifacts_dir: Directory to store output CSV files
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)

    def load_experiments(self, config_path: str) -> List[ExperimentConfig]:
        """Load experiment configurations from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            List of experiment configurations

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is malformed
        """
        try:
            with open(config_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Experiment config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

        if not isinstance(data, list):
            raise ValueError(
                "Config file must contain a list of experiment configurations"
            )

        experiments = []
        for i, config_data in enumerate(data):
            try:
                config = ExperimentConfig(
                    config_id=config_data.get("config_id", f"config_{i}"),
                    model=config_data["model"],
                    rounds=config_data["rounds"],
                    params=config_data["params"],
                    firms=config_data["firms"],
                    segments=config_data.get("segments"),
                    policies=config_data.get("policies"),
                )
                experiments.append(config)
            except KeyError as e:
                raise ValueError(f"Missing required field in config {i}: {e}")

        return experiments

    def run_experiment_batch(
        self,
        experiments: List[ExperimentConfig],
        seeds_per_config: int,
        db_url: str,
        parallel: bool = False,
    ) -> str:
        """Run batch of experiments with multiple seeds.

        Args:
            experiments: List of experiment configurations
            seeds_per_config: Number of seeds to run per configuration
            db: Database session

        Returns:
            Path to the generated CSV file

        Raises:
            RuntimeError: If any simulation fails
        """
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.artifacts_dir / f"exp_{timestamp}.csv"

        # Prepare CSV headers
        headers = [
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
            "cartel_duration",
            "total_defections",
        ]

        # Add strategy types for each firm
        if experiments:
            max_firms = max(len(exp.firms) for exp in experiments)
            for i in range(max_firms):
                headers.append(f"firm_{i}_strategy")
                headers.append(f"firm_{i}_profit")
                headers.append(f"firm_{i}_defections")

        # Prepare tasks for execution
        tasks = []
        for exp_config in experiments:
            for seed in range(seeds_per_config):
                tasks.append((exp_config, seed))

        results = []
        total_tasks = len(tasks)

        if parallel:
            print(f"Running {total_tasks} simulations in parallel...")
            worker_args = [(t[0], t[1], db_url, self._calculate_summary_metrics) for t in tasks]

            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                futures = [executor.submit(_simulation_worker, arg) for arg in worker_args]
                for future in tqdm(as_completed(futures), total=total_tasks, desc="Experiments"):
                    results.append(future.result())
        else:
            print(f"Running {total_tasks} simulations sequentially...")
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            engine = create_engine(db_url)
            SessionLocal = sessionmaker(bind=engine)
            db = SessionLocal()
            try:
                for exp_config, seed in tqdm(tasks, desc="Experiments"):
                    config = exp_config.to_simulation_config(seed)
                    run_id = run_game(exp_config.model, exp_config.rounds, config, db)
                    run_results = get_run_results(run_id, db)
                    metrics = self._calculate_summary_metrics(run_results, exp_config)
                    results.append({
                        "run_id": run_id,
                        "config_id": exp_config.config_id,
                        "seed": seed,
                        "model": exp_config.model,
                        "rounds": exp_config.rounds,
                        **metrics,
                    })
            finally:
                db.close()

        # Write results to CSV
        self._write_csv(csv_path, headers, results)

        print(f"Experiment batch completed. Results saved to: {csv_path}")
        return str(csv_path)

    def _calculate_summary_metrics(
        self, run_results: Dict[str, Any], exp_config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Calculate summary metrics for a simulation run.

        Args:
            run_results: Results from get_run_results
            exp_config: Original experiment configuration

        Returns:
            Dictionary of calculated metrics
        """
        rounds_data = run_results["rounds_data"]
        firms_data = run_results["firms_data"]

        if not rounds_data:
            # Handle empty results
            return {
                "avg_price": 0.0,
                "avg_hhi": 0.0,
                "avg_cs": 0.0,
                "total_profit": 0.0,
                "mean_profit_per_firm": 0.0,
                "num_firms": len(exp_config.firms),
                **{f"firm_{i}_profit": 0.0 for i in range(len(exp_config.firms))},
            }

        # Calculate averages across rounds
        avg_price = sum(round_data["price"] for round_data in rounds_data) / len(
            rounds_data
        )
        total_profit = sum(round_data["total_profit"] for round_data in rounds_data)
        mean_profit_per_firm = total_profit / len(exp_config.firms)

        # Calculate HHI and consumer surplus for each round
        hhi_values = []
        cs_values = []

        for round_data in rounds_data:
            round_idx = round_data["round"]

            # Get firm data for this round
            firm_quantities = []
            firm_prices = []
            for firm_data in firms_data:
                if round_idx < len(firm_data["quantities"]):
                    firm_quantities.append(firm_data["quantities"][round_idx])
                    firm_prices.append(
                        firm_data["actions"][round_idx]
                    )  # actions are prices in Bertrand

            if exp_config.model == "cournot":
                # For Cournot, calculate HHI from quantities
                market_shares = calculate_market_shares_cournot(firm_quantities)
                hhi = calculate_hhi(market_shares)

                # Consumer surplus for Cournot
                if exp_config.segments:
                    # For segmented demand, use weighted average of segment parameters
                    weighted_alpha = sum(
                        segment["weight"] * segment["alpha"]
                        for segment in exp_config.segments
                    )
                    cs = self._calculate_cs_cournot(
                        weighted_alpha, round_data["price"], round_data["total_qty"]
                    )
                else:
                    demand_a = exp_config.params.get("a", 100.0)
                    cs = self._calculate_cs_cournot(
                        demand_a, round_data["price"], round_data["total_qty"]
                    )

            else:  # bertrand
                # For Bertrand, calculate HHI from revenues
                market_shares = calculate_market_shares_bertrand(
                    firm_prices, firm_quantities
                )
                hhi = calculate_hhi(market_shares)

                # Consumer surplus for Bertrand
                if exp_config.segments:
                    # For segmented demand, use weighted average of segment parameters
                    weighted_alpha = sum(
                        segment["weight"] * segment["alpha"]
                        for segment in exp_config.segments
                    )
                    cs = self._calculate_cs_bertrand(
                        weighted_alpha, round_data["price"], round_data["total_qty"]
                    )
                else:
                    demand_alpha = exp_config.params.get("alpha", 100.0)
                    cs = self._calculate_cs_bertrand(
                        demand_alpha, round_data["price"], round_data["total_qty"]
                    )

            hhi_values.append(hhi)
            cs_values.append(cs)

        avg_hhi = sum(hhi_values) / len(hhi_values) if hhi_values else 0.0
        avg_cs = sum(cs_values) / len(cs_values) if cs_values else 0.0

        # Calculate firm-specific metrics
        firm_metrics = {}
        for i, firm_data in enumerate(firms_data):
            total_firm_profit = sum(firm_data["profits"])
            firm_metrics[f"firm_{i}_profit"] = total_firm_profit
            
            # Get strategy type from config if available
            if i < len(exp_config.firms):
                firm_metrics[f"firm_{i}_strategy"] = exp_config.firms[i].get("strategy_type", "nash")
                
            # Defections (placeholders until we fully integrate events in get_run_results)
            firm_metrics[f"firm_{i}_defections"] = 0 

        # Summary collusion metrics (placeholders)
        return {
            "avg_price": avg_price,
            "avg_hhi": avg_hhi,
            "avg_cs": avg_cs,
            "total_profit": total_profit,
            "mean_profit_per_firm": mean_profit_per_firm,
            "num_firms": len(exp_config.firms),
            "cartel_duration": 0,
            "total_defections": 0,
            **firm_metrics,
        }

    def _calculate_cs_cournot(
        self, demand_a: float, market_price: float, total_qty: float
    ) -> float:
        """Calculate consumer surplus for Cournot model."""
        if market_price >= demand_a or total_qty <= 0:
            return 0.0
        return 0.5 * (demand_a - market_price) * total_qty

    def _calculate_cs_bertrand(
        self, demand_alpha: float, market_price: float, total_qty: float
    ) -> float:
        """Calculate consumer surplus for Bertrand model."""
        if market_price >= demand_alpha or total_qty <= 0:
            return 0.0
        return 0.5 * (demand_alpha - market_price) * total_qty

    def _write_csv(
        self, csv_path: Path, headers: List[str], results: List[Dict[str, Any]]
    ) -> None:
        """Write results to CSV file.

        Args:
            csv_path: Path to output CSV file
            headers: Column headers
            results: List of result dictionaries
        """
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            for result in results:
                # Ensure all headers are present in each row
                row = {header: result.get(header, 0.0) for header in headers}
                writer.writerow(row)


def run_experiment_batch_from_file(
    config_path: str,
    seeds_per_config: int,
    db_url: str,
    artifacts_dir: str = "artifacts",
    parallel: bool = False,
) -> str:
    """Convenience function to run experiments from a config file.

    Args:
        config_path: Path to JSON experiment configuration file
        seeds_per_config: Number of seeds to run per configuration
        db: Database session
        artifacts_dir: Directory to store output CSV files

    Returns:
        Path to the generated CSV file
    """
    runner = ExperimentRunner(artifacts_dir)
    experiments = runner.load_experiments(config_path)
    return runner.run_experiment_batch(experiments, seeds_per_config, db_url, parallel)
