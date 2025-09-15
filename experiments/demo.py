#!/usr/bin/env python3
"""Demo script for the experiment runner.

This script demonstrates how to use the experiment runner to run batch simulations
and export results to CSV files.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils import (
    create_experiment_database,
    print_experiment_summary,
)
from sim.experiments.runner import ExperimentRunner


def get_demo_config() -> str:
    """Get the demo experiment configuration file path."""
    return str(Path(__file__).parent / "sample_config.json")


def run_demo() -> None:
    """Run the experiment runner demo."""
    print("=== Oligopoly Experiment Runner Demo ===")

    # Get demo configuration
    print("1. Loading demo experiment configuration")
    config_path = get_demo_config()
    print(f"   Configuration loaded from: {config_path}")

    # Setup database
    print("2. Setting up database")
    db_path = "data/demo_experiments.db"
    db = create_experiment_database(db_path)

    try:
        # Create experiment runner
        print("3. Running experiments")
        runner = ExperimentRunner("artifacts")

        # Load experiments
        experiments = runner.load_experiments(config_path)
        print(f"   Loaded {len(experiments)} experiment configurations")

        # Run experiments with 3 seeds each
        seeds_per_config = 3
        csv_path = runner.run_experiment_batch(experiments, seeds_per_config, db)

        print(f"\n4. Results saved to: {csv_path}")

        # Show summary
        print("5. Summary")
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print_experiment_summary(rows)

        print("\n=== Demo Complete ===")
        print(f"Check the CSV file for detailed results: {csv_path}")

    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback

        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    run_demo()
