#!/usr/bin/env python3
"""Demo script for the experiment runner.

This script demonstrates how to use the experiment runner to run batch simulations
and export results to CSV files.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.experiments.runner import ExperimentRunner
from sim.models.models import Base


def create_demo_config() -> str:
    """Create a demo experiment configuration file."""
    config_path = Path("experiments") / "demo_config.json"
    config_path.parent.mkdir(exist_ok=True)

    config_data = [
        {
            "config_id": "cournot_baseline",
            "model": "cournot",
            "rounds": 5,
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        },
        {
            "config_id": "bertrand_baseline",
            "model": "bertrand",
            "rounds": 5,
            "params": {"alpha": 100.0, "beta": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
        },
        {
            "config_id": "cournot_with_policy",
            "model": "cournot",
            "rounds": 5,
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0}, {"cost": 15.0}],
            "policies": [{"round_idx": 2, "policy_type": "TAX", "value": 0.1}],
        },
    ]

    import json

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    return str(config_path)


def run_demo() -> None:
    """Run the experiment runner demo."""
    print("=== Oligopoly Experiment Runner Demo ===\n")

    # Create demo configuration
    print("1. Creating demo experiment configuration...")
    config_path = create_demo_config()
    print(f"   Configuration saved to: {config_path}")

    # Setup database
    print("\n2. Setting up database...")
    db_path = "data/demo_experiments.db"
    Path(db_path).parent.mkdir(exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}")
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    db = session_local()

    try:
        # Create experiment runner
        print("\n3. Running experiments...")
        runner = ExperimentRunner("artifacts")

        # Load experiments
        experiments = runner.load_experiments(config_path)
        print(f"   Loaded {len(experiments)} experiment configurations")

        # Run experiments with 3 seeds each
        seeds_per_config = 3
        csv_path = runner.run_experiment_batch(experiments, seeds_per_config, db)

        print(f"\n4. Results saved to: {csv_path}")

        # Show summary
        print("\n5. Summary:")
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"   Total runs: {len(rows)}")

        # Show results by configuration
        config_results: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            config_id = row["config_id"]
            if config_id not in config_results:
                config_results[config_id] = []
            config_results[config_id].append(row)

        for config_id, results in config_results.items():
            print(f"\n   Configuration: {config_id}")
            print(f"     Runs: {len(results)}")

            # Calculate averages across seeds
            avg_prices = [float(r["avg_price"]) for r in results]
            avg_hhis = [float(r["avg_hhi"]) for r in results]
            avg_cs = [float(r["avg_cs"]) for r in results]
            total_profits = [float(r["total_profit"]) for r in results]

            print(f"     Average Price: ${sum(avg_prices)/len(avg_prices):.2f}")
            print(f"     Average HHI: {sum(avg_hhis)/len(avg_hhis):.3f}")
            print(f"     Average CS: ${sum(avg_cs)/len(avg_cs):.2f}")
            print(
                f"     Average Total Profit: ${sum(total_profits)/len(total_profits):.2f}"
            )

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
