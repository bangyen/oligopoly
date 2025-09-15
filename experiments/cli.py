#!/usr/bin/env python3
"""Command-line interface for running batch experiments.

This script provides a CLI for running experiment batches from JSON configuration files.
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.experiments.runner import run_experiment_batch_from_file
from sim.models.models import Base


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch experiments from JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments with 3 seeds per config
  python experiments/cli.py experiments/sample_config.json --seeds 3

  # Use custom artifacts directory
  python experiments/cli.py experiments/sample_config.json --seeds 5 --artifacts results/

  # Use custom database
  python experiments/cli.py experiments/sample_config.json --seeds 3 --db data/experiments.db
        """,
    )

    parser.add_argument(
        "config_file",
        help="Path to JSON experiment configuration file",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds to run per configuration (default: 3)",
    )

    parser.add_argument(
        "--artifacts",
        default="artifacts",
        help="Directory to store output CSV files (default: artifacts)",
    )

    parser.add_argument(
        "--db",
        default="data/experiments.db",
        help="Database file path (default: data/experiments.db)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Validate inputs
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    if args.seeds <= 0:
        print(f"Error: Number of seeds must be positive, got {args.seeds}")
        sys.exit(1)

    # Create artifacts directory if it doesn't exist
    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Setup database
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}")
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    db = session_local()

    try:
        if args.verbose:
            print(f"Configuration file: {config_path}")
            print(f"Seeds per config: {args.seeds}")
            print(f"Artifacts directory: {artifacts_dir}")
            print(f"Database: {db_path}")
            print()

        # Run experiments
        print("Starting experiment batch...")
        csv_path = run_experiment_batch_from_file(
            config_path=str(config_path),
            seeds_per_config=args.seeds,
            db=db,
            artifacts_dir=str(artifacts_dir),
        )

        print("\nExperiment batch completed successfully!")
        print(f"Results saved to: {csv_path}")

        # Show summary
        if args.verbose:
            import csv

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            print("\nSummary:")
            print(f"  Total runs: {len(rows)}")

            # Count by config
            config_counts: dict[str, int] = {}
            for row in rows:
                config_id = row["config_id"]
                config_counts[config_id] = config_counts.get(config_id, 0) + 1

            print("  Configurations:")
            for config_id, count in config_counts.items():
                print(f"    {config_id}: {count} runs")

    except Exception as e:
        print(f"Error running experiments: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        db.close()


if __name__ == "__main__":
    main()
