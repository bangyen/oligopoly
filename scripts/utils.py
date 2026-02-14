#!/usr/bin/env python3
"""Common utilities for demo and experiment scripts.

This module provides shared functionality for formatting output,
calculating metrics, and managing database connections across demo and experiment scripts.
"""

from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

try:
    from src.sim.models.models import Base
except ImportError:
    # Fallback to local import if src is not available
    from sim.models.models import Base  # type: ignore


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n=== {title} ===")


def print_summary(title: str, items: List[str]) -> None:
    """Print a summary section with checkmarks."""
    print(f"\n{title}:")
    for item in items:
        print(f"  ✓ {item}")


def format_currency(value: float) -> str:
    """Format a value as currency."""
    return f"${value:.1f}"


def format_list(values: List[float], formatter: str = "currency") -> str:
    """Format a list of values."""
    if formatter == "currency":
        return f"[{', '.join(format_currency(v) for v in values)}]"
    return f"[{', '.join(f'{v:.1f}' for v in values)}]"


def calculate_hhi(quantities: List[float]) -> float:
    """Calculate Herfindahl-Hirschman Index (HHI) for market concentration."""
    total_quantity = sum(quantities)
    if total_quantity == 0:
        return 0.0
    market_shares = [q / total_quantity for q in quantities]
    return sum(share**2 for share in market_shares) * 10000


def calculate_consumer_surplus(
    price: float, total_quantity: float, a: float, b: float
) -> float:
    """Calculate consumer surplus for linear demand curve."""
    if total_quantity == 0:
        return 0.0
    # CS = 0.5 * (a - price) * total_quantity
    return 0.5 * max(0, a - price) * total_quantity


def create_demo_database(database_url: str = "sqlite:///:memory:") -> Any:
    """Create a database session for demo purposes."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return session_local()


def print_demo_completion(topic: str, features: str) -> None:
    """Print standardized demo completion message."""
    print(f"\n✓ {topic} demonstrated successfully!")
    print(f"Features: {features}")


def print_round_results(results: dict, label: str) -> None:
    """Print simulation results in a concise format."""
    rounds_data = results["rounds_data"]

    # Show just the key metrics for each round
    for round_data in rounds_data:
        print(
            f"   Round {round_data['round']}: Price {format_currency(round_data['price'])}, Profit {format_currency(round_data['total_profit'])}"
        )


# Experiment-specific utilities
def create_experiment_database(db_path: str) -> Session:
    """Create and return a database session for experiments.

    Args:
        db_path: Path to the database file

    Returns:
        Database session
    """
    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create engine and session
    engine = create_engine(f"sqlite:///{db_path}")
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    return session_local()


def print_experiment_summary(results: List[Dict[str, Any]]) -> None:
    """Print a formatted experiment summary.

    Args:
        results: List of experiment result dictionaries
    """
    print("\nSummary:")
    print(f"  Total runs: {len(results)}")

    # Group results by configuration
    config_results: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        config_id = result["config_id"]
        if config_id not in config_results:
            config_results[config_id] = []
        config_results[config_id].append(result)

    # Print summary for each configuration
    for config_id, config_runs in config_results.items():
        print(f"\n  Configuration: {config_id}")
        print(f"    Runs: {len(config_runs)}")

        # Calculate averages
        avg_prices = [float(r["avg_price"]) for r in config_runs]
        avg_hhis = [float(r["avg_hhi"]) for r in config_runs]
        avg_cs = [float(r["avg_cs"]) for r in config_runs]
        total_profits = [float(r["total_profit"]) for r in config_runs]

        print(
            f"    Average Price: {format_currency(sum(avg_prices) / len(avg_prices))}"
        )
        print(f"    Average HHI: {sum(avg_hhis) / len(avg_hhis):.3f}")
        print(f"    Average CS: {format_currency(sum(avg_cs) / len(avg_cs))}")
        print(
            f"    Average Total Profit: {format_currency(sum(total_profits) / len(total_profits))}"
        )


def print_verbose_summary(results: List[Dict[str, Any]]) -> None:
    """Print a detailed experiment summary for verbose mode.

    Args:
        results: List of experiment result dictionaries
    """
    print("\nDetailed Summary:")
    print(f"  Total runs: {len(results)}")

    # Count by configuration
    config_counts: Dict[str, int] = {}
    for result in results:
        config_id = result["config_id"]
        config_counts[config_id] = config_counts.get(config_id, 0) + 1

    print("  Configurations:")
    for config_id, count in config_counts.items():
        print(f"    {config_id}: {count} runs")
