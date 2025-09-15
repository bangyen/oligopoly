#!/usr/bin/env python3
"""Common utilities for demo scripts.

This module provides shared functionality for formatting output,
calculating metrics, and managing database connections across demo scripts.
"""

from typing import Any, List

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.models.models import Base


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
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def print_demo_completion(topic: str, features: str) -> None:
    """Print standardized demo completion message."""
    print(f"\n✓ {topic} demonstrated successfully!")
    print(f"Features: {features}")


def print_round_results(results: dict, label: str) -> None:
    """Print simulation results in a concise format."""
    rounds_data = results["rounds_data"]
    
    # Show just the key metrics for each round
    for round_data in rounds_data:
        print(f"   Round {round_data['round']}: Price {format_currency(round_data['price'])}, Profit {format_currency(round_data['total_profit'])}")
