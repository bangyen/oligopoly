"""Test data generation utilities.

This module provides common test data configurations and fixtures
for consistent testing across the codebase.
"""

from typing import Any, Dict


def create_sample_bertrand_config() -> Dict[str, Any]:
    """Create a standard Bertrand test configuration.

    Returns:
        Dictionary with standard Bertrand test parameters
    """
    return {
        "alpha": 120.0,
        "beta": 1.2,
        "costs": [20.0, 20.0, 25.0],
        "prices": [22.0, 21.0, 24.0],
    }


def create_sample_cournot_config() -> Dict[str, Any]:
    """Create a standard Cournot test configuration.

    Returns:
        Dictionary with standard Cournot test parameters
    """
    return {
        "a": 100.0,
        "b": 1.0,
        "costs": [10.0, 15.0, 20.0],
        "quantities": [12.0, 14.0, 16.0],
    }


def create_sample_bertrand_cli_config() -> Dict[str, Any]:
    """Create a standard Bertrand CLI test configuration.

    Returns:
        Dictionary with standard CLI test parameters
    """
    return {
        "alpha": 100.0,
        "beta": 1.0,
        "costs": [10.0, 15.0, 20.0],
        "prices": [12.0, 14.0, 16.0],
        "expected_demand": 88.0,  # Q(12) = 100-1*12 = 88
        "expected_quantities": [88.0, 0.0, 0.0],  # Firm 0 gets all demand
        "expected_profits": [176.0, 0.0, 0.0],  # (12-10)*88 = 176
    }


def create_sample_cournot_cli_config() -> Dict[str, Any]:
    """Create a standard Cournot CLI test configuration.

    Returns:
        Dictionary with standard CLI test parameters
    """
    return {
        "a": 100.0,
        "b": 1.0,
        "costs": [10.0, 15.0, 20.0],
        "quantities": [12.0, 14.0, 16.0],
        "expected_price": 58.0,  # P = 100 - 1*(12+14+16) = 58
    }


def create_sample_strategy_config() -> Dict[str, Any]:
    """Create a standard strategy test configuration.

    Returns:
        Dictionary with standard strategy test parameters
    """
    return {
        "bounds": (0.0, 10.0),
        "initial_value": 5.0,
        "step_size": 1.0,
        "seed": 42,
        "rounds": 5,
    }


def create_sample_experiment_config() -> Dict[str, Any]:
    """Create a standard experiment test configuration.

    Returns:
        Dictionary with standard experiment test parameters
    """
    return {
        "model": "cournot",
        "rounds": 10,
        "firms": 3,
        "params": {
            "a": 100.0,
            "b": 1.0,
        },
        "costs": [10.0, 15.0, 20.0],
        "strategies": ["static", "titfortat", "randomwalk"],
        "seed": 42,
    }


def create_sample_policy_config() -> Dict[str, Any]:
    """Create a standard policy test configuration.

    Returns:
        Dictionary with standard policy test parameters
    """
    return {
        "model": "cournot",
        "rounds": 5,
        "firms": 3,
        "params": {
            "a": 100.0,
            "b": 1.0,
        },
        "costs": [10.0, 15.0, 20.0],
        "strategies": ["static", "static", "static"],
        "policies": [
            {"type": "tax", "rate": 0.1, "round": 1},
            {"type": "subsidy", "rate": 0.05, "round": 2},
            {"type": "price_cap", "cap": 50.0, "round": 3},
        ],
        "seed": 42,
    }


def create_sample_segmented_demand_config() -> Dict[str, Any]:
    """Create a standard segmented demand test configuration.

    Returns:
        Dictionary with standard segmented demand test parameters
    """
    return {
        "segments": [
            {"alpha": 100.0, "beta": 1.0, "weight": 0.6},
            {"alpha": 80.0, "beta": 2.0, "weight": 0.4},
        ],
        "costs": [10.0, 15.0],
        "quantities": [20.0, 15.0],  # For Cournot
        "prices": [25.0, 30.0],  # For Bertrand
    }


def create_sample_collusion_config() -> Dict[str, Any]:
    """Create a standard collusion test configuration.

    Returns:
        Dictionary with standard collusion test parameters
    """
    return {
        "model": "bertrand",
        "rounds": 10,
        "firms": 3,
        "params": {
            "alpha": 100.0,
            "beta": 1.0,
        },
        "costs": [20.0, 20.0, 20.0],
        "strategies": ["cartel", "collusive", "opportunistic"],
        "regulator_thresholds": {
            "hhi_threshold": 0.8,
            "price_threshold": 45.0,
        },
        "seed": 42,
    }
