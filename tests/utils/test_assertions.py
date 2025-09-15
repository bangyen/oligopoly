"""Test assertion utilities.

This module provides common assertion functions for consistent
testing patterns across the codebase.
"""

from typing import Any, List

import pytest

from src.sim.games.bertrand import BertrandResult
from src.sim.games.cournot import CournotResult


def assert_bertrand_output_format(result: BertrandResult, expected_firms: int) -> None:
    """Verify Bertrand simulation output structure.

    Args:
        result: Bertrand simulation result
        expected_firms: Expected number of firms
    """
    assert result.total_demand >= 0
    assert len(result.prices) == expected_firms
    assert len(result.quantities) == expected_firms
    assert len(result.profits) == expected_firms

    # Verify total demand equals sum of allocated quantities
    assert sum(result.quantities) == pytest.approx(result.total_demand, abs=1e-6)


def assert_cournot_output_format(result: CournotResult, expected_firms: int) -> None:
    """Verify Cournot simulation output structure.
    
    Args:
        result: Cournot simulation result
        expected_firms: Expected number of firms
    """
    assert result.price >= 0
    assert len(result.quantities) == expected_firms
    assert len(result.profits) == expected_firms
    
    # Verify all quantities are non-negative
    assert all(q >= 0 for q in result.quantities)


def assert_bertrand_cli_format(
    result: BertrandResult,
    expected_demand: float,
    expected_quantities: List[float],
    expected_profits: List[float],
) -> None:
    """Verify Bertrand CLI output format matches specification.

    Args:
        result: Bertrand simulation result
        expected_demand: Expected total demand
        expected_quantities: Expected quantities for each firm
        expected_profits: Expected profits for each firm
    """
    # Verify total demand
    assert result.total_demand == pytest.approx(expected_demand, abs=1e-6)

    # Verify quantities
    for i, expected_qty in enumerate(expected_quantities):
        assert result.quantities[i] == pytest.approx(expected_qty, abs=1e-6)

    # Verify profits
    for i, expected_profit in enumerate(expected_profits):
        assert result.profits[i] == pytest.approx(expected_profit, abs=1e-6)


def assert_cournot_cli_format(
    result: CournotResult,
    expected_price: float,
    expected_quantities: List[float],
    expected_profits: List[float],
) -> None:
    """Verify Cournot CLI output format matches specification.

    Args:
        result: Cournot simulation result
        expected_price: Expected market price
        expected_quantities: Expected quantities for each firm
        expected_profits: Expected profits for each firm
    """
    # Verify market price
    assert result.price == pytest.approx(expected_price, abs=1e-6)

    # Verify quantities
    for i, expected_qty in enumerate(expected_quantities):
        assert result.quantities[i] == pytest.approx(expected_qty, abs=1e-6)

    # Verify profits
    for i, expected_profit in enumerate(expected_profits):
        assert result.profits[i] == pytest.approx(expected_profit, abs=1e-6)


def assert_hhi_calculation(quantities: List[float], expected_hhi: float) -> None:
    """Verify HHI calculation is correct.

    Args:
        quantities: List of quantities produced by each firm
        expected_hhi: Expected HHI value
    """
    total_quantity = sum(quantities)
    if total_quantity == 0:
        assert expected_hhi == 0.0
        return

    market_shares = [q / total_quantity for q in quantities]
    calculated_hhi = sum(share**2 for share in market_shares) * 10000
    assert calculated_hhi == pytest.approx(expected_hhi, abs=1e-6)


def assert_consumer_surplus_calculation(
    price_intercept: float,
    market_price: float,
    market_quantity: float,
    expected_cs: float,
) -> None:
    """Verify consumer surplus calculation is correct.

    Args:
        price_intercept: Maximum price when quantity is zero
        market_price: Current market price
        market_quantity: Current market quantity
        expected_cs: Expected consumer surplus value
    """
    if market_quantity == 0:
        assert expected_cs == 0.0
        return

    calculated_cs = 0.5 * max(0, price_intercept - market_price) * market_quantity
    assert calculated_cs == pytest.approx(expected_cs, abs=1e-6)


def assert_strategy_trajectory(
    trajectory: List[float],
    expected_length: int,
    bounds: tuple[float, float],
) -> None:
    """Verify strategy trajectory is valid.

    Args:
        trajectory: List of actions taken by strategy
        expected_length: Expected number of actions
        bounds: Valid range for actions (min, max)
    """
    assert len(trajectory) == expected_length

    for action in trajectory:
        assert bounds[0] <= action <= bounds[1]


def assert_run_data_structure(run_data: dict[str, Any]) -> None:
    """Verify run data has expected structure.

    Args:
        run_data: Dictionary containing run data
    """
    required_keys = ["run_id", "rounds_data", "firms_data"]
    for key in required_keys:
        assert key in run_data, f"Missing required key: {key}"

    # Verify rounds_data structure
    rounds_data = run_data["rounds_data"]
    assert isinstance(rounds_data, list)
    assert len(rounds_data) > 0

    # Verify firms_data structure
    firms_data = run_data["firms_data"]
    assert isinstance(firms_data, list)
    assert len(firms_data) > 0


def assert_event_log_structure(events: List[dict[str, Any]]) -> None:
    """Verify event log has expected structure.

    Args:
        events: List of event dictionaries
    """
    assert isinstance(events, list)

    for event in events:
        required_keys = ["event_type", "round_idx", "description"]
        for key in required_keys:
            assert key in event, f"Missing required event key: {key}"

        assert isinstance(event["round_idx"], int)
        assert event["round_idx"] >= 0
        assert isinstance(event["description"], str)
        assert len(event["description"]) > 0
