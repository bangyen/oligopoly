"""Test policy price cap shocks with validation.

This module tests the price cap policy shock implementation, ensuring that
when unconstrained prices exceed the cap, prices are set to the cap and
profits are recalculated accordingly.
"""

import math

import pytest

from sim.bertrand import BertrandResult
from sim.cournot import CournotResult
from sim.policy_shocks import apply_price_cap_shock


def test_price_cap_cournot_no_cap_needed() -> None:
    """Test price cap when market price is below cap (no change)."""
    result = CournotResult(
        price=50.0, quantities=[10.0, 15.0, 20.0], profits=[400.0, 525.0, 600.0]
    )
    costs = [10.0, 15.0, 20.0]
    price_cap = 60.0

    capped_result = apply_price_cap_shock(result, price_cap, costs)

    # Price is below cap, so no change
    assert isinstance(capped_result, CournotResult)
    assert math.isclose(capped_result.price, 50.0, abs_tol=1e-6)
    assert capped_result.quantities == [10.0, 15.0, 20.0]
    assert capped_result.profits == [400.0, 525.0, 600.0]


def test_price_cap_cournot_cap_applied() -> None:
    """Test price cap when market price exceeds cap."""
    result = CournotResult(
        price=70.0,
        quantities=[10.0, 15.0, 20.0],
        profits=[600.0, 825.0, 1000.0],  # (70-10)*10, (70-15)*15, (70-20)*20
    )
    costs = [10.0, 15.0, 20.0]
    price_cap = 60.0

    capped_result = apply_price_cap_shock(result, price_cap, costs)

    # Price should be capped at 60.0
    assert isinstance(capped_result, CournotResult)
    assert math.isclose(capped_result.price, 60.0, abs_tol=1e-6)
    assert capped_result.quantities == [10.0, 15.0, 20.0]

    # Profits should be recalculated: (60-10)*10, (60-15)*15, (60-20)*20
    expected_profits = [500.0, 675.0, 800.0]
    for i, (actual, expected) in enumerate(
        zip(capped_result.profits, expected_profits)
    ):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"


def test_price_cap_cournot_exact_cap() -> None:
    """Test price cap when market price equals cap."""
    result = CournotResult(price=60.0, quantities=[10.0, 15.0], profits=[500.0, 675.0])
    costs = [10.0, 15.0]
    price_cap = 60.0

    capped_result = apply_price_cap_shock(result, price_cap, costs)

    # Price equals cap, so no change
    assert isinstance(capped_result, CournotResult)
    assert math.isclose(capped_result.price, 60.0, abs_tol=1e-6)
    assert capped_result.quantities == [10.0, 15.0]
    assert capped_result.profits == [500.0, 675.0]


def test_price_cap_bertrand_no_cap_needed() -> None:
    """Test price cap for Bertrand when prices are below cap."""
    result = BertrandResult(
        total_demand=100.0,
        prices=[50.0, 55.0, 60.0],
        quantities=[30.0, 35.0, 35.0],
        profits=[1200.0, 1400.0, 1400.0],
    )
    costs = [10.0, 15.0, 20.0]
    price_cap = 70.0

    capped_result = apply_price_cap_shock(result, price_cap, costs)

    # All prices are below cap, so no change
    assert isinstance(capped_result, BertrandResult)
    assert capped_result.prices == [50.0, 55.0, 60.0]
    assert capped_result.quantities == [30.0, 35.0, 35.0]
    assert capped_result.profits == [1200.0, 1400.0, 1400.0]


def test_price_cap_bertrand_cap_applied() -> None:
    """Test price cap for Bertrand when some prices exceed cap."""
    result = BertrandResult(
        total_demand=100.0,
        prices=[50.0, 70.0, 80.0],
        quantities=[30.0, 35.0, 35.0],
        profits=[1200.0, 1925.0, 2100.0],
    )
    costs = [10.0, 15.0, 20.0]
    price_cap = 60.0

    capped_result = apply_price_cap_shock(result, price_cap, costs)

    # Prices should be capped at 60.0
    assert isinstance(capped_result, BertrandResult)
    assert capped_result.prices == [50.0, 60.0, 60.0]
    assert capped_result.quantities == [30.0, 35.0, 35.0]

    # Profits should be recalculated: (50-10)*30, (60-15)*35, (60-20)*35
    expected_profits = [1200.0, 1575.0, 1400.0]
    for i, (actual, expected) in enumerate(
        zip(capped_result.profits, expected_profits)
    ):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"


def test_price_cap_negative_cap() -> None:
    """Test price cap with negative cap (should raise error)."""
    result = CournotResult(price=50.0, quantities=[10.0, 15.0], profits=[400.0, 525.0])
    costs = [10.0, 15.0]
    price_cap = -10.0

    with pytest.raises(ValueError, match="Price cap must be non-negative"):
        apply_price_cap_shock(result, price_cap, costs)


def test_price_cap_zero_cap() -> None:
    """Test price cap with zero cap."""
    result = CournotResult(price=50.0, quantities=[10.0, 15.0], profits=[400.0, 525.0])
    costs = [10.0, 15.0]
    price_cap = 0.0

    capped_result = apply_price_cap_shock(result, price_cap, costs)

    # Price should be capped at 0.0
    assert isinstance(capped_result, CournotResult)
    assert math.isclose(capped_result.price, 0.0, abs_tol=1e-6)
    assert capped_result.quantities == [10.0, 15.0]

    # Profits should be recalculated: (0-10)*10, (0-15)*15
    expected_profits = [-100.0, -225.0]
    for i, (actual, expected) in enumerate(
        zip(capped_result.profits, expected_profits)
    ):
        assert math.isclose(
            actual, expected, abs_tol=1e-6
        ), f"Firm {i}: expected {expected}, got {actual}"


def test_price_cap_unsupported_result_type() -> None:
    """Test price cap with unsupported result type."""

    class MockResult:
        pass

    result = MockResult()
    costs = [10.0, 15.0]
    price_cap = 60.0

    with pytest.raises(ValueError, match="Unsupported result type"):
        apply_price_cap_shock(result, price_cap, costs)


def test_price_cap_single_firm() -> None:
    """Test price cap with single firm."""
    result = CournotResult(price=70.0, quantities=[10.0], profits=[600.0])
    costs = [10.0]
    price_cap = 60.0

    capped_result = apply_price_cap_shock(result, price_cap, costs)

    assert isinstance(capped_result, CournotResult)
    assert math.isclose(capped_result.price, 60.0, abs_tol=1e-6)
    assert capped_result.quantities == [10.0]
    assert math.isclose(capped_result.profits[0], 500.0, abs_tol=1e-6)  # (60-10)*10
