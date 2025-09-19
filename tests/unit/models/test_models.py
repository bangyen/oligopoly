"""Tests for economic models.

This module tests the core economic models including the Demand
class and its methods to ensure proper functionality and stable
string representations.
"""

from src.sim.models.models import Demand


def test_demand_creation() -> None:
    """Test Demand creation with integer and float values."""
    # Test with integers
    demand_int = Demand(a=100, b=1)
    assert demand_int.a == 100
    assert demand_int.b == 1

    # Test with floats
    demand_float = Demand(a=100.0, b=1.5)
    assert demand_float.a == 100.0
    assert demand_float.b == 1.5


def test_demand_price_calculation() -> None:
    """Test price calculation using inverse demand function."""
    demand = Demand(a=100, b=1)

    # Test normal case
    assert demand.price(10) == 90.0
    assert demand.price(50) == 50.0
    assert demand.price(100) == 0.0

    # Test edge case where price would be negative
    assert demand.price(150) == 0.0


def test_demand_repr_stability() -> None:
    """Test that Demand repr() is stable and consistent."""
    demand1 = Demand(a=100, b=1)
    demand2 = Demand(a=100.0, b=1.0)

    # Should have stable string representation
    assert repr(demand1) == "Demand(a=100, b=1)"
    assert repr(demand2) == "Demand(a=100.0, b=1.0)"

    # Multiple calls should be consistent
    assert repr(demand1) == repr(demand1)
    assert repr(demand2) == repr(demand2)
