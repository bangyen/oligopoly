"""Tests for Cournot price calculation functionality.

This module tests the core price calculation logic in the Cournot simulation,
ensuring that market prices are computed correctly based on demand parameters
and total quantity supplied.
"""

import pytest
from src.sim.cournot import cournot_simulation


class TestCournotPrice:
    """Test cases for Cournot price calculation."""
    
    def test_cournot_price_basic(self) -> None:
        """Test basic price calculation: a=100, b=1, q=[10,20] -> P=70."""
        result = cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[10, 20])
        assert result.price == 70.0
        
    def test_cournot_price_zero_quantity(self) -> None:
        """Test price calculation when total quantity is zero."""
        result = cournot_simulation(a=100, b=1, costs=[10], quantities=[0])
        assert result.price == 100.0
        
    def test_cournot_price_high_quantity(self) -> None:
        """Test price calculation when quantity drives price to zero."""
        result = cournot_simulation(a=100, b=1, costs=[10], quantities=[150])
        assert result.price == 0.0
        
    def test_cournot_price_exact_boundary(self) -> None:
        """Test price calculation at exact boundary where P=0."""
        result = cournot_simulation(a=100, b=1, costs=[10], quantities=[100])
        assert result.price == 0.0
        
    def test_cournot_price_multiple_firms(self) -> None:
        """Test price calculation with multiple firms."""
        result = cournot_simulation(a=50, b=0.5, costs=[5, 10, 15], quantities=[20, 15, 10])
        expected_price = max(0, 50 - 0.5 * (20 + 15 + 10))
        assert result.price == expected_price
        
    def test_cournot_price_fractional_values(self) -> None:
        """Test price calculation with fractional values."""
        result = cournot_simulation(a=75.5, b=1.25, costs=[12.5], quantities=[30.2])
        expected_price = max(0, 75.5 - 1.25 * 30.2)
        assert abs(result.price - expected_price) < 1e-10
