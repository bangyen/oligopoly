"""Tests for consumer surplus calculations.

This module tests the consumer surplus calculation functionality including
edge cases, validation, and expected economic behavior.
"""

import math

import pytest

from src.sim.models.metrics import calculate_consumer_surplus


class TestConsumerSurplus:
    """Test cases for consumer surplus calculation."""

    def test_consumer_surplus_basic(self):
        """Test basic consumer surplus calculation."""
        # Example from spec: a=100, P=70, Q=30 -> CS = 0.5*(100-70)*30 = 450
        price_intercept = 100.0
        market_price = 70.0
        market_quantity = 30.0

        cs = calculate_consumer_surplus(price_intercept, market_price, market_quantity)
        expected = 0.5 * (100 - 70) * 30
        assert math.isclose(cs, expected, abs_tol=1e-10)
        assert math.isclose(cs, 450.0, abs_tol=1e-10)

    def test_consumer_surplus_zero_price(self):
        """Test consumer surplus when market price is zero."""
        price_intercept = 100.0
        market_price = 0.0
        market_quantity = 50.0

        cs = calculate_consumer_surplus(price_intercept, market_price, market_quantity)
        expected = 0.5 * (100 - 0) * 50
        assert math.isclose(cs, expected, abs_tol=1e-10)
        assert math.isclose(cs, 2500.0, abs_tol=1e-10)

    def test_consumer_surplus_zero_quantity(self):
        """Test consumer surplus when market quantity is zero."""
        price_intercept = 100.0
        market_price = 50.0
        market_quantity = 0.0

        cs = calculate_consumer_surplus(price_intercept, market_price, market_quantity)
        assert math.isclose(cs, 0.0, abs_tol=1e-10)

    def test_consumer_surplus_price_equals_intercept(self):
        """Test consumer surplus when price equals intercept."""
        price_intercept = 100.0
        market_price = 100.0
        market_quantity = 30.0

        cs = calculate_consumer_surplus(price_intercept, market_price, market_quantity)
        assert math.isclose(cs, 0.0, abs_tol=1e-10)

    def test_consumer_surplus_high_quantity(self):
        """Test consumer surplus with high quantity."""
        price_intercept = 200.0
        market_price = 50.0
        market_quantity = 100.0

        cs = calculate_consumer_surplus(price_intercept, market_price, market_quantity)
        expected = 0.5 * (200 - 50) * 100
        assert math.isclose(cs, expected, abs_tol=1e-10)
        assert math.isclose(cs, 7500.0, abs_tol=1e-10)

    def test_consumer_surplus_small_values(self):
        """Test consumer surplus with small values."""
        price_intercept = 1.0
        market_price = 0.5
        market_quantity = 0.1

        cs = calculate_consumer_surplus(price_intercept, market_price, market_quantity)
        expected = 0.5 * (1.0 - 0.5) * 0.1
        assert math.isclose(cs, expected, abs_tol=1e-10)
        assert math.isclose(cs, 0.025, abs_tol=1e-10)

    def test_consumer_surplus_negative_intercept(self):
        """Test consumer surplus with negative price intercept."""
        price_intercept = -10.0
        market_price = 5.0
        market_quantity = 10.0

        with pytest.raises(
            ValueError, match="Price intercept -10.000 must be positive"
        ):
            calculate_consumer_surplus(price_intercept, market_price, market_quantity)

    def test_consumer_surplus_zero_intercept(self):
        """Test consumer surplus with zero price intercept."""
        price_intercept = 0.0
        market_price = 5.0
        market_quantity = 10.0

        with pytest.raises(ValueError, match="Price intercept 0.000 must be positive"):
            calculate_consumer_surplus(price_intercept, market_price, market_quantity)

    def test_consumer_surplus_negative_price(self):
        """Test consumer surplus with negative market price."""
        price_intercept = 100.0
        market_price = -10.0
        market_quantity = 30.0

        with pytest.raises(
            ValueError, match="Market price -10.000 must be non-negative"
        ):
            calculate_consumer_surplus(price_intercept, market_price, market_quantity)

    def test_consumer_surplus_negative_quantity(self):
        """Test consumer surplus with negative market quantity."""
        price_intercept = 100.0
        market_price = 70.0
        market_quantity = -30.0

        with pytest.raises(
            ValueError, match="Market quantity -30.000 must be non-negative"
        ):
            calculate_consumer_surplus(price_intercept, market_price, market_quantity)

    def test_consumer_surplus_price_exceeds_intercept(self):
        """Test consumer surplus when price exceeds intercept."""
        price_intercept = 100.0
        market_price = 150.0
        market_quantity = 30.0

        with pytest.raises(
            ValueError, match="Market price 150.000 cannot exceed intercept 100.000"
        ):
            calculate_consumer_surplus(price_intercept, market_price, market_quantity)

    def test_consumer_surplus_economic_intuition(self):
        """Test that consumer surplus follows economic intuition."""
        # Higher quantity should generally increase CS (holding price constant)
        price_intercept = 100.0
        market_price = 50.0

        cs_low_qty = calculate_consumer_surplus(price_intercept, market_price, 10.0)
        cs_high_qty = calculate_consumer_surplus(price_intercept, market_price, 20.0)

        assert cs_high_qty > cs_low_qty

        # Lower price should generally increase CS (holding quantity constant)
        market_quantity = 30.0

        cs_high_price = calculate_consumer_surplus(
            price_intercept, 80.0, market_quantity
        )
        cs_low_price = calculate_consumer_surplus(
            price_intercept, 40.0, market_quantity
        )

        assert cs_low_price > cs_high_price

    def test_consumer_surplus_mathematical_properties(self):
        """Test mathematical properties of consumer surplus."""
        price_intercept = 100.0
        market_price = 60.0
        market_quantity = 20.0

        cs = calculate_consumer_surplus(price_intercept, market_price, market_quantity)

        # CS should be non-negative
        assert cs >= 0

        # CS should be proportional to quantity (when price difference is constant)
        cs_double_qty = calculate_consumer_surplus(price_intercept, market_price, 40.0)
        assert math.isclose(cs_double_qty, 2 * cs, abs_tol=1e-10)

        # CS should be proportional to price difference (when quantity is constant)
        cs_double_diff = calculate_consumer_surplus(
            price_intercept, 20.0, market_quantity
        )
        assert math.isclose(cs_double_diff, 2 * cs, abs_tol=1e-10)
