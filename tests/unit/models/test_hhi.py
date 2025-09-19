"""Tests for HHI (Herfindahl-Hirschman Index) calculations.

This module tests the HHI calculation functionality including edge cases,
validation, and expected economic behavior.
"""

import math

import pytest

from src.sim.models.metrics import (
    calculate_hhi,
    calculate_market_shares_bertrand,
    calculate_market_shares_cournot,
)


class TestHHI:
    """Test cases for HHI calculation."""

    def test_hhi_equal_shares(self):
        """Test HHI with equal market shares."""
        # Two firms with equal shares: [0.5, 0.5]
        shares = [0.5, 0.5]
        hhi = calculate_hhi(shares)
        assert math.isclose(hhi, 0.5, abs_tol=1e-10)

    def test_hhi_monopoly(self):
        """Test HHI with monopoly (single firm)."""
        # Monopoly: [1.0, 0.0]
        shares = [1.0, 0.0]
        hhi = calculate_hhi(shares)
        assert math.isclose(hhi, 1.0, abs_tol=1e-10)

    def test_hhi_perfect_competition(self):
        """Test HHI approaches zero with many equal firms."""
        # Many equal firms: each with 1/n share
        n_firms = 10
        shares = [1.0 / n_firms] * n_firms
        hhi = calculate_hhi(shares)
        expected = n_firms * (1.0 / n_firms) ** 2
        assert math.isclose(hhi, expected, abs_tol=1e-10)
        assert hhi < 0.2  # Should be low for many firms

    def test_hhi_three_firms(self):
        """Test HHI with three firms."""
        # Three firms: [0.4, 0.3, 0.3]
        shares = [0.4, 0.3, 0.3]
        hhi = calculate_hhi(shares)
        expected = 0.4**2 + 0.3**2 + 0.3**2
        assert math.isclose(hhi, expected, abs_tol=1e-10)
        assert math.isclose(hhi, 0.34, abs_tol=1e-10)

    def test_hhi_single_firm(self):
        """Test HHI with single firm."""
        shares = [1.0]
        hhi = calculate_hhi(shares)
        assert math.isclose(hhi, 1.0, abs_tol=1e-10)

    def test_hhi_empty_list(self):
        """Test HHI with empty shares list."""
        with pytest.raises(ValueError, match="Market shares list cannot be empty"):
            calculate_hhi([])

    def test_hhi_negative_shares(self):
        """Test HHI with negative shares."""
        shares = [0.6, -0.1, 0.5]
        with pytest.raises(
            ValueError, match="Market share 1 = -0.100 must be non-negative"
        ):
            calculate_hhi(shares)

    def test_hhi_shares_not_sum_to_one(self):
        """Test HHI when shares don't sum to 1."""
        shares = [0.5, 0.5, 0.5]  # Sums to 1.5
        with pytest.raises(ValueError, match="Market shares must sum to 1.0"):
            calculate_hhi(shares)

    def test_hhi_floating_point_precision(self):
        """Test HHI handles floating point precision correctly."""
        # Shares that sum to approximately 1.0 due to floating point
        shares = [0.3333333333333333, 0.3333333333333333, 0.3333333333333334]
        hhi = calculate_hhi(shares)
        expected = 3 * (1.0 / 3) ** 2
        assert math.isclose(hhi, expected, abs_tol=1e-10)


class TestMarketSharesCournot:
    """Test cases for Cournot market share calculation."""

    def test_cournot_shares_equal_quantities(self):
        """Test market shares with equal quantities."""
        quantities = [10.0, 10.0]
        shares = calculate_market_shares_cournot(quantities)
        assert len(shares) == 2
        assert math.isclose(shares[0], 0.5, abs_tol=1e-10)
        assert math.isclose(shares[1], 0.5, abs_tol=1e-10)
        assert math.isclose(sum(shares), 1.0, abs_tol=1e-10)

    def test_cournot_shares_unequal_quantities(self):
        """Test market shares with unequal quantities."""
        quantities = [20.0, 10.0, 5.0]
        shares = calculate_market_shares_cournot(quantities)
        assert len(shares) == 3
        assert math.isclose(shares[0], 20.0 / 35.0, abs_tol=1e-10)
        assert math.isclose(shares[1], 10.0 / 35.0, abs_tol=1e-10)
        assert math.isclose(shares[2], 5.0 / 35.0, abs_tol=1e-10)
        assert math.isclose(sum(shares), 1.0, abs_tol=1e-10)

    def test_cournot_shares_zero_quantity(self):
        """Test market shares with zero quantity."""
        quantities = [10.0, 0.0, 5.0]
        shares = calculate_market_shares_cournot(quantities)
        assert len(shares) == 3
        assert math.isclose(shares[0], 10.0 / 15.0, abs_tol=1e-10)
        assert math.isclose(shares[1], 0.0, abs_tol=1e-10)
        assert math.isclose(shares[2], 5.0 / 15.0, abs_tol=1e-10)

    def test_cournot_shares_empty_list(self):
        """Test market shares with empty quantities list."""
        with pytest.raises(ValueError, match="Quantities list cannot be empty"):
            calculate_market_shares_cournot([])

    def test_cournot_shares_negative_quantities(self):
        """Test market shares with negative quantities."""
        quantities = [10.0, -5.0, 5.0]
        with pytest.raises(
            ValueError, match="Quantity 1 = -5.000 must be non-negative"
        ):
            calculate_market_shares_cournot(quantities)

    def test_cournot_shares_zero_total(self):
        """Test market shares when total quantity is zero."""
        quantities = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="Total quantity cannot be zero"):
            calculate_market_shares_cournot(quantities)


class TestMarketSharesBertrand:
    """Test cases for Bertrand market share calculation."""

    def test_bertrand_shares_equal_revenue(self):
        """Test market shares with equal revenue."""
        prices = [10.0, 10.0]
        quantities = [5.0, 5.0]
        shares = calculate_market_shares_bertrand(prices, quantities)
        assert len(shares) == 2
        assert math.isclose(shares[0], 0.5, abs_tol=1e-10)
        assert math.isclose(shares[1], 0.5, abs_tol=1e-10)
        assert math.isclose(sum(shares), 1.0, abs_tol=1e-10)

    def test_bertrand_shares_unequal_revenue(self):
        """Test market shares with unequal revenue."""
        prices = [20.0, 10.0, 5.0]
        quantities = [2.0, 3.0, 4.0]
        shares = calculate_market_shares_bertrand(prices, quantities)
        revenues = [20.0 * 2.0, 10.0 * 3.0, 5.0 * 4.0]  # [40, 30, 20]
        total_revenue = sum(revenues)
        assert len(shares) == 3
        assert math.isclose(shares[0], 40.0 / total_revenue, abs_tol=1e-10)
        assert math.isclose(shares[1], 30.0 / total_revenue, abs_tol=1e-10)
        assert math.isclose(shares[2], 20.0 / total_revenue, abs_tol=1e-10)
        assert math.isclose(sum(shares), 1.0, abs_tol=1e-10)

    def test_bertrand_shares_zero_revenue(self):
        """Test market shares with zero revenue."""
        prices = [10.0, 0.0, 5.0]
        quantities = [2.0, 3.0, 0.0]
        shares = calculate_market_shares_bertrand(prices, quantities)
        assert len(shares) == 3
        assert math.isclose(shares[0], 1.0, abs_tol=1e-10)
        assert math.isclose(shares[1], 0.0, abs_tol=1e-10)
        assert math.isclose(shares[2], 0.0, abs_tol=1e-10)

    def test_bertrand_shares_empty_lists(self):
        """Test market shares with empty lists."""
        with pytest.raises(
            ValueError, match="Prices and quantities lists cannot be empty"
        ):
            calculate_market_shares_bertrand([], [])

    def test_bertrand_shares_mismatched_lengths(self):
        """Test market shares with mismatched list lengths."""
        prices = [10.0, 20.0]
        quantities = [5.0]
        with pytest.raises(
            ValueError,
            match="Prices \\(2\\) and quantities \\(1\\) must have same length",
        ):
            calculate_market_shares_bertrand(prices, quantities)

    def test_bertrand_shares_negative_prices(self):
        """Test market shares with negative prices."""
        prices = [10.0, -5.0, 5.0]
        quantities = [2.0, 3.0, 4.0]
        with pytest.raises(ValueError, match="Price 1 = -5.000 must be non-negative"):
            calculate_market_shares_bertrand(prices, quantities)

    def test_bertrand_shares_negative_quantities(self):
        """Test market shares with negative quantities."""
        prices = [10.0, 20.0, 5.0]
        quantities = [2.0, -3.0, 4.0]
        with pytest.raises(
            ValueError, match="Quantity 1 = -3.000 must be non-negative"
        ):
            calculate_market_shares_bertrand(prices, quantities)

    def test_bertrand_shares_zero_total_revenue(self):
        """Test market shares when total revenue is zero."""
        prices = [0.0, 0.0, 0.0]
        quantities = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="Total revenue cannot be zero"):
            calculate_market_shares_bertrand(prices, quantities)
