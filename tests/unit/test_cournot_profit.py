"""Tests for Cournot profit calculation functionality.

This module tests the profit calculation logic in the Cournot simulation,
ensuring that individual firm profits are computed correctly based on
market price, firm costs, and quantities chosen.
"""

from src.sim.cournot import cournot_simulation


class TestCournotProfit:
    """Test cases for Cournot profit calculation."""

    def test_cournot_profit_basic(self) -> None:
        """Test basic profit calculation: costs=[10,20] -> π1=600, π2=1000."""
        result = cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[10, 20])

        # P = 100 - 1 * (10 + 20) = 70
        # π1 = (70 - 10) * 10 = 600
        # π2 = (70 - 20) * 20 = 1000
        assert result.profits[0] == 600.0
        assert result.profits[1] == 1000.0

    def test_cournot_profit_zero_price(self) -> None:
        """Test profit calculation when market price is zero."""
        result = cournot_simulation(a=100, b=1, costs=[10, 20], quantities=[50, 50])

        # P = 0, so π1 = (0 - 10) * 50 = -500, π2 = (0 - 20) * 50 = -1000
        assert result.profits[0] == -500.0
        assert result.profits[1] == -1000.0

    def test_cournot_profit_equal_costs(self) -> None:
        """Test profit calculation when all firms have equal costs."""
        result = cournot_simulation(a=100, b=1, costs=[15, 15], quantities=[20, 30])

        # P = 100 - 1 * (20 + 30) = 50
        # Both firms: π = (50 - 15) * q = 35 * q
        assert result.profits[0] == 35.0 * 20  # 700
        assert result.profits[1] == 35.0 * 30  # 1050

    def test_cournot_profit_single_firm(self) -> None:
        """Test profit calculation with single firm."""
        result = cournot_simulation(a=100, b=1, costs=[25], quantities=[40])

        # P = 100 - 1 * 40 = 60
        # π = (60 - 25) * 40 = 1400
        assert result.profits[0] == 1400.0

    def test_cournot_profit_break_even(self) -> None:
        """Test profit calculation when price equals cost."""
        result = cournot_simulation(a=100, b=1, costs=[50], quantities=[50])

        # P = 100 - 1 * 50 = 50
        # π = (50 - 50) * 50 = 0
        assert result.profits[0] == 0.0

    def test_cournot_profit_negative_profit(self) -> None:
        """Test profit calculation when cost exceeds price."""
        result = cournot_simulation(a=100, b=1, costs=[80], quantities=[30])

        # P = 100 - 1 * 30 = 70
        # π = (70 - 80) * 30 = -300
        assert result.profits[0] == -300.0

    def test_cournot_profit_multiple_firms(self) -> None:
        """Test profit calculation with multiple firms."""
        result = cournot_simulation(
            a=60, b=0.5, costs=[5, 10, 15], quantities=[20, 15, 10]
        )

        # P = 60 - 0.5 * (20 + 15 + 10) = 60 - 22.5 = 37.5
        expected_profits = [
            (37.5 - 5) * 20,  # 650
            (37.5 - 10) * 15,  # 412.5
            (37.5 - 15) * 10,  # 225
        ]

        for i, expected in enumerate(expected_profits):
            assert abs(result.profits[i] - expected) < 1e-10
