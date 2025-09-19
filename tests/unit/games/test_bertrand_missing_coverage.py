"""Tests for missing coverage in bertrand.py module."""

import pytest

from src.sim.games.bertrand import (
    BertrandResult,
    allocate_demand,
    calculate_demand,
    parse_costs,
    parse_prices,
    validate_prices,
)


class TestBertrandResult:
    """Test the BertrandResult class."""

    def test_bertrand_result_repr(self) -> None:
        """Test the string representation of BertrandResult."""
        result = BertrandResult(
            total_demand=50.0,
            prices=[10.0, 15.0],
            quantities=[30.0, 20.0],
            profits=[100.0, 50.0],
        )

        repr_str = repr(result)
        expected = "BertrandResult(demand=50.0, prices=[10.0, 15.0], quantities=[30.0, 20.0], profits=[100.0, 50.0])"
        assert repr_str == expected


class TestAllocateDemand:
    """Test the allocate_demand function."""

    def test_allocate_demand_empty_prices(self) -> None:
        """Test allocate_demand with empty prices list."""
        result = allocate_demand(
            [], [10.0, 20.0], 100.0, 1.0, use_capacity_constraints=False
        )
        assert result == ([], 0.0)


class TestCalculateDemand:
    """Test the calculate_demand function."""

    def test_calculate_demand_edge_cases(self) -> None:
        """Test calculate_demand with edge cases."""
        # Test with alpha = 0 (no demand at any price)
        demand = calculate_demand(0.0, 1.0, 10.0)
        assert demand == 0.0

        # Test with beta = 0 (infinite demand at any price)
        demand = calculate_demand(100.0, 0.0, 10.0)
        assert demand == 100.0

        # Test with very high price (should result in zero demand)
        demand = calculate_demand(100.0, 1.0, 200.0)
        assert demand == 0.0


class TestValidatePrices:
    """Test the validate_prices function."""

    def test_validate_prices_negative_prices(self) -> None:
        """Test validate_prices with negative prices."""
        with pytest.raises(ValueError, match="Price p_1 = -5.0 must be non-negative"):
            validate_prices([10.0, -5.0, 15.0])


class TestParseCosts:
    """Test the parse_costs function."""

    def test_parse_costs_empty_string(self) -> None:
        """Test parse_costs with empty string."""
        with pytest.raises(ValueError, match="Costs list cannot be empty"):
            parse_costs("")

    def test_parse_costs_whitespace_only(self) -> None:
        """Test parse_costs with whitespace-only string."""
        with pytest.raises(ValueError, match="Costs list cannot be empty"):
            parse_costs("   ")

    def test_parse_costs_empty_after_split(self) -> None:
        """Test parse_costs with string that becomes empty after splitting."""
        with pytest.raises(ValueError, match="Costs list cannot be empty"):
            parse_costs(",,,")

    def test_parse_costs_invalid_format(self) -> None:
        """Test parse_costs with invalid format."""
        with pytest.raises(ValueError, match="Invalid costs format"):
            parse_costs("10,abc,20")


class TestParsePrices:
    """Test the parse_prices function."""

    def test_parse_prices_empty_string(self) -> None:
        """Test parse_prices with empty string."""
        with pytest.raises(ValueError, match="Prices list cannot be empty"):
            parse_prices("")

    def test_parse_prices_whitespace_only(self) -> None:
        """Test parse_prices with whitespace-only string."""
        with pytest.raises(ValueError, match="Prices list cannot be empty"):
            parse_prices("   ")

    def test_parse_prices_empty_after_split(self) -> None:
        """Test parse_prices with string that becomes empty after splitting."""
        with pytest.raises(ValueError, match="Prices list cannot be empty"):
            parse_prices(",,,")

    def test_parse_prices_invalid_format(self) -> None:
        """Test parse_prices with invalid format."""
        with pytest.raises(ValueError, match="Invalid prices format"):
            parse_prices("10,abc,20")
