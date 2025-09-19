"""Tests for product differentiation models.

This module tests the product differentiation implementations including
horizontal differentiation (Hotelling), vertical differentiation,
and differentiated Bertrand competition.
"""

import pytest

from src.sim.models.product_differentiation import (
    DifferentiatedBertrandResult,
    HotellingDemand,
    LogitDemand,
    ProductCharacteristics,
    VerticalDifferentiation,
    calculate_differentiated_nash_equilibrium,
    differentiated_bertrand_simulation,
)


class TestProductCharacteristics:
    """Test ProductCharacteristics class."""

    def test_product_characteristics_creation(self) -> None:
        """Test creating product characteristics."""
        product = ProductCharacteristics(
            quality=1.2, location=0.3, brand_strength=1.5, innovation_level=0.2
        )

        assert product.quality == 1.2
        assert product.location == 0.3
        assert product.brand_strength == 1.5
        assert product.innovation_level == 0.2

    def test_product_characteristics_validation(self) -> None:
        """Test product characteristics validation."""
        # Test negative quality
        with pytest.raises(ValueError, match="Quality must be positive"):
            ProductCharacteristics(quality=-1.0)

        # Test location out of bounds
        with pytest.raises(ValueError, match="Location must be in"):
            ProductCharacteristics(location=1.5)

        # Test negative brand strength
        with pytest.raises(ValueError, match="Brand strength must be positive"):
            ProductCharacteristics(brand_strength=-1.0)

        # Test negative innovation level
        with pytest.raises(ValueError, match="Innovation level must be non-negative"):
            ProductCharacteristics(innovation_level=-0.1)


class TestHotellingDemand:
    """Test HotellingDemand class."""

    def test_hotelling_demand_creation(self) -> None:
        """Test creating Hotelling demand."""
        demand = HotellingDemand(transportation_cost=1.0, consumer_density=1.0)

        assert demand.transportation_cost == 1.0
        assert demand.consumer_density == 1.0

    def test_hotelling_demand_validation(self) -> None:
        """Test Hotelling demand validation."""
        with pytest.raises(ValueError, match="Transportation cost must be positive"):
            HotellingDemand(transportation_cost=0.0)

        with pytest.raises(ValueError, match="Consumer density must be positive"):
            HotellingDemand(consumer_density=-1.0)

    def test_hotelling_demand_calculation(self) -> None:
        """Test Hotelling demand calculation."""
        demand = HotellingDemand(transportation_cost=1.0, consumer_density=1.0)

        # Test with two firms
        prices = [20.0, 25.0]
        locations = [0.2, 0.8]

        quantities = demand.calculate_demand(prices, locations)

        assert len(quantities) == 2
        assert all(q >= 0 for q in quantities)
        assert abs(sum(quantities) - 1.0) < 1e-6  # Should sum to total market

    def test_hotelling_demand_monopoly(self) -> None:
        """Test Hotelling demand with monopoly."""
        demand = HotellingDemand(transportation_cost=1.0, consumer_density=1.0)

        prices = [20.0]
        locations = [0.5]

        quantities = demand.calculate_demand(prices, locations)

        assert len(quantities) == 1
        assert abs(quantities[0] - 1.0) < 1e-6  # Monopoly gets all demand

    def test_hotelling_demand_mismatched_lengths(self) -> None:
        """Test Hotelling demand with mismatched input lengths."""
        demand = HotellingDemand(transportation_cost=1.0, consumer_density=1.0)

        with pytest.raises(
            ValueError, match="Prices and locations must have same length"
        ):
            demand.calculate_demand([20.0, 25.0], [0.2])  # Different lengths


class TestLogitDemand:
    """Test LogitDemand class."""

    def test_logit_demand_creation(self) -> None:
        """Test creating Logit demand."""
        demand = LogitDemand(price_sensitivity=1.0, quality_sensitivity=1.0)

        assert demand.price_sensitivity == 1.0
        assert demand.quality_sensitivity == 1.0

    def test_logit_demand_validation(self) -> None:
        """Test Logit demand validation."""
        with pytest.raises(ValueError, match="Price sensitivity must be positive"):
            LogitDemand(price_sensitivity=0.0)

        with pytest.raises(ValueError, match="Quality sensitivity must be positive"):
            LogitDemand(quality_sensitivity=-1.0)

    def test_logit_demand_calculation(self) -> None:
        """Test Logit demand calculation."""
        demand = LogitDemand(price_sensitivity=1.0, quality_sensitivity=1.0)

        prices = [2.0, 2.5]
        products = [
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
            ProductCharacteristics(quality=1.2, brand_strength=1.0),
        ]

        quantities = demand.calculate_demand(prices, products, 100.0)

        assert len(quantities) == 2
        assert all(q >= 0 for q in quantities)
        # Logit demand includes outside option, so quantities may not sum to total market
        assert sum(quantities) <= 100.0  # Should not exceed total market

    def test_logit_demand_market_shares(self) -> None:
        """Test Logit demand market shares calculation."""
        demand = LogitDemand(price_sensitivity=1.0, quality_sensitivity=1.0)

        prices = [2.0, 2.5]
        products = [
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
            ProductCharacteristics(quality=1.2, brand_strength=1.0),
        ]

        shares = demand.calculate_market_shares(prices, products)

        assert len(shares) == 2
        assert all(0 <= s <= 1 for s in shares)
        # Logit demand includes outside option, so shares may not sum to 1.0
        assert sum(shares) <= 1.0  # Should not exceed 1.0

    def test_logit_demand_mismatched_lengths(self) -> None:
        """Test Logit demand with mismatched input lengths."""
        demand = LogitDemand(price_sensitivity=1.0, quality_sensitivity=1.0)

        with pytest.raises(
            ValueError, match="Prices and products must have same length"
        ):
            demand.calculate_demand(
                [20.0, 25.0], [ProductCharacteristics()]
            )  # Different lengths


class TestVerticalDifferentiation:
    """Test VerticalDifferentiation class."""

    def test_vertical_differentiation_creation(self) -> None:
        """Test creating vertical differentiation."""
        diff = VerticalDifferentiation(consumer_heterogeneity=1.0)

        assert diff.consumer_heterogeneity == 1.0

    def test_vertical_differentiation_validation(self) -> None:
        """Test vertical differentiation validation."""
        with pytest.raises(ValueError, match="Consumer heterogeneity must be positive"):
            VerticalDifferentiation(consumer_heterogeneity=0.0)

    def test_vertical_differentiation_calculation(self) -> None:
        """Test vertical differentiation demand calculation."""
        diff = VerticalDifferentiation(consumer_heterogeneity=1.0)

        prices = [20.0, 25.0, 18.0]
        qualities = [1.0, 1.2, 0.8]

        quantities = diff.calculate_demand(prices, qualities, 100.0)

        assert len(quantities) == 3
        assert all(q >= 0 for q in quantities)
        assert abs(sum(quantities) - 100.0) < 1e-6  # Should sum to total market

    def test_vertical_differentiation_mismatched_lengths(self) -> None:
        """Test vertical differentiation with mismatched input lengths."""
        diff = VerticalDifferentiation(consumer_heterogeneity=1.0)

        with pytest.raises(
            ValueError, match="Prices and qualities must have same length"
        ):
            diff.calculate_demand([20.0, 25.0], [1.0])  # Different lengths


class TestDifferentiatedBertrandSimulation:
    """Test differentiated Bertrand simulation."""

    def test_differentiated_bertrand_simulation(self) -> None:
        """Test differentiated Bertrand simulation."""
        prices = [20.0, 25.0]
        products = [
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
            ProductCharacteristics(quality=1.2, brand_strength=1.0),
        ]
        costs = [10.0, 12.0]

        result = differentiated_bertrand_simulation(
            prices, products, costs, "logit", {}, 100.0
        )

        assert isinstance(result, DifferentiatedBertrandResult)
        assert len(result.prices) == 2
        assert len(result.quantities) == 2
        assert len(result.market_shares) == 2
        assert len(result.profits) == 2
        assert result.total_demand > 0
        assert result.consumer_surplus >= 0

    def test_differentiated_bertrand_validation(self) -> None:
        """Test differentiated Bertrand simulation validation."""
        with pytest.raises(
            ValueError, match="Prices, products, and costs must have same length"
        ):
            differentiated_bertrand_simulation(
                [20.0, 25.0], [ProductCharacteristics()], [10.0, 12.0, 15.0]
            )

        with pytest.raises(ValueError, match="Unknown demand model"):
            differentiated_bertrand_simulation(
                [20.0], [ProductCharacteristics()], [10.0], "unknown_model"
            )

    def test_differentiated_bertrand_different_models(self) -> None:
        """Test differentiated Bertrand with different demand models."""
        prices = [20.0, 25.0]
        products = [
            ProductCharacteristics(quality=1.0, location=0.2),
            ProductCharacteristics(quality=1.2, location=0.8),
        ]
        costs = [10.0, 12.0]

        # Test Logit model
        logit_result = differentiated_bertrand_simulation(
            prices, products, costs, "logit", {}, 100.0
        )

        # Test Hotelling model
        hotelling_result = differentiated_bertrand_simulation(
            prices, products, costs, "hotelling", {"locations": [0.2, 0.8]}, 100.0
        )

        # Test Vertical differentiation model
        vertical_result = differentiated_bertrand_simulation(
            prices, products, costs, "vertical", {}, 100.0
        )

        # All should produce valid results
        for result in [logit_result, hotelling_result, vertical_result]:
            assert isinstance(result, DifferentiatedBertrandResult)
            assert len(result.prices) == 2
            assert all(q >= 0 for q in result.quantities)
            assert all(0 <= s <= 1 for s in result.market_shares)


class TestDifferentiatedNashEquilibrium:
    """Test differentiated Nash equilibrium calculation."""

    def test_differentiated_nash_equilibrium(self) -> None:
        """Test differentiated Nash equilibrium calculation."""
        products = [
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
            ProductCharacteristics(quality=1.2, brand_strength=1.0),
        ]
        costs = [10.0, 12.0]

        equilibrium_prices, result = calculate_differentiated_nash_equilibrium(
            products, costs, "logit", {}, 100.0
        )

        assert len(equilibrium_prices) == 2
        assert isinstance(result, DifferentiatedBertrandResult)
        assert all(p > 0 for p in equilibrium_prices)
        assert all(p >= c for p, c in zip(equilibrium_prices, costs))  # Prices >= costs

    def test_differentiated_nash_equilibrium_validation(self) -> None:
        """Test differentiated Nash equilibrium validation."""
        with pytest.raises(
            ValueError, match="Products and costs must have same length"
        ):
            calculate_differentiated_nash_equilibrium(
                [ProductCharacteristics()], [10.0, 12.0]
            )

    def test_differentiated_nash_equilibrium_convergence(self) -> None:
        """Test that Nash equilibrium calculation converges."""
        products = [
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
            ProductCharacteristics(quality=1.2, brand_strength=1.0),
            ProductCharacteristics(quality=0.8, brand_strength=1.0),
        ]
        costs = [10.0, 12.0, 8.0]

        equilibrium_prices, result = calculate_differentiated_nash_equilibrium(
            products, costs, "logit", {}, 100.0, max_iterations=50, tolerance=1e-6
        )

        # Should converge to reasonable prices
        assert len(equilibrium_prices) == 3
        assert all(p > 0 for p in equilibrium_prices)
        assert all(p >= c for p, c in zip(equilibrium_prices, costs))

        # Profits should be non-negative at equilibrium
        assert all(p >= 0 for p in result.profits)


class TestDifferentiatedBertrandResult:
    """Test DifferentiatedBertrandResult class."""

    def test_result_creation(self) -> None:
        """Test creating differentiated Bertrand result."""
        result = DifferentiatedBertrandResult(
            prices=[20.0, 25.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            profits=[150.0, 156.0],
            total_demand=27.0,
            consumer_surplus=500.0,
        )

        assert result.prices == [20.0, 25.0]
        assert result.quantities == [15.0, 12.0]
        assert result.market_shares == [0.6, 0.4]
        assert result.profits == [150.0, 156.0]
        assert result.total_demand == 27.0
        assert result.consumer_surplus == 500.0

    def test_result_repr(self) -> None:
        """Test result string representation."""
        result = DifferentiatedBertrandResult(
            prices=[20.0, 25.0],
            quantities=[15.0, 12.0],
            market_shares=[0.6, 0.4],
            profits=[150.0, 156.0],
            total_demand=27.0,
            consumer_surplus=500.0,
        )

        repr_str = repr(result)
        assert "DifferentiatedBertrandResult" in repr_str
        assert "prices=" in repr_str
        assert "quantities=" in repr_str
        assert "market_shares=" in repr_str
        assert "profits=" in repr_str


class TestProductDifferentiationIntegration:
    """Test product differentiation integration."""

    def test_hotelling_vs_logit_demand(self) -> None:
        """Test that Hotelling and Logit demand produce different results."""
        prices = [20.0, 25.0]
        products = [
            ProductCharacteristics(quality=1.0, location=0.2),
            ProductCharacteristics(quality=1.2, location=0.8),
        ]
        costs = [10.0, 12.0]

        # Hotelling demand
        hotelling_result = differentiated_bertrand_simulation(
            prices, products, costs, "hotelling", {"locations": [0.2, 0.8]}, 100.0
        )

        # Logit demand
        logit_result = differentiated_bertrand_simulation(
            prices, products, costs, "logit", {}, 100.0
        )

        # Results should be different
        assert hotelling_result.quantities != logit_result.quantities
        assert hotelling_result.market_shares != logit_result.market_shares

    def test_quality_impact_on_demand(self) -> None:
        """Test that higher quality products get more demand."""
        prices = [20.0, 20.0]  # Same prices
        products_low_quality = [
            ProductCharacteristics(quality=0.8, brand_strength=1.0),
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
        ]
        products_high_quality = [
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
            ProductCharacteristics(quality=1.2, brand_strength=1.0),
        ]
        costs = [10.0, 12.0]

        result_low = differentiated_bertrand_simulation(
            prices, products_low_quality, costs, "logit", {}, 100.0
        )

        result_high = differentiated_bertrand_simulation(
            prices, products_high_quality, costs, "logit", {}, 100.0
        )

        # Higher quality should lead to higher market share
        assert result_high.market_shares[1] > result_low.market_shares[1]

    def test_price_impact_on_demand(self) -> None:
        """Test that lower prices lead to higher demand."""
        prices_low = [18.0, 22.0]
        prices_high = [22.0, 26.0]
        products = [
            ProductCharacteristics(quality=1.0, brand_strength=1.0),
            ProductCharacteristics(quality=1.2, brand_strength=1.0),
        ]
        costs = [10.0, 12.0]

        result_low = differentiated_bertrand_simulation(
            prices_low, products, costs, "logit", {}, 100.0
        )

        result_high = differentiated_bertrand_simulation(
            prices_high, products, costs, "logit", {}, 100.0
        )

        # Lower prices should lead to higher total demand
        assert result_low.total_demand > result_high.total_demand
