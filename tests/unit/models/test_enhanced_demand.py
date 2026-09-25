"""Tests for the enhanced demand functions."""

import math

import numpy as np
import pytest

from sim.models.enhanced_demand import (
    CESDemand,
    DynamicDemand,
    MultiSegmentDemand,
    NetworkEffectsDemand,
    calculate_enhanced_demand_elasticity,
    create_enhanced_demand_function,
)


class TestCESDemand:
    def test_expenditure_equals_market_size(self) -> None:
        """CES demand spends exactly scale * market_size across products."""
        demand = CESDemand(elasticity=3.0, scale_parameter=1.0, market_size=100.0)
        prices = [10.0, 20.0, 15.0]
        quantities = demand.calculate_demand(prices, [1.0, 1.0, 1.0])
        assert math.isclose(
            sum(p * q for p, q in zip(prices, quantities)), 100.0, rel_tol=1e-9
        )

    def test_cheaper_product_sells_more(self) -> None:
        demand = CESDemand(elasticity=2.0)
        low, high = demand.calculate_demand([5.0, 10.0], [1.0, 1.0])
        assert low > high

    def test_quality_offsets_price(self) -> None:
        """Doubling price and quality leaves the quality-adjusted price unchanged."""
        demand = CESDemand(elasticity=2.5)
        a, b = demand.calculate_demand([10.0, 20.0], [1.0, 2.0])
        assert math.isclose(a, b, rel_tol=1e-9)

    def test_market_shares_sum_to_one(self) -> None:
        shares = CESDemand().calculate_market_shares([5.0, 7.0, 9.0], [1.0, 1.2, 0.8])
        assert math.isclose(sum(shares), 1.0)

    def test_empty_and_mismatched_inputs(self) -> None:
        demand = CESDemand()
        assert demand.calculate_demand([], []) == []
        with pytest.raises(ValueError):
            demand.calculate_demand([1.0], [1.0, 2.0])

    @pytest.mark.parametrize(
        "kwargs",
        [{"elasticity": 1.0}, {"scale_parameter": 0.0}, {"market_size": -1.0}],
    )
    def test_invalid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            CESDemand(**kwargs)


class TestNetworkEffectsDemand:
    def test_network_effect_above_critical_mass_only(self) -> None:
        demand = NetworkEffectsDemand(
            network_strength=0.5, base_demand=100.0, critical_mass=10.0
        )
        below, above = demand.calculate_demand([1.0, 1.0], [5.0, 30.0], [1.0, 1.0])
        assert math.isclose(below, 50.0)
        assert math.isclose(above, 50.0 + 0.5 * 20.0)

    def test_network_value(self) -> None:
        demand = NetworkEffectsDemand(network_strength=0.1, critical_mass=10.0)
        assert demand.calculate_network_value([5.0, 20.0], [1.0, 2.0]) == [
            1.0,
            pytest.approx(3.0),
        ]
        with pytest.raises(ValueError):
            demand.calculate_network_value([1.0], [])

    def test_mismatched_inputs(self) -> None:
        with pytest.raises(ValueError):
            NetworkEffectsDemand().calculate_demand([1.0], [1.0, 2.0], [1.0])

    @pytest.mark.parametrize(
        "kwargs",
        [{"network_strength": -0.1}, {"base_demand": 0.0}, {"critical_mass": 0.0}],
    )
    def test_invalid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            NetworkEffectsDemand(**kwargs)


class TestDynamicDemand:
    def test_deterministic_without_volatility(self) -> None:
        demand = DynamicDemand(
            base_demand=100.0, growth_rate=0.0, volatility=0.0, seasonal_amplitude=0.0
        )
        quantities, size = demand.calculate_demand(0, [1.0], [1.0])
        assert math.isclose(size, 100.0)
        assert math.isclose(quantities[0], 50.0)

    def test_growth_over_a_year(self) -> None:
        demand = DynamicDemand(
            growth_rate=0.1, volatility=0.0, seasonal_amplitude=0.0, cycle_period=12
        )
        _, start = demand.calculate_demand(0, [1.0], [1.0])
        _, year = demand.calculate_demand(12, [1.0], [1.0])
        assert math.isclose(year / start, 1.1, rel_tol=1e-9)

    def test_market_size_stays_positive(self) -> None:
        np.random.seed(0)
        demand = DynamicDemand(volatility=5.0)
        for round_num in range(50):
            _, size = demand.calculate_demand(round_num, [1.0], [1.0])
            assert size >= 0.1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"base_demand": 0.0},
            {"volatility": -1.0},
            {"seasonal_amplitude": -1.0},
            {"cycle_period": 0},
        ],
    )
    def test_invalid_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            DynamicDemand(**kwargs)


class TestMultiSegmentDemand:
    SEGMENTS = [
        {"weight": 0.5, "price_sensitivity": 1.0, "quality_preference": 1.0},
        {"weight": 0.5, "price_sensitivity": 0.5, "quality_preference": 2.0},
    ]

    def test_demand_is_sum_of_segments(self) -> None:
        demand = MultiSegmentDemand(segments=self.SEGMENTS)
        (quantity,) = demand.calculate_demand([2.0], [1.0], total_market_size=100.0)
        assert math.isclose(quantity, 50.0 / 3.0 + 100.0 / 2.0)

    def test_market_shares_sum_to_one(self) -> None:
        demand = MultiSegmentDemand(segments=self.SEGMENTS)
        assert math.isclose(
            sum(demand.calculate_market_shares([1.0, 3.0], [1.0, 1.0])), 1.0
        )

    @pytest.mark.parametrize(
        "segments",
        [
            [],
            [{"weight": 1.0, "price_sensitivity": 1.0}],
            [{"weight": 1.5, "price_sensitivity": 1.0, "quality_preference": 1.0}],
            [{"weight": 1.0, "price_sensitivity": 0.0, "quality_preference": 1.0}],
            [{"weight": 0.4, "price_sensitivity": 1.0, "quality_preference": 1.0}],
        ],
    )
    def test_invalid_segments(self, segments: list) -> None:
        with pytest.raises(ValueError):
            MultiSegmentDemand(segments=segments)


class TestFactoryAndElasticity:
    @pytest.mark.parametrize(
        ("demand_type", "cls", "kwargs"),
        [
            ("ces", CESDemand, {}),
            ("network", NetworkEffectsDemand, {}),
            ("dynamic", DynamicDemand, {}),
            (
                "multi_segment",
                MultiSegmentDemand,
                {"segments": TestMultiSegmentDemand.SEGMENTS},
            ),
        ],
    )
    def test_factory(self, demand_type: str, cls: type, kwargs: dict) -> None:
        assert isinstance(create_enhanced_demand_function(demand_type, **kwargs), cls)

    def test_factory_unknown(self) -> None:
        with pytest.raises(ValueError):
            create_enhanced_demand_function("linear")

    def test_elasticity_is_negative(self) -> None:
        for demand in (
            CESDemand(elasticity=2.0),
            NetworkEffectsDemand(),
            DynamicDemand(volatility=0.0),
            MultiSegmentDemand(segments=TestMultiSegmentDemand.SEGMENTS),
        ):
            assert (
                calculate_enhanced_demand_elasticity(demand, [5.0, 5.0], [1.0, 1.0]) < 0
            )
