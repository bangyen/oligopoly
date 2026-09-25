"""Benchmark the isoelastic and CES markets against closed-form equilibria.

Each case derives the expected equilibrium by hand and then brute-forces
every firm's unilateral deviations through the market's own round engine.
"""

import math

import numpy as np
import pytest

from sim.markets import (
    CESBertrandMarket,
    IsoelasticBertrandMarket,
    IsoelasticCournotMarket,
)

TOL = 1e-6


def _assert_no_profitable_deviation(market, costs, rel_tol=1e-6):
    actions = market.nash(costs)
    for i, cost in enumerate(costs):
        base = market.profit(i, actions[i], actions, costs)
        lo, hi = market.action_bounds(cost, costs)
        for x in np.linspace(lo, hi, 4001):
            assert market.profit(i, float(x), actions, costs) <= base + rel_tol * max(
                1.0, abs(base)
            )


class TestIsoelasticCournot:
    @pytest.mark.parametrize(("n", "e"), [(1, 2.0), (2, 2.0), (3, 1.5), (5, 3.0)])
    def test_symmetric_closed_form(self, n: int, e: float) -> None:
        """Symmetric price is n c / (n - 1/e); n = 1 gives the monopoly markup."""
        scale, c = 100.0, 10.0
        market = IsoelasticCournotMarket({"A": scale, "elasticity": e})
        quantities, price = market.equilibrium([c] * n)
        expected_price = n * c / (n - 1 / e)
        assert price == pytest.approx(expected_price, rel=TOL)
        assert sum(quantities) == pytest.approx((scale / expected_price) ** e, rel=TOL)
        assert quantities == pytest.approx([quantities[0]] * n, rel=TOL)

    @pytest.mark.parametrize(
        "costs", [[10.0, 12.0], [5.0, 8.0, 9.0], [10.0, 10.0, 10.0]]
    )
    def test_no_profitable_deviation(self, costs: list[float]) -> None:
        market = IsoelasticCournotMarket({"A": 100.0, "elasticity": 2.0})
        _assert_no_profitable_deviation(market, costs)

    def test_inefficient_firm_inactive(self) -> None:
        """With e = 2 and c = [10, 30], firm 2's share e(1 - 30/P) is negative."""
        market = IsoelasticCournotMarket({"A": 100.0, "elasticity": 2.0})
        quantities, price = market.equilibrium([10.0, 30.0])
        assert quantities[1] == 0.0
        assert price == pytest.approx(20.0)  # monopoly: c e / (e - 1)


class TestIsoelasticBertrand:
    @pytest.mark.parametrize(
        ("costs", "price"),
        [
            ([10.0, 10.0], 10.0),  # Bertrand paradox
            ([10.0, 15.0], 15.0),  # limit price at rival cost
            ([10.0, 30.0], 20.0),  # monopoly price c e/(e-1) binds
        ],
    )
    def test_closed_form(self, costs: list[float], price: float) -> None:
        market = IsoelasticBertrandMarket({"A": 100.0, "elasticity": 2.0})
        assert min(market.nash(costs)) == pytest.approx(price)

    @pytest.mark.parametrize("costs", [[10.0, 10.0], [10.0, 15.0], [10.0, 30.0]])
    def test_no_profitable_deviation(self, costs: list[float]) -> None:
        market = IsoelasticBertrandMarket({"A": 100.0, "elasticity": 2.0})
        _assert_no_profitable_deviation(market, costs)


class TestCESBertrand:
    @pytest.mark.parametrize(
        ("n", "sigma", "eta"),
        [(1, 3.0, 2.0), (2, 2.0, 1.5), (3, 4.0, 2.0), (5, 1.5, 3.0)],
    )
    def test_symmetric_closed_form(self, n: int, sigma: float, eta: float) -> None:
        """Symmetric shares 1/n give e = sigma - (sigma-eta)/n, p = c e/(e-1).

        n = 1 is the monopoly price c eta / (eta - 1).
        """
        c = 10.0
        market = CESBertrandMarket({"elasticity": sigma, "market_elasticity": eta}, n)
        e = sigma - (sigma - eta) / n
        assert market.nash([c] * n) == pytest.approx([c * e / (e - 1)] * n, rel=TOL)

    @pytest.mark.parametrize(
        ("costs", "qualities"),
        [
            ([10.0], None),
            ([10.0, 10.0], None),
            ([8.0, 12.0], None),
            ([10.0, 10.0, 10.0], [1.0, 1.5, 0.8]),
        ],
    )
    def test_no_profitable_deviation(self, costs, qualities) -> None:
        params = {"elasticity": 3.0, "market_elasticity": 2.0, "market_size": 5000.0}
        if qualities:
            params["qualities"] = qualities
        market = CESBertrandMarket(params, len(costs))
        _assert_no_profitable_deviation(market, costs)

    def test_group_demand_and_consumer_surplus(self) -> None:
        """Spending is M P^(1-eta); surplus integrates Q(P) = M P^-eta above P."""
        sigma, eta, size = 2.5, 2.0, 250.0
        market = CESBertrandMarket(
            {"elasticity": sigma, "market_elasticity": eta, "market_size": size}, 3
        )
        prices = [12.0, 15.0, 20.0]
        result = market.play(prices, [10.0] * 3, [0.0] * 3)
        index = sum(p ** (1 - sigma) for p in prices) ** (1 / (1 - sigma))
        spend = sum(p * q for p, q in zip(result.prices, result.quantities))
        assert math.isclose(spend, size * index ** (1 - eta))
        _, _, surplus = market.metrics(result.prices, result.quantities)
        assert math.isclose(surplus, size * index ** (1 - eta) / (eta - 1))
