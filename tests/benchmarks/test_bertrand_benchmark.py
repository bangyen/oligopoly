"""Benchmark the Bertrand equilibrium against the textbook homogeneous-good model.

Demand Q(p) = alpha - beta p; the lowest price takes the market. Expected
values are derived by hand from the standard undercutting argument, and the
no-deviation check plays the engine's winner-take-all mode.
"""

import numpy as np
import pytest

from sim.games.bertrand import bertrand_simulation
from sim.strategies.nash_strategies import bertrand_nash_equilibrium

TOL = 1e-6
ALPHA, BETA = 200.0, 2.0


@pytest.mark.parametrize(
    ("costs", "price", "quantities"),
    [
        ([10.0], 55.0, [90.0]),  # monopoly: (alpha + beta c) / 2 beta
        ([10.0, 10.0], 10.0, [90.0, 90.0]),  # Bertrand paradox
        ([10.0, 10.0, 10.0], 10.0, [60.0, 60.0, 60.0]),
        ([10.0, 20.0], 20.0, [160.0, 0.0]),  # limit pricing
        ([30.0, 20.0, 25.0], 25.0, [0.0, 150.0, 0.0]),
        ([10.0, 80.0], 55.0, [90.0, 0.0]),  # monopoly price binds
        ([15.0, 15.0, 40.0], 15.0, [85.0, 85.0, 0.0]),
    ],
)
def test_matches_textbook(
    costs: list[float], price: float, quantities: list[float]
) -> None:
    _, eq_quantities, eq_profits, market_price = bertrand_nash_equilibrium(
        ALPHA, BETA, costs
    )
    assert market_price == pytest.approx(price, abs=TOL)
    assert eq_quantities == pytest.approx(quantities, abs=TOL)
    assert eq_profits == pytest.approx(
        [(price - c) * q for c, q in zip(costs, quantities)], abs=TOL
    )


@pytest.mark.parametrize(
    "costs", [[10.0, 10.0], [10.0, 20.0], [30.0, 20.0, 25.0], [10.0, 80.0]]
)
def test_no_profitable_unilateral_deviation(costs: list[float]) -> None:
    prices, _, profits, _ = bertrand_nash_equilibrium(ALPHA, BETA, costs)
    for i, cost in enumerate(costs):
        for p in np.linspace(cost, ALPHA / BETA, 2001):
            deviated = list(prices)
            deviated[i] = float(p)
            result = bertrand_simulation(
                ALPHA, BETA, costs, deviated, use_capacity_constraints=False
            )
            assert result.profits[i] <= profits[i] + TOL
