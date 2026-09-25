"""Benchmark the Cournot engine against textbook closed-form results.

Every expected value here is derived independently of the package, so these
tests back the README's accuracy claims rather than restating the code.
For inverse demand P = a - bQ and constant marginal costs c_i, the interior
Nash equilibrium is

    q_i* = (a - (n + 1) c_i + sum_j c_j) / (b (n + 1))
    P*   = (a + sum_j c_j) / (n + 1)
    pi_i = b q_i*^2
"""

import numpy as np
import pytest

from sim.games.cournot import cournot_simulation
from sim.strategies.nash_strategies import (
    cournot_best_response,
    cournot_nash_equilibrium,
)

TOL = 1e-6

# (a, b, costs) with every firm active at equilibrium
CASES = [
    (100.0, 1.0, [10.0, 10.0]),
    (100.0, 1.0, [10.0, 20.0]),
    (120.0, 2.0, [5.0, 15.0, 25.0]),
    (50.0, 0.5, [8.0, 8.0, 8.0, 8.0]),
    (200.0, 1.5, [30.0, 10.0, 20.0, 40.0, 25.0]),
]


def _closed_form(a: float, b: float, costs: list[float]) -> tuple[list[float], float]:
    n = len(costs)
    total = sum(costs)
    quantities = [(a - (n + 1) * c + total) / (b * (n + 1)) for c in costs]
    return quantities, (a + total) / (n + 1)


@pytest.mark.parametrize(("a", "b", "costs"), CASES)
def test_equilibrium_matches_closed_form(
    a: float, b: float, costs: list[float]
) -> None:
    quantities, price, profits = cournot_nash_equilibrium(a, b, costs)
    expected_q, expected_p = _closed_form(a, b, costs)

    assert quantities == pytest.approx(expected_q, abs=TOL)
    assert price == pytest.approx(expected_p, abs=TOL)
    assert profits == pytest.approx([b * q**2 for q in expected_q], abs=TOL)


def _textbook_profit(
    a: float, b: float, cost: float, own_q: float, rival_total: float
) -> float:
    return (max(0.0, a - b * (own_q + rival_total)) - cost) * own_q


@pytest.mark.parametrize(("a", "b", "costs"), CASES)
def test_no_profitable_unilateral_deviation(
    a: float, b: float, costs: list[float]
) -> None:
    """Brute-force check that no firm gains by deviating on a fine grid."""
    quantities, _, profits = cournot_nash_equilibrium(a, b, costs)
    grid = np.linspace(0.0, a / b, 2001)
    for i, cost in enumerate(costs):
        rival_total = sum(quantities) - quantities[i]
        best = max(_textbook_profit(a, b, cost, float(q), rival_total) for q in grid)
        assert best <= profits[i] + TOL


def test_engine_has_no_profitable_deviation_from_nash() -> None:
    """Regression: the engine once removed below-cost rivals after the round and
    recomputed the price, letting a low-cost firm flood the market for profit.
    """
    a, b, costs = 100.0, 1.0, [10.0, 20.0]
    quantities, _, profits = cournot_nash_equilibrium(a, b, costs)
    for q in np.linspace(0.0, a / b, 2001):
        result = cournot_simulation(a, b, costs, [float(q), quantities[1]])
        assert result.profits[0] <= profits[0] + TOL


@pytest.mark.parametrize(("a", "b", "costs"), CASES)
def test_one_round_simulation_at_equilibrium(
    a: float, b: float, costs: list[float]
) -> None:
    """The per-round game engine reproduces the equilibrium price and profits."""
    expected_q, expected_p = _closed_form(a, b, costs)
    result = cournot_simulation(a, b, costs, expected_q)

    assert result.price == pytest.approx(expected_p, abs=TOL)
    assert result.profits == pytest.approx([b * q**2 for q in expected_q], abs=TOL)


@pytest.mark.parametrize(("a", "b", "costs"), [c for c in CASES if len(c[2]) == 2])
def test_best_response_dynamics_converge(
    a: float, b: float, costs: list[float]
) -> None:
    """Duopoly best-response iteration converges to the Nash equilibrium."""
    quantities = [0.0, 0.0]
    for _ in range(100):
        quantities = [
            cournot_best_response(a, b, costs[i], [quantities[1 - i]]) for i in range(2)
        ]
    expected_q, _ = _closed_form(a, b, costs)
    assert quantities == pytest.approx(expected_q, abs=TOL)


def test_high_cost_firm_exits() -> None:
    """A firm priced out of the market produces nothing; the rest re-solve."""
    a, b, costs = 100.0, 1.0, [10.0, 10.0, 90.0]
    quantities, price, _ = cournot_nash_equilibrium(a, b, costs)
    expected_q, expected_p = _closed_form(a, b, costs[:2])

    assert quantities[2] == 0.0
    assert quantities[:2] == pytest.approx(expected_q, abs=TOL)
    assert price == pytest.approx(expected_p, abs=TOL)


@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_symmetric_outcome_approaches_competition(n: int) -> None:
    """Symmetric price is (a + n c) / (n + 1): monopoly at n=1, -> c as n grows."""
    a, b, c = 100.0, 1.0, 10.0
    _, price, _ = cournot_nash_equilibrium(a, b, [c] * n)
    assert price == pytest.approx((a + n * c) / (n + 1), abs=TOL)
