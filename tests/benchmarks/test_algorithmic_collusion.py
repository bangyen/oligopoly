"""Tests for the long-horizon Q-learning experiments (small, fast settings)."""

import pytest

from sim.experiments.algorithmic_collusion import (
    ExperimentConfig,
    impulse_response,
    run_experiment,
)
from sim.markets import build_market

CES = ("bertrand", {"demand_type": "ces", "elasticity": 3.0, "market_elasticity": 2.0})
COSTS = [10.0, 10.0]
FAST = {"grid_size": 7, "beta": 2e-4, "convergence_window": 5_000}


def _run(memory=1, seed=0, market=CES):
    model, params = market
    return run_experiment(
        build_market(model, dict(params), 2),
        COSTS,
        ExperimentConfig(seed=seed, memory=memory, **FAST),
    )


def test_converges_and_reports_benchmarks() -> None:
    result = _run()
    assert result.converged
    assert result.periods < ExperimentConfig().max_periods
    assert len(result.grid) == 7
    assert result.nash_action < result.monopoly_action  # Bertrand prices
    assert result.nash_profit < result.monopoly_profit
    assert result.limit_path and len(result.limit_actions) == len(result.limit_path)


def test_seeded_runs_are_reproducible() -> None:
    first, second = _run(seed=3), _run(seed=3)
    assert first.periods == second.periods
    assert first.limit_actions == second.limit_actions


def test_memory_enables_supracompetitive_prices() -> None:
    """With memory the learners settle well above Nash; without it they do not."""
    with_memory = [_run(seed=s).collusion_index for s in range(3)]
    memoryless = [_run(memory=0, seed=s).collusion_index for s in range(3)]
    assert sum(with_memory) / 3 > sum(memoryless) / 3 + 0.3


def test_impulse_response_shape() -> None:
    model, params = CES
    market = build_market(model, dict(params), 2)
    result = _run()
    ir = impulse_response(market, COSTS, result, periods=10)
    assert len(ir.actions) == 11 and len(ir.profits) == 11
    assert ir.actions[0] == result.limit_actions[0]
    # The deviation is a static best response, so it pays in the deviation period
    assert ir.profits[1][0] >= ir.profits[0][0]


def test_memoryless_limit_path_is_constant() -> None:
    result = _run(memory=0)
    assert len(result.limit_path) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"grid_size": 1}, {"alpha": 0.0}, {"delta": 1.0}, {"beta": 0.0}, {"memory": 2}],
)
def test_invalid_config(kwargs) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_requires_symmetric_costs() -> None:
    model, params = CES
    with pytest.raises(ValueError, match="symmetric"):
        run_experiment(build_market(model, dict(params), 2), [10.0, 12.0])
