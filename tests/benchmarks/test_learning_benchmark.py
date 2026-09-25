"""Regression checks for the learning benchmark (see docs/learning_benchmarks.md)."""

import pytest

from scripts import learning_benchmark as lb


@pytest.fixture
def fictitious_play_only(monkeypatch):
    monkeypatch.setattr(lb, "LEARNING_STRATEGY_TYPES", ("fictitious_play",))


@pytest.mark.parametrize(
    "market", ["Linear Cournot", "Isoelastic Cournot", "CES Bertrand"]
)
def test_fictitious_play_converges_to_nash(fictitious_play_only, monkeypatch, market):
    monkeypatch.setattr(lb, "MARKETS", {market: lb.MARKETS[market]})
    (outcome,) = lb.run_benchmark(rounds=80, seeds=1, window=10)
    assert outcome.nash_gap < 0.03
    assert abs(outcome.collusion_index) < 0.1


def test_markdown_covers_every_market_and_strategy(monkeypatch):
    monkeypatch.setattr(lb, "MARKETS", {"Linear Cournot": lb.MARKETS["Linear Cournot"]})
    outcomes = lb.run_benchmark(rounds=12, seeds=1, window=4)
    table = lb.to_markdown(outcomes, 12, 1, 4)
    assert len(outcomes) == len(lb.LEARNING_STRATEGY_TYPES)
    for strategy in lb.LEARNING_STRATEGY_TYPES:
        assert f"`{strategy}`" in table
