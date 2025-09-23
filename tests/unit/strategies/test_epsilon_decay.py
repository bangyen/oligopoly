"""Test epsilon decay mechanism in Q-learning strategy.

This test verifies that the exploration rate ε decreases over time
according to the decay rate, but never goes below the minimum bound ε_min.
"""

import math
from typing import List

import pytest

from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import QLearning


def test_epsilon_decay_basic() -> None:
    """Test basic epsilon decay functionality."""
    q_learning = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=1.0,  # Start with full exploration
        epsilon_min=0.05,  # Minimum exploration
        epsilon_decay=0.9,  # Decay rate
        seed=42,
    )

    # Verify initial epsilon
    assert q_learning.get_current_epsilon() == 1.0

    bounds = (0.0, 10.0)
    market_params: dict[str, float] = {}

    # Track epsilon over multiple rounds
    epsilon_values: List[float] = []

    # First round
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    epsilon_values.append(q_learning.get_current_epsilon())

    # Multiple rounds to observe decay
    for round_num in range(1, 10):
        # Create mock result
        mock_result = CournotResult(price=20.0, quantities=[action1], profits=[30.0])

        # Choose action (triggers epsilon decay)
        _ = q_learning.next_action(
            round_num=round_num,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        epsilon_values.append(q_learning.get_current_epsilon())

    # Verify epsilon decreases over time
    assert len(epsilon_values) == 10

    # Check that epsilon decreases
    for i in range(1, len(epsilon_values)):
        assert epsilon_values[i] <= epsilon_values[i - 1], (
            f"Epsilon should decrease: round {i - 1}={epsilon_values[i - 1]}, round {i}={epsilon_values[i]}"
        )

    # Check that epsilon never goes below minimum
    for i, eps in enumerate(epsilon_values):
        assert eps >= q_learning.epsilon_min, (
            f"Epsilon at round {i} ({eps}) should be >= epsilon_min ({q_learning.epsilon_min})"
        )

    # Check that epsilon approaches the minimum
    final_epsilon = epsilon_values[-1]
    assert final_epsilon >= q_learning.epsilon_min, (
        f"Final epsilon ({final_epsilon}) should be >= epsilon_min ({q_learning.epsilon_min})"
    )


def test_epsilon_decay_rate() -> None:
    """Test that epsilon decays at the correct rate."""
    decay_rate = 0.8
    q_learning = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        epsilon_0=1.0,
        epsilon_min=0.01,  # Very low minimum
        epsilon_decay=decay_rate,
        seed=123,
    )

    bounds = (0.0, 5.0)
    market_params: dict[str, float] = {}

    # Track epsilon over rounds
    epsilon_values: List[float] = []

    # First round
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    epsilon_values.append(q_learning.get_current_epsilon())

    # Multiple rounds
    for round_num in range(1, 6):
        mock_result = CournotResult(price=15.0, quantities=[action1], profits=[25.0])

        _ = q_learning.next_action(
            round_num=round_num,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        epsilon_values.append(q_learning.get_current_epsilon())

    # Verify decay rate
    for i in range(1, len(epsilon_values)):
        if epsilon_values[i - 1] > q_learning.epsilon_min:
            expected_epsilon = epsilon_values[i - 1] * decay_rate
            actual_epsilon = epsilon_values[i]
            assert math.isclose(actual_epsilon, expected_epsilon, abs_tol=1e-10), (
                f"Round {i}: expected {expected_epsilon}, got {actual_epsilon}"
            )


def test_epsilon_min_bound() -> None:
    """Test that epsilon never goes below epsilon_min."""
    q_learning = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        epsilon_0=1.0,
        epsilon_min=0.1,  # Higher minimum
        epsilon_decay=0.5,  # Fast decay
        seed=456,
    )

    bounds = (0.0, 5.0)
    market_params: dict[str, float] = {}

    # Run many rounds to ensure epsilon reaches minimum
    epsilon_values: List[float] = []

    # First round
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    epsilon_values.append(q_learning.get_current_epsilon())

    # Run enough rounds to reach minimum
    for round_num in range(1, 20):
        mock_result = CournotResult(price=10.0, quantities=[action1], profits=[20.0])

        _ = q_learning.next_action(
            round_num=round_num,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        epsilon_values.append(q_learning.get_current_epsilon())

    # Check that epsilon reaches and stays at minimum
    min_epsilon_reached = False
    for i, eps in enumerate(epsilon_values):
        assert eps >= q_learning.epsilon_min, (
            f"Epsilon at round {i} ({eps}) should be >= epsilon_min ({q_learning.epsilon_min})"
        )

        if math.isclose(eps, q_learning.epsilon_min, abs_tol=1e-10):
            min_epsilon_reached = True

    assert min_epsilon_reached, (
        f"Epsilon should reach minimum value {q_learning.epsilon_min}"
    )

    # Check that epsilon stays at minimum once reached
    final_epsilons = epsilon_values[-5:]  # Last 5 rounds
    for eps in final_epsilons:
        assert math.isclose(eps, q_learning.epsilon_min, abs_tol=1e-10), (
            f"Epsilon should stay at minimum {q_learning.epsilon_min}, got {eps}"
        )


def test_epsilon_decay_different_rates() -> None:
    """Test epsilon decay with different decay rates."""
    decay_rates = [0.5, 0.8, 0.95, 0.99]

    for decay_rate in decay_rates:
        q_learning = QLearning(
            min_action=0.0,
            max_action=5.0,
            step_size=1.0,
            epsilon_0=1.0,
            epsilon_min=0.01,
            epsilon_decay=decay_rate,
            seed=789,
        )

        bounds = (0.0, 5.0)
        market_params: dict[str, float] = {}

        # Check initial epsilon before any actions
        initial_epsilon = q_learning.get_current_epsilon()
        assert initial_epsilon == 1.0

        # First round
        action1 = q_learning.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        # Run several rounds
        for round_num in range(1, 8):
            mock_result = CournotResult(
                price=12.0, quantities=[action1], profits=[18.0]
            )

            _ = q_learning.next_action(
                round_num=round_num,
                my_history=[mock_result],
                rival_histories=[],
                bounds=bounds,
                market_params=market_params,
            )

        final_epsilon = q_learning.get_current_epsilon()

        # Verify epsilon decreased
        assert final_epsilon < initial_epsilon, (
            f"Decay rate {decay_rate}: epsilon should decrease from {initial_epsilon} to {final_epsilon}"
        )

        # Verify epsilon is above minimum
        assert final_epsilon >= q_learning.epsilon_min, (
            f"Decay rate {decay_rate}: final epsilon {final_epsilon} should be >= epsilon_min {q_learning.epsilon_min}"
        )


def test_epsilon_no_decay_when_zero() -> None:
    """Test that epsilon doesn't decay when already at minimum."""
    q_learning = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        epsilon_0=0.1,  # Start at minimum
        epsilon_min=0.1,  # Same as initial
        epsilon_decay=0.5,
        seed=999,
    )

    bounds = (0.0, 5.0)
    market_params: dict[str, float] = {}

    # First round
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )

    initial_epsilon = q_learning.get_current_epsilon()
    assert math.isclose(initial_epsilon, 0.1, abs_tol=1e-10)

    # Run multiple rounds
    for round_num in range(1, 10):
        mock_result = CournotResult(price=8.0, quantities=[action1], profits=[12.0])

        _ = q_learning.next_action(
            round_num=round_num,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        current_epsilon = q_learning.get_current_epsilon()
        assert math.isclose(current_epsilon, 0.1, abs_tol=1e-10), (
            f"Epsilon should stay at minimum 0.1, got {current_epsilon}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
