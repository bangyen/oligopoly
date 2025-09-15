"""Test Q-learning initialization with ε=1 produces random actions.

This test verifies that when epsilon is set to 1.0 (full exploration),
the Q-learning strategy chooses actions randomly from the action grid.
"""

from collections import Counter
from typing import List

import pytest

from src.sim.cournot import CournotResult
from src.sim.strategies import QLearning


def test_q_init_random_actions():
    """Test that ε=1 produces random actions from grid."""
    # Create Q-learning strategy with epsilon=1 (full exploration)
    q_learning = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=1.0,  # Full exploration
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=42,  # Fixed seed for reproducibility
    )

    # Expected action grid: [0.0, 1.0, 2.0, ..., 10.0]
    expected_actions = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    actual_actions = q_learning.get_action_grid()

    assert (
        actual_actions == expected_actions
    ), f"Expected {expected_actions}, got {actual_actions}"

    # Test multiple actions to verify randomness
    actions_chosen: List[float] = []
    bounds = (0.0, 10.0)
    market_params = {}

    # Generate 100 actions with empty history (first round behavior)
    for _ in range(100):
        action = q_learning.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        actions_chosen.append(action)

    # Verify all actions are from the grid
    for action in actions_chosen:
        assert (
            action in expected_actions
        ), f"Action {action} not in expected grid {expected_actions}"

    # Verify we get a reasonable distribution (not all the same action)
    action_counts = Counter(actions_chosen)
    unique_actions = len(action_counts)

    # With 100 samples and 11 possible actions, we should see multiple different actions
    assert (
        unique_actions > 1
    ), f"Expected multiple different actions, got only {unique_actions} unique actions"

    # Verify epsilon has decayed from 100 iterations (should be less than 1.0)
    current_epsilon = q_learning.get_current_epsilon()
    assert (
        current_epsilon < 1.0
    ), f"Expected epsilon < 1.0 after 100 iterations, got {current_epsilon}"
    assert (
        current_epsilon >= q_learning.epsilon_min
    ), f"Expected epsilon >= epsilon_min ({q_learning.epsilon_min}), got {current_epsilon}"


def test_q_init_with_history():
    """Test that ε=1 produces random actions even with history."""
    # Create Q-learning strategy with epsilon=1
    q_learning = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        epsilon_0=1.0,  # Full exploration
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=123,
    )

    expected_actions = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    actual_actions = q_learning.get_action_grid()
    assert actual_actions == expected_actions

    # Create some mock history
    mock_history = [CournotResult(price=50.0, quantities=[2.0], profits=[100.0])]

    # Generate actions with history
    actions_chosen: List[float] = []
    bounds = (0.0, 5.0)
    market_params = {}

    for _ in range(50):
        action = q_learning.next_action(
            round_num=1,
            my_history=mock_history,
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        actions_chosen.append(action)

    # Verify all actions are from the grid
    for action in actions_chosen:
        assert (
            action in expected_actions
        ), f"Action {action} not in expected grid {expected_actions}"

    # Verify randomness (should see multiple different actions)
    action_counts = Counter(actions_chosen)
    unique_actions = len(action_counts)
    assert (
        unique_actions > 1
    ), f"Expected multiple different actions, got only {unique_actions} unique actions"


def test_q_init_epsilon_verification():
    """Test that epsilon starts at 1.0 and Q-table is empty initially."""
    q_learning = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=2.0,
        epsilon_0=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=456,
    )

    # Verify initial epsilon
    assert q_learning.get_current_epsilon() == 1.0

    # Verify Q-table is initially empty
    q_table = q_learning.get_q_table()
    assert len(q_table) == 0, f"Expected empty Q-table, got {len(q_table)} states"

    # Verify action grid is correct
    expected_actions = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    actual_actions = q_learning.get_action_grid()
    assert actual_actions == expected_actions


if __name__ == "__main__":
    pytest.main([__file__])
