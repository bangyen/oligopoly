"""Test action bounds in Q-learning strategy.

This test verifies that over 200 steps, all actions chosen by the Q-learning
strategy stay within the specified action grid bounds.
"""

from typing import List

import pytest

from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import QLearning


def test_action_bounds_200_steps() -> None:
    """Test that all actions over 200 steps stay within grid bounds."""
    # Create Q-learning strategy with specific bounds
    min_action = 5.0
    max_action = 25.0
    step_size = 2.0

    q_learning = QLearning(
        min_action=min_action,
        max_action=max_action,
        step_size=step_size,
        epsilon_0=0.5,  # Some exploration
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=42,
    )

    # Expected action grid: [5.0, 7.0, 9.0, ..., 25.0]
    expected_grid = [5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0]
    actual_grid = q_learning.get_action_grid()
    assert actual_grid == expected_grid

    bounds = (min_action, max_action)
    market_params: dict[str, float] = {}

    # Track all actions over 200 steps
    actions_chosen: List[float] = []

    # First action (round 0)
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    actions_chosen.append(action1)

    # Run 200 steps
    for step in range(1, 200):
        # Create mock result with varying profits
        profit = 20.0 + (step % 10) * 5.0  # Vary profit between 20-65
        mock_result = CournotResult(
            price=15.0 + (step % 5) * 2.0,  # Vary price between 15-23
            quantities=[action1],
            profits=[profit],
        )

        # Choose next action
        action2 = q_learning.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        actions_chosen.append(action2)

    # Verify all actions are within bounds
    assert (
        len(actions_chosen) == 200
    ), f"Expected 200 actions, got {len(actions_chosen)}"

    for i, action in enumerate(actions_chosen):
        assert (
            min_action <= action <= max_action
        ), f"Action {i} ({action}) is outside bounds [{min_action}, {max_action}]"

    # Verify all actions are from the grid
    unique_actions = set(actions_chosen)
    expected_grid_set = set(expected_grid)

    for action in unique_actions:
        assert (
            action in expected_grid_set
        ), f"Action {action} is not in expected grid {expected_grid}"

    # Verify we used multiple different actions (exploration worked)
    assert (
        len(unique_actions) > 1
    ), f"Should use multiple different actions, got only {len(unique_actions)} unique actions"


def test_action_bounds_different_grids() -> None:
    """Test action bounds with different grid configurations."""
    test_configs = [
        {"min": 0.0, "max": 10.0, "step": 1.0},  # [0,1,2,...,10]
        {"min": 1.0, "max": 5.0, "step": 0.5},  # [1.0,1.5,2.0,...,5.0]
        {"min": 10.0, "max": 20.0, "step": 2.5},  # [10.0,12.5,15.0,17.5,20.0]
        {"min": 0.0, "max": 1.0, "step": 0.1},  # [0.0,0.1,0.2,...,1.0]
    ]

    for config in test_configs:
        min_action = config["min"]
        max_action = config["max"]
        step_size = config["step"]

        q_learning = QLearning(
            min_action=min_action,
            max_action=max_action,
            step_size=step_size,
            epsilon_0=0.3,
            epsilon_min=0.01,
            epsilon_decay=0.99,
            seed=123,
        )

        bounds = (min_action, max_action)
        market_params: dict[str, float] = {}

        # Generate 50 actions
        actions_chosen: List[float] = []

        # First action
        action1 = q_learning.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        actions_chosen.append(action1)

        # Generate more actions
        for step in range(1, 50):
            mock_result = CournotResult(
                price=10.0, quantities=[action1], profits=[15.0]
            )

            action2 = q_learning.next_action(
                round_num=step,
                my_history=[mock_result],
                rival_histories=[],
                bounds=bounds,
                market_params=market_params,
            )
            actions_chosen.append(action2)

        # Verify bounds
        for i, action in enumerate(actions_chosen):
            assert (
                min_action <= action <= max_action
            ), f"Config {config}: Action {i} ({action}) outside bounds [{min_action}, {max_action}]"

        # Verify grid membership
        expected_grid = q_learning.get_action_grid()
        for action in actions_chosen:
            assert (
                action in expected_grid
            ), f"Config {config}: Action {action} not in grid {expected_grid}"


def test_action_bounds_edge_cases() -> None:
    """Test action bounds with edge cases."""
    # Test with very small step size
    q_learning_small_step = QLearning(
        min_action=0.0,
        max_action=1.0,
        step_size=0.01,  # Very small step
        epsilon_0=0.2,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=456,
    )

    bounds = (0.0, 1.0)
    market_params: dict[str, float] = {}

    # Generate 100 actions
    actions_chosen: List[float] = []

    # First action
    action1 = q_learning_small_step.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    actions_chosen.append(action1)

    for step in range(1, 100):
        mock_result = CournotResult(price=5.0, quantities=[action1], profits=[10.0])

        action2 = q_learning_small_step.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        actions_chosen.append(action2)

    # Verify bounds
    for action in actions_chosen:
        assert 0.0 <= action <= 1.0, f"Action {action} outside bounds [0.0, 1.0]"

    # Test with large step size
    q_learning_large_step = QLearning(
        min_action=0.0,
        max_action=100.0,
        step_size=25.0,  # Large step
        epsilon_0=0.5,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=789,
    )

    bounds = (0.0, 100.0)

    # Generate 50 actions
    actions_chosen_large: List[float] = []

    # First action
    action1 = q_learning_large_step.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    actions_chosen_large.append(action1)

    for step in range(1, 50):
        mock_result = CournotResult(price=50.0, quantities=[action1], profits=[75.0])

        action2 = q_learning_large_step.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        actions_chosen_large.append(action2)

    # Verify bounds
    for action in actions_chosen_large:
        assert 0.0 <= action <= 100.0, f"Action {action} outside bounds [0.0, 100.0]"

    # Expected grid: [0.0, 25.0, 50.0, 75.0, 100.0]
    expected_large_grid = [0.0, 25.0, 50.0, 75.0, 100.0]
    actual_large_grid = q_learning_large_step.get_action_grid()
    assert actual_large_grid == expected_large_grid

    # Verify all actions are from grid
    for action in actions_chosen_large:
        assert (
            action in expected_large_grid
        ), f"Action {action} not in grid {expected_large_grid}"


def test_action_bounds_clamping() -> None:
    """Test that actions are properly clamped to bounds."""
    # Create strategy with internal bounds different from external bounds
    q_learning = QLearning(
        min_action=10.0,  # Internal min
        max_action=20.0,  # Internal max
        step_size=1.0,
        epsilon_0=0.1,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=111,
    )

    # Use external bounds that are wider than internal bounds
    external_bounds = (5.0, 25.0)
    market_params: dict[str, float] = {}

    # Generate actions
    actions_chosen: List[float] = []

    # First action
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=external_bounds,
        market_params=market_params,
    )
    actions_chosen.append(action1)

    for step in range(1, 50):
        mock_result = CournotResult(price=15.0, quantities=[action1], profits=[25.0])

        action2 = q_learning.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=external_bounds,
            market_params=market_params,
        )
        actions_chosen.append(action2)

    # All actions should be within external bounds
    for action in actions_chosen:
        assert (
            5.0 <= action <= 25.0
        ), f"Action {action} outside external bounds [5.0, 25.0]"

    # But actions should still be from the internal grid
    expected_grid = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    for action in actions_chosen:
        assert (
            action in expected_grid
        ), f"Action {action} not in internal grid {expected_grid}"


if __name__ == "__main__":
    pytest.main([__file__])
