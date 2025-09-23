"""Unit tests for ε-greedy strategy grid bounds functionality.

Tests that chosen actions are always within the discrete action grid
and that Q-values contain no NaN values.
"""

import math

from src.sim.strategies.strategies import EpsilonGreedy


class TestGridBounds:
    """Test action grid bounds and Q-value validity in ε-greedy strategy."""

    def test_actions_always_in_grid(self) -> None:
        """Test that all chosen actions are from the discrete action grid."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=20.0,
            step_size=2.0,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        action_grid = strategy.get_action_grid()

        # Run many rounds and check all actions
        for round_num in range(50):
            action = strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 20.0),
                market_params={},
            )

            # Action must be in the grid
            assert action in action_grid, (
                f"Action {action} not in grid {action_grid} at round {round_num}"
            )

    def test_actions_within_bounds(self) -> None:
        """Test that all actions are within the specified bounds."""
        min_action = 5.0
        max_action = 25.0
        strategy = EpsilonGreedy(
            min_action=min_action,
            max_action=max_action,
            step_size=1.0,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        # Run many rounds
        for round_num in range(50):
            action = strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(min_action, max_action),
                market_params={},
            )

            # Action must be within bounds
            assert min_action <= action <= max_action, (
                f"Action {action} outside bounds [{min_action}, {max_action}] at round {round_num}"
            )

    def test_q_values_no_nans(self) -> None:
        """Test that Q-values never contain NaN values."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=1.0,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        # Run many rounds with various rewards
        for round_num in range(50):
            # Check Q-values before action
            q_values = strategy.get_q_values()
            for i, q_val in enumerate(q_values):
                assert not math.isnan(q_val), f"Q-value {i} is NaN at round {round_num}"

            # Choose action (not used, just to trigger Q-value checks)
            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

            # Check Q-values after action
            q_values = strategy.get_q_values()
            for i, q_val in enumerate(q_values):
                assert not math.isnan(q_val), (
                    f"Q-value {i} is NaN after action at round {round_num}"
                )

    def test_q_values_finite(self) -> None:
        """Test that Q-values are always finite (not infinite)."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=1.0,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        # Run rounds with extreme rewards
        for round_num in range(20):
            # Check Q-values are finite
            q_values = strategy.get_q_values()
            for i, q_val in enumerate(q_values):
                assert math.isfinite(q_val), (
                    f"Q-value {i} is not finite at round {round_num}: {q_val}"
                )

            # Choose action (not used, just to trigger Q-value checks)
            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

    def test_action_grid_properties(self) -> None:
        """Test that action grid has correct properties."""
        min_action = 2.0
        max_action = 8.0
        step_size = 0.5

        strategy = EpsilonGreedy(
            min_action=min_action,
            max_action=max_action,
            step_size=step_size,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        action_grid = strategy.get_action_grid()

        # Grid should start at min_action
        assert action_grid[0] == min_action

        # Grid should end at or near max_action
        assert action_grid[-1] <= max_action
        assert action_grid[-1] >= max_action - step_size

        # All actions should be spaced by step_size
        for i in range(1, len(action_grid)):
            diff = action_grid[i] - action_grid[i - 1]
            assert abs(diff - step_size) < 1e-10, (
                f"Step size incorrect: {diff} != {step_size}"
            )

        # All actions should be within bounds
        for action in action_grid:
            assert min_action <= action <= max_action

    def test_different_grid_sizes(self) -> None:
        """Test grids with different sizes and step sizes."""
        test_cases = [
            (0.0, 10.0, 1.0),  # Small grid
            (0.0, 100.0, 5.0),  # Large grid
            (0.0, 1.0, 0.1),  # Fine grid
            (10.0, 20.0, 0.5),  # Offset grid
        ]

        for min_action, max_action, step_size in test_cases:
            strategy = EpsilonGreedy(
                min_action=min_action,
                max_action=max_action,
                step_size=step_size,
                epsilon_0=0.1,
                epsilon_min=0.01,
                seed=42,
            )

            action_grid = strategy.get_action_grid()

            # Grid should have reasonable size
            assert len(action_grid) > 0, "Grid should not be empty"
            assert len(action_grid) <= (max_action - min_action) / step_size + 2, (
                "Grid should not be too large"
            )

            # All actions should be valid
            for action in action_grid:
                assert min_action <= action <= max_action
                assert not math.isnan(action)
                assert math.isfinite(action)

    def test_bounds_clamping(self) -> None:
        """Test that actions are properly clamped to bounds."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        # Test with bounds that are different from grid bounds
        tight_bounds = (1.0, 9.0)  # Tighter than grid bounds

        for round_num in range(20):
            action = strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=tight_bounds,
                market_params={},
            )

            # Action should be clamped to tight bounds
            assert tight_bounds[0] <= action <= tight_bounds[1], (
                f"Action {action} not clamped to bounds {tight_bounds}"
            )

    def test_q_values_initialization(self) -> None:
        """Test that Q-values are properly initialized."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=1.0,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        q_values = strategy.get_q_values()

        # Q-values should be initialized to 0.0
        for i, q_val in enumerate(q_values):
            assert q_val == 0.0, f"Q-value {i} not initialized to 0.0: {q_val}"
            assert not math.isnan(q_val)
            assert math.isfinite(q_val)

        # Q-values should have same length as action grid
        assert len(q_values) == len(strategy.get_action_grid())

    def test_action_index_mapping(self) -> None:
        """Test that action-to-index mapping works correctly."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=6.0,
            step_size=2.0,
            epsilon_0=0.1,
            epsilon_min=0.01,
            seed=42,
        )

        action_grid = strategy.get_action_grid()

        # Test mapping for each action in grid
        for i, action in enumerate(action_grid):
            mapped_index = strategy._get_action_index(action)
            assert mapped_index == i, (
                f"Action {action} mapped to index {mapped_index}, expected {i}"
            )

        # Test mapping for actions not exactly in grid
        test_actions = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        for action in test_actions:
            mapped_index = strategy._get_action_index(action)
            # Should map to closest action
            closest_action = action_grid[mapped_index]
            distances = [abs(action - grid_action) for grid_action in action_grid]
            min_distance = min(distances)
            actual_distance = abs(action - closest_action)

            assert actual_distance <= min_distance + 1e-10, (
                f"Action {action} not mapped to closest grid action"
            )
