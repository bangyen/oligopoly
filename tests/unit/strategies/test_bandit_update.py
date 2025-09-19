"""Unit tests for ε-greedy strategy bandit update functionality.

Tests that Q-values are properly updated based on immediate rewards
and that positive rewards increase Q-values within tolerance.
"""

import math

from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import EpsilonGreedy


class TestBanditUpdate:
    """Test Q-value updates in ε-greedy strategy."""

    def test_positive_reward_increases_q_value(self) -> None:
        """Test that positive reward increases Q-value for selected action."""
        # Create ε-greedy strategy with small grid for testing
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            learning_rate=0.1,
            epsilon_0=0.0,  # No exploration for deterministic testing
            epsilon_min=0.0,  # Must be <= epsilon_0
            seed=42,
        )

        # Get initial Q-values
        initial_q_values = strategy.get_q_values()

        # Choose an action (index 2 = action 4.0)
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Verify action is from grid
        assert action in strategy.get_action_grid()

        # Simulate positive reward by calling next_action again
        # This will trigger Q-value update
        strategy.next_action(
            round_num=1,
            my_history=[CournotResult(price=50.0, quantities=[action], profits=[10.0])],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Get updated Q-values
        updated_q_values = strategy.get_q_values()

        # Find the action index that was chosen
        action_index = strategy.get_action_grid().index(action)

        # Q-value for chosen action should increase
        assert updated_q_values[action_index] > initial_q_values[action_index]

        # Verify the increase is within expected tolerance
        expected_increase = 0.1 * (
            10.0 - initial_q_values[action_index]
        )  # learning_rate * (reward - old_q)
        actual_increase = (
            updated_q_values[action_index] - initial_q_values[action_index]
        )

        assert math.isclose(actual_increase, expected_increase, abs_tol=1e-6)

    def test_negative_reward_decreases_q_value(self) -> None:
        """Test that negative reward decreases Q-value for selected action."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            learning_rate=0.1,
            epsilon_0=0.0,
            epsilon_min=0.0,
            seed=42,
        )

        # Start with positive Q-value
        strategy.q_values = [5.0] * len(strategy.q_values)

        # Choose an action
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Simulate negative reward
        strategy.next_action(
            round_num=1,
            my_history=[CournotResult(price=50.0, quantities=[action], profits=[-5.0])],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Get updated Q-values
        updated_q_values = strategy.get_q_values()

        # Find the action index that was chosen
        action_index = strategy.get_action_grid().index(action)

        # Q-value for chosen action should decrease
        assert updated_q_values[action_index] < 5.0

        # Verify the decrease is within expected tolerance
        expected_decrease = 0.1 * (-5.0 - 5.0)  # learning_rate * (reward - old_q)
        actual_decrease = updated_q_values[action_index] - 5.0

        assert math.isclose(actual_decrease, expected_decrease, abs_tol=1e-6)

    def test_zero_reward_decreases_q_value(self) -> None:
        """Test that zero reward decreases Q-value towards zero for selected action."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            learning_rate=0.1,
            epsilon_0=0.0,
            epsilon_min=0.0,
            seed=42,
        )

        # Set initial Q-value
        initial_q_value = 3.0
        strategy.q_values = [initial_q_value] * len(strategy.q_values)

        # Choose an action
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Simulate zero reward
        strategy.next_action(
            round_num=1,
            my_history=[CournotResult(price=50.0, quantities=[action], profits=[0.0])],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Get updated Q-values
        updated_q_values = strategy.get_q_values()

        # Find the action index that was chosen
        action_index = strategy.get_action_grid().index(action)

        # Q-value for chosen action should decrease towards zero reward
        # Q_new = Q_old + α * (reward - Q_old) = 3.0 + 0.1 * (0.0 - 3.0) = 2.7
        expected_q_value = initial_q_value + 0.1 * (0.0 - initial_q_value)
        assert math.isclose(
            updated_q_values[action_index], expected_q_value, abs_tol=1e-6
        )

    def test_multiple_updates_cumulative(self) -> None:
        """Test that multiple updates accumulate correctly."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            learning_rate=0.1,
            epsilon_0=0.0,
            epsilon_min=0.0,
            seed=42,
        )

        # Choose same action multiple times with same reward
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Apply same reward multiple times
        for round_num in range(1, 6):  # 5 updates
            strategy.next_action(
                round_num=round_num,
                my_history=[
                    CournotResult(price=50.0, quantities=[action], profits=[2.0])
                ],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

        # Get final Q-values
        final_q_values = strategy.get_q_values()
        action_index = strategy.get_action_grid().index(action)

        # Q-value should converge towards the reward value (2.0)
        # With learning rate 0.1, after 5 updates starting from 0:
        # Q = 0 + 0.1*(2-0) + 0.1*(2-0.2) + 0.1*(2-0.38) + 0.1*(2-0.542) + 0.1*(2-0.6878)
        # This should be approximately 1.0 (converging towards 2.0)
        assert final_q_values[action_index] > 0.5
        assert final_q_values[action_index] < 2.0

        # Verify it's converging towards the reward
        assert abs(final_q_values[action_index] - 2.0) < abs(0.0 - 2.0)

    def test_different_actions_independent_updates(self) -> None:
        """Test that updates to different actions are independent."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            learning_rate=0.1,
            epsilon_0=0.0,
            epsilon_min=0.0,
            seed=42,
        )

        # Choose first action
        action1 = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Update with positive reward
        strategy.next_action(
            round_num=1,
            my_history=[CournotResult(price=50.0, quantities=[action1], profits=[5.0])],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Q-values after first update (not used, just for completeness)

        # Choose different action (force by manipulating internal state)
        # We'll manually set the previous action index to a different value
        action_grid = strategy.get_action_grid()
        action1_index = action_grid.index(action1)
        action2_index = (action1_index + 1) % len(action_grid)
        action2 = action_grid[action2_index]

        # Manually set previous action index to simulate choosing different action
        strategy._previous_action_index = action2_index

        # Update with different reward
        strategy.next_action(
            round_num=2,
            my_history=[
                CournotResult(price=50.0, quantities=[action1], profits=[5.0]),
                CournotResult(price=50.0, quantities=[action2], profits=[-2.0]),
            ],
            rival_histories=[],
            bounds=(0.0, 10.0),
            market_params={},
        )

        # Get final Q-values
        final_q_values = strategy.get_q_values()

        # First action should still have positive Q-value
        assert final_q_values[action1_index] > 0

        # Second action should have negative Q-value
        assert final_q_values[action2_index] < 0

        # Actions should be independent
        assert final_q_values[action1_index] != final_q_values[action2_index]
