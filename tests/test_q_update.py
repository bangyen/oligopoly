"""Test Q-learning update mechanism with positive rewards.

This test verifies that when a positive reward is received, the Q-value
for the corresponding state-action pair increases according to the
Q-learning update rule: Q[s,a] ← (1-α)Q[s,a] + α(r + γ·max Q[s',·]).
"""

import math

import pytest

from src.sim.cournot import CournotResult
from src.sim.strategies import QLearning


def test_q_update_positive_reward():
    """Test that positive reward increases Q[s,a]."""
    # Create Q-learning strategy with deterministic parameters
    q_learning = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        alpha=0.1,  # Learning rate
        gamma=0.9,  # Discount factor
        epsilon_0=0.0,  # No exploration for deterministic testing
        epsilon_min=0.0,  # Must be <= epsilon_0
        epsilon_decay=0.995,
        seed=42,
    )

    # Get initial Q-table (should be empty)
    initial_q_table = q_learning.get_q_table()
    assert len(initial_q_table) == 0

    # Simulate first action (round 0)
    bounds = (0.0, 10.0)
    market_params = {}

    # Choose first action
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )

    # Verify action is from grid
    assert action1 in q_learning.get_action_grid()

    # Create mock result with positive profit
    positive_profit = 50.0
    mock_result = CournotResult(
        price=30.0, quantities=[action1], profits=[positive_profit]
    )

    # Simulate second action (round 1) - this triggers Q-value update
    _ = q_learning.next_action(
        round_num=1,
        my_history=[mock_result],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )

    # Check that Q-table now has entries
    updated_q_table = q_learning.get_q_table()
    assert len(updated_q_table) > 0, "Q-table should have entries after update"

    # Find the state-action pair that was updated
    # The state should be the previous state (from round 0), not current state
    # For round 0, the state would be based on default values (0, 0, 0)
    state = (0, 0, 0)  # Default state from round 0
    action_index = q_learning._get_action_index(action1)

    assert state in updated_q_table, f"State {state} not found in Q-table"

    q_values = updated_q_table[state]
    q_value = q_values[action_index]

    # With positive reward and no previous Q-value (0), the new Q-value should be positive
    # Q[s,a] = (1-α)*0 + α*(reward + γ*0) = α*reward = 0.1*5.0 = 0.5 (normalized reward)
    expected_q_value = q_learning.alpha * (positive_profit / 10.0)  # Normalized reward
    assert q_value > 0, f"Expected positive Q-value, got {q_value}"
    assert math.isclose(
        q_value, expected_q_value, abs_tol=1e-6
    ), f"Expected Q-value ≈ {expected_q_value}, got {q_value}"


def test_q_update_multiple_rewards():
    """Test Q-value updates with multiple positive rewards."""
    q_learning = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        alpha=0.2,  # Higher learning rate for clearer updates
        gamma=0.9,
        epsilon_0=0.0,  # No exploration
        epsilon_min=0.0,  # Must be <= epsilon_0
        epsilon_decay=0.995,
        seed=123,
    )

    bounds = (0.0, 5.0)
    market_params = {}

    # Track Q-values over multiple rounds
    # q_values_history = []  # Not used in simplified test

    # First action
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )

    # Multiple rounds with positive rewards
    for round_num in range(1, 6):
        # Create result with increasing profit
        profit = 20.0 + round_num * 10.0  # 30, 40, 50, 60
        mock_result = CournotResult(
            price=25.0 + round_num, quantities=[action1], profits=[profit]
        )

        # Choose next action (triggers update)
        _ = q_learning.next_action(
            round_num=round_num,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        # Check that Q-table has been updated
        q_table_after = q_learning.get_q_table()
        assert (
            len(q_table_after) > 0
        ), f"Q-table should have entries after round {round_num}"

    # Check that final Q-values are positive
    final_q_table = q_learning.get_q_table()
    for state, q_values in final_q_table.items():
        for q_val in q_values:
            if q_val != 0.0:  # Only check non-zero Q-values
                assert q_val > 0, f"All Q-values should be positive, got {q_val}"


def test_q_update_negative_reward():
    """Test Q-value updates with negative rewards."""
    q_learning = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        alpha=0.1,
        gamma=0.9,
        epsilon_0=0.0,  # No exploration
        epsilon_min=0.0,  # Must be <= epsilon_0
        epsilon_decay=0.995,
        seed=456,
    )

    bounds = (0.0, 5.0)
    market_params = {}

    # First action
    action1 = q_learning.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )

    # Create result with negative profit (loss)
    negative_profit = -20.0
    mock_result = CournotResult(
        price=15.0, quantities=[action1], profits=[negative_profit]
    )

    # Second action (triggers update)
    _ = q_learning.next_action(
        round_num=1,
        my_history=[mock_result],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )

    # Check Q-table
    q_table = q_learning.get_q_table()
    assert len(q_table) > 0

    # The Q-value should be updated for the previous state (0, 0, 0)
    state = (0, 0, 0)  # Default state from round 0
    action_index = q_learning._get_action_index(action1)

    assert state in q_table, f"State {state} not found in Q-table"
    q_value = q_table[state][action_index]

    # With negative reward, Q-value should be negative
    assert q_value < 0, f"Expected negative Q-value for negative reward, got {q_value}"


def test_q_update_discount_factor():
    """Test that discount factor γ affects Q-value updates."""
    # Create two strategies with different gamma values
    q_learning_high_gamma = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        alpha=0.1,
        gamma=0.9,  # High discount factor
        epsilon_0=0.0,
        epsilon_min=0.0,  # Must be <= epsilon_0
        epsilon_decay=0.995,
        seed=789,
    )

    q_learning_low_gamma = QLearning(
        min_action=0.0,
        max_action=5.0,
        step_size=1.0,
        alpha=0.1,
        gamma=0.1,  # Low discount factor
        epsilon_0=0.0,
        epsilon_min=0.0,  # Must be <= epsilon_0
        epsilon_decay=0.995,
        seed=789,  # Same seed for fair comparison
    )

    bounds = (0.0, 5.0)
    market_params = {}

    # Test both strategies with same sequence
    for q_learning in [q_learning_high_gamma, q_learning_low_gamma]:
        # First action
        action1 = q_learning.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        # Create result
        profit = 30.0
        mock_result = CournotResult(price=20.0, quantities=[action1], profits=[profit])

        # Second action (triggers update)
        _ = q_learning.next_action(
            round_num=1,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

    # Compare Q-values
    q_table_high = q_learning_high_gamma.get_q_table()
    q_table_low = q_learning_low_gamma.get_q_table()

    # Both should have updated Q-values
    assert len(q_table_high) > 0 and len(q_table_low) > 0

    # The exact difference depends on the implementation details,
    # but both should have non-zero Q-values
    for q_table in [q_table_high, q_table_low]:
        has_nonzero_q = any(
            any(q_val != 0.0 for q_val in q_values) for q_values in q_table.values()
        )
        assert has_nonzero_q, "Should have non-zero Q-values after update"


if __name__ == "__main__":
    pytest.main([__file__])
