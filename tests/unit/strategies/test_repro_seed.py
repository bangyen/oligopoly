"""Test reproducibility with fixed seed in Q-learning strategy.

This test verifies that when using a fixed seed, the Q-learning strategy
produces identical sequences of actions across multiple runs.
"""

from typing import List

import pytest

from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import QLearning


def test_repro_seed_identical_sequences():
    """Test that fixed seed produces identical action sequences."""
    seed = 42

    # Create two identical strategies with same seed
    q_learning1 = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=0.5,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=seed,
    )

    q_learning2 = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=0.5,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=seed,
    )

    bounds = (0.0, 10.0)
    market_params = {}

    # Generate sequences from both strategies
    sequence1: List[float] = []
    sequence2: List[float] = []

    # First action (round 0)
    action1_1 = q_learning1.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    sequence1.append(action1_1)

    action1_2 = q_learning2.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    sequence2.append(action1_2)

    # Generate 50 more actions
    for step in range(1, 51):
        # Create mock result
        mock_result = CournotResult(
            price=15.0 + (step % 3) * 2.0,  # Vary price
            quantities=[action1_1],
            profits=[20.0 + (step % 5) * 3.0],  # Vary profit
        )

        # Get action from first strategy
        action2_1 = q_learning1.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        sequence1.append(action2_1)

        # Get action from second strategy
        action2_2 = q_learning2.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        sequence2.append(action2_2)

    # Verify sequences are identical
    assert len(sequence1) == len(
        sequence2
    ), f"Sequences should have same length: {len(sequence1)} vs {len(sequence2)}"

    for i, (action1, action2) in enumerate(zip(sequence1, sequence2)):
        assert action1 == action2, f"Action {i} differs: {action1} vs {action2}"

    # Verify epsilon values are also identical
    assert (
        q_learning1.get_current_epsilon() == q_learning2.get_current_epsilon()
    ), f"Epsilon values differ: {q_learning1.get_current_epsilon()} vs {q_learning2.get_current_epsilon()}"


def test_repro_seed_different_seeds():
    """Test that different seeds produce different sequences."""
    # Create strategies with different seeds
    q_learning_seed1 = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=0.3,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=123,
    )

    q_learning_seed2 = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=0.3,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=456,
    )

    bounds = (0.0, 10.0)
    market_params = {}

    # Generate sequences from both strategies
    sequence1: List[float] = []
    sequence2: List[float] = []

    # First action
    action1_1 = q_learning_seed1.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    sequence1.append(action1_1)

    action1_2 = q_learning_seed2.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    sequence2.append(action1_2)

    # Generate more actions
    for step in range(1, 30):
        mock_result = CournotResult(price=12.0, quantities=[action1_1], profits=[18.0])

        action2_1 = q_learning_seed1.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        sequence1.append(action2_1)

        action2_2 = q_learning_seed2.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        sequence2.append(action2_2)

    # Verify sequences are different
    assert len(sequence1) == len(sequence2)

    # Check if sequences differ (they should with different seeds)
    sequences_differ = any(
        action1 != action2 for action1, action2 in zip(sequence1, sequence2)
    )

    # With different seeds and exploration, sequences should differ
    # (though it's theoretically possible they could be the same by chance)
    if not sequences_differ:
        # If sequences are identical, verify that epsilon values are different
        # or that we're in a deterministic phase
        epsilon1 = q_learning_seed1.get_current_epsilon()
        epsilon2 = q_learning_seed2.get_current_epsilon()

        # If epsilons are very low, we might be in exploitation phase
        if epsilon1 < 0.1 and epsilon2 < 0.1:
            # In exploitation phase, sequences might be identical if Q-values converge
            pass  # This is acceptable
        else:
            # In exploration phase, sequences should differ
            assert sequences_differ, "Sequences should differ with different seeds"


def test_repro_seed_multiple_runs():
    """Test reproducibility across multiple independent runs."""
    seed = 789

    # Run the same strategy multiple times
    all_sequences: List[List[float]] = []

    for run in range(5):  # 5 independent runs
        q_learning = QLearning(
            min_action=0.0,
            max_action=5.0,
            step_size=1.0,
            epsilon_0=0.4,
            epsilon_min=0.01,
            epsilon_decay=0.98,
            seed=seed,
        )

        bounds = (0.0, 5.0)
        market_params = {}

        sequence: List[float] = []

        # First action
        action1 = q_learning.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        sequence.append(action1)

        # Generate more actions
        for step in range(1, 20):
            mock_result = CournotResult(price=8.0, quantities=[action1], profits=[12.0])

            action2 = q_learning.next_action(
                round_num=step,
                my_history=[mock_result],
                rival_histories=[],
                bounds=bounds,
                market_params=market_params,
            )
            sequence.append(action2)

        all_sequences.append(sequence)

    # Verify all sequences are identical
    assert len(all_sequences) == 5

    reference_sequence = all_sequences[0]
    for i, sequence in enumerate(all_sequences[1:], 1):
        assert (
            sequence == reference_sequence
        ), f"Run {i} differs from reference sequence"


def test_repro_seed_q_table_consistency():
    """Test that Q-tables are identical with same seed."""
    seed = 555

    # Create two strategies with same seed
    q_learning1 = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=2.0,
        epsilon_0=0.2,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=seed,
    )

    q_learning2 = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=2.0,
        epsilon_0=0.2,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=seed,
    )

    bounds = (0.0, 10.0)
    market_params = {}

    # Run both strategies through same sequence
    for step in range(20):
        mock_result = CournotResult(
            price=10.0 + (step % 3) * 1.0,
            quantities=[2.0],
            profits=[15.0 + (step % 4) * 2.0],
        )

        action1 = q_learning1.next_action(
            round_num=step,
            my_history=[mock_result] if step > 0 else [],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        action2 = q_learning2.next_action(
            round_num=step,
            my_history=[mock_result] if step > 0 else [],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )

        # Actions should be identical
        assert action1 == action2, f"Step {step}: actions differ {action1} vs {action2}"

    # Q-tables should be identical
    q_table1 = q_learning1.get_q_table()
    q_table2 = q_learning2.get_q_table()

    assert q_table1 == q_table2, "Q-tables should be identical with same seed"

    # Epsilon values should be identical
    assert q_learning1.get_current_epsilon() == q_learning2.get_current_epsilon()


def test_repro_seed_no_seed_vs_seed():
    """Test that no seed produces different results than fixed seed."""
    # Strategy with no seed (None)
    q_learning_no_seed = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=0.3,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=None,
    )

    # Strategy with fixed seed
    q_learning_seed = QLearning(
        min_action=0.0,
        max_action=10.0,
        step_size=1.0,
        epsilon_0=0.3,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        seed=999,
    )

    bounds = (0.0, 10.0)
    market_params = {}

    # Generate sequences
    sequence_no_seed: List[float] = []
    sequence_seed: List[float] = []

    # First actions
    action1_no_seed = q_learning_no_seed.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    sequence_no_seed.append(action1_no_seed)

    action1_seed = q_learning_seed.next_action(
        round_num=0,
        my_history=[],
        rival_histories=[],
        bounds=bounds,
        market_params=market_params,
    )
    sequence_seed.append(action1_seed)

    # Generate more actions
    for step in range(1, 15):
        mock_result = CournotResult(
            price=7.0, quantities=[action1_no_seed], profits=[11.0]
        )

        action2_no_seed = q_learning_no_seed.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        sequence_no_seed.append(action2_no_seed)

        action2_seed = q_learning_seed.next_action(
            round_num=step,
            my_history=[mock_result],
            rival_histories=[],
            bounds=bounds,
            market_params=market_params,
        )
        sequence_seed.append(action2_seed)

    # Sequences should be different (very high probability)
    sequences_identical = all(
        action1 == action2 for action1, action2 in zip(sequence_no_seed, sequence_seed)
    )

    # It's theoretically possible but extremely unlikely that sequences are identical
    if sequences_identical:
        # If they happen to be identical, verify it's not due to deterministic behavior
        epsilon_no_seed = q_learning_no_seed.get_current_epsilon()
        epsilon_seed = q_learning_seed.get_current_epsilon()

        # If both are in exploitation phase, they might converge to same actions
        if epsilon_no_seed < 0.05 and epsilon_seed < 0.05:
            pass  # Acceptable if both are exploiting
        else:
            # In exploration phase, sequences should differ
            assert (
                not sequences_identical
            ), "Sequences should differ between no seed and fixed seed"


if __name__ == "__main__":
    pytest.main([__file__])
