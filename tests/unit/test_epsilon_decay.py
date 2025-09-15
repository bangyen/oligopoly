"""Unit tests for ε-greedy strategy epsilon decay functionality.

Tests that the exploration rate ε decays monotonically from ε₀ to ε_min
over rounds of the simulation.
"""

from sim.strategies import EpsilonGreedy


class TestEpsilonDecay:
    """Test ε decay behavior in ε-greedy strategy."""

    def test_epsilon_starts_at_initial_value(self) -> None:
        """Test that ε starts at the specified initial value."""
        epsilon_0 = 0.3
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=epsilon_0,
            epsilon_min=0.01,
            decay_rate=0.95,
            seed=42,
        )

        assert strategy.get_current_epsilon() == epsilon_0

    def test_epsilon_decays_monotonically(self) -> None:
        """Test that ε decreases monotonically over rounds."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=0.2,
            epsilon_min=0.01,
            decay_rate=0.9,
            seed=42,
        )

        epsilon_values = []

        # Run multiple rounds and track epsilon values
        for round_num in range(10):
            epsilon_values.append(strategy.get_current_epsilon())

            # Call next_action to trigger epsilon decay
            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

        # Check monotonicity: each value should be <= previous value
        for i in range(1, len(epsilon_values)):
            assert (
                epsilon_values[i] <= epsilon_values[i - 1]
            ), f"Epsilon increased from {epsilon_values[i-1]} to {epsilon_values[i]} at round {i}"

    def test_epsilon_reaches_minimum(self) -> None:
        """Test that ε eventually reaches the minimum value."""
        epsilon_min = 0.05
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=0.2,
            epsilon_min=epsilon_min,
            decay_rate=0.8,  # Faster decay for testing
            seed=42,
        )

        # Run enough rounds to reach minimum
        for round_num in range(20):
            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

        # Epsilon should be at or near minimum
        assert strategy.get_current_epsilon() == epsilon_min

    def test_epsilon_never_below_minimum(self) -> None:
        """Test that ε never goes below the minimum value."""
        epsilon_min = 0.02
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=0.1,
            epsilon_min=epsilon_min,
            decay_rate=0.5,  # Very fast decay
            seed=42,
        )

        # Run many rounds
        for round_num in range(50):
            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

            # Epsilon should never go below minimum
            assert strategy.get_current_epsilon() >= epsilon_min

    def test_epsilon_decay_rate_correct(self) -> None:
        """Test that ε decays by the correct rate each round."""
        decay_rate = 0.9
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=0.2,
            epsilon_min=0.01,
            decay_rate=decay_rate,
            seed=42,
        )

        # Track epsilon values
        epsilon_values = [strategy.get_current_epsilon()]

        # Run a few rounds
        for round_num in range(5):
            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )
            epsilon_values.append(strategy.get_current_epsilon())

        # Check decay rate
        for i in range(1, len(epsilon_values)):
            if epsilon_values[i - 1] > strategy.epsilon_min:
                expected = epsilon_values[i - 1] * decay_rate
                # Allow for floating point precision
                assert (
                    abs(epsilon_values[i] - expected) < 1e-10
                ), f"Expected {expected}, got {epsilon_values[i]}"

    def test_epsilon_minimum_equals_initial_no_decay(self) -> None:
        """Test that when ε_min = ε₀, epsilon doesn't decay."""
        epsilon_0 = 0.15
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=epsilon_0,
            epsilon_min=epsilon_0,  # Same as initial
            decay_rate=0.5,
            seed=42,
        )

        # Run multiple rounds
        for round_num in range(10):
            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )

            # Epsilon should remain at initial value
            assert strategy.get_current_epsilon() == epsilon_0

    def test_epsilon_decay_with_different_rates(self) -> None:
        """Test epsilon decay with different decay rates."""
        decay_rates = [0.5, 0.8, 0.95, 0.99]

        for decay_rate in decay_rates:
            strategy = EpsilonGreedy(
                min_action=0.0,
                max_action=10.0,
                step_size=2.0,
                epsilon_0=0.2,
                epsilon_min=0.01,
                decay_rate=decay_rate,
                seed=42,
            )

            initial_epsilon = strategy.get_current_epsilon()

            # Run 5 rounds
            for round_num in range(5):
                strategy.next_action(
                    round_num=round_num,
                    my_history=[],
                    rival_histories=[],
                    bounds=(0.0, 10.0),
                    market_params={},
                )

            final_epsilon = strategy.get_current_epsilon()

            # Faster decay rates should result in lower final epsilon
            # (assuming we haven't hit the minimum)
            if final_epsilon > strategy.epsilon_min:
                expected_final = initial_epsilon * (decay_rate**5)
                assert (
                    abs(final_epsilon - expected_final) < 1e-10
                ), f"Decay rate {decay_rate}: expected {expected_final}, got {final_epsilon}"

    def test_epsilon_values_are_valid_probabilities(self) -> None:
        """Test that all epsilon values are valid probabilities [0, 1]."""
        strategy = EpsilonGreedy(
            min_action=0.0,
            max_action=10.0,
            step_size=2.0,
            epsilon_0=0.2,
            epsilon_min=0.01,
            decay_rate=0.9,
            seed=42,
        )

        # Run many rounds
        for round_num in range(100):
            epsilon = strategy.get_current_epsilon()

            # Epsilon should be a valid probability
            assert (
                0.0 <= epsilon <= 1.0
            ), f"Epsilon {epsilon} is not a valid probability at round {round_num}"

            strategy.next_action(
                round_num=round_num,
                my_history=[],
                rival_histories=[],
                bounds=(0.0, 10.0),
                market_params={},
            )
