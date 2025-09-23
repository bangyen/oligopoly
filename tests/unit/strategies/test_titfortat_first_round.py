"""Test TitForTat first round behavior."""

from src.sim.strategies.strategies import TitForTat


def test_titfortat_first_round() -> None:
    """Test that TitForTat first action equals midpoint of bounds."""
    strategy = TitForTat()

    # Test with different bound ranges
    test_cases = [
        (0, 20),  # Midpoint = 10
        (5, 15),  # Midpoint = 10
        (0, 100),  # Midpoint = 50
        (-10, 10),  # Midpoint = 0
    ]

    for min_bound, max_bound in test_cases:
        action = strategy.next_action(
            round_num=0,
            my_history=[],
            rival_histories=[],
            bounds=(min_bound, max_bound),
            market_params={},
        )

        expected = (min_bound + max_bound) / 2.0

        assert action == expected, (
            f"Expected {expected}, got {action} for bounds ({min_bound}, {max_bound})"
        )
