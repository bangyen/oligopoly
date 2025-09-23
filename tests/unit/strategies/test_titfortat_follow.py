"""Test TitForTat follow behavior."""

import math
from typing import List, Sequence, Union

from src.sim.games.bertrand import BertrandResult
from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import TitForTat


def test_titfortat_follow() -> None:
    """Test that TitForTat on round 2 mirrors rival mean from round 1 (±1e-6)."""
    strategy = TitForTat()

    # Test cases with different rival action combinations
    test_cases = [
        # Cournot results with quantities
        (
            [
                CournotResult(price=10.0, quantities=[5.0], profits=[25.0]),
                CournotResult(price=10.0, quantities=[15.0], profits=[75.0]),
            ],
            10.0,
        ),  # Mean = 10.0
        (
            [
                CournotResult(price=10.0, quantities=[0.0], profits=[0.0]),
                CournotResult(price=10.0, quantities=[20.0], profits=[100.0]),
            ],
            10.0,
        ),  # Mean = 10.0
        (
            [
                CournotResult(price=10.0, quantities=[1.0], profits=[5.0]),
                CournotResult(price=10.0, quantities=[2.0], profits=[10.0]),
                CournotResult(price=10.0, quantities=[3.0], profits=[15.0]),
            ],
            2.0,
        ),  # Mean = 2.0
        (
            [
                CournotResult(price=10.0, quantities=[7.5], profits=[37.5]),
                CournotResult(price=10.0, quantities=[12.5], profits=[62.5]),
            ],
            10.0,
        ),  # Mean = 10.0
        (
            [CournotResult(price=10.0, quantities=[15.0], profits=[75.0])],
            15.0,
        ),  # Single rival
    ]

    for rival_results, expected_mean in test_cases:
        rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]] = [
            [result] for result in rival_results
        ]

        action = strategy.next_action(
            round_num=1,  # Round 2 (0-indexed)
            my_history=[],
            rival_histories=rival_histories,
            bounds=(0, 20),
            market_params={},
        )

        assert math.isclose(action, expected_mean, abs_tol=1e-6), (
            f"Expected {expected_mean}, got {action} for rival results {rival_results}"
        )

    # Test with bounds clamping
    rival_result = CournotResult(price=5.0, quantities=[15.0], profits=[0.0])

    action = strategy.next_action(
        round_num=1,
        my_history=[],
        rival_histories=[[rival_result]],
        bounds=(0, 10),  # Tight bounds
        market_params={},
    )

    assert action == 10.0, f"Expected 10.0 (clamped), got {action}"
