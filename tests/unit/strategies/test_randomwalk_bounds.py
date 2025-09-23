"""Test RandomWalk bounds behavior."""

from typing import List, Union

from src.sim.games.bertrand import BertrandResult
from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import RandomWalk


def test_randomwalk_bounds() -> None:
    """Test that 100 steps always stay within [min,max] bounds."""
    # Test with different bound ranges and step sizes
    test_cases = [
        (0.0, 10.0, 1.0),  # Standard case
        (5.0, 15.0, 0.5),  # Smaller step size
        (-10.0, 10.0, 2.0),  # Negative lower bound
        (0.0, 100.0, 5.0),  # Large range
    ]

    for min_bound, max_bound, step_size in test_cases:
        strategy = RandomWalk(
            step=step_size, min_bound=min_bound, max_bound=max_bound, seed=42
        )

        # Run 100 steps and verify all actions are within bounds
        current_action = None
        for round_num in range(100):
            my_history: List[Union[CournotResult, BertrandResult]] = []
            if current_action is not None:
                # Create a mock result with the current action
                if round_num % 2 == 0:
                    result: Union[CournotResult, BertrandResult] = CournotResult(
                        price=10.0, quantities=[current_action], profits=[50.0]
                    )
                else:
                    result = BertrandResult(
                        total_demand=50.0,
                        prices=[current_action],
                        quantities=[25.0],
                        profits=[125.0],
                    )
                my_history = [result]

            current_action = strategy.next_action(
                round_num=round_num,
                my_history=my_history,
                rival_histories=[],
                bounds=(min_bound, max_bound),
                market_params={},
            )

            assert min_bound <= current_action <= max_bound, (
                f"Action {current_action} outside bounds [{min_bound}, {max_bound}] at round {round_num}"
            )

    # Test edge case: very small step size
    strategy = RandomWalk(step=0.01, min_bound=0.0, max_bound=1.0, seed=123)
    current_action = None

    for round_num in range(50):
        my_history = []
        if current_action is not None:
            result = CournotResult(
                price=10.0, quantities=[current_action], profits=[50.0]
            )
            my_history = [result]

        current_action = strategy.next_action(
            round_num=round_num,
            my_history=my_history,
            rival_histories=[],
            bounds=(0.0, 1.0),
            market_params={},
        )

        assert 0.0 <= current_action <= 1.0, (
            f"Action {current_action} outside bounds [0.0, 1.0] at round {round_num}"
        )
