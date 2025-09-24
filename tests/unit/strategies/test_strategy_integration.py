"""Test strategy integration with simulation framework."""

from typing import List, Sequence, Union

from src.sim.games.bertrand import BertrandResult
from src.sim.games.cournot import CournotResult
from src.sim.strategies.strategies import RandomWalk, Static, Strategy, TitForTat


def test_strategy_integration() -> None:
    """Test that different strategies produce different behaviors in multi-round simulation."""
    strategies: List[Strategy] = [
        Static(value=5.0),
        TitForTat(),
        RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=42),
    ]

    bounds = (0.0, 10.0)
    rounds = 5

    # Simulate multiple rounds manually
    trajectories: List[List[float]] = [[] for _ in strategies]

    for round_num in range(rounds):
        round_actions = []

        for firm_idx, strategy in enumerate(strategies):
            # Build histories for this firm
            my_history = []
            rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]] = []

            if round_num > 0:
                # Add previous results to histories
                for i, traj in enumerate(trajectories):
                    if i == firm_idx:
                        # My history
                        if traj:
                            prev_action = traj[-1]
                            result = CournotResult(
                                price=8.0, quantities=[prev_action], profits=[40.0]
                            )
                            my_history = [result]
                    else:
                        # Rival history
                        if traj:
                            prev_action = traj[-1]
                            result = CournotResult(
                                price=8.0, quantities=[prev_action], profits=[40.0]
                            )
                            rival_histories.append([result])
                        else:
                            rival_histories.append([])

            action = strategy.next_action(
                round_num=round_num,
                my_history=my_history,
                rival_histories=rival_histories,
                bounds=bounds,
                market_params={},
            )

            round_actions.append(action)
            trajectories[firm_idx].append(action)

    # Verify we got trajectories for all 3 firms
    assert len(trajectories) == 3, f"Expected 3 trajectories, got {len(trajectories)}"
    assert all(
        len(traj) == rounds for traj in trajectories
    ), f"Expected all trajectories to have {rounds} rounds"

    # Verify trajectories are pairwise different (not all equal)
    # Check that at least one pair of firms has different trajectories
    all_equal = True
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            if trajectories[i] != trajectories[j]:
                all_equal = False
                break
        if not all_equal:
            break

    assert not all_equal, "All trajectories should not be identical"

    # Verify all actions are within bounds
    for i, traj in enumerate(trajectories):
        for round_idx, action in enumerate(traj):
            assert (
                bounds[0] <= action <= bounds[1]
            ), f"Firm {i} action {action} outside bounds {bounds} at round {round_idx}"

    # Print trajectories for debugging (optional)
    print(f"Static strategy trajectory: {trajectories[0]}")
    print(f"TitForTat strategy trajectory: {trajectories[1]}")
    print(f"RandomWalk strategy trajectory: {trajectories[2]}")


def test_strategy_integration_different_seeds() -> None:
    """Test that different seeds produce different RandomWalk trajectories."""
    strategies1: List[Strategy] = [
        Static(value=5.0),
        TitForTat(),
        RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=123),
    ]

    strategies2: List[Strategy] = [
        Static(value=5.0),
        TitForTat(),
        RandomWalk(step=1.0, min_bound=0.0, max_bound=10.0, seed=456),
    ]

    bounds = (0.0, 10.0)
    rounds = 5

    # Simulate both sets of strategies
    trajectories1: List[List[float]] = [[] for _ in strategies1]
    trajectories2: List[List[float]] = [[] for _ in strategies2]

    for round_num in range(rounds):
        for strategies, trajectories in [
            (strategies1, trajectories1),
            (strategies2, trajectories2),
        ]:
            for firm_idx, strategy in enumerate(strategies):
                my_history = []
                rival_histories: List[
                    Sequence[Union[CournotResult, BertrandResult]]
                ] = []

                if round_num > 0:
                    # Add previous results to histories
                    for i, traj in enumerate(trajectories):
                        if i == firm_idx:
                            if traj:
                                prev_action = traj[-1]
                                result = CournotResult(
                                    price=8.0, quantities=[prev_action], profits=[40.0]
                                )
                                my_history = [result]
                        else:
                            if traj:
                                prev_action = traj[-1]
                                result = CournotResult(
                                    price=8.0, quantities=[prev_action], profits=[40.0]
                                )
                                rival_histories.append([result])
                            else:
                                rival_histories.append([])

                action = strategy.next_action(
                    round_num=round_num,
                    my_history=my_history,
                    rival_histories=rival_histories,
                    bounds=bounds,
                    market_params={},
                )

                trajectories[firm_idx].append(action)

    # Static should be identical (doesn't depend on other strategies)
    assert trajectories1[0] == trajectories2[0], "Static strategies should be identical"

    # TitForTat will be different because it responds to different RandomWalk trajectories
    assert (
        trajectories1[1] != trajectories2[1]
    ), "TitForTat strategies should differ due to different RandomWalk inputs"

    # RandomWalk should be different due to different seeds
    assert (
        trajectories1[2] != trajectories2[2]
    ), "RandomWalk strategies should differ with different seeds"


def test_strategy_integration_mixed_bounds() -> None:
    """Test integration with different bound ranges."""
    strategies: List[Strategy] = [
        Static(value=7.5),
        TitForTat(),
        RandomWalk(step=0.5, min_bound=0.0, max_bound=15.0, seed=789),
    ]

    bounds = (0.0, 15.0)
    rounds = 8

    # Simulate multiple rounds
    trajectories: List[List[float]] = [[] for _ in strategies]

    for round_num in range(rounds):
        for firm_idx, strategy in enumerate(strategies):
            my_history = []
            rival_histories: List[Sequence[Union[CournotResult, BertrandResult]]] = []

            if round_num > 0:
                # Add previous results to histories
                for i, traj in enumerate(trajectories):
                    if i == firm_idx:
                        if traj:
                            prev_action = traj[-1]
                            result = CournotResult(
                                price=10.0, quantities=[prev_action], profits=[50.0]
                            )
                            my_history = [result]
                    else:
                        if traj:
                            prev_action = traj[-1]
                            result = CournotResult(
                                price=10.0, quantities=[prev_action], profits=[50.0]
                            )
                            rival_histories.append([result])
                        else:
                            rival_histories.append([])

            action = strategy.next_action(
                round_num=round_num,
                my_history=my_history,
                rival_histories=rival_histories,
                bounds=bounds,
                market_params={},
            )

            trajectories[firm_idx].append(action)

    # Verify trajectories are different
    all_equal = True
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            if trajectories[i] != trajectories[j]:
                all_equal = False
                break
        if not all_equal:
            break

    assert not all_equal, "Trajectories should differ even with different bounds"

    # Verify Static strategy respects bounds
    static_trajectory = trajectories[0]
    assert all(
        action == 7.5 for action in static_trajectory
    ), "Static strategy should maintain constant value"
