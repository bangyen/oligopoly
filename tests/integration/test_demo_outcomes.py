"""Integration tests for ε-greedy demo outcomes.

Tests the full demo scenario with directional assertions about market outcomes
after entry of a 4th firm at round 10.
"""

import sys
from typing import List

import pytest

# Add src to path for imports
sys.path.insert(0, "src")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.models import Base
from sim.strategies import EpsilonGreedy
from sim.strategy_runner import get_strategy_run_results, run_strategy_game


def calculate_hhi(quantities: List[float]) -> float:
    """Calculate Herfindahl-Hirschman Index (HHI) for market concentration."""
    total_quantity = sum(quantities)
    if total_quantity == 0:
        return 0.0
    market_shares = [q / total_quantity for q in quantities]
    return sum(share**2 for share in market_shares) * 10000


def calculate_consumer_surplus(
    price: float, total_quantity: float, a: float, b: float
) -> float:
    """Calculate consumer surplus for linear demand curve."""
    if total_quantity == 0:
        return 0.0
    # CS = 0.5 * (a - price) * total_quantity
    return 0.5 * max(0, a - price) * total_quantity


class TestDemoOutcomes:
    """Test integration outcomes for ε-greedy demo scenario."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session for testing."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = session_local()
        yield db
        db.close()

    def test_entrant_lowers_average_price_after_round_12(self, db_session) -> None:
        """Test that entrant at round 10 lowers average price after round 12."""
        # Market parameters
        market_params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        # Grid parameters for ε-greedy
        grid_params = {
            "min_action": 0.0,
            "max_action": 50.0,
            "step_size": 2.0,
            "epsilon_0": 0.2,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "decay_rate": 0.95,
        }

        # Initial 3 firms
        initial_costs = [10.0, 12.0, 15.0]
        strategies = [
            EpsilonGreedy(**grid_params, seed=42),
            EpsilonGreedy(**grid_params, seed=43),
            EpsilonGreedy(**grid_params, seed=44),
        ]

        # Run first 10 rounds
        run_id_1 = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=initial_costs,
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Add 4th firm
        new_cost = 8.0  # Lower cost
        strategies.append(EpsilonGreedy(**grid_params, seed=45))
        all_costs = initial_costs + [new_cost]

        # Run remaining 10 rounds
        run_id_2 = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=all_costs,
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Get results
        results_1 = get_strategy_run_results(run_id_1, db_session)
        results_2 = get_strategy_run_results(run_id_2, db_session)

        # Calculate average prices for different periods
        # Pre-entry: rounds 7-9 (last 3 rounds before entry)
        pre_entry_prices = []
        for round_idx in range(7, 10):
            round_data = results_1["results"][round_idx]
            price = round_data[0]["price"]  # All firms have same price in Cournot
            pre_entry_prices.append(price)

        # Post-entry: rounds 13-19 (after entry effects settle)
        post_entry_prices = []
        for round_idx in range(3, 10):  # rounds 13-19 in second run
            round_data = results_2["results"][round_idx]
            price = round_data[0]["price"]
            post_entry_prices.append(price)

        pre_avg_price = sum(pre_entry_prices) / len(pre_entry_prices)
        post_avg_price = sum(post_entry_prices) / len(post_entry_prices)

        # Price should decrease after entry (more competition)
        assert (
            post_avg_price < pre_avg_price
        ), f"Price should decrease after entry: {pre_avg_price:.2f} -> {post_avg_price:.2f}"

        # The decrease should be meaningful (at least 1%)
        price_decrease = (pre_avg_price - post_avg_price) / pre_avg_price
        assert price_decrease > 0.01, f"Price decrease too small: {price_decrease:.3f}"

    def test_hhi_decreases_after_entry(self, db_session) -> None:
        """Test that HHI decreases after entry (less concentration)."""
        # Same setup as previous test
        market_params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        grid_params = {
            "min_action": 0.0,
            "max_action": 50.0,
            "step_size": 2.0,
            "epsilon_0": 0.2,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "decay_rate": 0.95,
        }

        initial_costs = [10.0, 12.0, 15.0]
        strategies = [
            EpsilonGreedy(**grid_params, seed=42),
            EpsilonGreedy(**grid_params, seed=43),
            EpsilonGreedy(**grid_params, seed=44),
        ]

        # Run first 10 rounds
        run_id_1 = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=initial_costs,
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Add 4th firm
        strategies.append(EpsilonGreedy(**grid_params, seed=45))
        all_costs = initial_costs + [8.0]

        # Run remaining 10 rounds
        run_id_2 = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=all_costs,
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Get results
        results_1 = get_strategy_run_results(run_id_1, db_session)
        results_2 = get_strategy_run_results(run_id_2, db_session)

        # Calculate average HHI for different periods
        # Pre-entry: rounds 7-9
        pre_entry_hhi = []
        for round_idx in range(7, 10):
            round_data = results_1["results"][round_idx]
            quantities = [
                round_data[firm_id]["quantity"] for firm_id in sorted(round_data.keys())
            ]
            hhi = calculate_hhi(quantities)
            pre_entry_hhi.append(hhi)

        # Post-entry: rounds 13-19
        post_entry_hhi = []
        for round_idx in range(3, 10):
            round_data = results_2["results"][round_idx]
            quantities = [
                round_data[firm_id]["quantity"] for firm_id in sorted(round_data.keys())
            ]
            hhi = calculate_hhi(quantities)
            post_entry_hhi.append(hhi)

        pre_avg_hhi = sum(pre_entry_hhi) / len(pre_entry_hhi)
        post_avg_hhi = sum(post_entry_hhi) / len(post_entry_hhi)

        # HHI should decrease after entry (less concentration)
        assert (
            post_avg_hhi < pre_avg_hhi
        ), f"HHI should decrease after entry: {pre_avg_hhi:.0f} -> {post_avg_hhi:.0f}"

        # The decrease should be meaningful
        hhi_decrease = (pre_avg_hhi - post_avg_hhi) / pre_avg_hhi
        assert hhi_decrease > 0.05, f"HHI decrease too small: {hhi_decrease:.3f}"

    def test_consumer_surplus_increases_after_entry(self, db_session) -> None:
        """Test that consumer surplus increases after entry."""
        # Same setup
        market_params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        grid_params = {
            "min_action": 0.0,
            "max_action": 50.0,
            "step_size": 2.0,
            "epsilon_0": 0.2,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "decay_rate": 0.95,
        }

        initial_costs = [10.0, 12.0, 15.0]
        strategies = [
            EpsilonGreedy(**grid_params, seed=42),
            EpsilonGreedy(**grid_params, seed=43),
            EpsilonGreedy(**grid_params, seed=44),
        ]

        # Run first 10 rounds
        run_id_1 = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=initial_costs,
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Add 4th firm
        strategies.append(EpsilonGreedy(**grid_params, seed=45))
        all_costs = initial_costs + [8.0]

        # Run remaining 10 rounds
        run_id_2 = run_strategy_game(
            model="cournot",
            rounds=10,
            strategies=strategies,
            costs=all_costs,
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Get results
        results_1 = get_strategy_run_results(run_id_1, db_session)
        results_2 = get_strategy_run_results(run_id_2, db_session)

        # Calculate average consumer surplus for different periods
        # Pre-entry: rounds 7-9
        pre_entry_cs = []
        for round_idx in range(7, 10):
            round_data = results_1["results"][round_idx]
            quantities = [
                round_data[firm_id]["quantity"] for firm_id in sorted(round_data.keys())
            ]
            price = round_data[0]["price"]
            total_qty = sum(quantities)
            cs = calculate_consumer_surplus(
                price, total_qty, market_params["a"], market_params["b"]
            )
            pre_entry_cs.append(cs)

        # Post-entry: rounds 13-19
        post_entry_cs = []
        for round_idx in range(3, 10):
            round_data = results_2["results"][round_idx]
            quantities = [
                round_data[firm_id]["quantity"] for firm_id in sorted(round_data.keys())
            ]
            price = round_data[0]["price"]
            total_qty = sum(quantities)
            cs = calculate_consumer_surplus(
                price, total_qty, market_params["a"], market_params["b"]
            )
            post_entry_cs.append(cs)

        pre_avg_cs = sum(pre_entry_cs) / len(pre_entry_cs)
        post_avg_cs = sum(post_entry_cs) / len(post_entry_cs)

        # Consumer surplus should increase after entry (lower prices)
        assert (
            post_avg_cs > pre_avg_cs
        ), f"Consumer surplus should increase after entry: {pre_avg_cs:.2f} -> {post_avg_cs:.2f}"

        # The increase should be meaningful
        cs_increase = (post_avg_cs - pre_avg_cs) / pre_avg_cs
        assert (
            cs_increase > 0.01
        ), f"Consumer surplus increase too small: {cs_increase:.3f}"

    def test_epsilon_greedy_learning_progress(self, db_session) -> None:
        """Test that ε-greedy agents show learning progress."""
        market_params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        grid_params = {
            "min_action": 0.0,
            "max_action": 50.0,
            "step_size": 2.0,
            "epsilon_0": 0.2,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "decay_rate": 0.95,
        }

        # Single firm for focused testing
        strategy = EpsilonGreedy(**grid_params, seed=42)

        # Run simulation
        run_id = run_strategy_game(
            model="cournot",
            rounds=20,
            strategies=[strategy],
            costs=[10.0],
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Check that epsilon decayed
        assert (
            strategy.get_current_epsilon() < strategy.epsilon_0
        ), "Epsilon should have decayed from initial value"

        # Check that epsilon reached minimum
        assert (
            strategy.get_current_epsilon() >= strategy.epsilon_min
        ), "Epsilon should not go below minimum"

        # Check that Q-values are not all zero (some learning occurred)
        q_values = strategy.get_q_values()
        non_zero_q_values = [q for q in q_values if abs(q) > 1e-6]
        assert (
            len(non_zero_q_values) > 0
        ), "Some Q-values should be non-zero after learning"

        # Check that actions are from grid
        results = get_strategy_run_results(run_id, db_session)
        action_grid = strategy.get_action_grid()

        for round_idx in range(20):
            round_data = results["results"][round_idx]
            action = round_data[0]["action"]
            assert (
                action in action_grid
            ), f"Action {action} not in grid at round {round_idx}"

    def test_multiple_firms_independent_learning(self, db_session) -> None:
        """Test that multiple ε-greedy firms learn independently."""
        market_params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        grid_params = {
            "min_action": 0.0,
            "max_action": 50.0,
            "step_size": 2.0,
            "epsilon_0": 0.2,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "decay_rate": 0.95,
        }

        # Create 3 firms with different seeds
        strategies = [
            EpsilonGreedy(**grid_params, seed=42),
            EpsilonGreedy(**grid_params, seed=43),
            EpsilonGreedy(**grid_params, seed=44),
        ]

        costs = [10.0, 12.0, 15.0]

        # Run simulation
        run_id = run_strategy_game(
            model="cournot",
            rounds=15,
            strategies=strategies,
            costs=costs,
            params=market_params,
            bounds=bounds,
            db=db_session,
            seed=42,
        )

        # Check that all firms learned independently
        for i, strategy in enumerate(strategies):
            # Each firm should have decayed epsilon
            assert (
                strategy.get_current_epsilon() < strategy.epsilon_0
            ), f"Firm {i} epsilon should have decayed"

            # Each firm should have some non-zero Q-values
            q_values = strategy.get_q_values()
            non_zero_q_values = [q for q in q_values if abs(q) > 1e-6]
            assert (
                len(non_zero_q_values) > 0
            ), f"Firm {i} should have learned some Q-values"

            # Actions should be from grid
            action_grid = strategy.get_action_grid()
            results = get_strategy_run_results(run_id, db_session)

            for round_idx in range(15):
                round_data = results["results"][round_idx]
                action = round_data[i]["action"]
                assert (
                    action in action_grid
                ), f"Firm {i} action {action} not in grid at round {round_idx}"

        # Check that firms have different final Q-values (due to different experiences)
        q_values_0 = strategies[0].get_q_values()
        q_values_1 = strategies[1].get_q_values()
        q_values_2 = strategies[2].get_q_values()

        # At least one pair should be different
        assert (
            q_values_0 != q_values_1
            or q_values_0 != q_values_2
            or q_values_1 != q_values_2
        ), "Firms should have different Q-values due to different experiences"
