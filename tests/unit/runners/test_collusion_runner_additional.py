"""Additional tests for collusion_runner.py to improve coverage.

This module tests additional edge cases, error handling, and validation
scenarios in the collusion runner implementation.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

from src.sim.runners.collusion_runner import (
    _run_bertrand_round,
    _run_cournot_round,
    create_collusion_simulation_config,
    get_collusion_run_results,
    run_collusion_game,
)
from src.sim.strategies.strategies import Strategy


class TestRunCournotRound:
    """Test the _run_cournot_round function."""

    def test_run_cournot_round_basic(self):
        """Test _run_cournot_round with basic parameters."""
        params = {"a": 100.0, "b": 1.0}
        costs = [10.0, 12.0]
        quantities = [20.0, 15.0]

        result = _run_cournot_round(params, costs, quantities)

        assert result.price > 0
        assert len(result.quantities) == 2
        assert len(result.profits) == 2
        assert result.quantities == [20.0, 15.0]

    def test_run_cournot_round_default_params(self):
        """Test _run_cournot_round with default parameters."""
        params = {}  # Empty params
        costs = [10.0, 12.0]
        quantities = [20.0, 15.0]

        result = _run_cournot_round(params, costs, quantities)

        # Should use default values a=100.0, b=1.0
        assert result.price > 0
        assert len(result.quantities) == 2
        assert len(result.profits) == 2


class TestRunBertrandRound:
    """Test the _run_bertrand_round function."""

    def test_run_bertrand_round_basic(self):
        """Test _run_bertrand_round with basic parameters."""
        params = {"alpha": 200.0, "beta": 2.0}
        costs = [10.0, 12.0]
        prices = [50.0, 45.0]

        result = _run_bertrand_round(params, costs, prices)

        assert len(result.quantities) == 2
        assert len(result.profits) == 2
        assert len(result.prices) == 2
        assert result.prices == [50.0, 45.0]

    def test_run_bertrand_round_default_params(self):
        """Test _run_bertrand_round with default parameters."""
        params = {}  # Empty params
        costs = [10.0, 12.0]
        prices = [50.0, 45.0]

        result = _run_bertrand_round(params, costs, prices)

        # Should use default values alpha=100.0, beta=1.0
        assert len(result.quantities) == 2
        assert len(result.profits) == 2
        assert len(result.prices) == 2


class TestGetCollusionRunResults:
    """Test the get_collusion_run_results function."""

    def test_get_collusion_run_results_success(self):
        """Test get_collusion_run_results with successful data retrieval."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_123"
        mock_run.model = "cournot"
        mock_run.rounds = 10
        mock_run.created_at = "2023-01-01T00:00:00"

        # Mock results
        mock_result1 = Mock()
        mock_result1.round_idx = 0
        mock_result1.firm_id = 0
        mock_result1.action = 20.0
        mock_result1.price = 50.0
        mock_result1.qty = 20.0
        mock_result1.profit = 800.0

        mock_result2 = Mock()
        mock_result2.round_idx = 0
        mock_result2.firm_id = 1
        mock_result2.action = 15.0
        mock_result2.price = 50.0
        mock_result2.qty = 15.0
        mock_result2.profit = 570.0

        # Mock events
        mock_event = Mock()
        mock_event.round_idx = 0
        mock_event.id = 1
        mock_event.event_type = "cartel_formed"
        mock_event.firm_id = 0
        mock_event.description = "Cartel formed"
        mock_event.event_data = {"participating_firms": [0, 1]}
        mock_event.created_at = datetime(2023, 1, 1, 0, 0, 0)

        # Set up mock queries
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                mock_run_query = Mock()
                mock_run_query.filter.return_value.first.return_value = mock_run
                return mock_run_query
            elif model.__name__ == "Result":
                mock_result_query = Mock()
                mock_result_query.filter.return_value.order_by.return_value.all.return_value = [
                    mock_result1,
                    mock_result2,
                ]
                return mock_result_query
            elif model.__name__ == "CollusionEvent":
                mock_event_query = Mock()
                mock_event_query.filter.return_value.order_by.return_value.all.return_value = [
                    mock_event
                ]
                return mock_event_query
            else:
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        result = get_collusion_run_results("run_123", mock_db)

        assert result["run_id"] == "run_123"
        assert result["model"] == "cournot"
        assert result["rounds"] == 10
        assert result["created_at"] == "2023-01-01T00:00:00"
        assert "results" in result
        assert "events" in result

        # Check results structure
        assert 0 in result["results"]
        assert 0 in result["results"][0]
        assert 1 in result["results"][0]
        assert result["results"][0][0]["action"] == 20.0
        assert result["results"][0][0]["price"] == 50.0
        assert result["results"][0][0]["quantity"] == 20.0
        assert result["results"][0][0]["profit"] == 800.0

        # Check events structure
        assert 0 in result["events"]
        assert len(result["events"][0]) == 1
        assert result["events"][0][0]["event_type"] == "cartel_formed"
        assert result["events"][0][0]["firm_id"] == 0
        assert result["events"][0][0]["description"] == "Cartel formed"

    def test_get_collusion_run_results_not_found(self):
        """Test get_collusion_run_results with non-existent run."""
        mock_db = Mock(spec=Session)

        # Mock query returning None for run
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                mock_run_query = Mock()
                mock_run_query.filter.return_value.first.return_value = None
                return mock_run_query
            else:
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        with pytest.raises(ValueError) as exc_info:
            get_collusion_run_results("nonexistent", mock_db)
        assert "Run nonexistent not found" in str(exc_info.value)

    def test_get_collusion_run_results_with_none_event_data(self):
        """Test get_collusion_run_results with None event data."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_123"
        mock_run.model = "cournot"
        mock_run.rounds = 10
        mock_run.created_at = "2023-01-01T00:00:00"

        # Mock event with None event_data
        mock_event = Mock()
        mock_event.round_idx = 0
        mock_event.id = 1
        mock_event.event_type = "cartel_formed"
        mock_event.firm_id = 0
        mock_event.description = "Cartel formed"
        mock_event.event_data = None  # None event data
        mock_event.created_at = None  # None created_at

        # Set up mock queries
        def mock_query_side_effect(model):
            if model.__name__ == "Run":
                mock_run_query = Mock()
                mock_run_query.filter.return_value.first.return_value = mock_run
                return mock_run_query
            elif model.__name__ == "Result":
                mock_result_query = Mock()
                mock_result_query.filter.return_value.order_by.return_value.all.return_value = (
                    []
                )
                return mock_result_query
            elif model.__name__ == "CollusionEvent":
                mock_event_query = Mock()
                mock_event_query.filter.return_value.order_by.return_value.all.return_value = [
                    mock_event
                ]
                return mock_event_query
            else:
                return Mock()

        mock_db.query.side_effect = mock_query_side_effect

        result = get_collusion_run_results("run_123", mock_db)

        # Check that None event_data is handled
        assert result["events"][0][0]["data"] == {}
        assert result["events"][0][0]["created_at"] is None


class TestCreateCollusionSimulationConfig:
    """Test the create_collusion_simulation_config function."""

    def test_create_collusion_simulation_config_defaults(self):
        """Test create_collusion_simulation_config with default values."""
        config = create_collusion_simulation_config()

        assert config["hhi_threshold"] == 0.8
        assert config["price_threshold_multiplier"] == 1.5
        assert config["baseline_price"] == 0.0
        assert config["intervention_probability"] == 0.8
        assert config["penalty_amount"] == 100.0
        assert config["price_cap_multiplier"] == 0.9
        assert config["auto_form_cartel"] is False

    def test_create_collusion_simulation_config_custom(self):
        """Test create_collusion_simulation_config with custom values."""
        config = create_collusion_simulation_config(
            hhi_threshold=0.9,
            price_threshold_multiplier=2.0,
            baseline_price=10.0,
            intervention_probability=0.9,
            penalty_amount=200.0,
            price_cap_multiplier=0.8,
            auto_form_cartel=True,
        )

        assert config["hhi_threshold"] == 0.9
        assert config["price_threshold_multiplier"] == 2.0
        assert config["baseline_price"] == 10.0
        assert config["intervention_probability"] == 0.9
        assert config["penalty_amount"] == 200.0
        assert config["price_cap_multiplier"] == 0.8
        assert config["auto_form_cartel"] is True


class TestRunCollusionGame:
    """Test the run_collusion_game function."""

    def test_run_collusion_game_invalid_model(self):
        """Test run_collusion_game with invalid model."""
        mock_db = Mock(spec=Session)
        strategies = [Mock(spec=Strategy), Mock(spec=Strategy)]
        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with pytest.raises(ValueError) as exc_info:
            run_collusion_game(
                model="invalid",
                rounds=10,
                strategies=strategies,
                costs=costs,
                params=params,
                bounds=bounds,
                db=mock_db,
            )
        assert "Model must be 'cournot' or 'bertrand'" in str(exc_info.value)

    def test_run_collusion_game_invalid_rounds(self):
        """Test run_collusion_game with invalid rounds."""
        mock_db = Mock(spec=Session)
        strategies = [Mock(spec=Strategy), Mock(spec=Strategy)]
        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with pytest.raises(ValueError) as exc_info:
            run_collusion_game(
                model="cournot",
                rounds=0,  # Invalid rounds
                strategies=strategies,
                costs=costs,
                params=params,
                bounds=bounds,
                db=mock_db,
            )
        assert "Rounds must be positive" in str(exc_info.value)

    def test_run_collusion_game_mismatched_strategies_costs(self):
        """Test run_collusion_game with mismatched strategies and costs."""
        mock_db = Mock(spec=Session)
        strategies = [Mock(spec=Strategy)]  # 1 strategy
        costs = [10.0, 12.0]  # 2 costs
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with pytest.raises(ValueError) as exc_info:
            run_collusion_game(
                model="cournot",
                rounds=10,
                strategies=strategies,
                costs=costs,
                params=params,
                bounds=bounds,
                db=mock_db,
            )
        assert "Number of strategies (1) must match number of firms (2)" in str(
            exc_info.value
        )

    def test_run_collusion_game_empty_strategies(self):
        """Test run_collusion_game with empty strategies."""
        mock_db = Mock(spec=Session)
        strategies = []  # Empty strategies
        costs = []
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with pytest.raises(ValueError) as exc_info:
            run_collusion_game(
                model="cournot",
                rounds=10,
                strategies=strategies,
                costs=costs,
                params=params,
                bounds=bounds,
                db=mock_db,
            )
        assert "At least one strategy is required" in str(exc_info.value)

    def test_run_collusion_game_success_cournot(self):
        """Test run_collusion_game with successful Cournot execution."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_123"
        mock_db.add.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        # Mock strategies
        mock_strategy = Mock(spec=Strategy)
        mock_strategy.next_action.return_value = 20.0
        strategies = [mock_strategy, mock_strategy]

        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with patch("src.sim.runners.collusion_runner.Run") as mock_run_class:
            mock_run_class.return_value = mock_run
            with patch(
                "src.sim.runners.collusion_runner.CollusionManager"
            ) as mock_collusion_manager:
                with patch(
                    "src.sim.runners.collusion_runner.EventLogger"
                ) as mock_event_logger:
                    with patch(
                        "src.sim.runners.collusion_runner.cournot_simulation"
                    ) as mock_cournot_sim:
                        # Mock cournot simulation result
                        from src.sim.games.cournot import CournotResult

                        mock_result = Mock(spec=CournotResult)
                        mock_result.price = 50.0
                        mock_result.quantities = [20.0, 15.0]
                        mock_result.profits = [800.0, 570.0]
                        mock_cournot_sim.return_value = mock_result

                        # Mock collusion manager
                        mock_collusion_mgr = Mock()
                        mock_collusion_mgr.is_cartel_active.return_value = False
                        mock_collusion_mgr.get_events_for_round.return_value = []
                        mock_collusion_mgr.check_regulator_intervention.return_value = (
                            False,
                            None,
                            None,
                        )
                        mock_collusion_manager.return_value = mock_collusion_mgr

                        # Mock event logger
                        mock_event_log = Mock()
                        mock_event_logger.return_value = mock_event_log

                        run_id = run_collusion_game(
                            model="cournot",
                            rounds=2,  # Small number for testing
                            strategies=strategies,
                            costs=costs,
                            params=params,
                            bounds=bounds,
                            db=mock_db,
                        )

                        assert run_id == "run_123"
                        assert mock_db.add.call_count >= 1  # Run and rounds added
                        mock_db.commit.assert_called_once()

    def test_run_collusion_game_success_bertrand(self):
        """Test run_collusion_game with successful Bertrand execution."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_456"
        mock_db.add.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        # Mock strategies
        mock_strategy = Mock(spec=Strategy)
        mock_strategy.next_action.return_value = 50.0
        strategies = [mock_strategy, mock_strategy]

        costs = [10.0, 12.0]
        params = {"alpha": 200.0, "beta": 2.0}
        bounds = (0.0, 200.0)

        with patch("src.sim.runners.collusion_runner.Run") as mock_run_class:
            mock_run_class.return_value = mock_run
            with patch(
                "src.sim.runners.collusion_runner.CollusionManager"
            ) as mock_collusion_manager:
                with patch(
                    "src.sim.runners.collusion_runner.EventLogger"
                ) as mock_event_logger:
                    with patch(
                        "src.sim.runners.collusion_runner.bertrand_simulation"
                    ) as mock_bertrand_sim:
                        # Mock bertrand simulation result
                        from src.sim.games.bertrand import BertrandResult

                        mock_result = Mock(spec=BertrandResult)
                        mock_result.prices = [50.0, 45.0]
                        mock_result.quantities = [20.0, 25.0]
                        mock_result.profits = [800.0, 825.0]
                        mock_bertrand_sim.return_value = mock_result

                        # Mock collusion manager
                        mock_collusion_mgr = Mock()
                        mock_collusion_mgr.is_cartel_active.return_value = False
                        mock_collusion_mgr.get_events_for_round.return_value = []
                        mock_collusion_mgr.check_regulator_intervention.return_value = (
                            False,
                            None,
                            None,
                        )
                        mock_collusion_manager.return_value = mock_collusion_mgr

                        # Mock event logger
                        mock_event_log = Mock()
                        mock_event_logger.return_value = mock_event_log

                        run_id = run_collusion_game(
                            model="bertrand",
                            rounds=2,  # Small number for testing
                            strategies=strategies,
                            costs=costs,
                            params=params,
                            bounds=bounds,
                            db=mock_db,
                        )

                        assert run_id == "run_456"
                        assert mock_db.add.call_count >= 1  # Run and rounds added
                        mock_db.commit.assert_called_once()

    def test_run_collusion_game_with_collusion_config(self):
        """Test run_collusion_game with collusion configuration."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_789"
        mock_db.add.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        # Mock strategies
        mock_strategy = Mock(spec=Strategy)
        mock_strategy.next_action.return_value = 20.0
        strategies = [mock_strategy, mock_strategy]

        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)
        collusion_config = {
            "hhi_threshold": 0.9,
            "price_threshold_multiplier": 2.0,
            "baseline_price": 10.0,
            "intervention_probability": 0.9,
            "penalty_amount": 200.0,
            "price_cap_multiplier": 0.8,
            "auto_form_cartel": True,
        }

        with patch("src.sim.runners.collusion_runner.Run") as mock_run_class:
            mock_run_class.return_value = mock_run
            with patch(
                "src.sim.runners.collusion_runner.CollusionManager"
            ) as mock_collusion_manager:
                with patch(
                    "src.sim.runners.collusion_runner.EventLogger"
                ) as mock_event_logger:
                    with patch(
                        "src.sim.runners.collusion_runner.cournot_simulation"
                    ) as mock_cournot_sim:
                        # Mock cournot simulation result
                        from src.sim.games.cournot import CournotResult

                        mock_result = Mock(spec=CournotResult)
                        mock_result.price = 50.0
                        mock_result.quantities = [20.0, 15.0]
                        mock_result.profits = [800.0, 570.0]
                        mock_cournot_sim.return_value = mock_result

                        # Mock collusion manager
                        mock_collusion_mgr = Mock()
                        mock_collusion_mgr.is_cartel_active.return_value = False
                        mock_collusion_mgr.get_events_for_round.return_value = []
                        mock_collusion_mgr.check_regulator_intervention.return_value = (
                            False,
                            None,
                            None,
                        )
                        mock_collusion_manager.return_value = mock_collusion_mgr

                        # Mock event logger
                        mock_event_log = Mock()
                        mock_event_logger.return_value = mock_event_log

                        run_id = run_collusion_game(
                            model="cournot",
                            rounds=2,
                            strategies=strategies,
                            costs=costs,
                            params=params,
                            bounds=bounds,
                            db=mock_db,
                            collusion_config=collusion_config,
                        )

                        assert run_id == "run_789"
                        # Check that CollusionManager was called with the config
                        mock_collusion_manager.assert_called_once()
                        call_args = mock_collusion_manager.call_args[0][0]
                        assert call_args.hhi_threshold == 0.9
                        assert call_args.price_threshold_multiplier == 2.0
                        assert call_args.baseline_price == 10.0

    def test_run_collusion_game_with_collusion_strategies(self):
        """Test run_collusion_game with collusion-aware strategies."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_collusion"
        mock_db.add.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        # Mock collusion-aware strategy
        from src.sim.strategies.collusion_strategies import CollusiveStrategy

        mock_collusive_strategy = Mock(spec=CollusiveStrategy)
        mock_collusive_strategy.next_action.return_value = 20.0
        strategies = [mock_collusive_strategy, mock_collusive_strategy]

        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with patch("src.sim.runners.collusion_runner.Run") as mock_run_class:
            mock_run_class.return_value = mock_run
            with patch(
                "src.sim.runners.collusion_runner.CollusionManager"
            ) as mock_collusion_manager:
                with patch(
                    "src.sim.runners.collusion_runner.EventLogger"
                ) as mock_event_logger:
                    with patch(
                        "src.sim.runners.collusion_runner.cournot_simulation"
                    ) as mock_cournot_sim:
                        # Mock cournot simulation result
                        from src.sim.games.cournot import CournotResult

                        mock_result = Mock(spec=CournotResult)
                        mock_result.price = 50.0
                        mock_result.quantities = [20.0, 15.0]
                        mock_result.profits = [800.0, 570.0]
                        mock_cournot_sim.return_value = mock_result

                        # Mock collusion manager
                        mock_collusion_mgr = Mock()
                        mock_collusion_mgr.is_cartel_active.return_value = False
                        mock_collusion_mgr.get_events_for_round.return_value = []
                        mock_collusion_mgr.check_regulator_intervention.return_value = (
                            False,
                            None,
                            None,
                        )
                        mock_collusion_manager.return_value = mock_collusion_mgr

                        # Mock event logger
                        mock_event_log = Mock()
                        mock_event_logger.return_value = mock_event_log

                        run_id = run_collusion_game(
                            model="cournot",
                            rounds=2,
                            strategies=strategies,
                            costs=costs,
                            params=params,
                            bounds=bounds,
                            db=mock_db,
                        )

                        assert run_id == "run_collusion"
                        # Check that collusion-aware strategy was called with collusion_manager
                        mock_collusive_strategy.next_action.assert_called()
                        call_args = mock_collusive_strategy.next_action.call_args
                        assert "collusion_manager" in call_args.kwargs

    def test_run_collusion_game_with_opportunistic_strategy(self):
        """Test run_collusion_game with OpportunisticStrategy."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_opportunistic"
        mock_db.add.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        # Mock opportunistic strategy
        from src.sim.strategies.collusion_strategies import OpportunisticStrategy

        mock_opportunistic_strategy = Mock(spec=OpportunisticStrategy)
        mock_opportunistic_strategy.next_action.return_value = 20.0
        strategies = [mock_opportunistic_strategy, mock_opportunistic_strategy]

        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with patch("src.sim.runners.collusion_runner.Run") as mock_run_class:
            mock_run_class.return_value = mock_run
            with patch(
                "src.sim.runners.collusion_runner.CollusionManager"
            ) as mock_collusion_manager:
                with patch(
                    "src.sim.runners.collusion_runner.EventLogger"
                ) as mock_event_logger:
                    with patch(
                        "src.sim.runners.collusion_runner.cournot_simulation"
                    ) as mock_cournot_sim:
                        # Mock cournot simulation result
                        from src.sim.games.cournot import CournotResult

                        mock_result = Mock(spec=CournotResult)
                        mock_result.price = 50.0
                        mock_result.quantities = [20.0, 15.0]
                        mock_result.profits = [800.0, 570.0]
                        mock_cournot_sim.return_value = mock_result

                        # Mock collusion manager
                        mock_collusion_mgr = Mock()
                        mock_collusion_mgr.is_cartel_active.return_value = False
                        mock_collusion_mgr.get_events_for_round.return_value = []
                        mock_collusion_mgr.check_regulator_intervention.return_value = (
                            False,
                            None,
                            None,
                        )
                        mock_collusion_manager.return_value = mock_collusion_mgr

                        # Mock event logger
                        mock_event_log = Mock()
                        mock_event_logger.return_value = mock_event_log

                        run_id = run_collusion_game(
                            model="cournot",
                            rounds=2,
                            strategies=strategies,
                            costs=costs,
                            params=params,
                            bounds=bounds,
                            db=mock_db,
                        )

                        assert run_id == "run_opportunistic"
                        # Check that opportunistic strategy was called with my_cost
                        mock_opportunistic_strategy.next_action.assert_called()
                        call_args = mock_opportunistic_strategy.next_action.call_args
                        assert "my_cost" in call_args.kwargs
                        assert call_args.kwargs["my_cost"] in [
                            10.0,
                            12.0,
                        ]  # Either firm's cost

    def test_run_collusion_game_with_cartel_formation(self):
        """Test run_collusion_game with automatic cartel formation."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_cartel"
        mock_db.add.return_value = None
        mock_db.flush.return_value = None
        mock_db.commit.return_value = None

        # Mock strategies
        mock_strategy = Mock(spec=Strategy)
        mock_strategy.next_action.return_value = 20.0
        strategies = [mock_strategy, mock_strategy]

        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)
        collusion_config = {"auto_form_cartel": True}

        with patch("src.sim.runners.collusion_runner.Run") as mock_run_class:
            mock_run_class.return_value = mock_run
            with patch(
                "src.sim.runners.collusion_runner.CollusionManager"
            ) as mock_collusion_manager:
                with patch(
                    "src.sim.runners.collusion_runner.EventLogger"
                ) as mock_event_logger:
                    with patch(
                        "src.sim.runners.collusion_runner.cournot_simulation"
                    ) as mock_cournot_sim:
                        # Mock cournot simulation result
                        from src.sim.games.cournot import CournotResult

                        mock_result = Mock(spec=CournotResult)
                        mock_result.price = 50.0
                        mock_result.quantities = [20.0, 15.0]
                        mock_result.profits = [800.0, 570.0]
                        mock_cournot_sim.return_value = mock_result

                        # Mock collusion manager
                        mock_collusion_mgr = Mock()
                        mock_collusion_mgr.is_cartel_active.return_value = False
                        mock_collusion_mgr.get_events_for_round.return_value = []
                        mock_collusion_mgr.check_regulator_intervention.return_value = (
                            False,
                            None,
                            None,
                        )
                        mock_collusion_manager.return_value = mock_collusion_mgr

                        # Mock event logger
                        mock_event_log = Mock()
                        mock_event_logger.return_value = mock_event_log

                        run_id = run_collusion_game(
                            model="cournot",
                            rounds=10,  # Enough rounds for cartel formation
                            strategies=strategies,
                            costs=costs,
                            params=params,
                            bounds=bounds,
                            db=mock_db,
                            collusion_config=collusion_config,
                        )

                        assert run_id == "run_cartel"
                        # Check that form_cartel was called (should be called on round 5)
                        mock_collusion_mgr.form_cartel.assert_called()

    def test_run_collusion_game_exception_handling(self):
        """Test run_collusion_game with exception handling."""
        mock_db = Mock(spec=Session)

        # Mock run
        mock_run = Mock()
        mock_run.id = "run_error"
        mock_db.add.return_value = None
        mock_db.flush.return_value = None
        mock_db.rollback.return_value = None

        # Mock strategies
        mock_strategy = Mock(spec=Strategy)
        mock_strategy.next_action.side_effect = Exception("Strategy error")
        strategies = [mock_strategy, mock_strategy]

        costs = [10.0, 12.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 100.0)

        with patch(
            "src.sim.runners.collusion_runner.CollusionManager"
        ) as mock_collusion_manager:
            with patch(
                "src.sim.runners.collusion_runner.EventLogger"
            ) as mock_event_logger:
                # Mock collusion manager
                mock_collusion_mgr = Mock()
                mock_collusion_mgr.is_cartel_active.return_value = False
                mock_collusion_mgr.get_events_for_round.return_value = []
                mock_collusion_manager.return_value = mock_collusion_mgr

                # Mock event logger
                mock_event_log = Mock()
                mock_event_logger.return_value = mock_event_log

                with pytest.raises(RuntimeError) as exc_info:
                    run_collusion_game(
                        model="cournot",
                        rounds=2,
                        strategies=strategies,
                        costs=costs,
                        params=params,
                        bounds=bounds,
                        db=mock_db,
                    )
                assert "Simulation failed" in str(exc_info.value)
                mock_db.rollback.assert_called_once()
