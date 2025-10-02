"""Additional tests for main.py to improve coverage.

This module tests additional edge cases, error handling, and validation
scenarios in the main FastAPI application.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import (
    _calculate_comparison_metrics,
    app,
    get_db,
)


class TestSimulateEndpointAdditional:
    """Test additional scenarios for the simulate endpoint."""

    def test_simulate_with_negative_costs(self):
        """Test simulate endpoint with negative costs."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": -10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 422
                # Check that it's a validation error
                response_data = response.json()
                assert "detail" in response_data
            finally:
                app.dependency_overrides.clear()

    def test_simulate_with_negative_fixed_costs(self):
        """Test simulate endpoint with negative fixed costs."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": -5.0}],
                "params": {"a": 100.0, "b": 1.0},
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 422
                # Check that it's a validation error for fixed_cost
                response_data = response.json()
                assert "detail" in response_data
            finally:
                app.dependency_overrides.clear()

    def test_simulate_cournot_cost_too_high(self):
        """Test simulate endpoint with costs exceeding demand intercept."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 150.0, "fixed_cost": 0.0}],  # Cost > a=100
                "params": {"a": 100.0, "b": 1.0},
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 500
                assert (
                    "Firm costs cannot exceed demand intercept"
                    in response.json()["detail"]
                )
            finally:
                app.dependency_overrides.clear()

    def test_simulate_bertrand_cost_too_high(self):
        """Test simulate endpoint with costs exceeding demand intercept."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "bertrand",
                "rounds": 10,
                "firms": [{"cost": 150.0, "fixed_cost": 0.0}],  # Cost > alpha=100
                "params": {"alpha": 100.0, "beta": 1.0},
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 500
                assert (
                    "Firm costs cannot exceed demand intercept"
                    in response.json()["detail"]
                )
            finally:
                app.dependency_overrides.clear()

    def test_simulate_cournot_flat_demand(self):
        """Test simulate endpoint with too flat demand curve."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 0.05},  # Too flat
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 500
                assert "Demand slope" in response.json()["detail"]
                assert "too flat" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_simulate_bertrand_flat_demand(self):
        """Test simulate endpoint with too flat demand curve."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "bertrand",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"alpha": 100.0, "beta": 0.05},  # Too flat
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 500
                assert "Demand slope" in response.json()["detail"]
                assert "too flat" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_simulate_with_segments_invalid_weights(self):
        """Test simulate endpoint with segments that don't sum to 1."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
                "segments": [
                    {"alpha": 100.0, "beta": 1.0, "weight": 0.6},
                    {"alpha": 80.0, "beta": 1.2, "weight": 0.3},  # Sums to 0.9
                ],
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 500
                assert "sum to 1.0" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_simulate_with_segments_invalid_alpha(self):
        """Test simulate endpoint with invalid segment alpha."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
                "segments": [
                    {"alpha": -10.0, "beta": 1.0, "weight": 0.6},  # Invalid alpha
                    {"alpha": 80.0, "beta": 1.2, "weight": 0.4},
                ],
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 422
                assert (
                    "Input should be greater than 0"
                    in response.json()["detail"][0]["msg"]
                )
            finally:
                app.dependency_overrides.clear()

    def test_simulate_with_segments_invalid_beta(self):
        """Test simulate endpoint with invalid segment beta."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
                "segments": [
                    {"alpha": 100.0, "beta": -1.0, "weight": 0.6},  # Invalid beta
                    {"alpha": 80.0, "beta": 1.2, "weight": 0.4},
                ],
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 422
                assert (
                    "Input should be greater than 0"
                    in response.json()["detail"][0]["msg"]
                )
            finally:
                app.dependency_overrides.clear()

    def test_simulate_with_segments_invalid_weight(self):
        """Test simulate endpoint with invalid segment weight."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
                "segments": [
                    {"alpha": 100.0, "beta": 1.0, "weight": 1.5},  # Invalid weight > 1
                    {"alpha": 80.0, "beta": 1.2, "weight": 0.4},
                ],
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 422
                assert (
                    "Input should be less than or equal to 1"
                    in response.json()["detail"][0]["msg"]
                )
            finally:
                app.dependency_overrides.clear()

    def test_simulate_with_segments_unrealistic_elasticity(self):
        """Test simulate endpoint with unrealistic elasticity ratio."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
                "segments": [
                    {
                        "alpha": 10.0,
                        "beta": 25.0,
                        "weight": 0.6,
                    },  # beta/alpha = 2.5 > 2.0
                    {"alpha": 80.0, "beta": 1.2, "weight": 0.4},
                ],
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                response = client.post("/simulate", json=simulation_data)
                assert response.status_code == 500
                assert "unrealistic elasticity" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_simulate_with_policy_events(self):
        """Test simulate endpoint with policy events."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
                "events": [
                    {"round_idx": 5, "policy_type": "tax", "value": 0.1},
                    {"round_idx": 8, "policy_type": "subsidy", "value": 2.0},
                ],
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                with patch("src.main.run_game") as mock_run_game:
                    mock_run_game.return_value = "run_123"
                    response = client.post("/simulate", json=simulation_data)
                    assert response.status_code == 200
                    data = response.json()
                    assert "run_id" in data
            finally:
                app.dependency_overrides.clear()

    def test_simulate_value_error_handling(self):
        """Test simulate endpoint with ValueError."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                with patch("src.main.run_game") as mock_run_game:
                    mock_run_game.side_effect = ValueError("Invalid configuration")
                    response = client.post("/simulate", json=simulation_data)
                    assert response.status_code == 400
                    assert "Invalid configuration" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()

    def test_simulate_runtime_error_handling(self):
        """Test simulate endpoint with RuntimeError."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 10,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"a": 100.0, "b": 1.0},
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                with patch("src.main.run_game") as mock_run_game:
                    mock_run_game.side_effect = RuntimeError("Simulation failed")
                    response = client.post("/simulate", json=simulation_data)
                    assert response.status_code == 500
                    assert "Simulation failed" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()


class TestDifferentiatedBertrandAdditional:
    """Test additional scenarios for differentiated Bertrand endpoint."""

    def test_differentiated_bertrand_runtime_error(self):
        """Test differentiated Bertrand endpoint with RuntimeError."""
        with TestClient(app) as client:
            simulation_data = {
                "model": "cournot",
                "rounds": 1,
                "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                "params": {"demand_model": "logit"},
            }

            with patch(
                "src.main.calculate_differentiated_nash_equilibrium"
            ) as mock_calc:
                mock_calc.side_effect = RuntimeError("Computation failed")

                response = client.post("/differentiated-bertrand", json=simulation_data)
                assert response.status_code == 500
                assert "Unexpected error" in response.json()["detail"]


class TestGetRunAdditional:
    """Test additional scenarios for get run endpoint."""

    def test_get_run_with_bertrand_model(self):
        """Test get run endpoint with Bertrand model."""
        with TestClient(app) as client:
            with patch("src.main.get_run_results") as mock_get_run_results:
                mock_get_run_results.return_value = {
                    "id": 1,
                    "model": "bertrand",
                    "rounds": 10,
                    "results": {
                        "0": {
                            "firm_0": {
                                "price": 50.0,
                                "quantity": 20.0,
                                "profit": 800.0,
                            },
                            "firm_1": {
                                "price": 45.0,
                                "quantity": 25.0,
                                "profit": 875.0,
                            },
                        }
                    },
                }

                response = client.get("/runs/1")
                assert response.status_code == 200
                data = response.json()
                assert data["model"] == "bertrand"

    def test_get_run_with_empty_results(self):
        """Test get run endpoint with empty results."""
        with TestClient(app) as client:
            with patch("src.main.get_run_results") as mock_get_run_results:
                mock_get_run_results.return_value = {
                    "id": 1,
                    "model": "cournot",
                    "rounds": 10,
                    "results": {},
                }

                response = client.get("/runs/1")
                assert response.status_code == 200
                data = response.json()
                assert "metrics" in data

    def test_get_run_runtime_error(self):
        """Test get run endpoint with RuntimeError."""
        with TestClient(app) as client:
            with patch("src.main.get_run_results") as mock_get_run_results:
                mock_get_run_results.side_effect = RuntimeError("Database error")

                response = client.get("/runs/1")
                assert response.status_code == 500
                assert "Unexpected error" in response.json()["detail"]


class TestCompareScenariosAdditional:
    """Test additional scenarios for compare endpoints."""

    def test_compare_scenarios_with_events(self):
        """Test compare scenarios with policy events."""
        with TestClient(app) as client:
            comparison_data = {
                "left_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 42,
                    "events": [{"round_idx": 5, "policy_type": "tax", "value": 0.1}],
                },
                "right_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 12.0, "fixed_cost": 0.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 43,
                    "events": [
                        {"round_idx": 5, "policy_type": "subsidy", "value": 2.0}
                    ],
                },
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                with patch("src.main.run_game") as mock_run_game:
                    mock_run_game.side_effect = ["run_1", "run_2"]
                    response = client.post("/compare", json=comparison_data)
                    assert response.status_code == 200
            finally:
                app.dependency_overrides.clear()

    def test_compare_scenarios_runtime_error(self):
        """Test compare scenarios with RuntimeError."""
        with TestClient(app) as client:
            comparison_data = {
                "left_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 10.0, "fixed_cost": 0.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 42,
                    "events": [],
                },
                "right_config": {
                    "model": "cournot",
                    "rounds": 10,
                    "firms": [{"cost": 12.0, "fixed_cost": 0.0}],
                    "params": {"a": 100.0, "b": 1.0},
                    "seed": 43,
                    "events": [],
                },
            }

            mock_db = Mock()
            app.dependency_overrides[get_db] = lambda: mock_db
            try:
                with patch("src.main.run_game") as mock_run_game:
                    mock_run_game.side_effect = RuntimeError("Simulation failed")
                    response = client.post("/compare", json=comparison_data)
                    assert response.status_code == 500
                    assert "Simulation failed" in response.json()["detail"]
            finally:
                app.dependency_overrides.clear()


class TestCalculateComparisonMetrics:
    """Test the _calculate_comparison_metrics function."""

    def test_calculate_comparison_metrics_new_format(self):
        """Test _calculate_comparison_metrics with new format."""
        run_data = {
            "results": {
                "0": {
                    "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0},
                    "firm_1": {"price": 50.0, "quantity": 15.0, "profit": 600.0},
                },
                "1": {
                    "firm_0": {"price": 51.0, "quantity": 19.0, "profit": 779.0},
                    "firm_1": {"price": 51.0, "quantity": 14.0, "profit": 546.0},
                },
            },
            "model": "cournot",
        }

        metrics = _calculate_comparison_metrics(run_data)

        assert "market_price" in metrics
        assert "total_quantity" in metrics
        assert "total_profit" in metrics
        assert "hhi" in metrics
        assert "consumer_surplus" in metrics
        assert len(metrics["market_price"]) == 2
        assert len(metrics["total_quantity"]) == 2

    def test_calculate_comparison_metrics_old_format(self):
        """Test _calculate_comparison_metrics with old format."""
        run_data = {
            "rounds_data": [
                {"round": 0, "price": 50.0, "total_qty": 35.0, "total_profit": 1400.0},
                {"round": 1, "price": 51.0, "total_qty": 33.0, "total_profit": 1325.0},
            ],
            "model": "cournot",
            "firms_data": [
                {"quantities": [20.0, 19.0], "profits": [800.0, 779.0]},
                {"quantities": [15.0, 14.0], "profits": [600.0, 546.0]},
            ],
        }

        metrics = _calculate_comparison_metrics(run_data)

        assert "market_price" in metrics
        assert "total_quantity" in metrics
        assert "total_profit" in metrics
        assert "hhi" in metrics
        assert "consumer_surplus" in metrics
        assert len(metrics["market_price"]) == 2

    def test_calculate_comparison_metrics_bertrand(self):
        """Test _calculate_comparison_metrics with Bertrand model."""
        run_data = {
            "results": {
                "0": {
                    "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0},
                    "firm_1": {"price": 45.0, "quantity": 25.0, "profit": 875.0},
                }
            },
            "model": "bertrand",
        }

        metrics = _calculate_comparison_metrics(run_data)

        assert "market_price" in metrics
        assert len(metrics["market_price"]) == 1
        # For Bertrand, market price should be the minimum price
        assert metrics["market_price"][0] == 45.0

    def test_calculate_comparison_metrics_empty_results(self):
        """Test _calculate_comparison_metrics with empty results."""
        run_data = {
            "results": {},
            "model": "cournot",
        }

        metrics = _calculate_comparison_metrics(run_data)

        assert "market_price" in metrics
        assert "total_quantity" in metrics
        assert "total_profit" in metrics
        assert "hhi" in metrics
        assert "consumer_surplus" in metrics
        assert len(metrics["market_price"]) == 0


class TestDatabaseDependencyAdditional:
    """Test additional database dependency scenarios."""

    def test_get_db_exception_handling(self):
        """Test get_db dependency with exception."""
        with patch("src.main.SessionLocal") as mock_session_local:
            mock_session_local.side_effect = Exception("Database connection failed")

            # The dependency should still yield and close properly
            db_gen = get_db()
            try:
                next(db_gen)
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Database connection failed" in str(e)
            finally:
                # Ensure cleanup happens
                try:
                    db_gen.close()
                except StopIteration:
                    pass


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_demand_segment_config_validation(self):
        """Test DemandSegmentConfig validation."""
        from src.main import DemandSegmentConfig

        # Valid configuration
        segment = DemandSegmentConfig(alpha=100.0, beta=1.0, weight=0.5)
        assert segment.alpha == 100.0
        assert segment.beta == 1.0
        assert segment.weight == 0.5

        # Test validation errors
        with pytest.raises(ValueError):
            DemandSegmentConfig(alpha=-10.0, beta=1.0, weight=0.5)

        with pytest.raises(ValueError):
            DemandSegmentConfig(alpha=100.0, beta=-1.0, weight=0.5)

        with pytest.raises(ValueError):
            DemandSegmentConfig(alpha=100.0, beta=1.0, weight=1.5)

    def test_firm_config_validation(self):
        """Test FirmConfig validation."""
        from src.main import FirmConfig

        # Valid configuration
        firm = FirmConfig(cost=10.0, fixed_cost=5.0)
        assert firm.cost == 10.0
        assert firm.fixed_cost == 5.0

        # Test validation errors
        with pytest.raises(ValueError):
            FirmConfig(cost=-10.0, fixed_cost=5.0)

        with pytest.raises(ValueError):
            FirmConfig(cost=10.0, fixed_cost=-5.0)

    def test_policy_event_request_validation(self):
        """Test PolicyEventRequest validation."""
        from src.main import PolicyEventRequest
        from src.sim.policy.policy_shocks import PolicyType

        # Valid configuration
        event = PolicyEventRequest(round_idx=5, policy_type=PolicyType.TAX, value=10.0)
        assert event.round_idx == 5
        assert event.policy_type == PolicyType.TAX
        assert event.value == 10.0

        # Test validation errors
        with pytest.raises(ValueError):
            PolicyEventRequest(round_idx=-1, policy_type=PolicyType.TAX, value=10.0)

        with pytest.raises(ValueError):
            PolicyEventRequest(round_idx=5, policy_type=PolicyType.TAX, value=-10.0)

    def test_advanced_strategy_config_validation(self):
        """Test AdvancedStrategyConfig validation."""
        from src.main import AdvancedStrategyConfig

        # Valid configuration
        strategy = AdvancedStrategyConfig(
            strategy_type="fictitious_play", learning_rate=0.1, memory_length=10
        )
        assert strategy.strategy_type == "fictitious_play"
        assert strategy.learning_rate == 0.1
        assert strategy.memory_length == 10

        # Test validation errors
        with pytest.raises(ValueError):
            AdvancedStrategyConfig(
                strategy_type="invalid", learning_rate=0.1, memory_length=10
            )

        with pytest.raises(ValueError):
            AdvancedStrategyConfig(
                strategy_type="fictitious_play", learning_rate=1.5, memory_length=10
            )

        with pytest.raises(ValueError):
            AdvancedStrategyConfig(
                strategy_type="fictitious_play", learning_rate=0.1, memory_length=0
            )

    def test_enhanced_demand_config_validation(self):
        """Test EnhancedDemandConfig validation."""
        from src.main import EnhancedDemandConfig

        # Valid configuration
        demand = EnhancedDemandConfig(demand_type="linear", elasticity=2.0)
        assert demand.demand_type == "linear"
        assert demand.elasticity == 2.0

        # Test validation errors
        with pytest.raises(ValueError):
            EnhancedDemandConfig(demand_type="invalid", elasticity=2.0)

        with pytest.raises(ValueError):
            EnhancedDemandConfig(demand_type="ces", elasticity=0.5)

    def test_simulation_request_validation(self):
        """Test SimulationRequest validation."""
        from src.main import SimulationRequest

        # Valid configuration
        request = SimulationRequest(
            model="cournot",
            rounds=10,
            firms=[{"cost": 10.0, "fixed_cost": 0.0}],
            params={"a": 100.0, "b": 1.0},
        )
        assert request.model == "cournot"
        assert request.rounds == 10
        assert len(request.firms) == 1

        # Test validation errors
        with pytest.raises(ValueError):
            SimulationRequest(
                model="invalid", rounds=10, firms=[{"cost": 10.0}], params={}
            )

        with pytest.raises(ValueError):
            SimulationRequest(
                model="cournot", rounds=0, firms=[{"cost": 10.0}], params={}
            )

        with pytest.raises(ValueError):
            SimulationRequest(model="cournot", rounds=10, firms=[], params={})

        with pytest.raises(ValueError):
            SimulationRequest(
                model="cournot", rounds=10, firms=[{"cost": 10.0}] * 11, params={}
            )

    def test_heatmap_request_validation(self):
        """Test HeatmapRequest validation."""
        from src.main import HeatmapRequest

        # Valid configuration
        request = HeatmapRequest(
            model="cournot",
            firm_i=0,
            firm_j=1,
            grid_size=20,
            action_range=(10.0, 50.0),
            other_actions=[15.0],
            params={"a": 100.0, "b": 1.0},
            firms=[
                {"cost": 10.0, "fixed_cost": 0.0},
                {"cost": 12.0, "fixed_cost": 0.0},
                {"cost": 15.0, "fixed_cost": 0.0},
            ],
        )
        assert request.model == "cournot"
        assert request.firm_i == 0
        assert request.firm_j == 1
        assert request.grid_size == 20

        # Test validation errors
        with pytest.raises(ValueError):
            HeatmapRequest(
                model="invalid",
                firm_i=0,
                firm_j=1,
                grid_size=20,
                action_range=(10.0, 50.0),
                other_actions=[],
                params={},
                firms=[{"cost": 10.0}, {"cost": 12.0}],
            )

        with pytest.raises(ValueError):
            HeatmapRequest(
                model="cournot",
                firm_i=-1,
                firm_j=1,
                grid_size=20,
                action_range=(10.0, 50.0),
                other_actions=[],
                params={},
                firms=[{"cost": 10.0}, {"cost": 12.0}],
            )

        with pytest.raises(ValueError):
            HeatmapRequest(
                model="cournot",
                firm_i=0,
                firm_j=1,
                grid_size=4,  # Too small
                action_range=(10.0, 50.0),
                other_actions=[],
                params={},
                firms=[{"cost": 10.0}, {"cost": 12.0}],
            )

        with pytest.raises(ValueError):
            HeatmapRequest(
                model="cournot",
                firm_i=0,
                firm_j=1,
                grid_size=20,
                action_range=(10.0, 50.0),
                other_actions=[],
                params={},
                firms=[{"cost": 10.0}],  # Too few firms
            )
