"""Tests for market evolution models.

This module tests the dynamic market evolution functionality including
entry/exit dynamics, innovation, and market growth.
"""

import math
from unittest.mock import patch

import pytest

from src.sim.models.market_evolution import (
    FirmEvolution,
    MarketEvolutionConfig,
    MarketEvolutionEngine,
    MarketEvolutionState,
    create_market_evolution_engine,
)


class TestMarketEvolutionConfig:
    """Test market evolution configuration validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MarketEvolutionConfig()
        assert config.growth_rate == 0.02
        assert config.entry_cost == 100.0
        assert config.exit_threshold == -50.0
        assert config.innovation_rate == 0.1

    def test_config_validation_positive_entry_cost(self):
        """Test that entry cost must be positive."""
        with pytest.raises(ValueError, match="Entry cost must be positive"):
            MarketEvolutionConfig(entry_cost=0.0)

        with pytest.raises(ValueError, match="Entry cost must be positive"):
            MarketEvolutionConfig(entry_cost=-10.0)

    def test_config_validation_innovation_rate_bounds(self):
        """Test that innovation rate is in [0, 1]."""
        with pytest.raises(ValueError, match="Innovation rate must be in \\[0, 1\\]"):
            MarketEvolutionConfig(innovation_rate=-0.1)

        with pytest.raises(ValueError, match="Innovation rate must be in \\[0, 1\\]"):
            MarketEvolutionConfig(innovation_rate=1.1)

    def test_config_validation_valid_values(self):
        """Test valid configuration values."""
        config = MarketEvolutionConfig(
            growth_rate=0.05,
            entry_cost=200.0,
            exit_threshold=-100.0,
            innovation_rate=0.3,
        )
        assert config.growth_rate == 0.05
        assert config.entry_cost == 200.0
        assert config.exit_threshold == -100.0
        assert config.innovation_rate == 0.3


class TestFirmEvolution:
    """Test firm evolution state management."""

    def test_firm_evolution_initialization(self):
        """Test firm evolution initialization."""
        firm = FirmEvolution(firm_id=1)
        assert firm.firm_id == 1
        assert firm.age == 0
        assert firm.innovation_level == 0.0
        assert firm.experience == 0.0
        assert firm.market_share_history == []
        assert firm.profit_history == []

    def test_update_round(self):
        """Test updating firm state after a round."""
        firm = FirmEvolution(firm_id=1)
        firm.update_round(market_share=0.3, profit=50.0)

        assert firm.age == 1
        assert firm.market_share_history == [0.3]
        assert firm.profit_history == [50.0]
        assert firm.experience == 0.3

    def test_update_round_multiple_times(self):
        """Test updating firm state multiple times."""
        firm = FirmEvolution(firm_id=1)

        # Update multiple rounds
        firm.update_round(market_share=0.2, profit=30.0)
        firm.update_round(market_share=0.4, profit=60.0)
        firm.update_round(market_share=0.3, profit=45.0)

        assert firm.age == 3
        assert firm.market_share_history == [0.2, 0.4, 0.3]
        assert firm.profit_history == [30.0, 60.0, 45.0]
        assert math.isclose(firm.experience, 0.9, abs_tol=1e-6)  # Sum of market shares

    def test_history_length_limit(self):
        """Test that history length is limited to 20 entries."""
        firm = FirmEvolution(firm_id=1)

        # Add 25 rounds
        for i in range(25):
            firm.update_round(market_share=0.1, profit=10.0)

        assert len(firm.market_share_history) == 20
        assert len(firm.profit_history) == 20
        assert firm.age == 25


class TestMarketEvolutionState:
    """Test market evolution state management."""

    def test_state_initialization(self):
        """Test state initialization."""
        state = MarketEvolutionState()
        assert state.round_num == 0
        assert state.total_market_size == 100.0
        assert state.technology_level == 1.0
        assert state.num_firms == 0
        assert state.firm_evolutions == {}
        assert state.entry_history == []
        assert state.exit_history == []

    def test_add_firm(self):
        """Test adding a firm to the market."""
        state = MarketEvolutionState()
        state.add_firm(firm_id=1)

        assert state.num_firms == 1
        assert 1 in state.firm_evolutions
        assert state.entry_history == [0]
        assert isinstance(state.firm_evolutions[1], FirmEvolution)

    def test_remove_firm(self):
        """Test removing a firm from the market."""
        state = MarketEvolutionState()
        state.add_firm(firm_id=1)
        state.add_firm(firm_id=2)

        state.remove_firm(firm_id=1)

        assert state.num_firms == 1
        assert 1 not in state.firm_evolutions
        assert 2 in state.firm_evolutions
        assert state.exit_history == [0]

    def test_remove_nonexistent_firm(self):
        """Test removing a firm that doesn't exist."""
        state = MarketEvolutionState()
        state.add_firm(firm_id=1)

        # Should not raise error
        state.remove_firm(firm_id=999)

        assert state.num_firms == 1
        assert state.exit_history == []


class TestMarketEvolutionEngine:
    """Test market evolution engine functionality."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        assert engine.config == config
        assert engine.rng is not None
        assert isinstance(engine.state, MarketEvolutionState)

    def test_engine_initialization_with_seed(self):
        """Test engine initialization with specific seed."""
        config = MarketEvolutionConfig()
        engine1 = MarketEvolutionEngine(config, seed=42)
        engine2 = MarketEvolutionEngine(config, seed=42)

        # Should produce same random sequence
        assert engine1.rng.random() == engine2.rng.random()

    def test_evolve_market_basic(self):
        """Test basic market evolution."""
        config = MarketEvolutionConfig(growth_rate=0.01)
        engine = MarketEvolutionEngine(config, seed=42)

        current_firms = [1, 2]
        current_profits = [50.0, 30.0]
        current_market_shares = [0.6, 0.4]
        current_costs = [10.0, 12.0]
        current_qualities = [1.0, 1.0]
        demand_params = {"a": 100.0, "b": 1.0}

        new_firms, new_costs, new_qualities, new_demand_params = engine.evolve_market(
            current_firms,
            current_profits,
            current_market_shares,
            current_costs,
            current_qualities,
            demand_params,
        )

        assert engine.state.round_num == 1
        assert len(new_firms) >= len(current_firms)  # May have entry
        assert len(new_costs) == len(new_firms)
        assert len(new_qualities) == len(new_firms)
        assert (
            new_demand_params["a"] >= demand_params["a"]
        )  # Market growth (may be equal due to rounding)

    def test_evolve_market_growth(self):
        """Test market growth evolution."""
        config = MarketEvolutionConfig(growth_rate=0.05)
        engine = MarketEvolutionEngine(config, seed=42)

        demand_params = {"a": 100.0, "b": 1.0}
        initial_size = engine.state.total_market_size

        engine._evolve_market_growth(demand_params)

        expected_size = initial_size * 1.05
        assert engine.state.total_market_size == expected_size
        assert demand_params["a"] == 100.0 * 1.05

    def test_evolve_market_growth_alpha_beta(self):
        """Test market growth with alpha/beta parameters."""
        config = MarketEvolutionConfig(growth_rate=0.02)
        engine = MarketEvolutionEngine(config, seed=42)

        demand_params = {"alpha": 200.0, "beta": 2.0}
        engine._evolve_market_growth(demand_params)

        assert demand_params["alpha"] == 200.0 * 1.02

    def test_evolve_technology_empty_costs(self):
        """Test technology evolution with empty costs."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        costs = []
        qualities = []
        initial_tech = engine.state.technology_level

        engine._evolve_technology(costs, qualities)

        assert engine.state.technology_level == initial_tech

    @patch("random.Random.random")
    def test_evolve_technology_innovation(self, mock_random):
        """Test technology evolution with innovation."""
        config = MarketEvolutionConfig(innovation_rate=0.5)
        engine = MarketEvolutionEngine(config, seed=42)

        # Mock random to trigger innovation
        mock_random.return_value = 0.1  # Below innovation probability

        costs = [10.0, 12.0]
        qualities = [1.0, 1.0]
        initial_tech = engine.state.technology_level

        engine._evolve_technology(costs, qualities)

        assert engine.state.technology_level > initial_tech
        assert all(cost < original for cost, original in zip(costs, [10.0, 12.0]))
        assert all(
            quality > original for quality, original in zip(qualities, [1.0, 1.0])
        )

    def test_evolve_entry_exit_no_exit(self):
        """Test entry/exit evolution with no exits."""
        config = MarketEvolutionConfig(exit_threshold=-100.0)
        engine = MarketEvolutionEngine(config, seed=42)

        current_firms = [1, 2]
        current_profits = [50.0, 30.0]  # Profitable
        current_costs = [10.0, 12.0]
        current_qualities = [1.0, 1.0]

        new_firms = engine._evolve_entry_exit(
            current_firms, current_profits, current_costs, current_qualities
        )

        assert len(new_firms) >= len(current_firms)  # May have entry, no exit

    @patch("random.Random.random")
    def test_evolve_entry_exit_with_exit(self, mock_random):
        """Test entry/exit evolution with firm exits."""
        config = MarketEvolutionConfig(exit_threshold=-10.0)
        engine = MarketEvolutionEngine(config, seed=42)

        # Mock random to trigger exit
        mock_random.return_value = 0.1  # Low random value triggers exit

        current_firms = [1, 2]
        current_profits = [-20.0, 30.0]  # Firm 1 unprofitable
        current_costs = [10.0, 12.0]
        current_qualities = [1.0, 1.0]

        new_firms = engine._evolve_entry_exit(
            current_firms, current_profits, current_costs, current_qualities
        )

        # Firm 1 should exit
        assert len(new_firms) < len(current_firms)

    def test_should_enter_empty_market(self):
        """Test entry decision for empty market."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        result = engine._should_enter([], [], [])
        assert result is True

    def test_should_enter_profitable_market(self):
        """Test entry decision for profitable market."""
        config = MarketEvolutionConfig(entry_cost=50.0)
        engine = MarketEvolutionEngine(config, seed=42)

        current_firms = [1, 2]
        current_profits = [60.0, 70.0]  # Above entry cost
        current_costs = [10.0, 12.0]

        result = engine._should_enter(current_firms, current_profits, current_costs)
        assert result is True

    def test_should_enter_unprofitable_market(self):
        """Test entry decision for unprofitable market."""
        config = MarketEvolutionConfig(entry_cost=100.0)
        engine = MarketEvolutionEngine(config, seed=42)

        current_firms = [1, 2]
        current_profits = [30.0, 40.0]  # Below entry cost
        current_costs = [10.0, 12.0]

        result = engine._should_enter(current_firms, current_profits, current_costs)
        assert result is False

    def test_generate_entrant_cost_empty_costs(self):
        """Test generating entrant cost with no existing costs."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        cost = engine._generate_entrant_cost([])
        assert cost == 10.0

    def test_generate_entrant_cost_with_existing_costs(self):
        """Test generating entrant cost with existing costs."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        current_costs = [10.0, 12.0, 8.0]
        cost = engine._generate_entrant_cost(current_costs)

        # Should be around the average (10.0) with some variation
        assert cost > 0
        assert abs(cost - 10.0) < 5.0  # Within reasonable range

    def test_generate_entrant_quality_empty_qualities(self):
        """Test generating entrant quality with no existing qualities."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        quality = engine._generate_entrant_quality([])
        assert quality == 1.0

    def test_generate_entrant_quality_with_existing_qualities(self):
        """Test generating entrant quality with existing qualities."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        current_qualities = [1.0, 1.2, 0.8]
        quality = engine._generate_entrant_quality(current_qualities)

        # Should be around the average (1.0) with some variation
        assert quality > 0
        assert abs(quality - 1.0) < 0.5  # Within reasonable range

    @patch("random.Random.random")
    def test_evolve_innovation_success(self, mock_random):
        """Test successful innovation evolution."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        # Mock random to trigger innovation and success
        mock_random.side_effect = [
            0.05,
            0.1,
            0.1,
        ]  # Innovation attempt, then success for both firms

        firms = [1, 2]
        costs = [10.0, 12.0]
        qualities = [1.0, 1.0]

        # Add firm evolutions
        engine.state.add_firm(1)
        engine.state.add_firm(2)

        new_costs, new_qualities = engine._evolve_innovation(firms, costs, qualities)

        # At least one firm should have improved
        assert any(new_cost < original for new_cost, original in zip(new_costs, costs))
        assert any(
            new_quality > original
            for new_quality, original in zip(new_qualities, qualities)
        )

    def test_calculate_innovation_probability_with_history(self):
        """Test innovation probability calculation with market share history."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        firm_evolution = FirmEvolution(firm_id=1)
        firm_evolution.market_share_history = [
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
        ]  # High market share

        prob = engine._calculate_innovation_probability(firm_evolution)

        # Should be higher than default due to high market share
        assert prob > 0.1
        assert prob <= 0.3

    def test_calculate_innovation_probability_no_history(self):
        """Test innovation probability calculation with no history."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        firm_evolution = FirmEvolution(firm_id=1)

        prob = engine._calculate_innovation_probability(firm_evolution)

        assert prob == 0.1  # Default probability

    def test_apply_innovation(self):
        """Test applying innovation to a firm."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        firm_evolution = FirmEvolution(firm_id=1)
        costs = [10.0, 12.0]
        qualities = [1.0, 1.0]

        engine._apply_innovation(firm_evolution, costs, qualities, 0)

        assert firm_evolution.innovation_level == 0.1
        assert costs[0] == 10.0 * 0.95  # 5% cost reduction
        assert qualities[0] == 1.0 * 1.05  # 5% quality increase

    def test_get_evolution_metrics(self):
        """Test getting evolution metrics."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        # Add some firms and evolve
        engine.state.add_firm(1)
        engine.state.add_firm(2)
        engine.state.round_num = 5
        engine.state.total_market_size = 120.0
        engine.state.technology_level = 1.1

        metrics = engine.get_evolution_metrics()

        assert metrics["round_num"] == 5
        assert metrics["total_market_size"] == 120.0
        assert metrics["technology_level"] == 1.1
        assert metrics["num_firms"] == 2
        assert metrics["total_entries"] == 2
        assert metrics["total_exits"] == 0
        assert metrics["net_entries"] == 2

    def test_get_firm_evolution_metrics_existing_firm(self):
        """Test getting firm evolution metrics for existing firm."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        engine.state.add_firm(1)
        firm_evolution = engine.state.firm_evolutions[1]
        firm_evolution.update_round(0.3, 50.0)
        firm_evolution.update_round(0.4, 60.0)

        metrics = engine.get_firm_evolution_metrics(1)

        assert metrics is not None
        assert metrics["firm_id"] == 1
        assert metrics["age"] == 2
        assert metrics["innovation_level"] == 0.0
        assert metrics["experience"] == 0.7
        assert metrics["avg_market_share"] == 0.35
        assert metrics["avg_profit"] == 55.0

    def test_get_firm_evolution_metrics_nonexistent_firm(self):
        """Test getting firm evolution metrics for nonexistent firm."""
        config = MarketEvolutionConfig()
        engine = MarketEvolutionEngine(config, seed=42)

        metrics = engine.get_firm_evolution_metrics(999)

        assert metrics is None


class TestCreateMarketEvolutionEngine:
    """Test factory function for creating market evolution engine."""

    def test_create_with_default_config(self):
        """Test creating engine with default configuration."""
        engine = create_market_evolution_engine()

        assert isinstance(engine, MarketEvolutionEngine)
        assert isinstance(engine.config, MarketEvolutionConfig)

    def test_create_with_custom_config(self):
        """Test creating engine with custom configuration."""
        config = MarketEvolutionConfig(growth_rate=0.05, entry_cost=200.0)
        engine = create_market_evolution_engine(config)

        assert engine.config == config

    def test_create_with_seed(self):
        """Test creating engine with specific seed."""
        engine1 = create_market_evolution_engine(seed=42)
        engine2 = create_market_evolution_engine(seed=42)

        # Should produce same random sequence
        assert engine1.rng.random() == engine2.rng.random()

    def test_create_with_config_and_seed(self):
        """Test creating engine with both config and seed."""
        config = MarketEvolutionConfig(innovation_rate=0.3)
        engine = create_market_evolution_engine(config, seed=123)

        assert engine.config == config
        assert engine.rng.random() is not None
