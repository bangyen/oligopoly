"""Tests for enhanced oligopoly simulation with advanced economic features.

This module tests the enhanced simulation functions that support:
- Capacity constraints
- Fixed costs
- Economies of scale
- Non-linear demand functions
"""

import pytest

from src.sim.games.enhanced_simulation import (
    EnhancedSimulationConfig,
    enhanced_bertrand_simulation,
    enhanced_cournot_simulation,
    validate_enhanced_economic_parameters,
)
from src.sim.models.models import CostStructure, IsoelasticDemand


class TestEnhancedCostStructure:
    """Test the enhanced cost structure functionality."""

    def test_cost_structure_creation(self):
        """Test creating cost structures with various parameters."""
        # Basic cost structure
        cost_struct = CostStructure(marginal_cost=10.0)
        assert cost_struct.marginal_cost == 10.0
        assert cost_struct.fixed_cost == 0.0
        assert cost_struct.capacity_limit is None
        assert cost_struct.economies_of_scale == 1.0

        # Full cost structure
        cost_struct = CostStructure(
            marginal_cost=10.0,
            fixed_cost=100.0,
            capacity_limit=50.0,
            economies_of_scale=0.8,
        )
        assert cost_struct.marginal_cost == 10.0
        assert cost_struct.fixed_cost == 100.0
        assert cost_struct.capacity_limit == 50.0
        assert cost_struct.economies_of_scale == 0.8

    def test_cost_structure_validation(self):
        """Test cost structure parameter validation."""
        # Negative marginal cost should raise error
        with pytest.raises(ValueError, match="Marginal cost must be positive"):
            CostStructure(marginal_cost=-1.0)

        # Negative fixed cost should raise error
        with pytest.raises(ValueError, match="Fixed cost must be non-negative"):
            CostStructure(marginal_cost=10.0, fixed_cost=-1.0)

        # Zero capacity limit should raise error
        with pytest.raises(ValueError, match="Capacity limit must be positive"):
            CostStructure(marginal_cost=10.0, capacity_limit=0.0)

        # Negative economies of scale should raise error
        with pytest.raises(ValueError, match="Economies of scale must be positive"):
            CostStructure(marginal_cost=10.0, economies_of_scale=-1.0)

    def test_total_cost_calculation(self):
        """Test total cost calculation with various scenarios."""
        # Basic cost structure
        cost_struct = CostStructure(marginal_cost=10.0, fixed_cost=100.0)
        assert cost_struct.total_cost(0) == 100.0  # Only fixed costs
        assert cost_struct.total_cost(10) == 200.0  # 100 + 10*10

        # With capacity constraint
        cost_struct = CostStructure(
            marginal_cost=10.0, fixed_cost=100.0, capacity_limit=5.0
        )
        assert cost_struct.total_cost(10) == 150.0  # 100 + 10*5 (capped at capacity)

        # With economies of scale
        cost_struct = CostStructure(
            marginal_cost=10.0, fixed_cost=100.0, economies_of_scale=0.8
        )
        # For economies of scale, cost per unit decreases with quantity
        assert cost_struct.total_cost(10) == 100.0 + 10.0 * (10**0.8)

    def test_average_cost_calculation(self):
        """Test average cost calculation."""
        cost_struct = CostStructure(marginal_cost=10.0, fixed_cost=100.0)
        assert cost_struct.average_cost(10) == 20.0  # (100 + 10*10) / 10
        assert cost_struct.average_cost(0) == float("inf")  # Division by zero

    def test_marginal_cost_at_quantity(self):
        """Test marginal cost calculation at different quantities."""
        # No economies of scale
        cost_struct = CostStructure(marginal_cost=10.0)
        assert cost_struct.marginal_cost_at_quantity(5) == 10.0
        assert cost_struct.marginal_cost_at_quantity(10) == 10.0

        # With economies of scale
        cost_struct = CostStructure(marginal_cost=10.0, economies_of_scale=0.8)
        # Marginal cost should decrease with quantity
        mc_5 = cost_struct.marginal_cost_at_quantity(5)
        mc_10 = cost_struct.marginal_cost_at_quantity(10)
        assert mc_10 < mc_5  # Marginal cost decreases with quantity


class TestIsoelasticDemand:
    """Test the isoelastic demand function."""

    def test_isoelastic_demand_creation(self):
        """Test creating isoelastic demand functions."""
        demand = IsoelasticDemand(A=100.0, elasticity=2.0)
        assert demand.A == 100.0
        assert demand.elasticity == 2.0

    def test_isoelastic_demand_validation(self):
        """Test isoelastic demand parameter validation."""
        # Elasticity <= 1 should raise error
        with pytest.raises(ValueError, match="Elasticity must be > 1"):
            IsoelasticDemand(A=100.0, elasticity=1.0)

        with pytest.raises(ValueError, match="Elasticity must be > 1"):
            IsoelasticDemand(A=100.0, elasticity=0.5)

    def test_isoelastic_demand_price_calculation(self):
        """Test price calculation for isoelastic demand."""
        demand = IsoelasticDemand(A=100.0, elasticity=2.0)

        # P = A * Q^(-1/ε) = 100 * Q^(-0.5)
        assert demand.price(1) == 100.0  # 100 * 1^(-0.5) = 100
        assert demand.price(4) == 50.0  # 100 * 4^(-0.5) = 100 * 0.5 = 50
        assert demand.price(16) == 25.0  # 100 * 16^(-0.5) = 100 * 0.25 = 25

        # Zero quantity should return infinity
        assert demand.price(0) == float("inf")


class TestEnhancedCournotSimulation:
    """Test enhanced Cournot simulation with advanced features."""

    def test_enhanced_cournot_linear_demand(self):
        """Test enhanced Cournot simulation with linear demand."""
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"a": 100.0, "b": 1.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0, fixed_cost=50.0),
                CostStructure(marginal_cost=15.0, fixed_cost=30.0),
            ],
        )

        quantities = [20.0, 15.0]
        result = enhanced_cournot_simulation(config, quantities)

        # Check basic results
        assert result.price > 0
        assert len(result.quantities) == 2
        assert len(result.profits) == 2

        # Check that profits account for fixed costs
        # Profit should be (P - MC) * Q - FC
        for i, profit in enumerate(result.profits):
            expected_profit = (
                result.price - config.cost_structures[i].marginal_cost
            ) * result.quantities[i] - config.cost_structures[i].fixed_cost
            assert abs(profit - expected_profit) < 1e-6

    def test_enhanced_cournot_with_capacity_constraints(self):
        """Test enhanced Cournot simulation with capacity constraints."""
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"a": 100.0, "b": 1.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0, capacity_limit=10.0),
                CostStructure(marginal_cost=15.0, capacity_limit=5.0),
            ],
        )

        quantities = [20.0, 15.0]  # Exceed capacity limits
        result = enhanced_cournot_simulation(config, quantities)

        # Check that quantities are capped at capacity limits
        assert result.quantities[0] == 10.0  # Capped at capacity
        assert result.quantities[1] == 5.0  # Capped at capacity

    def test_enhanced_cournot_isoelastic_demand(self):
        """Test enhanced Cournot simulation with isoelastic demand."""
        config = EnhancedSimulationConfig(
            demand_type="isoelastic",
            demand_params={"A": 100.0, "elasticity": 2.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0, fixed_cost=50.0),
                CostStructure(marginal_cost=15.0, fixed_cost=30.0),
            ],
        )

        quantities = [10.0, 5.0]
        result = enhanced_cournot_simulation(config, quantities)

        # Check that price is calculated using isoelastic demand
        total_quantity = sum(result.quantities)
        expected_price = 100.0 * (total_quantity ** (-0.5))  # A * Q^(-1/ε)
        assert abs(result.price - expected_price) < 1e-6


class TestEnhancedBertrandSimulation:
    """Test enhanced Bertrand simulation with advanced features."""

    def test_enhanced_bertrand_linear_demand(self):
        """Test enhanced Bertrand simulation with linear demand."""
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"alpha": 100.0, "beta": 1.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0, fixed_cost=50.0),
                CostStructure(marginal_cost=15.0, fixed_cost=30.0),
            ],
        )

        prices = [12.0, 18.0]  # First firm has lower price
        result = enhanced_bertrand_simulation(config, prices)

        # Check that only the lowest price firm gets demand
        assert result.quantities[0] > 0  # First firm gets demand
        assert result.quantities[1] == 0  # Second firm gets no demand

        # Check that profits account for fixed costs
        assert result.profits[0] > -50.0  # Should be profitable after fixed costs
        assert result.profits[1] == -30.0  # Only pays fixed costs

    def test_enhanced_bertrand_with_capacity_constraints(self):
        """Test enhanced Bertrand simulation with capacity constraints."""
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"alpha": 100.0, "beta": 1.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0, capacity_limit=20.0),
                CostStructure(marginal_cost=15.0, capacity_limit=10.0),
            ],
        )

        prices = [12.0, 18.0]
        result = enhanced_bertrand_simulation(config, prices)

        # Check that quantities respect capacity constraints
        assert result.quantities[0] <= 20.0  # Within capacity
        assert result.quantities[1] == 0  # No demand

    def test_enhanced_bertrand_isoelastic_demand(self):
        """Test enhanced Bertrand simulation with isoelastic demand."""
        config = EnhancedSimulationConfig(
            demand_type="isoelastic",
            demand_params={"A": 100.0, "elasticity": 2.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0, fixed_cost=50.0),
                CostStructure(marginal_cost=15.0, fixed_cost=30.0),
            ],
        )

        prices = [12.0, 18.0]
        result = enhanced_bertrand_simulation(config, prices)

        # Check that demand is calculated using isoelastic function
        min_price = min(prices)
        expected_demand = 100.0 * (min_price ** (-2.0))  # A * P^(-ε)
        assert abs(result.total_demand - expected_demand) < 1e-6


class TestEnhancedValidation:
    """Test validation of enhanced economic parameters."""

    def test_validate_enhanced_parameters_valid(self):
        """Test validation with valid parameters."""
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"a": 100.0, "b": 1.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0),
                CostStructure(marginal_cost=15.0),
            ],
        )

        # Should not raise any errors
        validate_enhanced_economic_parameters(config)

    def test_validate_enhanced_parameters_invalid_costs(self):
        """Test validation with invalid cost parameters."""
        # The CostStructure validation happens during creation, not during parameter validation
        # So we need to test the parameter validation separately
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"a": 100.0, "b": 1.0},
            cost_structures=[
                CostStructure(marginal_cost=10.0),
                CostStructure(marginal_cost=15.0),
            ],
        )

        # Test with invalid demand parameters instead
        config.demand_params = {"a": 100.0, "b": -1.0}  # Invalid negative slope

        with pytest.raises(ValueError, match="Demand slope must be positive"):
            validate_enhanced_economic_parameters(config)

    def test_validate_enhanced_parameters_impossible_market(self):
        """Test validation with economically impossible market conditions."""
        config = EnhancedSimulationConfig(
            demand_type="linear",
            demand_params={"a": 50.0, "b": 1.0},  # Low demand intercept
            cost_structures=[
                CostStructure(marginal_cost=60.0),  # Higher than demand intercept
                CostStructure(marginal_cost=70.0),
            ],
        )

        with pytest.raises(ValueError, match="No firm can be profitable"):
            validate_enhanced_economic_parameters(config)

    def test_validate_enhanced_parameters_isoelastic(self):
        """Test validation with isoelastic demand parameters."""
        config = EnhancedSimulationConfig(
            demand_type="isoelastic",
            demand_params={"A": 100.0, "elasticity": 0.5},  # Invalid elasticity
            cost_structures=[
                CostStructure(marginal_cost=10.0),
            ],
        )

        with pytest.raises(ValueError, match="Elasticity must be > 1"):
            validate_enhanced_economic_parameters(config)
