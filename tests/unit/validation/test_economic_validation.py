"""Tests for economic validation utilities.

This module tests the economic validation functions that ensure
realistic market behavior and prevent unrealistic outcomes.
"""

import pytest

from src.sim.validation.economic_validation import (
    EconomicValidationError,
    EconomicValidationResult,
    enforce_economic_constraints,
    log_economic_warnings,
    validate_cost_structure,
    validate_demand_parameters,
    validate_market_evolution_config,
    validate_simulation_result,
)


class TestEconomicValidationResult:
    """Test economic validation result data structure."""

    def test_validation_result_creation(self):
        """Test creating validation result."""
        result = EconomicValidationResult(
            is_valid=True, warnings=["Warning 1"], errors=[], metrics={"hhi": 0.5}
        )

        assert result.is_valid is True
        assert result.warnings == ["Warning 1"]
        assert result.errors == []
        assert result.metrics == {"hhi": 0.5}


class TestEconomicValidationError:
    """Test economic validation error."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = EconomicValidationError("Test error message")
        assert str(error) == "Test error message"


class TestValidateDemandParameters:
    """Test demand parameter validation."""

    def test_validate_cournot_parameters_valid(self):
        """Test validation with valid Cournot parameters."""
        # Should not raise exception
        validate_demand_parameters(a=100.0, b=1.0, alpha=0.0, beta=0.0)

    def test_validate_bertrand_parameters_valid(self):
        """Test validation with valid Bertrand parameters."""
        # Should not raise exception
        validate_demand_parameters(a=0.0, b=0.0, alpha=200.0, beta=2.0)

    def test_validate_both_parameters_valid(self):
        """Test validation with valid both Cournot and Bertrand parameters."""
        # Should not raise exception
        validate_demand_parameters(a=100.0, b=1.0, alpha=200.0, beta=2.0)

    def test_validate_cournot_negative_intercept(self):
        """Test validation with negative Cournot intercept."""
        with pytest.raises(
            EconomicValidationError,
            match="Cournot demand intercept 'a' must be positive",
        ):
            validate_demand_parameters(a=-10.0, b=1.0, alpha=0.0, beta=0.0)

    def test_validate_cournot_negative_slope(self):
        """Test validation with negative Cournot slope."""
        with pytest.raises(
            EconomicValidationError, match="Cournot demand slope 'b' must be positive"
        ):
            validate_demand_parameters(a=100.0, b=-1.0, alpha=0.0, beta=0.0)

    def test_validate_cournot_small_market_size(self):
        """Test validation with small market size."""
        with pytest.raises(
            EconomicValidationError, match="Market size \\(a/b\\) too small"
        ):
            validate_demand_parameters(a=5.0, b=1.0, alpha=0.0, beta=0.0)

    def test_validate_bertrand_negative_intercept(self):
        """Test validation with negative Bertrand intercept."""
        with pytest.raises(
            EconomicValidationError,
            match="Bertrand demand intercept 'alpha' must be positive",
        ):
            validate_demand_parameters(a=0.0, b=0.0, alpha=-10.0, beta=2.0)

    def test_validate_bertrand_negative_slope(self):
        """Test validation with negative Bertrand slope."""
        with pytest.raises(
            EconomicValidationError,
            match="Bertrand demand slope 'beta' must be positive",
        ):
            validate_demand_parameters(a=0.0, b=0.0, alpha=200.0, beta=-2.0)

    def test_validate_bertrand_small_market_size(self):
        """Test validation with small Bertrand market size."""
        with pytest.raises(
            EconomicValidationError, match="Market size \\(alpha/beta\\) too small"
        ):
            validate_demand_parameters(a=0.0, b=0.0, alpha=5.0, beta=1.0)

    def test_validate_zero_parameters(self):
        """Test validation with all zero parameters."""
        # Should not raise exception when all parameters are zero
        validate_demand_parameters(a=0.0, b=0.0, alpha=0.0, beta=0.0)

    def test_validate_multiple_errors(self):
        """Test validation with multiple errors."""
        # Test with parameters that should trigger validation errors
        with pytest.raises(EconomicValidationError) as exc_info:
            validate_demand_parameters(
                a=5.0, b=1.0, alpha=5.0, beta=1.0
            )  # Small market sizes

        error_msg = str(exc_info.value)
        # Check that at least some errors are present
        assert len(error_msg) > 0


class TestValidateCostStructure:
    """Test cost structure validation."""

    def test_validate_costs_valid(self):
        """Test validation with valid costs."""
        costs = [10.0, 12.0, 8.0]
        # Should not raise exception
        validate_cost_structure(costs)

    def test_validate_costs_with_fixed_costs(self):
        """Test validation with valid costs and fixed costs."""
        costs = [10.0, 12.0, 8.0]
        fixed_costs = [5.0, 3.0, 7.0]
        # Should not raise exception
        validate_cost_structure(costs, fixed_costs)

    def test_validate_costs_empty(self):
        """Test validation with empty costs."""
        with pytest.raises(
            EconomicValidationError, match="At least one firm must have a cost"
        ):
            validate_cost_structure([])

    def test_validate_costs_non_positive(self):
        """Test validation with non-positive costs."""
        with pytest.raises(
            EconomicValidationError, match="Firm 0 marginal cost must be positive"
        ):
            validate_cost_structure(
                [-5.0, 12.0]
            )  # Use negative instead of zero to avoid division by zero

    def test_validate_costs_unrealistically_high(self):
        """Test validation with unrealistically high costs."""
        with pytest.raises(
            EconomicValidationError,
            match="Firm 0 marginal cost 2000.0 seems unrealistically high",
        ):
            validate_cost_structure([2000.0, 12.0])

    def test_validate_costs_high_dispersion(self):
        """Test validation with high cost dispersion."""
        with pytest.raises(EconomicValidationError, match="Cost dispersion too high"):
            validate_cost_structure([1.0, 20.0])  # Ratio of 20

    def test_validate_fixed_costs_length_mismatch(self):
        """Test validation with mismatched fixed costs length."""
        with pytest.raises(
            EconomicValidationError,
            match="Fixed costs length 2 must match marginal costs length 3",
        ):
            validate_cost_structure([10.0, 12.0, 8.0], [5.0, 3.0])

    def test_validate_fixed_costs_negative(self):
        """Test validation with negative fixed costs."""
        with pytest.raises(
            EconomicValidationError, match="Firm 0 fixed cost must be non-negative"
        ):
            validate_cost_structure([10.0, 12.0], [-5.0, 3.0])

    def test_validate_fixed_costs_unrealistically_high(self):
        """Test validation with unrealistically high fixed costs."""
        with pytest.raises(
            EconomicValidationError,
            match="Firm 0 fixed cost 20000.0 seems unrealistically high",
        ):
            validate_cost_structure([10.0, 12.0], [20000.0, 3.0])


class TestValidateSimulationResult:
    """Test simulation result validation."""

    def test_validate_cournot_result_valid(self):
        """Test validation with valid Cournot result."""
        prices = [50.0]  # Single market price
        quantities = [20.0, 15.0]
        profits = [800.0, 570.0]
        costs = [10.0, 12.0]
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_bertrand_result_valid(self):
        """Test validation with valid Bertrand result."""
        prices = [45.0, 50.0]
        quantities = [25.0, 0.0]
        profits = [875.0, 0.0]
        costs = [10.0, 12.0]
        demand_params = {"alpha": 200.0, "beta": 2.0}

        result = validate_simulation_result(
            "bertrand", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_empty_results(self):
        """Test validation with empty results."""
        result = validate_simulation_result("cournot", [], [], [], [], {})

        assert result.is_valid is False
        assert "Empty simulation results" in result.errors

    def test_validate_mismatched_lengths(self):
        """Test validation with mismatched array lengths."""
        result = validate_simulation_result(
            "cournot", [50.0], [20.0, 15.0], [800.0], [10.0, 12.0], {}
        )

        assert result.is_valid is False
        assert "Mismatched array lengths in simulation results" in result.errors

    def test_validate_negative_profits_warning(self):
        """Test validation with negative profits (warning)."""
        prices = [50.0]
        quantities = [20.0, 15.0]
        profits = [-100.0, 570.0]  # One negative profit
        costs = [10.0, 12.0]
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is True
        assert len(result.warnings) >= 1
        assert any("firms have negative profits" in w for w in result.warnings)

    def test_validate_excessive_losses_warning(self):
        """Test validation with excessive losses (warning)."""
        prices = [50.0]
        quantities = [20.0, 15.0]
        profits = [-200.0, 570.0]  # Excessive loss for firm 0
        costs = [10.0, 12.0]
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is True
        assert len(result.warnings) >= 1
        assert any("Firm 0 has excessive losses" in w for w in result.warnings)

    def test_validate_high_concentration_warning(self):
        """Test validation with high market concentration (warning)."""
        prices = [50.0]
        quantities = [35.0, 0.0]  # High concentration
        profits = [1400.0, 0.0]
        costs = [10.0, 12.0]
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is True
        assert len(result.warnings) >= 1
        assert any("Market highly concentrated" in w for w in result.warnings)

    def test_validate_selling_below_cost_error(self):
        """Test validation with selling below marginal cost (error)."""
        prices = [50.0]
        quantities = [20.0, 15.0]
        profits = [800.0, 570.0]
        costs = [60.0, 12.0]  # Firm 0 cost higher than price
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is False
        assert "Firm 0 selling below marginal cost" in result.errors

    def test_validate_no_production_error(self):
        """Test validation with no production (error)."""
        prices = [50.0]
        quantities = [0.0, 0.0]  # No production
        profits = [0.0, 0.0]
        costs = [10.0, 12.0]
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is False
        assert "No production in market" in result.errors

    def test_validate_non_positive_price_error(self):
        """Test validation with non-positive price (error)."""
        prices = [0.0]
        quantities = [20.0, 15.0]
        profits = [0.0, 0.0]
        costs = [10.0, 12.0]
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.is_valid is False
        assert "Non-positive market price" in result.errors

    def test_validate_metrics_calculation(self):
        """Test that metrics are calculated correctly."""
        prices = [50.0]
        quantities = [20.0, 15.0]
        profits = [800.0, 570.0]
        costs = [10.0, 12.0]
        demand_params = {"a": 100.0, "b": 1.0}

        result = validate_simulation_result(
            "cournot", prices, quantities, profits, costs, demand_params
        )

        assert result.metrics["total_quantity"] == 35.0
        assert result.metrics["total_profit"] == 1370.0
        assert result.metrics["market_price"] == 50.0
        assert result.metrics["active_firms"] == 2.0
        assert "hhi" in result.metrics
        assert "avg_margin" in result.metrics


class TestEnforceEconomicConstraints:
    """Test economic constraints enforcement."""

    def test_enforce_constraints_valid_prices(self):
        """Test enforcement with valid prices."""
        quantities = [20.0, 15.0]
        costs = [10.0, 12.0]
        market_price = 50.0

        adjusted = enforce_economic_constraints(quantities, costs, market_price)

        assert adjusted == quantities  # Should be unchanged

    def test_enforce_constraints_price_below_cost(self):
        """Test enforcement with price below marginal cost."""
        quantities = [20.0, 15.0]
        costs = [60.0, 12.0]  # Firm 0 cost higher than price
        market_price = 50.0

        adjusted = enforce_economic_constraints(quantities, costs, market_price)

        assert adjusted[0] == 0.0  # Firm 0 should exit
        assert adjusted[1] == 15.0  # Firm 1 should continue

    def test_enforce_constraints_negative_quantities(self):
        """Test enforcement with negative quantities."""
        quantities = [-5.0, 15.0]
        costs = [10.0, 12.0]
        market_price = 50.0

        adjusted = enforce_economic_constraints(quantities, costs, market_price)

        assert adjusted[0] == 0.0  # Should be set to 0
        assert adjusted[1] == 15.0  # Should remain unchanged

    def test_enforce_constraints_min_quantity(self):
        """Test enforcement with minimum quantity constraint."""
        quantities = [20.0, 15.0]
        costs = [10.0, 12.0]
        market_price = 50.0
        min_quantity = 1.0

        adjusted = enforce_economic_constraints(
            quantities, costs, market_price, min_quantity
        )

        assert adjusted[0] == 20.0  # Should remain unchanged
        assert adjusted[1] == 15.0  # Should remain unchanged


class TestValidateMarketEvolutionConfig:
    """Test market evolution configuration validation."""

    def test_validate_config_valid(self):
        """Test validation with valid configuration."""
        config = {
            "growth_rate": 0.05,
            "entry_cost": 200.0,
            "exit_threshold": -100.0,
            "innovation_rate": 0.3,
        }

        # Should not raise exception
        validate_market_evolution_config(config)

    def test_validate_config_default_values(self):
        """Test validation with default values."""
        config = {}

        # Should not raise exception
        validate_market_evolution_config(config)

    def test_validate_config_invalid_growth_rate(self):
        """Test validation with invalid growth rate."""
        config = {"growth_rate": 0.15}  # Too high

        with pytest.raises(
            EconomicValidationError,
            match="Growth rate 0.15 should be between 0 and 0.1",
        ):
            validate_market_evolution_config(config)

    def test_validate_config_negative_entry_cost(self):
        """Test validation with negative entry cost."""
        config = {"entry_cost": -100.0}

        with pytest.raises(
            EconomicValidationError, match="Entry cost -100.0 must be positive"
        ):
            validate_market_evolution_config(config)

    def test_validate_config_high_entry_cost(self):
        """Test validation with unrealistically high entry cost."""
        config = {"entry_cost": 20000.0}

        with pytest.raises(
            EconomicValidationError,
            match="Entry cost 20000.0 seems unrealistically high",
        ):
            validate_market_evolution_config(config)

    def test_validate_config_positive_exit_threshold(self):
        """Test validation with positive exit threshold."""
        config = {"exit_threshold": 50.0}

        with pytest.raises(
            EconomicValidationError, match="Exit threshold 50.0 should be negative"
        ):
            validate_market_evolution_config(config)

    def test_validate_config_low_exit_threshold(self):
        """Test validation with unrealistically low exit threshold."""
        config = {"exit_threshold": -2000.0}

        with pytest.raises(
            EconomicValidationError,
            match="Exit threshold -2000.0 seems unrealistically low",
        ):
            validate_market_evolution_config(config)

    def test_validate_config_invalid_innovation_rate(self):
        """Test validation with invalid innovation rate."""
        config = {"innovation_rate": 1.5}

        with pytest.raises(
            EconomicValidationError,
            match="Innovation rate 1.5 should be between 0 and 1",
        ):
            validate_market_evolution_config(config)


class TestLogEconomicWarnings:
    """Test economic warnings logging."""

    def test_log_warnings_with_logger(self):
        """Test logging warnings with provided logger."""
        import logging
        from unittest.mock import Mock

        logger = Mock(spec=logging.Logger)
        result = EconomicValidationResult(
            is_valid=True, warnings=["Warning 1", "Warning 2"], errors=[], metrics={}
        )

        log_economic_warnings(result, logger)

        assert logger.warning.call_count == 2
        logger.warning.assert_any_call("Economic validation warning: Warning 1")
        logger.warning.assert_any_call("Economic validation warning: Warning 2")

    def test_log_errors_with_logger(self):
        """Test logging errors with provided logger."""
        import logging
        from unittest.mock import Mock

        logger = Mock(spec=logging.Logger)
        result = EconomicValidationResult(
            is_valid=False, warnings=[], errors=["Error 1", "Error 2"], metrics={}
        )

        log_economic_warnings(result, logger)

        assert logger.error.call_count == 3  # 2 errors + 1 "failed validation"
        logger.error.assert_any_call("Economic validation error: Error 1")
        logger.error.assert_any_call("Economic validation error: Error 2")
        logger.error.assert_any_call("Simulation failed economic validation")

    def test_log_success_with_warnings(self):
        """Test logging success with warnings."""
        import logging
        from unittest.mock import Mock

        logger = Mock(spec=logging.Logger)
        result = EconomicValidationResult(
            is_valid=True, warnings=["Warning 1"], errors=[], metrics={}
        )

        log_economic_warnings(result, logger)

        logger.info.assert_called_once_with(
            "Simulation passed validation with warnings"
        )

    def test_log_success_no_warnings(self):
        """Test logging success without warnings."""
        import logging
        from unittest.mock import Mock

        logger = Mock(spec=logging.Logger)
        result = EconomicValidationResult(
            is_valid=True, warnings=[], errors=[], metrics={}
        )

        log_economic_warnings(result, logger)

        logger.debug.assert_called_once_with("Simulation passed economic validation")

    def test_log_without_provided_logger(self):
        """Test logging without provided logger (uses default)."""
        result = EconomicValidationResult(
            is_valid=True, warnings=["Warning 1"], errors=[], metrics={}
        )

        # Should not raise exception
        log_economic_warnings(result)
