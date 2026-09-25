"""Tests for simulation validation utilities.

This module tests the simulation pipeline validation functions
that ensure economic consistency across all stages.
"""

import pytest

from sim.validation.economic_validation import EconomicValidationError
from sim.validation.simulation_validation import validate_simulation_config


class TestValidateSimulationConfig:
    """Test simulation configuration validation."""

    def test_validate_config_with_valid_params(self):
        """Test validation with valid configuration."""
        config = {
            "params": {"a": 100.0, "b": 1.0, "alpha": 200.0, "beta": 2.0},
            "firms": [
                {"cost": 10.0, "fixed_cost": 5.0},
                {"cost": 12.0, "fixed_cost": 3.0},
            ],
        }

        # Should not raise any exception
        validate_simulation_config(config)

    def test_validate_config_missing_firms(self):
        """Test validation with missing firms."""
        config = {"params": {"a": 100.0, "b": 1.0}, "firms": []}

        with pytest.raises(
            EconomicValidationError, match="Simulation must have at least one firm"
        ):
            validate_simulation_config(config)

    def test_validate_config_no_firms_key(self):
        """Test validation with no firms key."""
        config = {"params": {"a": 100.0, "b": 1.0}}

        with pytest.raises(
            EconomicValidationError, match="Simulation must have at least one firm"
        ):
            validate_simulation_config(config)

    def test_validate_config_invalid_firm_structure(self):
        """Test validation with invalid firm structure."""
        config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": ["invalid", {"cost": 10.0}],
        }

        with pytest.raises(
            EconomicValidationError, match="Firm 0 must be a dictionary"
        ):
            validate_simulation_config(config)

    def test_validate_config_missing_cost(self):
        """Test validation with missing cost field."""
        config = {"params": {"a": 100.0, "b": 1.0}, "firms": [{"fixed_cost": 5.0}]}

        with pytest.raises(
            EconomicValidationError, match="Firm 0 must have a 'cost' field"
        ):
            validate_simulation_config(config)

    def test_validate_config_non_positive_cost(self):
        """Test validation with non-positive cost."""
        config = {"params": {"a": 100.0, "b": 1.0}, "firms": [{"cost": 0.0}]}

        with pytest.raises(
            EconomicValidationError, match="Firm 0 cost 0.0 must be positive"
        ):
            validate_simulation_config(config)

    def test_validate_config_negative_fixed_cost(self):
        """Test validation with negative fixed cost."""
        config = {
            "params": {"a": 100.0, "b": 1.0},
            "firms": [{"cost": 10.0, "fixed_cost": -5.0}],
        }

        with pytest.raises(
            EconomicValidationError, match="Firm 0 fixed cost -5.0 must be non-negative"
        ):
            validate_simulation_config(config)

    def test_validate_config_no_params(self):
        """Test validation with no params."""
        config = {"firms": [{"cost": 10.0}]}

        # Should not raise exception when no demand params
        validate_simulation_config(config)
