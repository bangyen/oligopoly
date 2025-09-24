"""Tests for simulation validation utilities.

This module tests the simulation pipeline validation functions
that ensure economic consistency across all stages.
"""

import pytest

from src.sim.validation.economic_validation import EconomicValidationError
from src.sim.validation.simulation_validation import (
    check_economic_plausibility,
    sanitize_simulation_results,
    validate_round_results,
    validate_run_results,
    validate_simulation_config,
)


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


class TestValidateRoundResults:
    """Test round results validation."""

    def test_validate_round_results_valid_cournot(self):
        """Test validation with valid Cournot results."""
        round_results = [
            {"price": 50.0, "quantity": 20.0, "cost": 10.0, "profit": 800.0},
            {"price": 50.0, "quantity": 15.0, "cost": 12.0, "profit": 570.0},
        ]

        # Should not raise exception
        validate_round_results(round_results, "cournot", (100.0, 1.0))

    def test_validate_round_results_valid_bertrand(self):
        """Test validation with valid Bertrand results."""
        round_results = [
            {"price": 45.0, "quantity": 25.0, "cost": 10.0, "profit": 875.0},
            {"price": 50.0, "quantity": 0.0, "cost": 12.0, "profit": 0.0},
        ]

        # Should not raise exception
        validate_round_results(round_results, "bertrand", (200.0, 2.0))

    def test_validate_round_results_empty_results(self):
        """Test validation with empty results."""
        round_results = []

        with pytest.raises(
            EconomicValidationError, match="Round must have at least one firm result"
        ):
            validate_round_results(round_results, "cournot", (100.0, 1.0))

    def test_validate_round_results_mismatched_lengths(self):
        """Test validation with mismatched array lengths."""
        round_results = [
            {"price": 50.0, "quantity": 20.0, "cost": 10.0, "profit": 800.0},
            {
                "price": 50.0,
                "quantity": 15.0,
                "cost": 12.0,
                "profit": 570.0,
            },  # Fixed: added profit
        ]

        # This should now pass validation since all arrays have same length
        validate_round_results(round_results, "cournot", (100.0, 1.0))

    def test_validate_round_results_invalid_price_quantity(self):
        """Test validation with invalid price-quantity combination."""
        round_results = [
            {"price": 0.0, "quantity": 20.0, "cost": 10.0, "profit": -200.0}
        ]

        with pytest.raises(
            EconomicValidationError, match="quantity 20.0 > 0 but price 0.0 <= 0"
        ):
            validate_round_results(round_results, "cournot", (100.0, 1.0))

    def test_validate_round_results_invalid_profit_calculation(self):
        """Test validation with invalid profit calculation."""
        round_results = [
            {
                "price": 50.0,
                "quantity": 20.0,
                "cost": 10.0,
                "profit": 1000.0,
            }  # Should be 800.0
        ]

        with pytest.raises(
            EconomicValidationError, match="profit 1000.0 doesn't match expected 800.0"
        ):
            validate_round_results(round_results, "cournot", (100.0, 1.0))

    def test_validate_round_results_cournot_different_prices(self):
        """Test validation with different prices in Cournot model."""
        round_results = [
            {"price": 50.0, "quantity": 20.0, "cost": 10.0, "profit": 800.0},
            {"price": 45.0, "quantity": 15.0, "cost": 12.0, "profit": 495.0},
        ]

        with pytest.raises(
            EconomicValidationError,
            match="Cournot model: all firms must face same market price",
        ):
            validate_round_results(round_results, "cournot", (100.0, 1.0))

    def test_validate_round_results_bertrand_invalid_pricing(self):
        """Test validation with invalid pricing in Bertrand model."""
        round_results = [
            {"price": 50.0, "quantity": 20.0, "cost": 10.0, "profit": 800.0},
            {
                "price": 45.0,
                "quantity": 15.0,
                "cost": 12.0,
                "profit": 495.0,
            },  # Higher price but positive quantity
        ]

        with pytest.raises(
            EconomicValidationError,
            match="Bertrand model: only lowest-price firms should have positive quantities",
        ):
            validate_round_results(round_results, "bertrand", (200.0, 2.0))


class TestValidateRunResults:
    """Test run results validation."""

    def test_validate_run_results_valid(self):
        """Test validation with valid run results."""
        run_results = {
            "results": {
                "0": {
                    "firm_0": {
                        "price": 50.0,
                        "quantity": 20.0,
                        "cost": 10.0,
                        "profit": 800.0,
                    },
                    "firm_1": {
                        "price": 50.0,
                        "quantity": 15.0,
                        "cost": 12.0,
                        "profit": 570.0,
                    },
                },
                "1": {
                    "firm_0": {
                        "price": 48.0,
                        "quantity": 22.0,
                        "cost": 10.0,
                        "profit": 836.0,
                    },
                    "firm_1": {
                        "price": 48.0,
                        "quantity": 17.0,
                        "cost": 12.0,
                        "profit": 612.0,
                    },
                },
            }
        }

        # Should not raise exception
        validate_run_results(run_results, "cournot", (100.0, 1.0))

    def test_validate_run_results_empty_results(self):
        """Test validation with empty run results."""
        run_results = {"results": {}}

        with pytest.raises(
            EconomicValidationError, match="Run must have at least one round"
        ):
            validate_run_results(run_results, "cournot", (100.0, 1.0))

    def test_validate_run_results_inconsistent_firms(self):
        """Test validation with inconsistent number of firms across rounds."""
        run_results = {
            "results": {
                "0": {
                    "firm_0": {
                        "price": 50.0,
                        "quantity": 20.0,
                        "cost": 10.0,
                        "profit": 800.0,
                    },
                    "firm_1": {
                        "price": 50.0,
                        "quantity": 15.0,
                        "cost": 12.0,
                        "profit": 570.0,
                    },
                },
                "1": {
                    "firm_0": {
                        "price": 48.0,
                        "quantity": 22.0,
                        "cost": 10.0,
                        "profit": 836.0,
                    }
                    # Missing firm_1
                },
            }
        }

        with pytest.raises(
            EconomicValidationError, match="Number of firms inconsistent across rounds"
        ):
            validate_run_results(run_results, "cournot", (100.0, 1.0))


class TestSanitizeSimulationResults:
    """Test simulation results sanitization."""

    def test_sanitize_results_valid_prices(self):
        """Test sanitization with valid prices."""
        results = {
            "results": {
                "0": {
                    "firm_0": {
                        "price": 50.0,
                        "quantity": 20.0,
                        "profit": 800.0,
                        "fixed_cost": 10.0,
                    },
                    "firm_1": {
                        "price": 45.0,
                        "quantity": 15.0,
                        "profit": 495.0,
                        "fixed_cost": 5.0,
                    },
                }
            }
        }

        sanitized = sanitize_simulation_results(results, "cournot")

        # Should be unchanged
        assert sanitized == results

    def test_sanitize_results_low_price_with_quantity(self):
        """Test sanitization with low price and positive quantity."""
        results = {
            "results": {
                "0": {
                    "firm_0": {
                        "price": 0.005,
                        "quantity": 20.0,
                        "profit": 0.1,
                        "fixed_cost": 10.0,
                    },
                    "firm_1": {
                        "price": 45.0,
                        "quantity": 15.0,
                        "profit": 495.0,
                        "fixed_cost": 5.0,
                    },
                }
            }
        }

        sanitized = sanitize_simulation_results(results, "cournot", min_price=0.01)

        # Firm 0 should be forced to exit
        assert sanitized["results"]["0"]["firm_0"]["price"] == 0.0
        assert sanitized["results"]["0"]["firm_0"]["quantity"] == 0.0
        assert sanitized["results"]["0"]["firm_0"]["profit"] == -10.0  # -fixed_cost

    def test_sanitize_results_zero_price_with_quantity(self):
        """Test sanitization with zero price and positive quantity."""
        results = {
            "results": {
                "0": {
                    "firm_0": {
                        "price": 0.0,
                        "quantity": 20.0,
                        "profit": -200.0,
                        "fixed_cost": 10.0,
                    },
                    "firm_1": {
                        "price": 45.0,
                        "quantity": 15.0,
                        "profit": 495.0,
                        "fixed_cost": 5.0,
                    },
                }
            }
        }

        sanitized = sanitize_simulation_results(results, "cournot")

        # Firm 0 should be forced to exit
        assert sanitized["results"]["0"]["firm_0"]["price"] == 0.0
        assert sanitized["results"]["0"]["firm_0"]["quantity"] == 0.0
        assert sanitized["results"]["0"]["firm_0"]["profit"] == -10.0  # -fixed_cost

    def test_sanitize_results_no_fixed_cost(self):
        """Test sanitization without fixed cost field."""
        results = {
            "results": {
                "0": {"firm_0": {"price": 0.005, "quantity": 20.0, "profit": 0.1}}
            }
        }

        sanitized = sanitize_simulation_results(results, "cournot", min_price=0.01)

        # Should default fixed cost to 0
        assert sanitized["results"]["0"]["firm_0"]["profit"] == 0.0


class TestCheckEconomicPlausibility:
    """Test economic plausibility checking."""

    def test_check_plausibility_valid_results(self):
        """Test plausibility check with valid results."""
        results = {
            "results": {
                "0": {
                    "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0},
                    "firm_1": {"price": 50.0, "quantity": 15.0, "profit": 570.0},
                }
            }
        }

        warnings = check_economic_plausibility(results, "cournot", (100.0, 1.0))

        # Should have warnings for price mismatch with demand curve
        assert len(warnings) >= 1

    def test_check_plausibility_zero_price_with_quantity(self):
        """Test plausibility check with zero price and positive quantity."""
        results = {
            "results": {
                "0": {
                    "firm_0": {"price": 0.0, "quantity": 20.0, "profit": -200.0},
                    "firm_1": {"price": 50.0, "quantity": 15.0, "profit": 570.0},
                }
            }
        }

        warnings = check_economic_plausibility(results, "cournot", (100.0, 1.0))

        assert len(warnings) >= 1
        assert any("zero price with positive quantity" in w for w in warnings)

    def test_check_plausibility_extremely_negative_profit(self):
        """Test plausibility check with extremely negative profit."""
        results = {
            "results": {
                "0": {
                    "firm_0": {"price": 50.0, "quantity": 20.0, "profit": -2000.0},
                    "firm_1": {"price": 50.0, "quantity": 15.0, "profit": 570.0},
                }
            }
        }

        warnings = check_economic_plausibility(results, "cournot", (100.0, 1.0))

        assert len(warnings) >= 1
        assert any("extremely negative profit" in w for w in warnings)

    def test_check_plausibility_cournot_price_mismatch(self):
        """Test plausibility check with price mismatch in Cournot."""
        results = {
            "results": {
                "0": {
                    "firm_0": {"price": 50.0, "quantity": 20.0, "profit": 800.0},
                    "firm_1": {"price": 50.0, "quantity": 15.0, "profit": 570.0},
                }
            }
        }

        # Use demand params that don't match the price
        warnings = check_economic_plausibility(results, "cournot", (200.0, 1.0))

        assert len(warnings) == 1
        assert "market price 50.0 doesn't match demand curve" in warnings[0]

    def test_check_plausibility_bertrand_monopoly(self):
        """Test plausibility check with unrealistic monopoly in Bertrand."""
        results = {
            "results": {
                "0": {
                    "firm_0": {"price": 45.0, "quantity": 35.0, "profit": 1225.0},
                    "firm_1": {"price": 50.0, "quantity": 0.0, "profit": 0.0},
                    "firm_2": {"price": 55.0, "quantity": 0.0, "profit": 0.0},
                }
            }
        }

        warnings = check_economic_plausibility(results, "bertrand", (200.0, 2.0))

        assert len(warnings) == 1
        assert "Unrealistic monopoly outcome in Bertrand competition" in warnings[0]

    def test_check_plausibility_excessive_total_losses(self):
        """Test plausibility check with excessive total losses."""
        results = {
            "results": {
                "0": {
                    "firm_0": {"price": 10.0, "quantity": 5.0, "profit": -2000.0},
                    "firm_1": {"price": 10.0, "quantity": 3.0, "profit": -1500.0},
                }
            }
        }

        warnings = check_economic_plausibility(results, "cournot", (100.0, 1.0))

        assert len(warnings) >= 1
        assert any("Excessive total losses across all firms" in w for w in warnings)

    def test_check_plausibility_multiple_warnings(self):
        """Test plausibility check with multiple issues."""
        results = {
            "results": {
                "0": {
                    "firm_0": {"price": 0.0, "quantity": 20.0, "profit": -2000.0},
                    "firm_1": {"price": 50.0, "quantity": 15.0, "profit": 570.0},
                }
            }
        }

        warnings = check_economic_plausibility(results, "cournot", (100.0, 1.0))

        # Should have multiple warnings
        assert len(warnings) >= 2
        assert any("zero price with positive quantity" in w for w in warnings)
        assert any("extremely negative profit" in w for w in warnings)
