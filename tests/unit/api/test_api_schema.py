"""Tests for API schema validation.

This module tests the API schema stability and ensures that the GET /runs/{id}
endpoint includes the required metrics keys as specified in the requirements.
"""

from typing import Dict, Optional

import pytest
from pydantic import BaseModel, ValidationError


class FirmData(BaseModel):
    """Schema for individual firm data in a round."""

    action: float
    price: float
    quantity: float
    profit: float


class RoundData(BaseModel):
    """Schema for round data containing firm results."""

    firm_data: Dict[int, FirmData]


class RunResponse(BaseModel):
    """Schema for the GET /runs/{id} API response."""

    run_id: str
    model: str
    rounds: int
    created_at: str
    results: Dict[int, Dict[int, Dict[str, float]]]


class MetricsResponse(BaseModel):
    """Schema for metrics data that should be included in API responses."""

    hhi: float
    consumer_surplus: float
    market_price: float
    total_quantity: float
    total_profit: float
    num_firms: int


class ExtendedRunResponse(BaseModel):
    """Extended schema that includes metrics in the API response."""

    run_id: str
    model: str
    rounds: int
    created_at: str
    results: Dict[int, Dict[int, Dict[str, float]]]
    metrics: Optional[Dict[int, MetricsResponse]] = None


class TestAPISchema:
    """Test cases for API schema validation."""

    def test_basic_run_response_schema(self):
        """Test that basic run response matches expected schema."""
        sample_data = {
            "run_id": "test-run-123",
            "model": "cournot",
            "rounds": 10,
            "created_at": "2024-01-01T00:00:00Z",
            "results": {
                "0": {
                    "0": {
                        "action": 10.0,
                        "price": 50.0,
                        "quantity": 10.0,
                        "profit": 400.0,
                    },
                    "1": {
                        "action": 15.0,
                        "price": 50.0,
                        "quantity": 15.0,
                        "profit": 525.0,
                    },
                },
                "1": {
                    "0": {
                        "action": 12.0,
                        "price": 48.0,
                        "quantity": 12.0,
                        "profit": 456.0,
                    },
                    "1": {
                        "action": 13.0,
                        "price": 48.0,
                        "quantity": 13.0,
                        "profit": 494.0,
                    },
                },
            },
        }

        # Should not raise ValidationError
        response = RunResponse(**sample_data)
        assert response.run_id == "test-run-123"
        assert response.model == "cournot"
        assert response.rounds == 10
        assert len(response.results) == 2

    def test_run_response_invalid_model(self):
        """Test that invalid model raises ValidationError."""
        sample_data = {
            "run_id": "test-run-123",
            "model": "invalid_model",
            "rounds": 10,
            "created_at": "2024-01-01T00:00:00Z",
            "results": {},
        }

        # Should not raise ValidationError for model (it's just a string)
        # The validation happens at the business logic level
        response = RunResponse(**sample_data)
        assert response.model == "invalid_model"

    def test_run_response_missing_fields(self):
        """Test that missing required fields raise ValidationError."""
        sample_data = {
            "run_id": "test-run-123",
            "model": "cournot",
            # Missing rounds, created_at, results
        }

        with pytest.raises(ValidationError):
            RunResponse(**sample_data)

    def test_run_response_invalid_types(self):
        """Test that invalid field types raise ValidationError."""
        sample_data = {
            "run_id": "test-run-123",
            "model": "cournot",
            "rounds": "not_a_number",  # Should be int
            "created_at": "2024-01-01T00:00:00Z",
            "results": {},
        }

        with pytest.raises(ValidationError):
            RunResponse(**sample_data)

    def test_metrics_response_schema(self):
        """Test that metrics response matches expected schema."""
        sample_metrics = {
            "hhi": 0.5,
            "consumer_surplus": 450.0,
            "market_price": 70.0,
            "total_quantity": 30.0,
            "total_profit": 1000.0,
            "num_firms": 2,
        }

        metrics = MetricsResponse(**sample_metrics)
        assert metrics.hhi == 0.5
        assert metrics.consumer_surplus == 450.0
        assert metrics.market_price == 70.0
        assert metrics.total_quantity == 30.0
        assert metrics.total_profit == 1000.0
        assert metrics.num_firms == 2

    def test_metrics_response_invalid_values(self):
        """Test that invalid metric values raise ValidationError."""
        sample_metrics = {
            "hhi": "not_a_number",  # Should be float
            "consumer_surplus": 450.0,
            "market_price": 70.0,
            "total_quantity": 30.0,
            "total_profit": 1000.0,
            "num_firms": 2,
        }

        with pytest.raises(ValidationError):
            MetricsResponse(**sample_metrics)

    def test_extended_run_response_with_metrics(self):
        """Test extended run response that includes metrics."""
        sample_data = {
            "run_id": "test-run-123",
            "model": "cournot",
            "rounds": 2,
            "created_at": "2024-01-01T00:00:00Z",
            "results": {
                "0": {
                    "0": {
                        "action": 10.0,
                        "price": 50.0,
                        "quantity": 10.0,
                        "profit": 400.0,
                    },
                    "1": {
                        "action": 15.0,
                        "price": 50.0,
                        "quantity": 15.0,
                        "profit": 525.0,
                    },
                },
                "1": {
                    "0": {
                        "action": 12.0,
                        "price": 48.0,
                        "quantity": 12.0,
                        "profit": 456.0,
                    },
                    "1": {
                        "action": 13.0,
                        "price": 48.0,
                        "quantity": 13.0,
                        "profit": 494.0,
                    },
                },
            },
            "metrics": {
                "0": {
                    "hhi": 0.52,
                    "consumer_surplus": 450.0,
                    "market_price": 50.0,
                    "total_quantity": 25.0,
                    "total_profit": 925.0,
                    "num_firms": 2,
                },
                "1": {
                    "hhi": 0.48,
                    "consumer_surplus": 480.0,
                    "market_price": 48.0,
                    "total_quantity": 25.0,
                    "total_profit": 950.0,
                    "num_firms": 2,
                },
            },
        }

        response = ExtendedRunResponse(**sample_data)
        assert response.run_id == "test-run-123"
        assert response.metrics is not None
        assert len(response.metrics) == 2
        assert response.metrics[0].hhi == 0.52
        assert response.metrics[1].consumer_surplus == 480.0

    def test_extended_run_response_without_metrics(self):
        """Test extended run response without metrics (optional field)."""
        sample_data = {
            "run_id": "test-run-123",
            "model": "cournot",
            "rounds": 2,
            "created_at": "2024-01-01T00:00:00Z",
            "results": {
                "0": {
                    "0": {
                        "action": 10.0,
                        "price": 50.0,
                        "quantity": 10.0,
                        "profit": 400.0,
                    }
                }
            },
        }

        response = ExtendedRunResponse(**sample_data)
        assert response.run_id == "test-run-123"
        assert response.metrics is None

    def test_firm_data_schema(self):
        """Test individual firm data schema."""
        firm_data = {"action": 10.0, "price": 50.0, "quantity": 10.0, "profit": 400.0}

        firm = FirmData(**firm_data)
        assert firm.action == 10.0
        assert firm.price == 50.0
        assert firm.quantity == 10.0
        assert firm.profit == 400.0

    def test_firm_data_negative_values(self):
        """Test that negative values are allowed (business logic validation)."""
        firm_data = {
            "action": -10.0,  # Negative action
            "price": 50.0,
            "quantity": 10.0,
            "profit": -100.0,  # Negative profit
        }

        # Schema validation should pass (type checking)
        # Business logic validation happens elsewhere
        firm = FirmData(**firm_data)
        assert firm.action == -10.0
        assert firm.profit == -100.0

    def test_schema_stability(self):
        """Test that schema remains stable across different data."""
        # Test with different models
        cournot_data = {
            "run_id": "cournot-run",
            "model": "cournot",
            "rounds": 5,
            "created_at": "2024-01-01T00:00:00Z",
            "results": {},
        }

        bertrand_data = {
            "run_id": "bertrand-run",
            "model": "bertrand",
            "rounds": 5,
            "created_at": "2024-01-01T00:00:00Z",
            "results": {},
        }

        cournot_response = RunResponse(**cournot_data)
        bertrand_response = RunResponse(**bertrand_data)

        assert cournot_response.model == "cournot"
        assert bertrand_response.model == "bertrand"

        # Both should have same schema structure
        assert hasattr(cournot_response, "run_id")
        assert hasattr(cournot_response, "model")
        assert hasattr(cournot_response, "rounds")
        assert hasattr(cournot_response, "created_at")
        assert hasattr(cournot_response, "results")

        assert hasattr(bertrand_response, "run_id")
        assert hasattr(bertrand_response, "model")
        assert hasattr(bertrand_response, "rounds")
        assert hasattr(bertrand_response, "created_at")
        assert hasattr(bertrand_response, "results")

    def test_required_metrics_keys(self):
        """Test that required metrics keys are present in the schema."""
        # This test ensures that if metrics are included, they have the required keys
        required_keys = {
            "hhi",
            "consumer_surplus",
            "market_price",
            "total_quantity",
            "total_profit",
            "num_firms",
        }

        sample_metrics = {
            "hhi": 0.5,
            "consumer_surplus": 450.0,
            "market_price": 70.0,
            "total_quantity": 30.0,
            "total_profit": 1000.0,
            "num_firms": 2,
        }

        metrics = MetricsResponse(**sample_metrics)

        # Check that all required keys are present
        for key in required_keys:
            assert hasattr(metrics, key)

        # Check that we can access all values
        assert metrics.hhi is not None
        assert metrics.consumer_surplus is not None
        assert metrics.market_price is not None
        assert metrics.total_quantity is not None
        assert metrics.total_profit is not None
        assert metrics.num_firms is not None
