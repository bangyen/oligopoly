"""Tests for FastAPI health endpoint.

This module tests the health check endpoint to ensure it returns
the expected status and response format.
"""

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test that /healthz endpoint returns 200 and {"ok": true}."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
