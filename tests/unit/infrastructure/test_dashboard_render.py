"""Tests that the dashboard root route actually renders.

The rest of the dashboard "smoke" coverage asserts on mock dictionaries and
never imports dashboard.main, so a broken template call could reach CI
undetected. These tests exercise the real ASGI app.
"""

import warnings

import pytest
from fastapi.testclient import TestClient

from dashboard.main import app


@pytest.fixture
def client() -> TestClient:
    """Return a test client for the dashboard app."""
    return TestClient(app, raise_server_exceptions=False)


def test_root_renders_html(client: TestClient) -> None:
    """GET / returns a rendered HTML page, not a 500."""
    response = client.get("/")

    assert response.status_code == 200, response.text[:500]
    assert "<html" in response.text.lower()


def test_root_uses_non_deprecated_template_call(client: TestClient) -> None:
    """The route must use the request-first TemplateResponse signature.

    The legacy ``TemplateResponse(name, {"request": request})`` order is
    deprecated, and starlette dispatches between the two forms with an
    isinstance check on the first argument. When two starlette copies are
    importable at once that check fails, the template name is read from the
    wrong position and rendering dies with
    ``TypeError: unhashable type: 'dict'``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        response = client.get("/")

    assert response.status_code == 200
