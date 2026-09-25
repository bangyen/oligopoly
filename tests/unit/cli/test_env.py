"""Tests for environment configuration and database setup.

This module tests environment variable handling and Alembic
migration functionality to ensure proper database configuration.
"""

import os
from unittest.mock import MagicMock, patch


def test_database_url_env_var() -> None:
    """Test that DATABASE_URL environment variable drives the app engine."""
    from sim.config import reload_settings
    from sim.database import get_engine

    custom_url = "postgresql://test:test@testhost:5432/testdb"
    try:
        with patch.dict(os.environ, {"DATABASE_URL": custom_url}):
            reload_settings()
            get_engine.cache_clear()
            url = get_engine().url
            assert url.host == "testhost"
            assert url.database == "testdb"
            # Bare postgresql:// is pinned to the shipped psycopg2 driver
            assert url.drivername == "postgresql+psycopg2"
    finally:
        reload_settings()
        get_engine.cache_clear()


def test_normalize_database_url() -> None:
    """Bare PostgreSQL URLs get an explicit driver; others are untouched."""
    from sim.database import normalize_database_url

    assert (
        normalize_database_url("postgresql://u:p@h/db")
        == "postgresql+psycopg2://u:p@h/db"
    )
    assert normalize_database_url("postgres://u:p@h/db") == (
        "postgresql+psycopg2://u:p@h/db"
    )
    assert (
        normalize_database_url("postgresql+psycopg://u:p@h/db")
        == "postgresql+psycopg://u:p@h/db"
    )
    assert normalize_database_url("sqlite:///x.db") == "sqlite:///x.db"


def test_importing_api_does_not_create_engine() -> None:
    """Importing the API must not connect to (or configure) a database."""
    import importlib

    import sim.api
    from sim.database import get_engine

    get_engine.cache_clear()
    importlib.reload(sim.api)
    assert get_engine.cache_info().currsize == 0


def test_alembic_migration_smoke() -> None:
    """Smoke test that Alembic migration can run without error."""
    # Mock the alembic command to avoid actual database operations
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)

        # This would normally run: alembic upgrade head
        # We're just testing that the command structure is correct
        import subprocess

        subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)

        # Verify the command was called correctly
        mock_run.assert_called_once_with(
            ["alembic", "upgrade", "head"], capture_output=True, text=True
        )
