"""Tests for configuration management module."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.sim.config import Settings, get_settings, reload_settings


class TestSettings:
    """Test the Settings class functionality."""

    def test_default_settings(self) -> None:
        """Test that default settings are correctly set."""
        settings = Settings()

        assert settings.app_name == "Oligopoly Simulation"
        assert settings.version == "0.1.0"
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.database_url == "postgresql://user:password@localhost/oligopoly"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.cors_origins == ["*"]
        assert settings.max_rounds == 1000
        assert settings.max_firms == 20

    def test_custom_settings(self) -> None:
        """Test that custom settings can be set."""
        settings = Settings(
            app_name="Test App",
            version="1.0.0",
            environment="production",
            debug=True,
            database_url="sqlite:///test.db",
            api_host="127.0.0.1",
            api_port=9000,
            cors_origins=["http://localhost:3000"],
            max_rounds=500,
            max_firms=10,
        )

        assert settings.app_name == "Test App"
        assert settings.version == "1.0.0"
        assert settings.environment == "production"
        assert settings.debug is True
        assert settings.database_url == "sqlite:///test.db"
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 9000
        assert settings.cors_origins == ["http://localhost:3000"]
        assert settings.max_rounds == 500
        assert settings.max_firms == 10

    def test_api_port_validation(self) -> None:
        """Test that API port validation works correctly."""
        # Valid ports
        Settings(api_port=1)
        Settings(api_port=65535)
        Settings(api_port=8080)

        # Invalid ports
        with pytest.raises(ValidationError):
            Settings(api_port=0)

        with pytest.raises(ValidationError):
            Settings(api_port=65536)

        with pytest.raises(ValidationError):
            Settings(api_port=-1)

    def test_max_rounds_validation(self) -> None:
        """Test that max_rounds validation works correctly."""
        # Valid values
        Settings(max_rounds=1)
        Settings(max_rounds=10000)
        Settings(max_rounds=1000)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(max_rounds=0)

        with pytest.raises(ValidationError):
            Settings(max_rounds=10001)

    def test_max_firms_validation(self) -> None:
        """Test that max_firms validation works correctly."""
        # Valid values
        Settings(max_firms=1)
        Settings(max_firms=100)
        Settings(max_firms=20)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(max_firms=0)

        with pytest.raises(ValidationError):
            Settings(max_firms=101)

    @patch.dict(
        os.environ,
        {
            "APP_NAME": "Env Test App",
            "VERSION": "2.0.0",
            "ENVIRONMENT": "testing",
            "DEBUG": "true",
            "DATABASE_URL": "postgresql://test:test@localhost/test",
            "API_HOST": "192.168.1.1",
            "API_PORT": "3000",
            "CORS_ORIGINS": '["http://test.com", "https://test.com"]',
            "MAX_ROUNDS": "2000",
            "MAX_FIRMS": "50",
        },
    )
    def test_environment_variables(self) -> None:
        """Test that environment variables are properly loaded."""
        settings = Settings()

        assert settings.app_name == "Env Test App"
        assert settings.version == "2.0.0"
        assert settings.environment == "testing"
        assert settings.debug is True
        assert settings.database_url == "postgresql://test:test@localhost/test"
        assert settings.api_host == "192.168.1.1"
        assert settings.api_port == 3000
        assert settings.cors_origins == ["http://test.com", "https://test.com"]
        assert settings.max_rounds == 2000
        assert settings.max_firms == 50


class TestGetSettings:
    """Test the get_settings function."""

    def test_get_settings_caching(self) -> None:
        """Test that get_settings returns cached instance."""
        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance due to caching
        assert settings1 is settings2

    def test_get_settings_returns_settings_instance(self) -> None:
        """Test that get_settings returns a Settings instance."""
        get_settings.cache_clear()
        settings = get_settings()

        assert isinstance(settings, Settings)
        assert settings.app_name == "Oligopoly Simulation"


class TestReloadSettings:
    """Test the reload_settings function."""

    def test_reload_settings_clears_cache(self) -> None:
        """Test that reload_settings clears the cache and returns new instance."""
        # Get initial settings
        settings1 = get_settings()

        # Reload settings
        settings2 = reload_settings()

        # Should be different instances
        assert settings1 is not settings2
        assert isinstance(settings2, Settings)

    def test_reload_settings_returns_settings_instance(self) -> None:
        """Test that reload_settings returns a Settings instance."""
        settings = reload_settings()

        assert isinstance(settings, Settings)
        assert settings.app_name == "Oligopoly Simulation"
