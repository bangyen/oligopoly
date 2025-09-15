"""Simple configuration management for oligopoly simulation."""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # App info
    app_name: str = Field(default="Oligopoly Simulation")
    version: str = Field(default="0.1.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # Database
    database_url: str = Field(
        default="postgresql://user:password@localhost/oligopoly",
        description="Database connection URL",
    )

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: List[str] = Field(default=["*"])

    # Simulation
    max_rounds: int = Field(default=1000, ge=1, le=10000)
    max_firms: int = Field(default=20, ge=1, le=100)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()
