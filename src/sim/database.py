"""Database configuration and session management for oligopoly simulation.

The engine is created lazily on first use so that importing the package (or
the API) never opens a database connection or requires a driver.
"""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings


def normalize_database_url(url: str) -> str:
    """Pin bare ``postgresql://`` URLs to the psycopg2 driver.

    SQLAlchemy 2.1 changed the default PostgreSQL driver from psycopg2 to
    psycopg (v3); this package ships psycopg2, so make the choice explicit.
    """
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix) :]
    return url


def make_engine(url: str, **kwargs: object) -> Engine:
    """Create an engine for ``url`` with the driver pinned."""
    return create_engine(normalize_database_url(url), **kwargs)  # type: ignore[arg-type]


@lru_cache
def get_engine() -> Engine:
    """Return the application engine, creating it on first call."""
    settings = get_settings()
    return make_engine(settings.database_url, echo=settings.debug)


SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal(bind=get_engine())
    try:
        yield db
    finally:
        db.close()
