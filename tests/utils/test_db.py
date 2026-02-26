"""Test utilities for database management.

This module provides shared database setup and teardown functionality
for tests, eliminating duplication across test files.
"""

import atexit
import os
import tempfile
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.sim.models.models import Base


class TestDatabaseManager:
    """Manages temporary test databases with automatic cleanup."""

    def __init__(self):
        self._temp_files = []
        self._cleanup_registered = False

    def create_temp_database(self) -> tuple[str, Session]:
        """Create a temporary database file and return URL and session."""
        # Create temporary database file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        temp_file.close()
        self._temp_files.append(temp_file.name)

        # Register cleanup if not already done
        if not self._cleanup_registered:
            atexit.register(self._cleanup_all_temp_files)
            self._cleanup_registered = True

        # Create database URL
        database_url = f"sqlite:///{temp_file.name}"

        # Create engine and session
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = session_local()

        # Create tables
        Base.metadata.create_all(bind=engine)

        return database_url, session

    def _cleanup_all_temp_files(self) -> None:
        """Clean up all temporary database files."""
        for temp_file in self._temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        self._temp_files.clear()


# Global instance for shared use
_db_manager = TestDatabaseManager()


def create_test_database() -> tuple[str, Session]:
    """Create a temporary test database and return URL and session.

    Returns:
        Tuple of (database_url, session)
    """
    return _db_manager.create_temp_database()


def create_test_session() -> Generator[Session, None, None]:
    """Create a test database session with automatic cleanup.

    Yields:
        Database session for testing
    """
    database_url, session = create_test_database()
    try:
        yield session
    finally:
        session.close()


def override_get_db_for_testing(app, get_db_func) -> None:
    """Override FastAPI database dependency for testing.

    Args:
        app: FastAPI application instance
        get_db_func: Original get_db function to override
    """
    # Create the test database ONCE for this override
    database_url, _ = create_test_database()
    # Create a sessionmaker for this specific database
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def test_get_db() -> Generator[Session, None, None]:
        """Test database dependency override."""
        session = testing_session_local()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_func] = test_get_db
