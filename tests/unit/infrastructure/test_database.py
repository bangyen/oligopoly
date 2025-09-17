"""Tests for database configuration and session management."""

from unittest.mock import Mock, patch

from sqlalchemy.orm import Session

from src.sim.database import SessionLocal, get_db, get_settings


class TestDatabaseConfiguration:
    """Test database configuration and engine creation."""

    def test_engine_creation(self) -> None:
        """Test that engine is created and configured."""
        from src.sim.database import engine

        assert engine is not None
        assert hasattr(engine, "url")
        assert hasattr(engine, "echo")

    def test_engine_configuration(self) -> None:
        """Test that engine has proper configuration."""
        from src.sim.database import engine

        # Test that engine has expected attributes
        assert engine is not None
        assert str(engine.url) is not None
        assert isinstance(engine.echo, bool)

    def test_session_local_creation(self) -> None:
        """Test that SessionLocal is properly configured."""
        assert SessionLocal is not None
        assert hasattr(SessionLocal, "__call__")


class TestGetDb:
    """Test the get_db function."""

    def test_get_db_generator(self) -> None:
        """Test that get_db returns a generator."""
        db_gen = get_db()

        # Should be a generator
        assert hasattr(db_gen, "__next__")
        assert hasattr(db_gen, "__iter__")

    @patch("src.sim.database.SessionLocal")
    def test_get_db_session_management(self, mock_session_local: Mock) -> None:
        """Test that get_db properly manages database sessions."""
        # Mock session
        mock_session = Mock(spec=Session)
        mock_session_local.return_value = mock_session

        # Get database session
        db_gen = get_db()
        session = next(db_gen)

        # Verify session was created
        mock_session_local.assert_called_once()
        assert session is mock_session

        # Verify session is closed when generator is exhausted
        try:
            next(db_gen)
        except StopIteration:
            pass

        mock_session.close.assert_called_once()

    @patch("src.sim.database.SessionLocal")
    def test_get_db_exception_handling(self, mock_session_local: Mock) -> None:
        """Test that get_db properly handles exceptions."""
        # Mock session that raises an exception
        mock_session = Mock(spec=Session)
        mock_session_local.return_value = mock_session

        # Simulate an exception during session usage
        def raise_exception(*args, **kwargs):
            raise ValueError("Database error")

        mock_session.execute.side_effect = raise_exception

        # Get database session
        db_gen = get_db()
        session = next(db_gen)

        # Verify session is still closed even if exception occurs
        try:
            from sqlalchemy import text

            session.execute(text("SELECT 1"))
        except ValueError:
            pass

        # Close the generator
        try:
            next(db_gen)
        except StopIteration:
            pass

        mock_session.close.assert_called_once()

    def test_get_db_multiple_calls(self) -> None:
        """Test that multiple calls to get_db work correctly."""
        db_gen1 = get_db()
        db_gen2 = get_db()

        # Should be different generators
        assert db_gen1 is not db_gen2

        # Both should be valid generators
        assert hasattr(db_gen1, "__next__")
        assert hasattr(db_gen2, "__next__")


class TestDatabaseIntegration:
    """Test database integration scenarios."""

    def test_database_url_configuration(self) -> None:
        """Test that database URL is properly configured."""
        from src.sim.database import engine

        # Test that engine has a valid URL
        assert str(engine.url) is not None
        assert len(str(engine.url)) > 0

    def test_settings_dependency(self) -> None:
        """Test that database module properly depends on settings."""
        # This test ensures that the database module imports and uses settings
        # The actual settings value depends on the environment
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, "database_url")
        assert hasattr(settings, "debug")
