"""Tests for database configuration and session management."""

from unittest.mock import Mock, patch

from sqlalchemy.orm import Session

from src.sim.database import SessionLocal, get_db, get_settings


class TestDatabaseConfiguration:
    """Test database configuration and engine creation."""

    @patch("src.sim.database.get_settings")
    def test_engine_creation_with_default_settings(
        self, mock_get_settings: Mock
    ) -> None:
        """Test that engine is created with default settings."""
        mock_settings = Mock()
        mock_settings.database_url = "sqlite:///:memory:"
        mock_settings.debug = False
        mock_get_settings.return_value = mock_settings

        # Import after mocking to get the mocked settings
        from src.sim.database import engine

        assert engine is not None
        assert str(engine.url) == "sqlite:///:memory:"

    @patch("src.sim.database.get_settings")
    def test_engine_creation_with_debug_enabled(self, mock_get_settings: Mock) -> None:
        """Test that engine is created with debug enabled."""
        mock_settings = Mock()
        mock_settings.database_url = "sqlite:///:memory:"
        mock_settings.debug = True
        mock_get_settings.return_value = mock_settings

        # Import after mocking to get the mocked settings
        from src.sim.database import engine

        assert engine is not None
        assert engine.echo is True

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
        def raise_exception():
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

    @patch("src.sim.database.get_settings")
    def test_database_url_configuration(self, mock_get_settings: Mock) -> None:
        """Test that database URL is properly configured."""
        test_url = "postgresql://user:pass@localhost:5432/testdb"
        mock_settings = Mock()
        mock_settings.database_url = test_url
        mock_settings.debug = False
        mock_get_settings.return_value = mock_settings

        # Import after mocking
        from src.sim.database import engine

        assert str(engine.url) == test_url

    def test_settings_dependency(self) -> None:
        """Test that database module properly depends on settings."""
        # This test ensures that the database module imports and uses settings
        # The actual settings value depends on the environment
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, "database_url")
        assert hasattr(settings, "debug")
