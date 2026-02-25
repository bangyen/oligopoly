"""Tests for logging configuration and utilities."""

import logging

from src.sim.logging import get_logger


class TestGetLogger:
    """Test the get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a logger instance."""
        logger = get_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_configures_handler(self) -> None:
        """Test that get_logger configures handlers and formatters."""
        logger = get_logger("test_logger_with_handler")

        # Should have at least one handler
        assert len(logger.handlers) > 0

        # Check handler configuration
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.formatter is not None

    def test_get_logger_sets_level(self) -> None:
        """Test that get_logger sets the correct log level."""
        logger = get_logger("test_logger_level")

        assert logger.level == logging.INFO

    def test_get_logger_reuses_existing_handlers(self) -> None:
        """Test that get_logger doesn't add duplicate handlers."""
        logger1 = get_logger("test_duplicate_handlers")
        initial_handler_count = len(logger1.handlers)

        logger2 = get_logger("test_duplicate_handlers")

        # Should be the same logger instance
        assert logger1 is logger2
        # Should not have added more handlers
        assert len(logger2.handlers) == initial_handler_count

    def test_get_logger_different_names(self) -> None:
        """Test that different logger names create different loggers."""
        logger1 = get_logger("logger_one")
        logger2 = get_logger("logger_two")

        assert logger1 is not logger2
        assert logger1.name != logger2.name
