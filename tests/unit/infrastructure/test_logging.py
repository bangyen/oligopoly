"""Tests for logging configuration and utilities."""

import logging
import time
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from src.sim.logging import (
    get_logger,
    handle_generic_error,
    handle_oligopoly_error,
    log_execution_time,
)


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


class TestLogExecutionTime:
    """Test the log_execution_time context manager."""

    def test_log_execution_time_success(self) -> None:
        """Test log_execution_time with successful operation."""
        logger = Mock(spec=logging.Logger)

        with log_execution_time(logger, "test_operation"):
            time.sleep(0.01)  # Small delay to ensure measurable time

        # Should log start and completion
        assert logger.info.call_count >= 2

        # Check that start message was logged
        start_calls = [
            call
            for call in logger.info.call_args_list
            if "Starting test_operation" in str(call)
        ]
        assert len(start_calls) > 0

        # Check that completion message was logged
        completion_calls = [
            call
            for call in logger.info.call_args_list
            if "Completed test_operation" in str(call)
        ]
        assert len(completion_calls) > 0

    def test_log_execution_time_with_exception(self) -> None:
        """Test log_execution_time with exception during operation."""
        logger = Mock(spec=logging.Logger)

        with pytest.raises(ValueError):
            with log_execution_time(logger, "failing_operation"):
                raise ValueError("Test error")

        # Should log start and error
        assert logger.info.call_count >= 1
        assert logger.error.call_count >= 1

        # Check that start message was logged
        start_calls = [
            call
            for call in logger.info.call_args_list
            if "Starting failing_operation" in str(call)
        ]
        assert len(start_calls) > 0

        # Check that error message was logged
        error_calls = [
            call
            for call in logger.error.call_args_list
            if "Error in failing_operation" in str(call)
        ]
        assert len(error_calls) > 0

    def test_log_execution_time_timing(self) -> None:
        """Test that log_execution_time measures and logs execution time."""
        logger = Mock(spec=logging.Logger)

        with log_execution_time(logger, "timing_test"):
            time.sleep(0.05)  # 50ms delay

        # Find the completion call
        completion_calls = [
            call
            for call in logger.info.call_args_list
            if "Completed timing_test" in str(call)
        ]
        assert len(completion_calls) > 0

        # Check that timing information is included
        completion_message = str(completion_calls[0])
        assert "in" in completion_message
        assert "s" in completion_message

    def test_log_execution_time_context_manager_interface(self) -> None:
        """Test that log_execution_time works as a proper context manager."""
        logger = Mock(spec=logging.Logger)

        # Should be usable as context manager
        cm = log_execution_time(logger, "context_test")
        assert hasattr(cm, "__enter__")
        assert hasattr(cm, "__exit__")

        # Should work with 'with' statement
        with cm:
            pass

        assert logger.info.call_count >= 2


class TestHandleGenericError:
    """Test the handle_generic_error function."""

    def test_handle_generic_error_logs_and_returns_http_exception(self) -> None:
        """Test that handle_generic_error logs error and returns HTTPException."""
        logger = Mock(spec=logging.Logger)
        error = ValueError("Test error")

        result = handle_generic_error(logger, error)

        # Should log the error
        logger.error.assert_called_once()
        error_call = logger.error.call_args[0][0]
        assert "Unexpected error" in error_call
        assert "Test error" in error_call

        # Should return HTTPException
        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert result.detail == "Internal server error"

    def test_handle_generic_error_with_different_errors(self) -> None:
        """Test handle_generic_error with different types of errors."""
        logger = Mock(spec=logging.Logger)

        # Test with different error types
        errors = [
            ValueError("Value error"),
            RuntimeError("Runtime error"),
            KeyError("Key error"),
            Exception("Generic exception"),
        ]

        for error in errors:
            result = handle_generic_error(logger, error)

            assert isinstance(result, HTTPException)
            assert result.status_code == 500
            assert result.detail == "Internal server error"

        # Should have logged each error
        assert logger.error.call_count == len(errors)


class TestHandleOligopolyError:
    """Test the handle_oligopoly_error function."""

    def test_handle_oligopoly_error_logs_and_returns_http_exception(self) -> None:
        """Test that handle_oligopoly_error logs error and returns HTTPException."""
        logger = Mock(spec=logging.Logger)
        error = ValueError("Oligopoly specific error")

        result = handle_oligopoly_error(logger, error)

        # Should log the error
        logger.error.assert_called_once()
        error_call = logger.error.call_args[0][0]
        assert "Application error" in error_call
        assert "Oligopoly specific error" in error_call

        # Should return HTTPException
        assert isinstance(result, HTTPException)
        assert result.status_code == 500
        assert result.detail == "Oligopoly specific error"

    def test_handle_oligopoly_error_preserves_error_message(self) -> None:
        """Test that handle_oligopoly_error preserves the original error message."""
        logger = Mock(spec=logging.Logger)
        error_message = "Custom oligopoly error message"
        error = RuntimeError(error_message)

        result = handle_oligopoly_error(logger, error)

        assert isinstance(result, HTTPException)
        assert result.detail == error_message

    def test_handle_oligopoly_error_with_different_errors(self) -> None:
        """Test handle_oligopoly_error with different types of errors."""
        logger = Mock(spec=logging.Logger)

        errors = [
            ValueError("Value error"),
            RuntimeError("Runtime error"),
            KeyError("Key error"),
            Exception("Generic exception"),
        ]

        for error in errors:
            result = handle_oligopoly_error(logger, error)

            assert isinstance(result, HTTPException)
            assert result.status_code == 500
            assert result.detail == str(error)

        # Should have logged each error
        assert logger.error.call_count == len(errors)
