"""Simple logging configuration for oligopoly simulation."""

import logging
import time
from contextlib import contextmanager
from typing import Generator

from fastapi import HTTPException


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def log_execution_time(
    logger: logging.Logger, operation: str
) -> Generator[None, None, None]:
    """Context manager to log execution time of operations."""
    start_time = time.time()
    logger.info(f"Starting {operation}")
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation}: {e}")
        raise
    finally:
        duration = time.time() - start_time
        logger.info(f"Completed {operation} in {duration:.3f}s")


def handle_generic_error(logger: logging.Logger, error: Exception) -> HTTPException:
    """Handle generic errors by logging and returning HTTP exception."""
    logger.error(f"Unexpected error: {error}")
    return HTTPException(status_code=500, detail="Internal server error")


def handle_oligopoly_error(logger: logging.Logger, error: Exception) -> HTTPException:
    """Handle custom errors by logging and returning HTTP exception."""
    logger.error(f"Application error: {error}")
    return HTTPException(status_code=500, detail=str(error))
