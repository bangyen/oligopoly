"""Simple monitoring and health checks for oligopoly simulation.

This module provides basic health checks and metrics collection
for the application.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm import Session

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status response model."""

    status: str
    timestamp: datetime
    checks: Dict[str, Any]
    uptime_seconds: float
    version: str


@dataclass
class Metrics:
    """Metrics data model."""

    timestamp: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    active_simulations: int
    total_simulations: int
    memory_usage_mb: float
    cpu_usage_percent: float


class HealthChecker:
    """Health check functionality."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.start_time = time.time()

    def check_database(self, db: Session) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            # Simple query to test connection
            db.execute("SELECT 1")
            return {
                "status": "healthy",
                "message": "Database connection successful",
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "message": "Database connection failed",
            }

    def get_overall_health(self, db: Session) -> HealthStatus:
        """Get overall application health status."""
        checks = {
            "database": self.check_database(db),
        }

        # Determine overall status
        statuses = [check["status"] for check in checks.values()]

        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        uptime = time.time() - self.start_time

        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow(),
            checks=checks,
            uptime_seconds=uptime,
            version=self.settings.version,
        )


class MetricsCollector:
    """Simple metrics collection."""

    def __init__(self) -> None:
        self.request_count = 0
        self.successful_requests = 0
        self.failed_requests = 0

    def record_request(self, response_time: float, success: bool = True) -> None:
        """Record a request metric."""
        self.request_count += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def collect_metrics(self, db: Session) -> Metrics:
        """Collect basic application metrics."""
        try:
            # Database metrics
            from .models.models import Run

            total_simulations = db.query(Run).count()

            metrics = Metrics(
                timestamp=datetime.utcnow(),
                total_requests=self.request_count,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                average_response_time=0.0,  # Simplified - no tracking
                active_simulations=0,  # Simplified - not tracked
                total_simulations=total_simulations,
                memory_usage_mb=0.0,  # Simplified - not tracked
                cpu_usage_percent=0.0,  # Simplified - not tracked
            )

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return default metrics on error
            return Metrics(
                timestamp=datetime.utcnow(),
                total_requests=self.request_count,
                successful_requests=self.successful_requests,
                failed_requests=self.failed_requests,
                average_response_time=0.0,
                active_simulations=0,
                total_simulations=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
            )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get simple metrics summary."""
        return {
            "total_requests": self.request_count,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.request_count
                if self.request_count > 0
                else 0.0
            ),
        }


# Global instances
health_checker = HealthChecker()
metrics_collector = MetricsCollector()


def get_health_status(db: Session) -> HealthStatus:
    """Get current health status."""
    return health_checker.get_overall_health(db)


def get_metrics(db: Session) -> Metrics:
    """Get current metrics."""
    return metrics_collector.collect_metrics(db)


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return metrics_collector.get_metrics_summary()


def record_request_metric(response_time: float, success: bool = True) -> None:
    """Record a request metric."""
    metrics_collector.record_request(response_time, success)
