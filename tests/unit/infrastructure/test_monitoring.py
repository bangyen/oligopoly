"""Tests for monitoring and health checks."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

from sqlalchemy.orm import Session

from src.sim.monitoring import (
    HealthChecker,
    HealthStatus,
    Metrics,
    MetricsCollector,
    get_health_status,
    get_metrics,
    get_metrics_summary,
    health_checker,
    metrics_collector,
    record_request_metric,
)


class TestHealthStatus:
    """Test the HealthStatus dataclass."""

    def test_health_status_creation(self) -> None:
        """Test that HealthStatus can be created with required fields."""
        timestamp = datetime.now(timezone.utc)
        checks = {"database": {"status": "healthy"}}

        health_status = HealthStatus(
            status="healthy",
            timestamp=timestamp,
            checks=checks,
            uptime_seconds=100.5,
            version="1.0.0",
        )

        assert health_status.status == "healthy"
        assert health_status.timestamp == timestamp
        assert health_status.checks == checks
        assert health_status.uptime_seconds == 100.5
        assert health_status.version == "1.0.0"

    def test_health_status_different_statuses(self) -> None:
        """Test HealthStatus with different status values."""
        timestamp = datetime.now(timezone.utc)
        checks = {"service": {"status": "unhealthy"}}

        for status in ["healthy", "degraded", "unhealthy"]:
            health_status = HealthStatus(
                status=status,
                timestamp=timestamp,
                checks=checks,
                uptime_seconds=0.0,
                version="1.0.0",
            )
            assert health_status.status == status


class TestMetrics:
    """Test the Metrics dataclass."""

    def test_metrics_creation(self) -> None:
        """Test that Metrics can be created with required fields."""
        timestamp = datetime.now(timezone.utc)

        metrics = Metrics(
            timestamp=timestamp,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            average_response_time=0.5,
            active_simulations=3,
            total_simulations=50,
            memory_usage_mb=128.5,
            cpu_usage_percent=25.0,
        )

        assert metrics.timestamp == timestamp
        assert metrics.total_requests == 100
        assert metrics.successful_requests == 95
        assert metrics.failed_requests == 5
        assert metrics.average_response_time == 0.5
        assert metrics.active_simulations == 3
        assert metrics.total_simulations == 50
        assert metrics.memory_usage_mb == 128.5
        assert metrics.cpu_usage_percent == 25.0

    def test_metrics_zero_values(self) -> None:
        """Test Metrics with zero values."""
        timestamp = datetime.now(timezone.utc)

        metrics = Metrics(
            timestamp=timestamp,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_response_time=0.0,
            active_simulations=0,
            total_simulations=0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
        )

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0


class TestHealthChecker:
    """Test the HealthChecker class."""

    def test_health_checker_initialization(self) -> None:
        """Test that HealthChecker initializes correctly."""
        with patch("src.sim.monitoring.get_settings") as mock_get_settings:
            mock_settings = Mock()
            mock_settings.version = "1.0.0"
            mock_get_settings.return_value = mock_settings

            checker = HealthChecker()

            assert checker.settings == mock_settings
            assert checker.start_time > 0

    @patch("src.sim.monitoring.get_settings")
    def test_check_database_healthy(self, mock_get_settings: Mock) -> None:
        """Test database health check when database is healthy."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        checker = HealthChecker()
        mock_db = Mock(spec=Session)

        # Mock successful database query
        mock_db.execute.return_value = None

        result = checker.check_database(mock_db)

        assert result["status"] == "healthy"
        assert "Database connection successful" in result["message"]
        mock_db.execute.assert_called_once()

    @patch("src.sim.monitoring.get_settings")
    def test_check_database_unhealthy(self, mock_get_settings: Mock) -> None:
        """Test database health check when database is unhealthy."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        checker = HealthChecker()
        mock_db = Mock(spec=Session)

        # Mock database error
        mock_db.execute.side_effect = Exception("Connection failed")

        result = checker.check_database(mock_db)

        assert result["status"] == "unhealthy"
        assert "Database connection failed" in result["message"]
        assert "Connection failed" in result["error"]

    @patch("src.sim.monitoring.get_settings")
    def test_get_overall_health_healthy(self, mock_get_settings: Mock) -> None:
        """Test overall health when all checks are healthy."""
        mock_settings = Mock()
        mock_settings.version = "1.0.0"
        mock_get_settings.return_value = mock_settings

        checker = HealthChecker()
        mock_db = Mock(spec=Session)
        mock_db.execute.return_value = None

        health_status = checker.get_overall_health(mock_db)

        assert health_status.status == "healthy"
        assert health_status.version == "1.0.0"
        assert "database" in health_status.checks
        assert health_status.checks["database"]["status"] == "healthy"
        assert health_status.uptime_seconds >= 0

    @patch("src.sim.monitoring.get_settings")
    def test_get_overall_health_unhealthy(self, mock_get_settings: Mock) -> None:
        """Test overall health when database check fails."""
        mock_settings = Mock()
        mock_settings.version = "1.0.0"
        mock_get_settings.return_value = mock_settings

        checker = HealthChecker()
        mock_db = Mock(spec=Session)
        mock_db.execute.side_effect = Exception("Database error")

        health_status = checker.get_overall_health(mock_db)

        assert health_status.status == "unhealthy"
        assert health_status.checks["database"]["status"] == "unhealthy"


class TestMetricsCollector:
    """Test the MetricsCollector class."""

    def test_metrics_collector_initialization(self) -> None:
        """Test that MetricsCollector initializes correctly."""
        collector = MetricsCollector()

        assert collector.request_count == 0
        assert collector.successful_requests == 0
        assert collector.failed_requests == 0

    def test_record_request_success(self) -> None:
        """Test recording successful requests."""
        collector = MetricsCollector()

        collector.record_request(0.5, success=True)
        collector.record_request(0.3, success=True)

        assert collector.request_count == 2
        assert collector.successful_requests == 2
        assert collector.failed_requests == 0

    def test_record_request_failure(self) -> None:
        """Test recording failed requests."""
        collector = MetricsCollector()

        collector.record_request(1.0, success=False)
        collector.record_request(0.8, success=False)

        assert collector.request_count == 2
        assert collector.successful_requests == 0
        assert collector.failed_requests == 2

    def test_record_request_mixed(self) -> None:
        """Test recording mixed success/failure requests."""
        collector = MetricsCollector()

        collector.record_request(0.5, success=True)
        collector.record_request(1.0, success=False)
        collector.record_request(0.3, success=True)

        assert collector.request_count == 3
        assert collector.successful_requests == 2
        assert collector.failed_requests == 1

    def test_collect_metrics_success(self) -> None:
        """Test collecting metrics successfully."""
        collector = MetricsCollector()
        collector.record_request(0.5, success=True)
        collector.record_request(0.3, success=False)

        mock_db = Mock(spec=Session)
        mock_query = Mock()
        mock_db.query.return_value = mock_query
        mock_query.count.return_value = 25

        with patch("src.sim.monitoring.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
            )

            metrics = collector.collect_metrics(mock_db)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.total_simulations == 25
        assert metrics.average_response_time == 0.0  # Simplified implementation
        assert metrics.active_simulations == 0  # Simplified implementation
        assert metrics.memory_usage_mb == 0.0  # Simplified implementation
        assert metrics.cpu_usage_percent == 0.0  # Simplified implementation

    def test_collect_metrics_database_error(self) -> None:
        """Test collecting metrics when database query fails."""
        collector = MetricsCollector()
        collector.record_request(0.5, success=True)

        mock_db = Mock(spec=Session)
        mock_db.query.side_effect = Exception("Database error")

        with patch("src.sim.monitoring.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
            )

            metrics = collector.collect_metrics(mock_db)

        # Should return default metrics on error
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.total_simulations == 0

    def test_get_metrics_summary(self) -> None:
        """Test getting metrics summary."""
        collector = MetricsCollector()
        collector.record_request(0.5, success=True)
        collector.record_request(0.3, success=True)
        collector.record_request(1.0, success=False)

        summary = collector.get_metrics_summary()

        assert summary["total_requests"] == 3
        assert summary["successful_requests"] == 2
        assert summary["failed_requests"] == 1
        assert summary["success_rate"] == 2 / 3

    def test_get_metrics_summary_zero_requests(self) -> None:
        """Test getting metrics summary with zero requests."""
        collector = MetricsCollector()

        summary = collector.get_metrics_summary()

        assert summary["total_requests"] == 0
        assert summary["successful_requests"] == 0
        assert summary["failed_requests"] == 0
        assert summary["success_rate"] == 0.0


class TestGlobalInstances:
    """Test global instances and convenience functions."""

    def test_global_instances_exist(self) -> None:
        """Test that global instances are created."""
        assert health_checker is not None
        assert metrics_collector is not None
        assert isinstance(health_checker, HealthChecker)
        assert isinstance(metrics_collector, MetricsCollector)

    @patch("src.sim.monitoring.health_checker")
    def test_get_health_status(self, mock_health_checker: Mock) -> None:
        """Test the get_health_status convenience function."""
        mock_db = Mock(spec=Session)
        expected_health = HealthStatus(
            status="healthy",
            timestamp=datetime.now(timezone.utc),
            checks={},
            uptime_seconds=100.0,
            version="1.0.0",
        )
        mock_health_checker.get_overall_health.return_value = expected_health

        result = get_health_status(mock_db)

        assert result == expected_health
        mock_health_checker.get_overall_health.assert_called_once_with(mock_db)

    @patch("src.sim.monitoring.metrics_collector")
    def test_get_metrics(self, mock_metrics_collector: Mock) -> None:
        """Test the get_metrics convenience function."""
        mock_db = Mock(spec=Session)
        expected_metrics = Metrics(
            timestamp=datetime.now(timezone.utc),
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            average_response_time=0.5,
            active_simulations=1,
            total_simulations=50,
            memory_usage_mb=128.0,
            cpu_usage_percent=25.0,
        )
        mock_metrics_collector.collect_metrics.return_value = expected_metrics

        result = get_metrics(mock_db)

        assert result == expected_metrics
        mock_metrics_collector.collect_metrics.assert_called_once_with(mock_db)

    @patch("src.sim.monitoring.metrics_collector")
    def test_get_metrics_summary(self, mock_metrics_collector: Mock) -> None:
        """Test the get_metrics_summary convenience function."""
        expected_summary = {
            "total_requests": 10,
            "successful_requests": 8,
            "failed_requests": 2,
            "success_rate": 0.8,
        }
        mock_metrics_collector.get_metrics_summary.return_value = expected_summary

        result = get_metrics_summary()

        assert result == expected_summary
        mock_metrics_collector.get_metrics_summary.assert_called_once()

    @patch("src.sim.monitoring.metrics_collector")
    def test_record_request_metric(self, mock_metrics_collector: Mock) -> None:
        """Test the record_request_metric convenience function."""
        record_request_metric(0.5, success=True)

        mock_metrics_collector.record_request.assert_called_once_with(0.5, True)

        # Test with default success parameter
        record_request_metric(0.3)
        mock_metrics_collector.record_request.assert_called_with(0.3, True)
