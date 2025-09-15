"""Tests for event logging and replay functionality.

This module provides comprehensive tests for the event system,
replay functionality, and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.main import app, get_db
from src.sim.events.event_logger import EventLogger
from src.sim.events.event_types import EventType
from src.sim.events.replay import ReplaySystem
from src.sim.models.models import Base, Run


# Test database setup
@pytest.fixture(scope="function")
def test_db():
    """Create a test database session."""
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_run(test_db):
    """Create a test run for testing."""
    run = Run(model="cournot", rounds=5)
    test_db.add(run)
    test_db.commit()
    return run


@pytest.fixture(scope="function")
def test_client(test_db):
    """Create a test client for API testing."""

    # Override the database dependency to use our test database
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestEventLogOrder:
    """Test that events are properly ordered and IDs are stable."""

    def test_event_log_order(self, test_db, test_run):
        """Test that events are sorted by round_idx and IDs are stable."""
        event_logger = EventLogger(str(test_run.id), test_db)

        # Log events in different order
        event_logger.log_event(EventType.CARTEL_FORMED, round_idx=2)
        event_logger.log_event(EventType.DEFECTION_DETECTED, round_idx=1)
        event_logger.log_event(EventType.PENALTY_IMPOSED, round_idx=3)

        test_db.commit()

        # Retrieve events
        events = event_logger.get_all_events()

        # Check ordering
        assert len(events) == 3
        assert events[0].round_idx == 1
        assert events[1].round_idx == 2
        assert events[2].round_idx == 3

        # Check IDs are stable
        event_ids = [event.id for event in events]
        assert len(set(event_ids)) == 3  # All unique

        # Check event types
        assert events[0].event_type == EventType.DEFECTION_DETECTED.value
        assert events[1].event_type == EventType.CARTEL_FORMED.value
        assert events[2].event_type == EventType.PENALTY_IMPOSED.value

    def test_event_log_with_firm_id(self, test_db, test_run):
        """Test event logging with firm IDs."""
        event_logger = EventLogger(str(test_run.id), test_db)

        event_logger.log_event(
            EventType.DEFECTION_DETECTED,
            round_idx=1,
            firm_id=2,
            event_data={"severity": "high"},
        )

        test_db.commit()

        events = event_logger.get_all_events()
        assert len(events) == 1
        assert events[0].firm_id == 2
        assert events[0].event_data["severity"] == "high"


class TestReplaySequence:
    """Test replay functionality returns correct frames and annotations."""

    def test_replay_returns_frames_1_to_t(self, test_db, test_run):
        """Test that replay returns frames 1..T with metric snapshots."""
        # Create some mock results data
        from src.sim.models.models import Result

        # Add mock results for rounds 0, 1, 2
        for round_idx in range(3):
            for firm_id in range(2):
                result = Result(
                    run_id=test_run.id,
                    round_idx=round_idx,
                    firm_id=firm_id,
                    action=10.0 + round_idx,
                    price=20.0 + round_idx,
                    qty=5.0 + round_idx,
                    profit=50.0 + round_idx * 10,
                )
                test_db.add(result)

        test_db.commit()

        # Create replay system
        replay_system = ReplaySystem(str(test_run.id), test_db)
        frames = replay_system.get_all_frames()

        # Check we get frames for rounds 0, 1, 2
        assert len(frames) == 3
        assert frames[0].round_idx == 0
        assert frames[1].round_idx == 1
        assert frames[2].round_idx == 2

        # Check frames contain metric snapshots
        for frame in frames:
            assert isinstance(frame.market_price, float)
            assert isinstance(frame.total_quantity, float)
            assert isinstance(frame.total_profit, float)
            assert isinstance(frame.hhi, float)
            assert isinstance(frame.consumer_surplus, float)
            assert isinstance(frame.firm_data, dict)
            assert len(frame.firm_data) == 2  # 2 firms

    def test_event_rounds_have_annotations(self, test_db, test_run):
        """Test that event rounds have non-empty annotations."""
        # Add events
        event_logger = EventLogger(str(test_run.id), test_db)
        event_logger.log_event(EventType.CARTEL_FORMED, round_idx=1)
        event_logger.log_event(EventType.DEFECTION_DETECTED, round_idx=2)
        test_db.commit()

        # Add mock results
        from src.sim.models.models import Result

        for round_idx in range(3):
            for firm_id in range(2):
                result = Result(
                    run_id=test_run.id,
                    round_idx=round_idx,
                    firm_id=firm_id,
                    action=10.0,
                    price=20.0,
                    qty=5.0,
                    profit=50.0,
                )
                test_db.add(result)

        test_db.commit()

        # Create replay system
        replay_system = ReplaySystem(str(test_run.id), test_db)
        frames = replay_system.get_all_frames()

        # Check event rounds have annotations
        event_frames = [f for f in frames if f.events]
        assert len(event_frames) == 2  # Rounds 1 and 2

        for frame in event_frames:
            assert len(frame.annotations) > 0
            assert len(frame.events) > 0


class TestFeedApiSchema:
    """Test that event feed API returns correct schema."""

    def test_feed_api_schema(self, test_client, test_db, test_run):
        """Test that event items include {round, type, details}."""
        # Add some events
        event_logger = EventLogger(str(test_run.id), test_db)
        event_logger.log_event(
            EventType.CARTEL_FORMED, round_idx=1, event_data={"participating_firms": 3}
        )
        event_logger.log_event(
            EventType.DEFECTION_DETECTED,
            round_idx=2,
            firm_id=1,
            event_data={"severity": "high"},
        )
        test_db.commit()

        # Test API endpoint
        response = test_client.get(f"/runs/{test_run.id}/events")
        assert response.status_code == 200

        data = response.json()
        assert "run_id" in data
        assert "total_events" in data
        assert "events" in data

        events = data["events"]
        assert len(events) == 2

        # Check event schema
        for event in events:
            assert "id" in event
            assert "round_idx" in event
            assert "event_type" in event
            assert "description" in event
            assert "created_at" in event

            # Check optional fields
            if event["round_idx"] == 2:
                assert event["firm_id"] == 1
                assert event["event_data"]["severity"] == "high"

    def test_replay_api_schema(self, test_client, test_db, test_run):
        """Test that replay API returns correct schema."""
        # Add mock results
        from src.sim.models.models import Result

        for round_idx in range(2):
            for firm_id in range(2):
                result = Result(
                    run_id=test_run.id,
                    round_idx=round_idx,
                    firm_id=firm_id,
                    action=10.0,
                    price=20.0,
                    qty=5.0,
                    profit=50.0,
                )
                test_db.add(result)

        test_db.commit()

        # Test API endpoint
        response = test_client.get(f"/runs/{test_run.id}/replay")
        assert response.status_code == 200

        data = response.json()
        assert "run_id" in data
        assert "total_frames" in data
        assert "frames_with_events" in data
        assert "event_rounds" in data
        assert "frames" in data

        frames = data["frames"]
        assert len(frames) == 2

        # Check frame schema
        for frame in frames:
            assert "round_idx" in frame
            assert "timestamp" in frame
            assert "market_price" in frame
            assert "total_quantity" in frame
            assert "total_profit" in frame
            assert "hhi" in frame
            assert "consumer_surplus" in frame
            assert "num_firms" in frame
            assert "firm_data" in frame
            assert "events" in frame
            assert "annotations" in frame


class TestEventLogger:
    """Test EventLogger functionality."""

    def test_log_collusion_event(self, test_db, test_run):
        """Test logging collusion events."""
        event_logger = EventLogger(str(test_run.id), test_db)

        event_logger.log_collusion_event(
            EventType.CARTEL_FORMED,
            round_idx=1,
            cartel_data={"participating_firms": 3, "collusive_price": 25.0},
        )

        test_db.commit()

        events = event_logger.get_all_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.CARTEL_FORMED.value
        assert events[0].event_data["category"] == "collusion"
        assert events[0].event_data["participating_firms"] == 3

    def test_log_policy_event(self, test_db, test_run):
        """Test logging policy events."""
        event_logger = EventLogger(str(test_run.id), test_db)

        event_logger.log_policy_event(
            EventType.TAX_APPLIED,
            round_idx=2,
            policy_value=0.2,
            policy_details={"tax_rate": 0.2},
        )

        test_db.commit()

        events = event_logger.get_all_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.TAX_APPLIED.value
        assert events[0].event_data["policy_value"] == 0.2
        assert events[0].event_data["category"] == "policy"

    def test_get_events_by_type(self, test_db, test_run):
        """Test filtering events by type."""
        event_logger = EventLogger(str(test_run.id), test_db)

        event_logger.log_event(EventType.CARTEL_FORMED, round_idx=1)
        event_logger.log_event(EventType.DEFECTION_DETECTED, round_idx=2)
        event_logger.log_event(EventType.CARTEL_FORMED, round_idx=3)

        test_db.commit()

        cartel_events = event_logger.get_events_by_type(EventType.CARTEL_FORMED)
        assert len(cartel_events) == 2

        defection_events = event_logger.get_events_by_type(EventType.DEFECTION_DETECTED)
        assert len(defection_events) == 1

    def test_get_event_summary(self, test_db, test_run):
        """Test event summary generation."""
        event_logger = EventLogger(str(test_run.id), test_db)

        event_logger.log_collusion_event(EventType.CARTEL_FORMED, round_idx=1)
        event_logger.log_collusion_event(EventType.DEFECTION_DETECTED, round_idx=2)
        event_logger.log_policy_event(
            EventType.TAX_APPLIED, round_idx=3, policy_value=0.2
        )

        test_db.commit()

        summary = event_logger.get_event_summary()

        assert summary["total_events"] == 3
        assert summary["events_by_type"]["cartel_formed"] == 1
        assert summary["events_by_type"]["defection_detected"] == 1
        assert summary["events_by_type"]["tax_applied"] == 1
        assert summary["events_by_category"]["collusion"] == 2
        assert summary["events_by_category"]["policy"] == 1
        assert summary["first_event_round"] == 1
        assert summary["last_event_round"] == 3


class TestReplaySystem:
    """Test ReplaySystem functionality."""

    def test_get_frame(self, test_db, test_run):
        """Test getting individual frames."""
        # Add mock results
        from src.sim.models.models import Result

        for firm_id in range(2):
            result = Result(
                run_id=test_run.id,
                round_idx=1,
                firm_id=firm_id,
                action=10.0,
                price=20.0,
                qty=5.0,
                profit=50.0,
            )
            test_db.add(result)

        test_db.commit()

        replay_system = ReplaySystem(str(test_run.id), test_db)
        frame = replay_system.get_frame(1)

        assert frame is not None
        assert frame.round_idx == 1
        assert frame.market_price == 20.0
        assert frame.total_quantity == 10.0
        assert frame.total_profit == 100.0
        assert len(frame.firm_data) == 2

    def test_get_frames_with_events(self, test_db, test_run):
        """Test getting frames that contain events."""
        # Add events
        event_logger = EventLogger(str(test_run.id), test_db)
        event_logger.log_event(EventType.CARTEL_FORMED, round_idx=1)
        test_db.commit()

        # Add mock results
        from src.sim.models.models import Result

        for round_idx in range(2):
            for firm_id in range(2):
                result = Result(
                    run_id=test_run.id,
                    round_idx=round_idx,
                    firm_id=firm_id,
                    action=10.0,
                    price=20.0,
                    qty=5.0,
                    profit=50.0,
                )
                test_db.add(result)

        test_db.commit()

        replay_system = ReplaySystem(str(test_run.id), test_db)
        event_frames = replay_system.get_frames_with_events()

        assert len(event_frames) == 1
        assert event_frames[0].round_idx == 1
        assert len(event_frames[0].events) == 1
        assert len(event_frames[0].annotations) == 1

    def test_get_event_rounds(self, test_db, test_run):
        """Test getting list of rounds with events."""
        event_logger = EventLogger(str(test_run.id), test_db)
        event_logger.log_event(EventType.CARTEL_FORMED, round_idx=1)
        event_logger.log_event(EventType.DEFECTION_DETECTED, round_idx=3)
        test_db.commit()

        replay_system = ReplaySystem(str(test_run.id), test_db)
        event_rounds = replay_system.get_event_rounds()

        assert event_rounds == [1, 3]

    def test_replay_summary(self, test_db, test_run):
        """Test replay summary generation."""
        # Add events and results
        event_logger = EventLogger(str(test_run.id), test_db)
        event_logger.log_event(EventType.CARTEL_FORMED, round_idx=1)
        test_db.commit()

        from src.sim.models.models import Result

        for round_idx in range(2):
            for firm_id in range(2):
                result = Result(
                    run_id=test_run.id,
                    round_idx=round_idx,
                    firm_id=firm_id,
                    action=10.0,
                    price=20.0,
                    qty=5.0,
                    profit=50.0,
                )
                test_db.add(result)

        test_db.commit()

        replay_system = ReplaySystem(str(test_run.id), test_db)
        summary = replay_system.get_replay_summary()

        assert summary["run_id"] == str(test_run.id)
        assert summary["model"] == "cournot"
        assert summary["total_rounds"] == 5
        assert summary["total_frames"] == 2
        assert summary["frames_with_events"] == 1
        assert summary["event_rounds"] == [1]
        assert summary["total_events"] == 1
        assert "events_by_type" in summary
