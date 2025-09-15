"""Tests for replay sequence functionality.

This module tests the replay system's ability to return frames
with proper ordering and event annotations.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.sim.events.event_types import EventType
from src.sim.events.replay import ReplaySystem
from src.sim.models.models import Base, Event, Result, Run


@pytest.fixture(scope="function")
def test_db():
    """Create a test database session."""
    engine = create_engine("sqlite:///:memory:")
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


class TestReplaySequence:
    """Test replay functionality returns frames 1..T with metric snapshots."""

    def test_replay_returns_frames_1_to_t(self, test_db, test_run):
        """Test that replay returns frames 1..T with metric snapshots."""
        # Create mock results for rounds 0, 1, 2
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

    def test_frames_contain_metric_snapshots(self, test_db, test_run):
        """Test that frames contain proper metric snapshots."""
        # Create mock results with known values
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
        frames = replay_system.get_all_frames()

        # Check metric calculations
        for frame in frames:
            # Market price should be consistent across firms
            assert frame.market_price == 20.0

            # Total quantity should be sum of firm quantities
            assert frame.total_quantity == 10.0  # 2 firms * 5.0 each

            # Total profit should be sum of firm profits
            assert frame.total_profit == 100.0  # 2 firms * 50.0 each

            # HHI and consumer surplus should be calculated
            assert frame.hhi >= 0.0
            assert frame.consumer_surplus >= 0.0

            # Firm data should contain all firms
            assert len(frame.firm_data) == 2
            for firm_id, firm_info in frame.firm_data.items():
                assert "action" in firm_info
                assert "price" in firm_info
                assert "quantity" in firm_info
                assert "profit" in firm_info

    def test_event_rounds_have_non_empty_annotations(self, test_db, test_run):
        """Test that event rounds have non-empty annotations."""
        # Add events
        event1 = Event(
            run_id=test_run.id,
            round_idx=1,
            event_type=EventType.CARTEL_FORMED.value,
            description="Cartel formed with 3 firms",
            event_data={"icon": "🤝", "category": "collusion"},
        )
        event2 = Event(
            run_id=test_run.id,
            round_idx=2,
            event_type=EventType.DEFECTION_DETECTED.value,
            firm_id=1,
            description="Firm 1 defected from cartel",
            event_data={"icon": "⚔️", "category": "collusion"},
        )
        test_db.add(event1)
        test_db.add(event2)
        test_db.commit()

        # Add mock results
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

            # Check annotation format
            for annotation in frame.annotations:
                assert isinstance(annotation, str)
                assert len(annotation) > 0
                # Should contain icon and description
                assert "🤝" in annotation or "⚔️" in annotation

    def test_replay_frame_ordering(self, test_db, test_run):
        """Test that replay frames are returned in correct order."""
        # Add results in reverse order
        for round_idx in reversed(range(3)):
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

        replay_system = ReplaySystem(str(test_run.id), test_db)
        frames = replay_system.get_all_frames()

        # Check ordering
        assert len(frames) == 3
        for i, frame in enumerate(frames):
            assert frame.round_idx == i

    def test_empty_replay_handling(self, test_db, test_run):
        """Test handling of runs with no results."""
        replay_system = ReplaySystem(str(test_run.id), test_db)

        frames = replay_system.get_all_frames()
        assert len(frames) == 0

        event_frames = replay_system.get_frames_with_events()
        assert len(event_frames) == 0

        event_rounds = replay_system.get_event_rounds()
        assert len(event_rounds) == 0

    def test_replay_with_mixed_event_types(self, test_db, test_run):
        """Test replay with different types of events."""
        # Add various event types
        events = [
            Event(
                run_id=test_run.id,
                round_idx=0,
                event_type=EventType.CARTEL_FORMED.value,
                description="Cartel formed",
                event_data={"icon": "🤝", "category": "collusion"},
            ),
            Event(
                run_id=test_run.id,
                round_idx=1,
                event_type=EventType.TAX_APPLIED.value,
                description="Tax applied",
                event_data={"icon": "📈", "category": "policy", "policy_value": 0.2},
            ),
            Event(
                run_id=test_run.id,
                round_idx=2,
                event_type=EventType.FIRM_ENTRY.value,
                description="New firm entered",
                event_data={"icon": "🚀", "category": "market", "cost": 15.0},
            ),
        ]

        for event in events:
            test_db.add(event)

        test_db.commit()

        # Add mock results
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

        replay_system = ReplaySystem(str(test_run.id), test_db)
        frames = replay_system.get_all_frames()

        # All frames should have events
        assert len(frames) == 3
        for frame in frames:
            assert len(frame.events) == 1
            assert len(frame.annotations) == 1

            # Check that annotations contain appropriate icons
            annotation = frame.annotations[0]
            if frame.round_idx == 0:
                assert "🤝" in annotation
            elif frame.round_idx == 1:
                assert "📈" in annotation
            elif frame.round_idx == 2:
                assert "🚀" in annotation
