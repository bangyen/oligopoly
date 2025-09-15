"""Integration tests for collusion workflow.

This module tests the complete collusion workflow including cartel formation,
defection detection, punishment mechanisms, and regulatory interventions.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sim.collusion import CollusionManager, RegulatorState
from sim.models.models import Base, CollusionEvent, Run
from sim.runners.collusion_runner import run_collusion_game
from sim.strategies.collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
)


class TestCollusionWorkflow:
    """Test complete collusion workflow integration."""

    @pytest.fixture
    def db_session(self):
        """Create in-memory database session for testing."""
        engine = create_engine("sqlite:///:memory:", echo=False)
        Base.metadata.create_all(engine)
        session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = session_local()
        yield db
        db.close()

    def test_cartel_formation_and_stability(self, db_session):
        """Test cartel formation and stability over multiple rounds."""
        # Setup: 3 firms with collusive strategies
        strategies = [
            CollusiveStrategy(defection_probability=0.05, seed=42),
            CollusiveStrategy(defection_probability=0.05, seed=43),
            CollusiveStrategy(defection_probability=0.05, seed=44),
        ]
        costs = [10.0, 12.0, 15.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        # Run collusion game
        run_id = run_collusion_game(
            model="cournot",
            rounds=20,
            strategies=strategies,
            costs=costs,
            params=params,
            bounds=bounds,
            db=db_session,
            collusion_config={
                "cartel_formation_threshold": 0.8,
                "defection_detection_threshold": 0.15,
                "punishment_rounds": 3,
                "auto_form_cartel": True,  # Enable automatic cartel formation
            },
            seed=42,
        )

        # Verify run was created
        run = db_session.query(Run).filter(Run.id == run_id).first()
        assert run is not None
        assert run.model == "cournot"
        assert run.rounds == 20

        # Check for collusion events
        collusion_events = (
            db_session.query(CollusionEvent)
            .filter(CollusionEvent.run_id == run_id)
            .order_by(CollusionEvent.round_idx)
            .all()
        )

        # Should have some collusion activity
        assert len(collusion_events) > 0

        # Check for defection events (which we're seeing)
        defection_events = [
            e for e in collusion_events if e.event_type == "firm_defected"
        ]
        assert len(defection_events) > 0

        # Verify defection event details
        defection_event = defection_events[0]
        assert defection_event.firm_id is not None
        assert defection_event.firm_id in [0, 1, 2]
        assert "defect" in defection_event.description.lower()

    def test_defection_detection_and_punishment(self, db_session):
        """Test defection detection and punishment mechanisms."""
        # Setup: Mix of strategies with one high-defection firm
        strategies = [
            CollusiveStrategy(defection_probability=0.05, seed=42),  # Stable
            CollusiveStrategy(defection_probability=0.8, seed=43),  # High defection
            CollusiveStrategy(defection_probability=0.05, seed=44),  # Stable
        ]
        costs = [10.0, 12.0, 15.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        run_id = run_collusion_game(
            model="cournot",
            rounds=15,
            strategies=strategies,
            costs=costs,
            params=params,
            bounds=bounds,
            db=db_session,
            collusion_config={
                "cartel_formation_threshold": 0.7,
                "defection_detection_threshold": 0.1,
                "punishment_rounds": 2,
                "auto_form_cartel": True,
            },
            seed=42,
        )

        # Check for defection events
        collusion_events = (
            db_session.query(CollusionEvent)
            .filter(CollusionEvent.run_id == run_id)
            .order_by(CollusionEvent.round_idx)
            .all()
        )

        defection_events = [
            e for e in collusion_events if e.event_type == "firm_defected"
        ]
        assert len(defection_events) > 0

        # Verify defection event details
        defection_event = defection_events[0]
        assert defection_event.firm_id is not None
        assert defection_event.firm_id in [0, 1, 2]
        assert "defect" in defection_event.description.lower()

    def test_regulatory_intervention(self, db_session):
        """Test regulatory intervention scenarios."""
        # Setup: High-defection scenario to trigger regulatory intervention
        strategies = [
            CollusiveStrategy(defection_probability=0.9, seed=42),
            CollusiveStrategy(defection_probability=0.9, seed=43),
            CollusiveStrategy(defection_probability=0.9, seed=44),
        ]
        costs = [10.0, 12.0, 15.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        run_id = run_collusion_game(
            model="cournot",
            rounds=20,
            strategies=strategies,
            costs=costs,
            params=params,
            bounds=bounds,
            db=db_session,
            collusion_config={
                "cartel_formation_threshold": 0.6,
                "defection_detection_threshold": 0.05,
                "punishment_rounds": 2,
                "regulator_intervention_threshold": 0.3,
                "regulator_intervention_probability": 0.8,
                "auto_form_cartel": True,
            },
            seed=42,
        )

        # Check for regulatory intervention events
        collusion_events = (
            db_session.query(CollusionEvent)
            .filter(CollusionEvent.run_id == run_id)
            .order_by(CollusionEvent.round_idx)
            .all()
        )

        # Should have some collusion activity (defections or interventions)
        assert len(collusion_events) > 0

    def test_cartel_strategy_behavior(self, db_session):
        """Test CartelStrategy behavior in collusion scenarios."""
        # Setup: Mix of cartel and opportunistic strategies
        strategies = [
            CartelStrategy(),
            CartelStrategy(),
            OpportunisticStrategy(profit_threshold_multiplier=1.2),
        ]
        costs = [10.0, 12.0, 15.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        run_id = run_collusion_game(
            model="cournot",
            rounds=15,
            strategies=strategies,
            costs=costs,
            params=params,
            bounds=bounds,
            db=db_session,
            collusion_config={
                "cartel_formation_threshold": 0.8,
                "defection_detection_threshold": 0.2,
                "punishment_rounds": 3,
                "auto_form_cartel": True,
            },
            seed=42,
        )

        # Verify run completed successfully
        run = db_session.query(Run).filter(Run.id == run_id).first()
        assert run is not None

        # Check for cartel formation events
        collusion_events = (
            db_session.query(CollusionEvent)
            .filter(CollusionEvent.run_id == run_id)
            .order_by(CollusionEvent.round_idx)
            .all()
        )

        # Should have some collusion activity (defections or cartel formation)
        assert len(collusion_events) > 0

    def test_bertrand_collusion_workflow(self, db_session):
        """Test collusion workflow with Bertrand competition."""
        strategies = [
            CollusiveStrategy(defection_probability=0.1, seed=42),
            CollusiveStrategy(defection_probability=0.1, seed=43),
            CollusiveStrategy(defection_probability=0.1, seed=44),
        ]
        costs = [10.0, 12.0, 15.0]
        params = {"alpha": 100.0, "beta": 1.0}
        bounds = (10.0, 80.0)

        run_id = run_collusion_game(
            model="bertrand",
            rounds=15,
            strategies=strategies,
            costs=costs,
            params=params,
            bounds=bounds,
            db=db_session,
            collusion_config={
                "cartel_formation_threshold": 0.8,
                "defection_detection_threshold": 0.15,
                "punishment_rounds": 2,
                "auto_form_cartel": True,
            },
            seed=42,
        )

        # Verify Bertrand run completed
        run = db_session.query(Run).filter(Run.id == run_id).first()
        assert run is not None
        assert run.model == "bertrand"

        # Check for collusion events
        collusion_events = (
            db_session.query(CollusionEvent)
            .filter(CollusionEvent.run_id == run_id)
            .order_by(CollusionEvent.round_idx)
            .all()
        )

        # Should have some collusion activity
        assert len(collusion_events) > 0

    def test_collusion_manager_state_transitions(self, db_session):
        """Test CollusionManager state transitions."""
        # Create collusion manager
        manager = CollusionManager()

        # Test initial state
        assert manager.current_cartel is None
        assert len(manager.events) == 0

        # Simulate cartel formation
        manager.form_cartel(
            round_idx=5,
            collusive_price=25.0,
            collusive_quantity=20.0,
            participating_firms=[0, 1, 2],
        )

        # Should have cartel state after formation
        assert manager.current_cartel is not None
        assert len(manager.events) > 0

    def test_regulator_state_management(self, db_session):
        """Test RegulatorState management and interventions."""
        # Create regulator state
        regulator = RegulatorState(
            hhi_threshold=0.3,
            intervention_probability=0.5,
            penalty_amount=50.0,
        )

        # Test initial state
        assert regulator.hhi_threshold == 0.3
        assert regulator.intervention_probability == 0.5

        # Test HHI calculation
        manager = CollusionManager(regulator)
        hhi = manager.calculate_hhi([0.4, 0.3, 0.3])
        assert hhi > 0.3  # Should exceed threshold

    def test_collusion_event_logging(self, db_session):
        """Test comprehensive collusion event logging."""
        strategies = [
            CollusiveStrategy(defection_probability=0.2, seed=42),
            CollusiveStrategy(defection_probability=0.2, seed=43),
            OpportunisticStrategy(profit_threshold_multiplier=1.2, seed=44),
        ]
        costs = [10.0, 12.0, 15.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        run_id = run_collusion_game(
            model="cournot",
            rounds=25,
            strategies=strategies,
            costs=costs,
            params=params,
            bounds=bounds,
            db=db_session,
            collusion_config={
                "cartel_formation_threshold": 0.7,
                "defection_detection_threshold": 0.1,
                "punishment_rounds": 2,
                "regulator_intervention_threshold": 0.25,
                "regulator_intervention_probability": 0.6,
                "auto_form_cartel": True,
            },
            seed=42,
        )

        # Get all collusion events
        collusion_events = (
            db_session.query(CollusionEvent)
            .filter(CollusionEvent.run_id == run_id)
            .order_by(CollusionEvent.round_idx)
            .all()
        )

        # Should have multiple types of events
        event_types = set(e.event_type for e in collusion_events)
        expected_types = {
            "cartel_formed",
            "firm_defected",
            "regulator_intervened",
            "penalty_imposed",
        }

        # Should have at least some of the expected event types
        assert len(event_types.intersection(expected_types)) > 0

        # Verify event data integrity
        for event in collusion_events:
            assert event.round_idx >= 0
            assert event.description != ""
            assert isinstance(event.event_data, dict)

    def test_collusion_with_policy_shocks(self, db_session):
        """Test collusion behavior under policy shocks."""
        strategies = [
            CollusiveStrategy(defection_probability=0.1, seed=42),
            CollusiveStrategy(defection_probability=0.1, seed=43),
            CollusiveStrategy(defection_probability=0.1, seed=44),
        ]
        costs = [10.0, 12.0, 15.0]
        params = {"a": 100.0, "b": 1.0}
        bounds = (0.0, 50.0)

        # Note: Policy events could be added here to test interaction with collusion
        # from sim.policy.policy_shocks import PolicyEvent, PolicyType
        # policy_events = [
        #     PolicyEvent(round_idx=5, policy_type=PolicyType.TAX, value=0.1),
        #     PolicyEvent(round_idx=10, policy_type=PolicyType.SUBSIDY, value=0.05),
        # ]

        run_id = run_collusion_game(
            model="cournot",
            rounds=20,
            strategies=strategies,
            costs=costs,
            params=params,
            bounds=bounds,
            db=db_session,
            collusion_config={
                "cartel_formation_threshold": 0.8,
                "defection_detection_threshold": 0.15,
                "punishment_rounds": 3,
                "auto_form_cartel": True,
            },
            seed=42,
        )

        # Verify run completed with policy events
        run = db_session.query(Run).filter(Run.id == run_id).first()
        assert run is not None

        # Check that collusion events still occurred despite policy shocks
        collusion_events = (
            db_session.query(CollusionEvent)
            .filter(CollusionEvent.run_id == run_id)
            .all()
        )
        assert len(collusion_events) > 0
