"""Event logging system for oligopoly simulation.

This module provides comprehensive event logging capabilities that integrate
with the database and enable detailed tracking of all simulation events.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models.models import Event, Run
from .event_types import (
    EventType,
    get_event_description,
    get_event_icon,
)


class EventLogger:
    """Comprehensive event logging system for simulation runs.

    Provides methods to log various types of events during simulation
    and retrieve them for analysis and replay functionality.
    """

    def __init__(self, run_id: str, db: Session):
        """Initialize event logger for a specific run.

        Args:
            run_id: Unique identifier for the simulation run
            db: Database session for persistence
        """
        self.run_id = run_id
        self.db = db

        # Verify run exists
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} not found")

    def log_event(
        self,
        event_type: EventType,
        round_idx: int,
        firm_id: Optional[int] = None,
        description: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Log an event to the database.

        Args:
            event_type: Type of event to log
            round_idx: Round index when event occurred
            firm_id: Firm involved in event (if applicable)
            description: Custom description (auto-generated if None)
            event_data: Additional event data

        Returns:
            Created Event record
        """
        if description is None:
            description = get_event_description(
                event_type, firm_id=firm_id, **(event_data or {})
            )

        event = Event(
            run_id=self.run_id,
            round_idx=round_idx,
            event_type=event_type.value,
            firm_id=firm_id,
            description=description,
            event_data=event_data or {},
        )

        self.db.add(event)
        self.db.flush()  # Get the ID

        return event

    def log_collusion_event(
        self,
        event_type: EventType,
        round_idx: int,
        firm_id: Optional[int] = None,
        cartel_data: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Log a collusion-related event.

        Args:
            event_type: Type of collusion event
            round_idx: Round index when event occurred
            firm_id: Firm involved in event
            cartel_data: Additional cartel-specific data

        Returns:
            Created Event record
        """
        event_data = cartel_data or {}
        event_data.update(
            {
                "category": "collusion",
                "icon": get_event_icon(event_type),
            }
        )

        return self.log_event(event_type, round_idx, firm_id, event_data=event_data)

    def log_policy_event(
        self,
        event_type: EventType,
        round_idx: int,
        policy_value: float,
        policy_details: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Log a policy intervention event.

        Args:
            event_type: Type of policy event
            round_idx: Round index when event occurred
            policy_value: Value of the policy intervention
            policy_details: Additional policy details

        Returns:
            Created Event record
        """
        event_data = policy_details or {}
        event_data.update(
            {
                "policy_value": policy_value,
                "category": "policy",
                "icon": get_event_icon(event_type),
            }
        )

        return self.log_event(event_type, round_idx, event_data=event_data)

    def log_market_event(
        self,
        event_type: EventType,
        round_idx: int,
        firm_id: Optional[int] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Log a market dynamics event.

        Args:
            event_type: Type of market event
            round_idx: Round index when event occurred
            firm_id: Firm involved in event (if applicable)
            market_data: Additional market data

        Returns:
            Created Event record
        """
        event_data = market_data or {}
        event_data.update(
            {
                "category": "market",
                "icon": get_event_icon(event_type),
            }
        )

        return self.log_event(event_type, round_idx, firm_id, event_data=event_data)

    def log_strategy_event(
        self,
        event_type: EventType,
        round_idx: int,
        firm_id: int,
        strategy_data: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """Log a strategy-related event.

        Args:
            event_type: Type of strategy event
            round_idx: Round index when event occurred
            firm_id: Firm whose strategy changed
            strategy_data: Additional strategy data

        Returns:
            Created Event record
        """
        event_data = strategy_data or {}
        event_data.update(
            {
                "category": "strategy",
                "icon": get_event_icon(event_type),
            }
        )

        return self.log_event(event_type, round_idx, firm_id, event_data=event_data)

    def get_events_for_round(self, round_idx: int) -> List[Event]:
        """Get all events for a specific round.

        Args:
            round_idx: Round index to query

        Returns:
            List of events for the specified round
        """
        return (
            self.db.query(Event)
            .filter(Event.run_id == self.run_id, Event.round_idx == round_idx)
            .order_by(Event.created_at)
            .all()
        )

    def get_all_events(self) -> List[Event]:
        """Get all events for this run.

        Returns:
            List of all events ordered by round and creation time
        """
        return (
            self.db.query(Event)
            .filter(Event.run_id == self.run_id)
            .order_by(Event.round_idx, Event.created_at)
            .all()
        )

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type.

        Args:
            event_type: Type of events to retrieve

        Returns:
            List of events of the specified type
        """
        return (
            self.db.query(Event)
            .filter(Event.run_id == self.run_id, Event.event_type == event_type.value)
            .order_by(Event.round_idx, Event.created_at)
            .all()
        )

    def get_events_by_category(self, category: str) -> List[Event]:
        """Get all events in a specific category.

        Args:
            category: Event category to filter by

        Returns:
            List of events in the specified category
        """
        return (
            self.db.query(Event)
            .filter(Event.run_id == self.run_id)
            .filter(Event.event_data["category"].astext == category)
            .order_by(Event.round_idx, Event.created_at)
            .all()
        )

    def get_event_summary(self) -> Dict[str, Any]:
        """Get a summary of all events for this run.

        Returns:
            Dictionary with event counts by type and category
        """
        events = self.get_all_events()

        summary: Dict[str, Any] = {
            "total_events": len(events),
            "events_by_type": {},
            "events_by_category": {},
            "events_by_round": {},
            "first_event_round": None,
            "last_event_round": None,
        }

        if events:
            summary["first_event_round"] = min(event.round_idx for event in events)
            summary["last_event_round"] = max(event.round_idx for event in events)

        for event in events:
            # Count by type
            event_type = event.event_type
            summary["events_by_type"][event_type] = (
                summary["events_by_type"].get(event_type, 0) + 1
            )

            # Count by category
            category = (
                event.event_data.get("category", "other")
                if event.event_data
                else "other"
            )
            summary["events_by_category"][category] = (
                summary["events_by_category"].get(category, 0) + 1
            )

            # Count by round
            round_idx = event.round_idx
            summary["events_by_round"][round_idx] = (
                summary["events_by_round"].get(round_idx, 0) + 1
            )

        return summary
