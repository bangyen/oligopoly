"""Replay system for oligopoly simulation.

This module provides comprehensive replay functionality that enables
frame-by-frame playback of simulation runs with event highlighting.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from sqlalchemy.orm import Session

from ..models.metrics import (
    calculate_round_metrics_bertrand,
    calculate_round_metrics_cournot,
)
from ..models.models import Event, Result, Run


@dataclass
class ReplayFrame:
    """A single frame in the simulation replay.

    Contains all data needed to display a specific round of the simulation
    including market metrics, firm actions, and any events that occurred.
    """

    round_idx: int
    timestamp: datetime
    market_price: float
    total_quantity: float
    total_profit: float
    hhi: float
    consumer_surplus: float
    num_firms: int
    firm_data: Dict[int, Dict[str, float]]  # firm_id -> {action, price, qty, profit}
    events: List[Dict[str, Any]]  # Events that occurred in this round
    annotations: List[str]  # Human-readable annotations for events

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to dictionary for API serialization."""
        return {
            "round_idx": self.round_idx,
            "timestamp": self.timestamp.isoformat(),
            "market_price": self.market_price,
            "total_quantity": self.total_quantity,
            "total_profit": self.total_profit,
            "hhi": self.hhi,
            "consumer_surplus": self.consumer_surplus,
            "num_firms": self.num_firms,
            "firm_data": self.firm_data,
            "events": self.events,
            "annotations": self.annotations,
        }


class ReplaySystem:
    """Comprehensive replay system for simulation runs.

    Provides frame-by-frame replay functionality with event highlighting
    and detailed market metrics for each round.
    """

    def __init__(self, run_id: str, db_session: Session) -> None:
        """Initialize replay system for a specific run.

        Args:
            run_id: Unique identifier for the simulation run
            db_session: Database session for data retrieval
        """
        self.run_id = run_id
        self.db = db_session

        # Load run metadata
        run = db_session.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} not found")
        self.run = run

        # Load all results and events
        self._load_data()

    def _load_data(self) -> None:
        """Load all simulation data for replay."""
        # Load results
        results = (
            self.db.query(Result)
            .filter(Result.run_id == self.run_id)
            .order_by(Result.round_idx, Result.firm_id)
            .all()
        )

        # Load events
        events = (
            self.db.query(Event)
            .filter(Event.run_id == self.run_id)
            .order_by(Event.round_idx, Event.created_at)
            .all()
        )

        # Organize results by round and firm
        self.results_by_round: Dict[int, Dict[int, Result]] = {}
        for result in results:
            round_idx = int(result.round_idx)
            firm_id = int(result.firm_id)

            if round_idx not in self.results_by_round:
                self.results_by_round[round_idx] = {}

            self.results_by_round[round_idx][firm_id] = result

        # Organize events by round
        self.events_by_round: Dict[int, List[Event]] = {}
        for event in events:
            round_idx = int(event.round_idx)

            if round_idx not in self.events_by_round:
                self.events_by_round[round_idx] = []

            self.events_by_round[round_idx].append(event)

    def get_frame(self, round_idx: int) -> Optional[ReplayFrame]:
        """Get a single replay frame for a specific round.

        Args:
            round_idx: Round index to retrieve

        Returns:
            ReplayFrame for the specified round, or None if not found
        """
        if round_idx not in self.results_by_round:
            return None

        round_results = self.results_by_round[round_idx]
        if not round_results:
            return None

        # Extract firm data
        firm_data = {}
        quantities = []
        prices = []
        profits = []

        for firm_id, result in round_results.items():
            firm_data[firm_id] = {
                "action": float(result.action),
                "price": float(result.price),
                "quantity": float(result.qty),
                "profit": float(result.profit),
            }
            quantities.append(float(result.qty))
            prices.append(float(result.price))
            profits.append(float(result.profit))

        # Calculate market metrics
        market_price = (
            prices[0] if prices else 0.0
        )  # All firms have same price in Cournot
        total_quantity = sum(quantities)
        total_profit = sum(profits)

        # Calculate HHI and consumer surplus
        if self.run.model == "cournot":
            demand_a = 100.0  # Default - should be stored in run config
            hhi, cs = calculate_round_metrics_cournot(
                quantities, market_price, demand_a
            )
        else:  # bertrand
            demand_alpha = 100.0  # Default - should be stored in run config
            hhi, cs = calculate_round_metrics_bertrand(
                prices, quantities, total_quantity, demand_alpha
            )
            market_price = min(prices) if prices else 0.0

        # Get events for this round
        events = self.events_by_round.get(round_idx, [])
        event_dicts = []
        annotations = []

        for event in events:
            event_dict = {
                "id": event.id,
                "type": event.event_type,
                "firm_id": event.firm_id,
                "description": event.description,
                "data": event.event_data or {},
                "timestamp": event.created_at.isoformat(),
            }
            event_dicts.append(event_dict)

            # Generate annotation
            icon = event.event_data.get("icon", "📝") if event.event_data else "📝"
            annotation = f"{icon} {event.description}"
            annotations.append(annotation)

        return ReplayFrame(
            round_idx=round_idx,
            timestamp=datetime.utcnow(),  # Use current time for replay
            market_price=market_price,
            total_quantity=total_quantity,
            total_profit=total_profit,
            hhi=hhi,
            consumer_surplus=cs,
            num_firms=len(firm_data),
            firm_data=firm_data,
            events=event_dicts,
            annotations=annotations,
        )

    def get_all_frames(self) -> List[ReplayFrame]:
        """Get all replay frames for the simulation.

        Returns:
            List of ReplayFrame objects for all rounds
        """
        frames = []
        for round_idx in sorted(self.results_by_round.keys()):
            frame = self.get_frame(round_idx)
            if frame:
                frames.append(frame)
        return frames

    def get_frames_with_events(self) -> List[ReplayFrame]:
        """Get replay frames that contain events.

        Returns:
            List of ReplayFrame objects for rounds with events
        """
        frames = []
        for round_idx in sorted(self.events_by_round.keys()):
            frame = self.get_frame(round_idx)
            if frame and frame.events:
                frames.append(frame)
        return frames

    def get_event_rounds(self) -> List[int]:
        """Get list of round indices that contain events.

        Returns:
            List of round indices with events
        """
        return sorted(self.events_by_round.keys())

    def replay_generator(
        self, delay_ms: int = 500
    ) -> Generator[ReplayFrame, None, None]:
        """Generate frames for replay with specified delay.

        Args:
            delay_ms: Delay between frames in milliseconds

        Yields:
            ReplayFrame objects in sequence
        """
        import time

        frames = self.get_all_frames()
        for frame in frames:
            yield frame
            time.sleep(delay_ms / 1000.0)

    def get_replay_summary(self) -> Dict[str, Any]:
        """Get summary information about the replay.

        Returns:
            Dictionary with replay metadata and statistics
        """
        frames = self.get_all_frames()
        event_frames = self.get_frames_with_events()

        summary = {
            "run_id": self.run_id,
            "model": self.run.model,
            "total_rounds": self.run.rounds,
            "total_frames": len(frames),
            "frames_with_events": len(event_frames),
            "event_rounds": self.get_event_rounds(),
            "first_round": (
                min(frames, key=lambda f: f.round_idx).round_idx if frames else None
            ),
            "last_round": (
                max(frames, key=lambda f: f.round_idx).round_idx if frames else None
            ),
        }

        # Event statistics
        total_events = sum(len(frame.events) for frame in frames)
        summary["total_events"] = total_events

        if total_events > 0:
            event_types: Dict[str, int] = {}
            for frame in frames:
                for event in frame.events:
                    event_type = event["type"]
                    event_types[event_type] = event_types.get(event_type, 0) + 1
            summary["events_by_type"] = event_types

        return summary
