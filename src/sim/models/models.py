"""Economic models for oligopoly market simulation.

This module defines the core economic models used in the oligopoly simulation,
including demand curves, market structures, firm behavior, and simulation configuration.
"""

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


@dataclass
class DemandSegment:
    """Individual consumer segment with linear demand: Q_k(p) = max(0, α_k - β_k*p).

    Represents a segment of consumers with homogeneous preferences.
    Each segment has its own demand parameters and market weight.
    """

    alpha: float  # Intercept parameter for segment demand curve
    beta: float  # Slope parameter for segment demand curve
    weight: float  # Market share weight (must sum to 1 across all segments)

    def demand(self, price: float) -> float:
        """Calculate segment demand at given price.

        Args:
            price: Market price

        Returns:
            Segment demand quantity
        """
        return max(0.0, self.alpha - self.beta * price)

    def __repr__(self) -> str:
        """Stable string representation for testing and debugging."""
        return (
            f"DemandSegment(alpha={self.alpha}, beta={self.beta}, weight={self.weight})"
        )


@dataclass
class SegmentedDemand:
    """Segmented market demand with K consumer segments.

    Represents a market with multiple consumer segments, each with
    different demand parameters and weights. Total demand is the
    weighted sum of segment demands.
    """

    segments: list[DemandSegment]  # List of consumer segments

    def __post_init__(self) -> None:
        """Validate segment weights sum to 1 and parameters are economically reasonable."""
        total_weight = sum(segment.weight for segment in self.segments)
        if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
            raise ValueError(f"Segment weights must sum to 1.0, got {total_weight:.6f}")

        # Validate that all segments have positive parameters
        for i, segment in enumerate(self.segments):
            if segment.alpha <= 0:
                raise ValueError(
                    f"Segment {i} alpha parameter must be positive, got {segment.alpha}"
                )
            if segment.beta <= 0:
                raise ValueError(
                    f"Segment {i} beta parameter must be positive, got {segment.beta}"
                )
            if segment.weight <= 0 or segment.weight > 1:
                raise ValueError(
                    f"Segment {i} weight must be in (0, 1], got {segment.weight}"
                )

        # Validate that effective demand parameters are reasonable
        weighted_alpha = sum(
            segment.weight * segment.alpha for segment in self.segments
        )
        weighted_beta = sum(segment.weight * segment.beta for segment in self.segments)

        if weighted_alpha <= 0:
            raise ValueError(
                f"Effective alpha parameter must be positive, got {weighted_alpha}"
            )
        if weighted_beta <= 0:
            raise ValueError(
                f"Effective beta parameter must be positive, got {weighted_beta}"
            )

        # Check for unrealistic elasticity (beta too high relative to alpha)
        max_elasticity_ratio = 2.0  # beta/alpha should not exceed this ratio
        for i, segment in enumerate(self.segments):
            if segment.beta / segment.alpha > max_elasticity_ratio:
                raise ValueError(
                    f"Segment {i} has unrealistic elasticity: beta/alpha = {segment.beta / segment.alpha:.3f} > {max_elasticity_ratio}"
                )

        # Check for economically viable market size
        # Ensure that at reasonable prices (10-50% of max price), there's positive demand
        for i, segment in enumerate(self.segments):
            max_price = segment.alpha / segment.beta  # Price where demand = 0
            test_price = max_price * 0.3  # 30% of max price
            test_demand = segment.demand(test_price)
            if test_demand <= 0:
                raise ValueError(
                    f"Segment {i} has no demand at reasonable prices. Max price: {max_price:.2f}, "
                    f"demand at 30% max price: {test_demand:.2f}"
                )

        # Check that weighted market size is reasonable
        effective_market_size = weighted_alpha / weighted_beta
        if effective_market_size < 5.0:
            raise ValueError(
                f"Effective market size too small: {effective_market_size:.2f}, should be >= 5.0"
            )

    def total_demand(self, price: float) -> float:
        """Calculate total market demand at given price.

        Args:
            price: Market price

        Returns:
            Total weighted demand across all segments
        """
        return sum(segment.weight * segment.demand(price) for segment in self.segments)

    def segment_demands(self, price: float) -> list[float]:
        """Calculate demand for each segment at given price.

        Args:
            price: Market price

        Returns:
            List of demand quantities for each segment
        """
        return [segment.demand(price) for segment in self.segments]

    def __repr__(self) -> str:
        """Stable string representation for testing and debugging."""
        return f"SegmentedDemand(segments={len(self.segments)})"


class Run(Base):  # type: ignore
    """Simulation run tracking.

    Stores metadata about multi-round simulation runs including
    the model type, number of rounds, creation timestamp, and the
    market demand parameters used (for faithful metric re-computation).
    """

    __tablename__ = "runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model = Column(String(20), nullable=False)  # "cournot" or "bertrand"
    rounds = Column(Integer, nullable=False)
    params = Column(JSON, nullable=True)  # Market demand parameters (a/b or alpha/beta)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    rounds_data = relationship(
        "Round", back_populates="run", cascade="all, delete-orphan"
    )
    results = relationship("Result", back_populates="run", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="run", cascade="all, delete-orphan")


class Round(Base):  # type: ignore
    """Individual round within a simulation run.

    Tracks each round of a multi-round simulation for
    time-series analysis and debugging.
    """

    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    idx = Column(Integer, nullable=False)  # Round index (0-based)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    run = relationship("Run", back_populates="rounds_data")
    results = relationship(
        "Result", back_populates="round", cascade="all, delete-orphan"
    )


class Result(Base):  # type: ignore
    """Individual firm results for each round.

    Stores the action (quantity/price), market price, quantity sold,
    and profit for each firm in each round of the simulation.
    """

    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    round_id = Column(
        Integer, ForeignKey("rounds.id"), nullable=True
    )  # Optional FK to rounds
    round_idx = Column(Integer, nullable=False)  # Round index (0-based)
    firm_id = Column(Integer, nullable=False)  # Firm identifier within the run
    action = Column(Float, nullable=False)  # Quantity (Cournot) or Price (Bertrand)
    price = Column(Float, nullable=False)  # Market price for this round
    qty = Column(Float, nullable=False)  # Quantity sold by this firm
    profit = Column(Float, nullable=False)  # Profit earned by this firm
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    run = relationship("Run", back_populates="results")
    round = relationship("Round", back_populates="results")


class Event(Base):  # type: ignore
    """Comprehensive event tracking for simulation runs.

    Stores all types of events that occur during simulation including collusion,
    defection, policy shocks, and market entry, exit and innovation.
    This unified event system enables comprehensive replay and analysis.
    """

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    round_idx = Column(Integer, nullable=False)  # Round index (0-based)
    event_type = Column(String(50), nullable=False)  # Type of event
    firm_id = Column(Integer, nullable=True)  # Firm involved (if applicable)
    description = Column(Text, nullable=False)  # Human-readable description
    event_data = Column(JSON, nullable=True)  # Additional event data
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    run = relationship("Run", back_populates="events")
