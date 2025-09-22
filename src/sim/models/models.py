"""Economic models for oligopoly market simulation.

This module defines the core economic models used in the oligopoly simulation,
including demand curves, market structures, firm behavior, and simulation configuration.
"""

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


@dataclass
class Demand:
    """Linear inverse demand curve: P(Q) = a - b*Q.

    Represents the market demand function where price decreases linearly
    with total quantity supplied. This is a fundamental building block
    for oligopoly market analysis.
    """

    a: float  # Maximum price when quantity is zero
    b: float  # Slope of demand curve (price sensitivity to quantity)

    def price(self, quantity: float) -> float:
        """Calculate market price for given total quantity.

        Args:
            quantity: Total quantity supplied by all firms

        Returns:
            Market price based on inverse demand function
        """
        return max(0.0, self.a - self.b * quantity)

    def __repr__(self) -> str:
        """Stable string representation for testing and debugging."""
        return f"Demand(a={self.a}, b={self.b})"


@dataclass
class IsoelasticDemand:
    """Isoelastic demand curve: P(Q) = A * Q^(-1/ε).

    Represents demand with constant price elasticity ε.
    This is more realistic for many markets than linear demand.
    """

    A: float  # Scale parameter
    elasticity: float  # Price elasticity of demand (must be > 1)

    def __post_init__(self) -> None:
        """Validate that elasticity is greater than 1 for economic realism."""
        if self.elasticity <= 1.0:
            raise ValueError(
                f"Elasticity must be > 1 for economic realism, got {self.elasticity}"
            )

    def price(self, quantity: float) -> float:
        """Calculate market price for given total quantity.

        Args:
            quantity: Total quantity supplied by all firms

        Returns:
            Market price based on isoelastic demand function
        """
        if quantity <= 0:
            return float("inf")
        return float(self.A * (quantity ** (-1 / self.elasticity)))

    def __repr__(self) -> str:
        """Stable string representation for testing and debugging."""
        return f"IsoelasticDemand(A={self.A}, elasticity={self.elasticity})"


@dataclass
class CostStructure:
    """Enhanced cost structure for firms including fixed costs and capacity constraints.

    This represents a more realistic cost function that includes:
    - Marginal costs (variable costs per unit)
    - Fixed costs (sunk costs that don't vary with output)
    - Capacity constraints (maximum production limits)
    - Economies of scale (cost reduction with higher output)
    """

    marginal_cost: float  # Variable cost per unit
    fixed_cost: float = 0.0  # Fixed cost per period
    capacity_limit: Optional[float] = None  # Maximum production capacity
    economies_of_scale: float = 1.0  # Cost reduction factor (1.0 = no economies)

    def __post_init__(self) -> None:
        """Validate cost structure parameters."""
        if self.marginal_cost <= 0:
            raise ValueError(
                f"Marginal cost must be positive, got {self.marginal_cost}"
            )
        if self.fixed_cost < 0:
            raise ValueError(f"Fixed cost must be non-negative, got {self.fixed_cost}")
        if self.capacity_limit is not None and self.capacity_limit <= 0:
            raise ValueError(
                f"Capacity limit must be positive, got {self.capacity_limit}"
            )
        if self.economies_of_scale <= 0:
            raise ValueError(
                f"Economies of scale must be positive, got {self.economies_of_scale}"
            )

    def total_cost(self, quantity: float) -> float:
        """Calculate total cost for given quantity.

        Args:
            quantity: Production quantity

        Returns:
            Total cost including fixed and variable costs
        """
        if quantity <= 0:
            return self.fixed_cost

        # Apply capacity constraint
        effective_quantity = (
            min(quantity, self.capacity_limit) if self.capacity_limit else quantity
        )

        # Calculate variable cost with economies of scale
        if self.economies_of_scale == 1.0:
            variable_cost = self.marginal_cost * effective_quantity
        else:
            # Economies of scale: cost per unit decreases with quantity
            variable_cost = self.marginal_cost * (
                effective_quantity**self.economies_of_scale
            )

        return self.fixed_cost + variable_cost

    def average_cost(self, quantity: float) -> float:
        """Calculate average cost per unit.

        Args:
            quantity: Production quantity

        Returns:
            Average cost per unit
        """
        if quantity <= 0:
            return float("inf")
        return float(self.total_cost(quantity) / quantity)

    def marginal_cost_at_quantity(self, quantity: float) -> float:
        """Calculate marginal cost at given quantity (accounting for economies of scale).

        Args:
            quantity: Production quantity

        Returns:
            Marginal cost at this quantity
        """
        if quantity <= 0:
            return self.marginal_cost

        if self.economies_of_scale == 1.0:
            return self.marginal_cost
        else:
            # Marginal cost decreases with economies of scale
            return float(
                self.marginal_cost
                * self.economies_of_scale
                * (quantity ** (self.economies_of_scale - 1))
            )

    def __repr__(self) -> str:
        """Stable string representation for testing and debugging."""
        return f"CostStructure(mc={self.marginal_cost}, fc={self.fixed_cost}, cap={self.capacity_limit}, scale={self.economies_of_scale})"


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
                    f"Segment {i} has unrealistic elasticity: beta/alpha = {segment.beta/segment.alpha:.3f} > {max_elasticity_ratio}"
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


class Market(Base):  # type: ignore
    """Market configuration and parameters for simulation.

    Stores market-level parameters including demand curve coefficients,
    number of firms, and other market characteristics needed for
    oligopoly simulation runs.
    """

    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    demand_a = Column(Float, nullable=False)
    demand_b = Column(Float, nullable=False)
    num_firms = Column(Integer, nullable=False)
    segments = Column(JSON, nullable=True)  # Segmented demand configuration
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to firms in this market
    firms = relationship("Firm", back_populates="market")

    def get_demand(self) -> Demand:
        """Get the demand curve for this market."""
        return Demand(a=float(self.demand_a), b=float(self.demand_b))

    def get_segmented_demand(self) -> SegmentedDemand:
        """Get the segmented demand for this market.

        Returns:
            SegmentedDemand object with configured segments

        Raises:
            ValueError: If segments configuration is invalid
        """
        if not self.segments:
            raise ValueError("No segments configured for this market")

        demand_segments = []
        for segment_config in self.segments:  # type: ignore[attr-defined]
            segment = DemandSegment(
                alpha=float(segment_config["alpha"]),
                beta=float(segment_config["beta"]),
                weight=float(segment_config["weight"]),
            )
            demand_segments.append(segment)

        return SegmentedDemand(segments=demand_segments)


class Firm(Base):  # type: ignore
    """Individual firm participating in the market.

    Represents a firm with its cost structure and strategic parameters.
    Each firm competes in the oligopoly market by choosing quantities
    or prices based on its cost function and market conditions.
    """

    __tablename__ = "firms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    cost = Column(Float, nullable=False)  # Marginal cost (legacy field)
    fixed_cost = Column(Float, default=0.0)  # Fixed cost per period
    capacity_limit = Column(Float, nullable=True)  # Maximum production capacity
    economies_of_scale = Column(Float, default=1.0)  # Economies of scale factor
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to market
    market = relationship("Market", back_populates="firms")

    def get_cost_structure(self) -> CostStructure:
        """Get the cost structure for this firm."""
        return CostStructure(
            marginal_cost=float(self.cost),
            fixed_cost=float(self.fixed_cost),
            capacity_limit=float(self.capacity_limit) if self.capacity_limit else None,
            economies_of_scale=float(self.economies_of_scale),
        )


class Run(Base):  # type: ignore
    """Simulation run tracking.

    Stores metadata about multi-round simulation runs including
    the model type, number of rounds, and creation timestamp.
    """

    __tablename__ = "runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model = Column(String(20), nullable=False)  # "cournot" or "bertrand"
    rounds = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    rounds_data = relationship(
        "Round", back_populates="run", cascade="all, delete-orphan"
    )
    results = relationship("Result", back_populates="run", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="run", cascade="all, delete-orphan")
    collusion_events = relationship(
        "CollusionEvent", back_populates="run", cascade="all, delete-orphan"
    )


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
    defection, regulator interventions, policy shocks, and market entry/exit.
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


class CollusionEvent(Base):  # type: ignore
    """Legacy collusion events table for backward compatibility.

    This table is maintained for existing data but new events should use
    the unified Event table. This enables gradual migration.
    """

    __tablename__ = "collusion_events"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    round_idx = Column(Integer, nullable=False)  # Round index (0-based)
    event_type = Column(String(50), nullable=False)  # Type of event
    firm_id = Column(Integer, nullable=True)  # Firm involved (if applicable)
    description = Column(Text, nullable=False)  # Human-readable description
    event_data = Column(JSON, nullable=True)  # Additional event data
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    run = relationship("Run", back_populates="collusion_events")


class RunConfig(Base):  # type: ignore
    """Configuration for simulation runs.

    Stores parameters that control how simulations are executed,
    including iteration counts, convergence criteria, and random
    seed for reproducible results.
    """

    __tablename__ = "run_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    max_iterations = Column(Integer, default=1000)
    convergence_threshold = Column(Float, default=1e-6)
    random_seed = Column(Integer, nullable=True)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to market
    market = relationship("Market")
