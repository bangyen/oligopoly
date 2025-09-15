"""Economic models for oligopoly market simulation.

This module defines the core economic models used in the oligopoly simulation,
including demand curves, market structures, firm behavior, and simulation configuration.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
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


class Market(Base):  # type: ignore[misc,valid-type]
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
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to firms in this market
    firms = relationship("Firm", back_populates="market")

    def get_demand(self) -> Demand:
        """Get the demand curve for this market."""
        return Demand(a=self.demand_a, b=self.demand_b)


class Firm(Base):  # type: ignore[misc,valid-type]
    """Individual firm participating in the market.

    Represents a firm with its cost structure and strategic parameters.
    Each firm competes in the oligopoly market by choosing quantities
    or prices based on its cost function and market conditions.
    """

    __tablename__ = "firms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    cost = Column(Float, nullable=False)  # Marginal cost
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to market
    market = relationship("Market", back_populates="firms")


class Run(Base):  # type: ignore[misc,valid-type]
    """Simulation run tracking.

    Stores metadata about multi-round simulation runs including
    the model type, number of rounds, and creation timestamp.
    """

    __tablename__ = "runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model = Column(String(20), nullable=False)  # "cournot" or "bertrand"
    rounds = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    rounds_data = relationship(
        "Round", back_populates="run", cascade="all, delete-orphan"
    )
    results = relationship("Result", back_populates="run", cascade="all, delete-orphan")


class Round(Base):  # type: ignore[misc,valid-type]
    """Individual round within a simulation run.

    Tracks each round of a multi-round simulation for
    time-series analysis and debugging.
    """

    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), ForeignKey("runs.id"), nullable=False)
    idx = Column(Integer, nullable=False)  # Round index (0-based)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    run = relationship("Run", back_populates="rounds_data")
    results = relationship(
        "Result", back_populates="round", cascade="all, delete-orphan"
    )


class Result(Base):  # type: ignore[misc,valid-type]
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
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    run = relationship("Run", back_populates="results")
    round = relationship("Round", back_populates="results")


class RunConfig(Base):  # type: ignore[misc,valid-type]
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
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to market
    market = relationship("Market")
