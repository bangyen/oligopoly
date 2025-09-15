"""Economic models for oligopoly market simulation.

This module defines the core economic models used in the oligopoly simulation,
including demand curves, market structures, firm behavior, and simulation configuration.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

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


class Market(Base):
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


class Firm(Base):
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


class RunConfig(Base):
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
