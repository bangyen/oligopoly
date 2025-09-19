"""Dynamic market evolution models for oligopoly simulation.

This module implements market dynamics including entry/exit, innovation,
market growth, and technological change.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MarketEvolutionConfig:
    """Simplified configuration for market evolution dynamics."""

    # Core market dynamics (reduced from 12 to 4 parameters)
    growth_rate: float = 0.02  # Market growth rate per round
    entry_cost: float = 100.0  # Cost to enter the market
    exit_threshold: float = -50.0  # Profit threshold for exit
    innovation_rate: float = 0.1  # Combined innovation parameter (simplified)

    def __post_init__(self) -> None:
        """Validate simplified evolution parameters."""
        if self.entry_cost <= 0:
            raise ValueError(f"Entry cost must be positive, got {self.entry_cost}")
        if not 0 <= self.innovation_rate <= 1:
            raise ValueError(
                f"Innovation rate must be in [0, 1], got {self.innovation_rate}"
            )


@dataclass
class FirmEvolution:
    """Evolution state for a single firm."""

    firm_id: int
    age: int = 0  # Number of rounds in market
    innovation_level: float = 0.0  # Cumulative innovation
    experience: float = 0.0  # Learning-by-doing experience
    market_share_history: List[float] = field(default_factory=list)
    profit_history: List[float] = field(default_factory=list)

    def update_round(self, market_share: float, profit: float) -> None:
        """Update firm state after a round."""
        self.age += 1
        self.market_share_history.append(market_share)
        self.profit_history.append(profit)

        # Update experience (learning-by-doing)
        self.experience += market_share

        # Limit history length
        if len(self.market_share_history) > 20:
            self.market_share_history.pop(0)
        if len(self.profit_history) > 20:
            self.profit_history.pop(0)


@dataclass
class MarketEvolutionState:
    """State of market evolution."""

    round_num: int = 0
    total_market_size: float = 100.0
    technology_level: float = 1.0  # Overall technology level
    num_firms: int = 0
    firm_evolutions: Dict[int, FirmEvolution] = field(default_factory=dict)
    entry_history: List[int] = field(default_factory=list)
    exit_history: List[int] = field(default_factory=list)

    def add_firm(self, firm_id: int) -> None:
        """Add a new firm to the market."""
        self.num_firms += 1
        self.firm_evolutions[firm_id] = FirmEvolution(firm_id=firm_id)
        self.entry_history.append(self.round_num)

    def remove_firm(self, firm_id: int) -> None:
        """Remove a firm from the market."""
        if firm_id in self.firm_evolutions:
            self.num_firms -= 1
            del self.firm_evolutions[firm_id]
            self.exit_history.append(self.round_num)


class MarketEvolutionEngine:
    """Engine for managing market evolution dynamics."""

    def __init__(self, config: MarketEvolutionConfig, seed: Optional[int] = None):
        """Initialize market evolution engine."""
        self.config = config
        self.rng = random.Random(seed)
        self.state = MarketEvolutionState()

    def evolve_market(
        self,
        current_firms: List[int],
        current_profits: List[float],
        current_market_shares: List[float],
        current_costs: List[float],
        current_qualities: List[float],
        demand_params: Dict[str, float],
    ) -> Tuple[List[int], List[float], List[float], Dict[str, float]]:
        """Evolve the market for one round.

        Args:
            current_firms: List of current firm IDs
            current_profits: List of current profits
            current_market_shares: List of current market shares
            current_costs: List of current costs
            current_qualities: List of current qualities
            demand_params: Current demand parameters

        Returns:
            Tuple of (new_firms, new_costs, new_qualities, new_demand_params)
        """
        self.state.round_num += 1

        # Update firm evolutions
        for i, firm_id in enumerate(current_firms):
            if firm_id not in self.state.firm_evolutions:
                self.state.firm_evolutions[firm_id] = FirmEvolution(firm_id=firm_id)

            self.state.firm_evolutions[firm_id].update_round(
                current_market_shares[i], current_profits[i]
            )

        # Market growth
        self._evolve_market_growth(demand_params)

        # Technological change
        self._evolve_technology(current_costs, current_qualities)

        # Entry/Exit dynamics
        new_firms = self._evolve_entry_exit(
            current_firms, current_profits, current_costs, current_qualities
        )

        # Innovation
        new_costs, new_qualities = self._evolve_innovation(
            new_firms, current_costs, current_qualities
        )

        return new_firms, new_costs, new_qualities, demand_params

    def _evolve_market_growth(self, demand_params: Dict[str, float]) -> None:
        """Evolve market size through growth (simplified)."""
        # Simple growth without volatility
        growth_factor = 1.0 + self.config.growth_rate
        self.state.total_market_size *= growth_factor

        # Update demand parameters to reflect market growth
        if "a" in demand_params:  # Linear demand
            demand_params["a"] *= growth_factor
        if "alpha" in demand_params:  # Linear demand (alternative)
            demand_params["alpha"] *= growth_factor

    def _evolve_technology(self, costs: List[float], qualities: List[float]) -> None:
        """Evolve overall technology level with realistic spillovers."""
        # Technology improvement probability depends on market concentration
        # More concentrated markets have higher innovation incentives
        if not costs:
            return

        # Calculate market concentration (HHI proxy)
        total_cost = sum(costs)
        if total_cost > 0:
            cost_shares = [c / total_cost for c in costs]
            concentration = sum(s**2 for s in cost_shares)  # HHI-like measure
        else:
            concentration = 1.0  # Monopoly case

        # Innovation probability increases with concentration but has diminishing returns
        innovation_prob = min(0.3, self.config.innovation_rate * (0.5 + concentration))

        if self.rng.random() < innovation_prob:
            # Technology improvement varies with market size and concentration
            base_improvement = 0.005 + 0.01 * concentration  # 0.5-1.5% improvement
            tech_improvement = self.rng.uniform(
                0.5 * base_improvement, 1.5 * base_improvement
            )

            self.state.technology_level *= 1.0 + tech_improvement

            # Technology spillovers are asymmetric - larger firms benefit more
            for i in range(len(costs)):
                if costs[i] > 0:
                    # Larger firms (lower costs) get better spillovers
                    firm_spillover = (
                        1.0 - (costs[i] / max(costs)) * 0.3
                    )  # 70-100% of improvement
                    effective_improvement = tech_improvement * firm_spillover

                    # Costs decrease with technology
                    costs[i] = max(0.1, costs[i] * (1.0 - effective_improvement))
                    # Qualities increase with technology
                    qualities[i] *= 1.0 + effective_improvement

    def _evolve_entry_exit(
        self,
        current_firms: List[int],
        current_profits: List[float],
        current_costs: List[float],
        current_qualities: List[float],
    ) -> List[int]:
        """Handle entry and exit of firms with economic constraints."""
        new_firms = current_firms.copy()

        # Exit decisions with economic constraints
        firms_to_exit = []
        for i, (firm_id, profit, cost) in enumerate(
            zip(current_firms, current_profits, current_costs)
        ):
            # Firm must exit if:
            # 1. Profit is below exit threshold (can't cover fixed costs)
            # 2. Profit is negative and below a reasonable threshold
            should_exit = (
                profit < self.config.exit_threshold
                or profit < -cost * 0.1  # Exit if losing more than 10% of marginal cost
            )

            if should_exit:
                # Higher probability of exit for more unprofitable firms
                exit_probability = min(0.9, 0.3 + abs(profit) / max(cost, 1.0) * 0.1)
                if self.rng.random() < exit_probability:
                    firms_to_exit.append(i)

        # Remove exiting firms
        for i in reversed(firms_to_exit):
            firm_id = new_firms.pop(i)
            self.state.remove_firm(firm_id)
            current_costs.pop(i)
            current_qualities.pop(i)

        # Entry decisions with economic realism
        if current_firms:
            # Entry probability depends on market profitability and concentration
            avg_profit = np.mean(current_profits)
            market_size = self.state.total_market_size

            # Calculate market concentration
            total_cost = sum(current_costs)
            if total_cost > 0:
                cost_shares = [c / total_cost for c in current_costs]
                concentration = sum(s**2 for s in cost_shares)
            else:
                concentration = 1.0

            # Entry is more likely in profitable, less concentrated markets
            # Scale entry cost with market size
            scaled_entry_cost = float(self.config.entry_cost * (market_size / 100.0))
            entry_probability = min(
                0.2,
                max(
                    0.01,
                    float(
                        (avg_profit / scaled_entry_cost) * (1.0 - concentration) * 0.1
                    ),
                ),
            )

            if self.rng.random() < entry_probability and avg_profit > scaled_entry_cost:
                new_firm_id = max(current_firms) + 1 if current_firms else 0
                new_firms.append(new_firm_id)
                self.state.add_firm(new_firm_id)

                # Add costs and qualities for new firm
                new_cost = self._generate_entrant_cost(current_costs)
                new_quality = self._generate_entrant_quality(current_qualities)
                current_costs.append(new_cost)
                current_qualities.append(new_quality)

        return new_firms

    def _should_enter(
        self,
        current_firms: List[int],
        current_profits: List[float],
        current_costs: List[float],
    ) -> bool:
        """Determine if a new firm should enter the market (simplified)."""
        if not current_firms:
            return True  # Enter empty market

        # Simple entry rule: enter if average profit exceeds entry cost
        avg_profit = float(np.mean(current_profits))
        return avg_profit > self.config.entry_cost

    def _generate_entrant_cost(self, current_costs: List[float]) -> float:
        """Generate cost for new entrant."""
        if not current_costs:
            return 10.0  # Default cost

        # New entrant has cost around the average
        avg_cost = float(np.mean(current_costs))
        std_cost = float(np.std(current_costs))

        # Add some randomness
        new_cost = float(self.rng.gauss(avg_cost, std_cost * 0.5))
        return max(1.0, new_cost)  # Ensure positive cost

    def _generate_entrant_quality(self, current_qualities: List[float]) -> float:
        """Generate quality for new entrant."""
        if not current_qualities:
            return 1.0  # Default quality

        # New entrant has quality around the average
        avg_quality = float(np.mean(current_qualities))
        std_quality = float(np.std(current_qualities))

        # Add some randomness
        new_quality = float(self.rng.gauss(avg_quality, std_quality * 0.5))
        return max(0.1, new_quality)  # Ensure positive quality

    def _evolve_innovation(
        self,
        firms: List[int],
        current_costs: List[float],
        current_qualities: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Handle innovation by firms."""
        new_costs = current_costs.copy()
        new_qualities = current_qualities.copy()

        for i, firm_id in enumerate(firms):
            if firm_id not in self.state.firm_evolutions:
                continue

            firm_evolution = self.state.firm_evolutions[firm_id]

            # Innovation probability depends on firm characteristics
            innovation_prob = self._calculate_innovation_probability(firm_evolution)

            if self.rng.random() < innovation_prob:
                # Firm attempts innovation
                if self.rng.random() < 0.3:  # Fixed success rate
                    # Successful innovation
                    self._apply_innovation(firm_evolution, new_costs, new_qualities, i)

        return new_costs, new_qualities

    def _calculate_innovation_probability(self, firm_evolution: FirmEvolution) -> float:
        """Calculate innovation probability for a firm (simplified)."""
        # Simple innovation probability based on market share
        if firm_evolution.market_share_history:
            avg_market_share = float(np.mean(firm_evolution.market_share_history[-5:]))
            # Higher market share = higher innovation probability
            return min(0.3, avg_market_share * 0.5)
        else:
            return 0.1  # Default probability

    def _apply_innovation(
        self,
        firm_evolution: FirmEvolution,
        costs: List[float],
        qualities: List[float],
        firm_index: int,
    ) -> None:
        """Apply successful innovation to a firm (simplified)."""
        # Update firm's innovation level
        firm_evolution.innovation_level += 0.1  # Fixed improvement

        # Reduce costs by 5%
        costs[firm_index] = max(1.0, costs[firm_index] * 0.95)

        # Increase quality by 5%
        qualities[firm_index] *= 1.05

    def get_evolution_metrics(self) -> Dict[str, float]:
        """Get metrics about market evolution."""
        return {
            "round_num": self.state.round_num,
            "total_market_size": self.state.total_market_size,
            "technology_level": self.state.technology_level,
            "num_firms": self.state.num_firms,
            "total_entries": len(self.state.entry_history),
            "total_exits": len(self.state.exit_history),
            "net_entries": len(self.state.entry_history) - len(self.state.exit_history),
        }

    def get_firm_evolution_metrics(self, firm_id: int) -> Optional[Dict[str, float]]:
        """Get evolution metrics for a specific firm."""
        if firm_id not in self.state.firm_evolutions:
            return None

        firm_evolution = self.state.firm_evolutions[firm_id]

        return {
            "firm_id": firm_id,
            "age": firm_evolution.age,
            "innovation_level": firm_evolution.innovation_level,
            "experience": firm_evolution.experience,
            "avg_market_share": float(
                np.mean(firm_evolution.market_share_history)
                if firm_evolution.market_share_history
                else 0.0
            ),
            "avg_profit": float(
                np.mean(firm_evolution.profit_history)
                if firm_evolution.profit_history
                else 0.0
            ),
        }


def create_market_evolution_engine(
    config: Optional[MarketEvolutionConfig] = None, seed: Optional[int] = None
) -> MarketEvolutionEngine:
    """Create a market evolution engine with default or custom configuration."""
    if config is None:
        config = MarketEvolutionConfig()

    return MarketEvolutionEngine(config, seed)
