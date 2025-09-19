"""Enhanced demand functions for oligopoly simulation.

This module implements advanced demand functions including CES demand,
network effects, and dynamic demand evolution.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CESDemand:
    """Constant Elasticity of Substitution (CES) demand function.

    CES demand is commonly used in monopolistic competition models
    and allows for different degrees of substitutability between products.
    """

    elasticity: float = 2.0  # Elasticity of substitution (must be > 1)
    scale_parameter: float = 1.0  # Scale parameter
    market_size: float = 100.0  # Total market size

    def __post_init__(self) -> None:
        """Validate CES parameters."""
        if self.elasticity <= 1.0:
            raise ValueError(f"Elasticity must be > 1, got {self.elasticity}")
        if self.scale_parameter <= 0:
            raise ValueError(
                f"Scale parameter must be positive, got {self.scale_parameter}"
            )
        if self.market_size <= 0:
            raise ValueError(f"Market size must be positive, got {self.market_size}")

    def calculate_demand(
        self,
        prices: List[float],
        qualities: List[float],
        market_share: Optional[List[float]] = None,
    ) -> List[float]:
        """Calculate demand using CES function.

        Args:
            prices: List of prices for each product
            qualities: List of quality levels for each product
            market_share: Optional market share weights

        Returns:
            List of demand quantities for each product
        """
        if len(prices) != len(qualities):
            raise ValueError("Prices and qualities must have same length")

        n = len(prices)
        if n == 0:
            return []

        # Calculate quality-adjusted prices
        quality_adjusted_prices = [p / q for p, q in zip(prices, qualities)]

        # Calculate CES demand
        # q_i = (A * Q * (p_i/q_i)^(-σ)) / (Σ_j (p_j/q_j)^(1-σ))
        # where σ is elasticity of substitution

        sigma = self.elasticity

        # Calculate denominator: sum of (p_j/q_j)^(1-σ)
        denominator = sum(p ** (1 - sigma) for p in quality_adjusted_prices)

        if denominator <= 0:
            # Fallback to equal shares
            return [self.market_size / n] * n

        # Calculate demand for each product
        demands = []
        for p in quality_adjusted_prices:
            if p <= 0:
                demands.append(0.0)
            else:
                demand = (
                    self.scale_parameter * self.market_size * p ** (-sigma)
                ) / denominator
                demands.append(demand)

        return demands

    def calculate_market_shares(
        self, prices: List[float], qualities: List[float]
    ) -> List[float]:
        """Calculate market shares using CES function."""
        demands = self.calculate_demand(prices, qualities)
        total_demand = sum(demands)

        if total_demand <= 0:
            return [1.0 / len(prices)] * len(prices)

        return [d / total_demand for d in demands]


@dataclass
class NetworkEffectsDemand:
    """Demand function with network effects.

    Network effects occur when the value of a product increases
    with the number of users (e.g., social networks, platforms).
    """

    network_strength: float = 0.1  # Strength of network effects
    base_demand: float = 100.0  # Base demand without network effects
    critical_mass: float = 10.0  # Critical mass for network effects

    def __post_init__(self) -> None:
        """Validate network effects parameters."""
        if self.network_strength < 0:
            raise ValueError(
                f"Network strength must be non-negative, got {self.network_strength}"
            )
        if self.base_demand <= 0:
            raise ValueError(f"Base demand must be positive, got {self.base_demand}")
        if self.critical_mass <= 0:
            raise ValueError(
                f"Critical mass must be positive, got {self.critical_mass}"
            )

    def calculate_demand(
        self,
        prices: List[float],
        current_users: List[float],
        qualities: List[float],
        price_sensitivity: float = 1.0,
    ) -> List[float]:
        """Calculate demand with network effects.

        Args:
            prices: List of prices for each product
            current_users: List of current user counts for each product
            qualities: List of quality levels for each product
            price_sensitivity: Price sensitivity parameter

        Returns:
            List of demand quantities for each product
        """
        if len(prices) != len(current_users) or len(prices) != len(qualities):
            raise ValueError(
                "Prices, current_users, and qualities must have same length"
            )

        demands = []
        for i, (price, users, quality) in enumerate(
            zip(prices, current_users, qualities)
        ):
            # Base demand from quality and price
            base_demand = self.base_demand * quality / (1.0 + price_sensitivity * price)

            # Network effects: demand increases with number of users
            if users >= self.critical_mass:
                network_effect = self.network_strength * (users - self.critical_mass)
            else:
                network_effect = 0.0

            # Total demand
            total_demand = base_demand + network_effect
            demands.append(max(0.0, total_demand))

        return demands

    def calculate_network_value(
        self, current_users: List[float], qualities: List[float]
    ) -> List[float]:
        """Calculate network value for each product."""
        if len(current_users) != len(qualities):
            raise ValueError("Current_users and qualities must have same length")

        network_values = []
        for users, quality in zip(current_users, qualities):
            # Network value = quality + network effects
            if users >= self.critical_mass:
                network_effect = self.network_strength * (users - self.critical_mass)
            else:
                network_effect = 0.0

            network_value = quality + network_effect
            network_values.append(network_value)

        return network_values


@dataclass
class DynamicDemand:
    """Dynamic demand function that evolves over time.

    Demand can change due to:
    - Market growth/decline
    - Seasonal effects
    - Economic cycles
    - Consumer learning
    """

    base_demand: float = 100.0
    growth_rate: float = 0.02  # Annual growth rate
    volatility: float = 0.1  # Demand volatility
    seasonal_amplitude: float = 0.1  # Seasonal variation amplitude
    cycle_period: int = 12  # Economic cycle period (rounds)

    def __post_init__(self) -> None:
        """Validate dynamic demand parameters."""
        if self.base_demand <= 0:
            raise ValueError(f"Base demand must be positive, got {self.base_demand}")
        if self.volatility < 0:
            raise ValueError(f"Volatility must be non-negative, got {self.volatility}")
        if self.seasonal_amplitude < 0:
            raise ValueError(
                f"Seasonal amplitude must be non-negative, got {self.seasonal_amplitude}"
            )
        if self.cycle_period <= 0:
            raise ValueError(f"Cycle period must be positive, got {self.cycle_period}")

    def calculate_demand(
        self,
        round_num: int,
        prices: List[float],
        qualities: List[float],
        price_sensitivity: float = 1.0,
    ) -> Tuple[List[float], float]:
        """Calculate dynamic demand for current round.

        Args:
            round_num: Current round number
            prices: List of prices for each product
            qualities: List of quality levels for each product
            price_sensitivity: Price sensitivity parameter

        Returns:
            Tuple of (demand_quantities, total_market_size)
        """
        if len(prices) != len(qualities):
            raise ValueError("Prices and qualities must have same length")

        # Calculate total market size with dynamics
        total_market_size = self._calculate_market_size(round_num)

        # Calculate demand for each product
        demands = []
        for price, quality in zip(prices, qualities):
            # Base demand from quality and price
            base_demand = (
                total_market_size * quality / (1.0 + price_sensitivity * price)
            )
            demands.append(max(0.0, base_demand))

        return demands, total_market_size

    def _calculate_market_size(self, round_num: int) -> float:
        """Calculate total market size for given round."""
        # Growth component
        growth_factor = (1.0 + self.growth_rate) ** (round_num / 12.0)  # Annual growth

        # Seasonal component
        seasonal_factor = 1.0 + self.seasonal_amplitude * math.sin(
            2 * math.pi * round_num / 12.0
        )

        # Economic cycle component
        cycle_factor = 1.0 + 0.1 * math.sin(2 * math.pi * round_num / self.cycle_period)

        # Random volatility
        volatility_factor = 1.0 + np.random.normal(0, self.volatility)

        # Total market size
        market_size = (
            self.base_demand
            * growth_factor
            * seasonal_factor
            * cycle_factor
            * volatility_factor
        )

        return float(max(0.1, market_size))  # Ensure positive market size


@dataclass
class MultiSegmentDemand:
    """Demand function with multiple consumer segments.

    Each segment has different preferences and price sensitivities.
    """

    segments: List[Dict[str, float]]  # List of segment parameters

    def __post_init__(self) -> None:
        """Validate multi-segment demand parameters."""
        if not self.segments:
            raise ValueError("At least one segment must be provided")

        for i, segment in enumerate(self.segments):
            required_keys = ["weight", "price_sensitivity", "quality_preference"]
            for key in required_keys:
                if key not in segment:
                    raise ValueError(f"Segment {i} missing required key: {key}")

            if not 0 <= segment["weight"] <= 1:
                raise ValueError(
                    f"Segment {i} weight must be in [0, 1], got {segment['weight']}"
                )
            if segment["price_sensitivity"] <= 0:
                raise ValueError(
                    f"Segment {i} price sensitivity must be positive, got {segment['price_sensitivity']}"
                )

        # Check weights sum to 1
        total_weight = sum(segment["weight"] for segment in self.segments)
        if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
            raise ValueError(f"Segment weights must sum to 1.0, got {total_weight}")

    def calculate_demand(
        self,
        prices: List[float],
        qualities: List[float],
        total_market_size: float = 100.0,
    ) -> List[float]:
        """Calculate demand across multiple segments.

        Args:
            prices: List of prices for each product
            qualities: List of quality levels for each product
            total_market_size: Total market size

        Returns:
            List of demand quantities for each product
        """
        if len(prices) != len(qualities):
            raise ValueError("Prices and qualities must have same length")

        n = len(prices)
        total_demands = [0.0] * n

        # Calculate demand for each segment
        for segment in self.segments:
            weight = segment["weight"]
            price_sensitivity = segment["price_sensitivity"]
            quality_preference = segment["quality_preference"]

            segment_market_size = weight * total_market_size

            # Calculate demand in this segment
            segment_demands = []
            for price, quality in zip(prices, qualities):
                # Demand = market_size * quality_preference * quality / (1 + price_sensitivity * price)
                demand = (
                    segment_market_size
                    * quality_preference
                    * quality
                    / (1.0 + price_sensitivity * price)
                )
                segment_demands.append(max(0.0, demand))

            # Add to total demands
            for i in range(n):
                total_demands[i] += segment_demands[i]

        return total_demands

    def calculate_market_shares(
        self,
        prices: List[float],
        qualities: List[float],
        total_market_size: float = 100.0,
    ) -> List[float]:
        """Calculate market shares across multiple segments."""
        demands = self.calculate_demand(prices, qualities, total_market_size)
        total_demand = sum(demands)

        if total_demand <= 0:
            return [1.0 / len(prices)] * len(prices)

        return [d / total_demand for d in demands]


def create_enhanced_demand_function(
    demand_type: str, **kwargs: Any
) -> Union[CESDemand, NetworkEffectsDemand, DynamicDemand, MultiSegmentDemand]:
    """Factory function to create enhanced demand functions.

    Args:
        demand_type: Type of demand function ("ces", "network", "dynamic", "multi_segment")
        **kwargs: Demand function specific parameters

    Returns:
        Enhanced demand function instance

    Raises:
        ValueError: If demand type is unknown
    """
    if demand_type == "ces":
        return CESDemand(**kwargs)
    elif demand_type == "network":
        return NetworkEffectsDemand(**kwargs)
    elif demand_type == "dynamic":
        return DynamicDemand(**kwargs)
    elif demand_type == "multi_segment":
        return MultiSegmentDemand(**kwargs)
    else:
        raise ValueError(f"Unknown enhanced demand type: {demand_type}")


def calculate_enhanced_demand_elasticity(
    demand_function: Union[
        CESDemand, NetworkEffectsDemand, DynamicDemand, MultiSegmentDemand
    ],
    prices: List[float],
    qualities: List[float],
    price_index: int = 0,
    price_change: float = 0.01,
) -> float:
    """Calculate price elasticity of demand for enhanced demand functions.

    Args:
        demand_function: Enhanced demand function
        prices: List of prices
        qualities: List of qualities
        price_index: Index of price to vary
        price_change: Small price change for elasticity calculation

    Returns:
        Price elasticity of demand
    """
    # Calculate demand at current prices
    if isinstance(demand_function, DynamicDemand):
        current_demands, _ = demand_function.calculate_demand(0, prices, qualities)
    elif isinstance(demand_function, NetworkEffectsDemand):
        # For network effects, we need current users - use dummy values
        current_users = [10.0] * len(prices)
        current_demands = demand_function.calculate_demand(
            prices, current_users, qualities
        )
    else:
        current_demands = demand_function.calculate_demand(prices, qualities)

    current_demand = current_demands[price_index]
    current_price = prices[price_index]

    # Calculate demand at new price
    new_prices = prices.copy()
    new_prices[price_index] = current_price * (1.0 + price_change)

    if isinstance(demand_function, DynamicDemand):
        new_demands, _ = demand_function.calculate_demand(0, new_prices, qualities)
    elif isinstance(demand_function, NetworkEffectsDemand):
        # For network effects, we need current users - use dummy values
        current_users = [10.0] * len(new_prices)
        new_demands = demand_function.calculate_demand(
            new_prices, current_users, qualities
        )
    else:
        new_demands = demand_function.calculate_demand(new_prices, qualities)

    new_demand = new_demands[price_index]

    # Calculate elasticity
    if current_demand <= 0:
        return 0.0

    elasticity = ((new_demand - current_demand) / current_demand) / price_change
    return elasticity
