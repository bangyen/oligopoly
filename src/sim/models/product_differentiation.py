"""Product differentiation models for oligopoly simulation.

This module implements various forms of product differentiation including
horizontal differentiation (Hotelling model), vertical differentiation,
and differentiated Bertrand competition.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ProductCharacteristics:
    """Product characteristics for differentiated products."""

    quality: float = 1.0  # Vertical differentiation (higher = better)
    location: float = 0.5  # Horizontal differentiation (0-1 scale)
    brand_strength: float = 1.0  # Brand loyalty factor
    innovation_level: float = 0.0  # Innovation/R&D level

    def __post_init__(self) -> None:
        """Validate product characteristics."""
        if self.quality <= 0:
            raise ValueError(f"Quality must be positive, got {self.quality}")
        if not 0 <= self.location <= 1:
            raise ValueError(f"Location must be in [0, 1], got {self.location}")
        if self.brand_strength <= 0:
            raise ValueError(
                f"Brand strength must be positive, got {self.brand_strength}"
            )
        if self.innovation_level < 0:
            raise ValueError(
                f"Innovation level must be non-negative, got {self.innovation_level}"
            )


@dataclass
class HotellingDemand:
    """Hotelling model for horizontal product differentiation.

    Consumers are uniformly distributed along a line [0, 1].
    Each firm has a location on this line, and consumers choose
    the firm that minimizes total cost (price + transportation cost).
    """

    transportation_cost: float = 1.0  # Cost per unit distance
    consumer_density: float = 1.0  # Density of consumers

    def __post_init__(self) -> None:
        """Validate Hotelling parameters."""
        if self.transportation_cost <= 0:
            raise ValueError(
                f"Transportation cost must be positive, got {self.transportation_cost}"
            )
        if self.consumer_density <= 0:
            raise ValueError(
                f"Consumer density must be positive, got {self.consumer_density}"
            )

    def calculate_demand(
        self, prices: List[float], locations: List[float]
    ) -> List[float]:
        """Calculate demand for each firm in Hotelling model.

        Args:
            prices: List of prices for each firm
            locations: List of locations for each firm (0-1 scale)

        Returns:
            List of demand quantities for each firm
        """
        n = len(prices)
        if len(locations) != n:
            raise ValueError("Prices and locations must have same length")

        if n == 1:
            # Monopoly case
            return [self.consumer_density]

        # Calculate market boundaries
        boundaries = self._calculate_boundaries(prices, locations)

        # Calculate demand for each firm
        demands = []
        for i in range(n):
            if i == 0:
                # Leftmost firm
                demand = boundaries[0] * self.consumer_density
            elif i == n - 1:
                # Rightmost firm
                demand = (1.0 - boundaries[-1]) * self.consumer_density
            else:
                # Middle firm
                demand = (boundaries[i] - boundaries[i - 1]) * self.consumer_density

            demands.append(max(0.0, demand))

        return demands

    def _calculate_boundaries(
        self, prices: List[float], locations: List[float]
    ) -> List[float]:
        """Calculate market boundaries between firms."""
        n = len(prices)
        boundaries = []

        for i in range(n - 1):
            # Boundary between firm i and firm i+1
            # Consumer at boundary is indifferent between firms
            # p_i + t * |x - l_i| = p_{i+1} + t * |x - l_{i+1}|

            p_i, p_j = prices[i], prices[i + 1]
            l_i, l_j = locations[i], locations[i + 1]
            t = self.transportation_cost

            # Solve for boundary location x
            if l_i < l_j:
                # Firm i is to the left of firm j
                x = (p_j - p_i + t * (l_i + l_j)) / (2 * t)
            else:
                # Firm i is to the right of firm j
                x = (p_i - p_j + t * (l_i + l_j)) / (2 * t)

            # Ensure boundary is between firm locations
            x = max(l_i, min(l_j, x))
            boundaries.append(x)

        return boundaries


@dataclass
class LogitDemand:
    """Logit demand model for differentiated products.

    Consumers choose products based on utility functions that include
    product characteristics and prices. Market shares are determined
    by the logit model.
    """

    price_sensitivity: float = 1.0  # Beta parameter
    quality_sensitivity: float = 1.0  # Alpha parameter
    outside_option_utility: float = 0.0  # Utility of outside option

    def __post_init__(self) -> None:
        """Validate logit parameters."""
        if self.price_sensitivity <= 0:
            raise ValueError(
                f"Price sensitivity must be positive, got {self.price_sensitivity}"
            )
        if self.quality_sensitivity <= 0:
            raise ValueError(
                f"Quality sensitivity must be positive, got {self.quality_sensitivity}"
            )

    def calculate_market_shares(
        self, prices: List[float], products: List[ProductCharacteristics]
    ) -> List[float]:
        """Calculate market shares using logit model.

        Args:
            prices: List of prices for each product
            products: List of product characteristics

        Returns:
            List of market shares for each product
        """
        if len(prices) != len(products):
            raise ValueError("Prices and products must have same length")

        # Calculate utilities
        utilities = []
        for price, product in zip(prices, products):
            utility = (
                self.quality_sensitivity * product.quality
                - self.price_sensitivity * price
                + product.brand_strength
                + product.innovation_level
            )
            utilities.append(utility)

        # Add outside option
        utilities.append(self.outside_option_utility)

        # Calculate market shares using logit formula
        exp_utilities = [math.exp(u) for u in utilities]
        total_exp_utility = sum(exp_utilities)

        market_shares = [exp_u / total_exp_utility for exp_u in exp_utilities[:-1]]

        return market_shares

    def calculate_demand(
        self,
        prices: List[float],
        products: List[ProductCharacteristics],
        total_market_size: float = 100.0,
    ) -> List[float]:
        """Calculate demand quantities using logit model.

        Args:
            prices: List of prices for each product
            products: List of product characteristics
            total_market_size: Total market size

        Returns:
            List of demand quantities for each product
        """
        market_shares = self.calculate_market_shares(prices, products)
        return [share * total_market_size for share in market_shares]


@dataclass
class VerticalDifferentiation:
    """Vertical differentiation model with quality competition.

    Products differ in quality, and consumers have heterogeneous
    willingness to pay for quality.
    """

    consumer_heterogeneity: float = 1.0  # Variance in consumer preferences
    base_utility: float = 0.0  # Base utility from consumption

    def __post_init__(self) -> None:
        """Validate vertical differentiation parameters."""
        if self.consumer_heterogeneity <= 0:
            raise ValueError(
                f"Consumer heterogeneity must be positive, got {self.consumer_heterogeneity}"
            )

    def calculate_demand(
        self,
        prices: List[float],
        qualities: List[float],
        total_market_size: float = 100.0,
    ) -> List[float]:
        """Calculate demand with vertical differentiation.

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
        demands = []

        for i in range(n):
            # Calculate demand for product i
            # Consumer with type theta buys product i if:
            # theta * quality_i - price_i >= theta * quality_j - price_j for all j

            demand = 0.0

            # For each consumer type theta
            num_consumer_types = 100  # Discretize consumer types
            for theta_idx in range(num_consumer_types):
                theta = theta_idx / num_consumer_types

                # Calculate utility from each product
                utilities = []
                for j in range(n):
                    utility = theta * qualities[j] - prices[j]
                    utilities.append(utility)

                # Consumer chooses product with highest utility
                if utilities[i] == max(utilities):
                    demand += 1.0 / num_consumer_types

            demands.append(demand * total_market_size)

        return demands


@dataclass
class DifferentiatedBertrandResult:
    """Result from differentiated Bertrand competition."""

    prices: List[float]
    quantities: List[float]
    market_shares: List[float]
    profits: List[float]
    total_demand: float
    consumer_surplus: float

    def __repr__(self) -> str:
        """String representation of result."""
        return (
            f"DifferentiatedBertrandResult("
            f"prices={[f'{p:.2f}' for p in self.prices]}, "
            f"quantities={[f'{q:.2f}' for q in self.quantities]}, "
            f"market_shares={[f'{s:.3f}' for s in self.market_shares]}, "
            f"profits={[f'{p:.2f}' for p in self.profits]})"
        )


def differentiated_bertrand_simulation(
    prices: List[float],
    products: List[ProductCharacteristics],
    costs: List[float],
    demand_model: str = "logit",
    demand_params: Optional[dict] = None,
    total_market_size: float = 100.0,
) -> DifferentiatedBertrandResult:
    """Run differentiated Bertrand competition simulation.

    Args:
        prices: List of prices set by each firm
        products: List of product characteristics
        costs: List of marginal costs for each firm
        demand_model: Type of demand model ("logit", "hotelling", "vertical")
        demand_params: Parameters for demand model
        total_market_size: Total market size

    Returns:
        DifferentiatedBertrandResult with market outcomes

    Raises:
        ValueError: If inputs are invalid
    """
    if len(prices) != len(products) or len(prices) != len(costs):
        raise ValueError("Prices, products, and costs must have same length")

    if demand_params is None:
        demand_params = {}

    # Calculate demand based on model
    if demand_model == "logit":
        logit_demand = LogitDemand(**demand_params)
        quantities = logit_demand.calculate_demand(prices, products, total_market_size)
        market_shares = logit_demand.calculate_market_shares(prices, products)

    elif demand_model == "hotelling":
        if "locations" not in demand_params:
            # Use product locations
            locations = [p.location for p in products]
        else:
            locations = demand_params["locations"]

        hotelling_demand = HotellingDemand(
            **{k: v for k, v in demand_params.items() if k != "locations"}
        )
        quantities = hotelling_demand.calculate_demand(prices, locations)
        total_qty = sum(quantities)
        market_shares = [q / total_qty if total_qty > 0 else 0.0 for q in quantities]

    elif demand_model == "vertical":
        qualities = [p.quality for p in products]
        vertical_demand = VerticalDifferentiation(**demand_params)
        quantities = vertical_demand.calculate_demand(
            prices, qualities, total_market_size
        )
        total_qty = sum(quantities)
        market_shares = [q / total_qty if total_qty > 0 else 0.0 for q in quantities]

    else:
        raise ValueError(f"Unknown demand model: {demand_model}")

    # Calculate profits
    profits = [(p - c) * q for p, c, q in zip(prices, costs, quantities)]

    # Calculate consumer surplus (simplified)
    total_demand = sum(quantities)
    if demand_model == "logit":
        # Consumer surplus from logit model
        logit_demand = LogitDemand(**demand_params)
        utilities = []
        for price, product in zip(prices, products):
            utility = (
                logit_demand.quality_sensitivity * product.quality
                - logit_demand.price_sensitivity * price
                + product.brand_strength
                + product.innovation_level
            )
            utilities.append(utility)

        # Add outside option
        utilities.append(logit_demand.outside_option_utility)

        # Consumer surplus = log(sum(exp(utilities))) / price_sensitivity
        exp_utilities = [math.exp(u) for u in utilities]
        consumer_surplus = math.log(sum(exp_utilities)) / logit_demand.price_sensitivity
    else:
        # Simplified consumer surplus
        consumer_surplus = 0.5 * total_demand * (100.0 - float(np.mean(prices)))

    return DifferentiatedBertrandResult(
        prices=prices,
        quantities=quantities,
        market_shares=market_shares,
        profits=profits,
        total_demand=total_demand,
        consumer_surplus=consumer_surplus,
    )


def calculate_differentiated_nash_equilibrium(
    products: List[ProductCharacteristics],
    costs: List[float],
    demand_model: str = "logit",
    demand_params: Optional[dict] = None,
    total_market_size: float = 100.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[List[float], DifferentiatedBertrandResult]:
    """Calculate Nash equilibrium for differentiated Bertrand competition.

    Args:
        products: List of product characteristics
        costs: List of marginal costs
        demand_model: Type of demand model
        demand_params: Parameters for demand model
        total_market_size: Total market size
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance

    Returns:
        Tuple of (equilibrium_prices, equilibrium_result)
    """
    n = len(products)
    if len(costs) != n:
        raise ValueError("Products and costs must have same length")

    # Initialize prices at marginal costs
    prices = costs.copy()

    for iteration in range(max_iterations):
        old_prices = prices.copy()

        # Update each firm's price (best response)
        for i in range(n):
            # Calculate best response for firm i
            best_price = _calculate_best_response_price(
                i,
                prices,
                products,
                costs,
                demand_model,
                demand_params,
                total_market_size,
            )
            prices[i] = best_price

        # Check convergence
        max_change = max(abs(p - old_p) for p, old_p in zip(prices, old_prices))
        if max_change < tolerance:
            break

    # Calculate final result
    result = differentiated_bertrand_simulation(
        prices, products, costs, demand_model, demand_params, total_market_size
    )

    return prices, result


def _calculate_best_response_price(
    firm_index: int,
    current_prices: List[float],
    products: List[ProductCharacteristics],
    costs: List[float],
    demand_model: str,
    demand_params: Optional[dict],
    total_market_size: float,
) -> float:
    """Calculate best response price for a single firm."""
    # Use numerical optimization to find best response
    # For simplicity, use a grid search

    min_price = costs[firm_index] + 0.1
    max_price = costs[firm_index] * 3.0  # Reasonable upper bound

    best_price = min_price
    best_profit: float = -float("inf")

    # Grid search over prices
    num_points = 50
    for i in range(num_points):
        test_price = min_price + (max_price - min_price) * i / (num_points - 1)

        # Create test prices
        test_prices = current_prices.copy()
        test_prices[firm_index] = test_price

        # Calculate profit
        result = differentiated_bertrand_simulation(
            test_prices, products, costs, demand_model, demand_params, total_market_size
        )

        profit = result.profits[firm_index]
        if profit > best_profit:
            best_profit = profit
            best_price = test_price

    return best_price
