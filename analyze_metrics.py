#!/usr/bin/env python3
"""Analyze oligopoly simulation metrics to find the most interesting visualization.

This script runs comprehensive simulations across parameter combinations and analyzes
different metric pairs to identify the most visually compelling and economically meaningful
combination for the frontend visualization.
"""

import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sim.games.cournot import cournot_simulation
from sim.models.metrics import (
    calculate_consumer_surplus,
    calculate_hhi,
    calculate_market_shares_cournot,
)


def compute_profit_variance(profits: List[float]) -> float:
    """Calculate variance in profits across firms."""
    if not profits:
        return 0.0
    mean_profit = sum(profits) / len(profits)
    variance = sum((p - mean_profit) ** 2 for p in profits) / len(profits)
    return variance


def compute_quantity_volatility(quantities: List[float]) -> float:
    """Calculate volatility in quantities across firms."""
    if not quantities:
        return 0.0
    mean_qty = sum(quantities) / len(quantities)
    variance = sum((q - mean_qty) ** 2 for q in quantities) / len(quantities)
    return variance


def compute_market_share_dispersion(shares: List[float]) -> float:
    """Calculate coefficient of variation for market shares."""
    if not shares or all(s == 0 for s in shares):
        return 0.0
    mean_share = sum(shares) / len(shares)
    if mean_share == 0:
        return 0.0
    std_dev = math.sqrt(sum((s - mean_share) ** 2 for s in shares) / len(shares))
    return std_dev / mean_share if mean_share > 0 else 0.0


def compute_profit_margin(
    price: float, costs: List[float], quantities: List[float]
) -> float:
    """Calculate average profit margin across firms."""
    if not quantities:
        return 0.0
    margins = []
    for cost, qty in zip(costs, quantities):
        if qty > 0:
            margin = (price - cost) / price  # as percentage
            margins.append(margin)
    return sum(margins) / len(margins) if margins else 0.0


def run_simulation_rounds(
    num_firms: int, demand_elasticity: float, base_price: float, rounds: int = 15
) -> List[Dict]:
    """Run multiple rounds of Cournot simulation."""
    # Set demand parameters based on elasticity
    # For linear demand: P = a - b*Q
    # Elasticity = -b * P / Q
    # We'll use a = base_price * 2 and adjust b based on elasticity
    a = base_price * 2
    b = demand_elasticity / 10.0  # Scale factor

    # Set costs with some variation
    costs = [10.0 + i * 2.0 for i in range(num_firms)]
    fixed_costs = [50.0] * num_firms

    # Initialize quantities near Nash equilibrium
    # Approximate Nash equilibrium for Cournot: q_i = (a - c_i) / (b * (n + 1))
    nash_quantities = [max(1.0, (a - cost) / (b * (num_firms + 1))) for cost in costs]

    # Add small randomness
    quantities = [max(0.1, q + random.uniform(-2, 2)) for q in nash_quantities]

    results = []
    for round_num in range(rounds):
        # Run simulation
        result = cournot_simulation(a, b, costs, quantities, fixed_costs)

        # Calculate metrics
        market_shares = calculate_market_shares_cournot(result.quantities)
        hhi = calculate_hhi(market_shares)
        consumer_surplus = calculate_consumer_surplus(
            a, result.price, sum(result.quantities)
        )
        profit_variance = compute_profit_variance(result.profits)
        quantity_volatility = compute_quantity_volatility(result.quantities)
        market_share_dispersion = compute_market_share_dispersion(market_shares)
        profit_margin = compute_profit_margin(result.price, costs, result.quantities)

        # Store results
        data_point = {
            "round": round_num + 1,
            "price": round(result.price, 2),
            "hhi": round(hhi, 4),
            "consumer_surplus": round(consumer_surplus, 2),
            "profit_variance": round(profit_variance, 2),
            "quantity_volatility": round(quantity_volatility, 2),
            "market_share_dispersion": round(market_share_dispersion, 4),
            "profit_margin": round(profit_margin, 4),
            "total_profit": round(sum(result.profits), 2),
            "num_firms": num_firms,
            "model_type": "cournot",
            "demand_elasticity": demand_elasticity,
            "base_price": base_price,
            "collusion_enabled": False,
        }
        results.append(data_point)

        # Update quantities for next round using adaptive strategy
        # Simple adaptive: adjust based on profit performance
        for i in range(num_firms):
            if result.profits[i] > 0:
                # Increase quantity if profitable
                quantities[i] = quantities[i] * 1.05
            else:
                # Decrease quantity if unprofitable
                quantities[i] = quantities[i] * 0.95

        # Add some randomness
        quantities = [max(0.1, q + random.uniform(-0.5, 0.5)) for q in quantities]

    return results


def analyze_metric_combinations(all_results: List[Dict]) -> Dict:
    """Analyze different metric combinations for visual interest."""
    # Define potential metric pairs
    metric_pairs = [
        ("price", "hhi", "Price vs Market Concentration"),
        ("price", "consumer_surplus", "Price vs Consumer Welfare"),
        ("profit_variance", "hhi", "Competition Intensity vs Concentration"),
        ("consumer_surplus", "hhi", "Welfare vs Concentration"),
        ("profit_margin", "quantity_volatility", "Profitability vs Volatility"),
        (
            "profit_variance",
            "quantity_volatility",
            "Profit Inequality vs Quantity Volatility",
        ),
        ("market_share_dispersion", "hhi", "Share Dispersion vs Concentration"),
        ("consumer_surplus", "profit_variance", "Welfare vs Inequality"),
    ]

    analysis = {}
    for metric_x, metric_y, name in metric_pairs:
        # Calculate correlation
        correlations = []
        x_ranges = []
        y_ranges = []

        for param_set in [r for r in all_results if r["round"] == 1]:
            # Get all rounds for this parameter set
            rounds = [
                r
                for r in all_results
                if r["num_firms"] == param_set["num_firms"]
                and r["demand_elasticity"] == param_set["demand_elasticity"]
                and r["base_price"] == param_set["base_price"]
            ]
            rounds = sorted(rounds, key=lambda x: x["round"])

            if len(rounds) < 2:
                continue

            x_values = [r[metric_x] for r in rounds]
            y_values = [r[metric_y] for r in rounds]

            if not x_values or not y_values:
                continue

            # Calculate correlation
            mean_x = sum(x_values) / len(x_values)
            mean_y = sum(y_values) / len(y_values)

            numerator = sum(
                (x_values[i] - mean_x) * (y_values[i] - mean_y)
                for i in range(len(x_values))
            )
            denom_x = sum((x_values[i] - mean_x) ** 2 for i in range(len(x_values)))
            denom_y = sum((y_values[i] - mean_y) ** 2 for i in range(len(y_values)))

            if denom_x > 0 and denom_y > 0:
                corr = numerator / math.sqrt(denom_x * denom_y)
                correlations.append(abs(corr))

            # Calculate ranges (variation)
            x_range = max(x_values) - min(x_values)
            y_range = max(y_values) - min(y_values)
            x_ranges.append(x_range)
            y_ranges.append(y_range)

        if correlations:
            avg_correlation = sum(correlations) / len(correlations)
            avg_x_range = sum(x_ranges) / len(x_ranges) if x_ranges else 0
            avg_y_range = sum(y_ranges) / len(y_ranges) if y_ranges else 0

            # Score based on correlation strength and variation
            score = avg_correlation * (avg_x_range + avg_y_range) / 2

            analysis[name] = {
                "metric_x": metric_x,
                "metric_y": metric_y,
                "correlation": round(avg_correlation, 3),
                "x_range": round(avg_x_range, 2),
                "y_range": round(avg_y_range, 2),
                "score": round(score, 2),
            }

    return analysis


def main() -> None:
    """Run comprehensive analysis."""
    print("Running comprehensive oligopoly simulations...")
    print("=" * 60)

    # Set random seed for reproducibility
    random.seed(42)

    # Run simulations across parameter space
    all_results = []
    num_firms_list = [2, 3, 4, 5]
    elasticities = [1.5, 2.0, 2.5]
    base_prices = [30, 40, 50]

    total_configs = len(num_firms_list) * len(elasticities) * len(base_prices)
    config_count = 0

    for num_firms in num_firms_list:
        for elasticity in elasticities:
            for base_price in base_prices:
                config_count += 1
                print(
                    f"Simulating config {config_count}/{total_configs}: "
                    f"{num_firms} firms, elasticity={elasticity}, base=${base_price}"
                )

                results = run_simulation_rounds(num_firms, elasticity, base_price)
                all_results.extend(results)

    print(f"\nCompleted {len(all_results)} simulation rounds")
    print("=" * 60)

    # Analyze metric combinations
    print("\nAnalyzing metric combinations...")
    analysis = analyze_metric_combinations(all_results)

    # Sort by score
    sorted_analysis = sorted(
        analysis.items(), key=lambda x: x[1]["score"], reverse=True
    )

    print("\nTop metric combinations by visual interest score:")
    print("-" * 80)
    print(
        f"{'Metric Pair':<40} {'Correlation':<12} {'X Range':<12} {'Y Range':<12} {'Score':<8}"
    )
    print("-" * 80)

    for name, metrics in sorted_analysis[:10]:
        print(
            f"{name:<40} {metrics['correlation']:<12.3f} {metrics['x_range']:<12.2f} "
            f"{metrics['y_range']:<12.2f} {metrics['score']:<8.2f}"
        )

    # Print recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    if sorted_analysis:
        best = sorted_analysis[0]
        print(f"Best metric pair: {best[0]}")
        print(f"  X-axis: {best[1]['metric_x']}")
        print(f"  Y-axis: {best[1]['metric_y']}")
        print(
            f"  Reason: High correlation ({best[1]['correlation']:.3f}) with good variation"
        )
        print(
            f"           (X range: {best[1]['x_range']:.2f}, Y range: {best[1]['y_range']:.2f})"
        )

    # Export full dataset
    output_file = Path(__file__).parent / "oligopoly_data.json"
    print(f"\nExporting full dataset to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Exported {len(all_results)} data points")
    print("Done!")


if __name__ == "__main__":
    main()
