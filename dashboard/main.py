"""Flask application serving the oligopoly simulation dashboard.

Provides endpoints for simulation data visualization and real-time metrics.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, Response, jsonify, render_template

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sim.games.bertrand import (  # type: ignore[import-not-found]
    BertrandResult,
    bertrand_simulation,
)
from sim.games.cournot import (  # type: ignore[import-not-found]
    CournotResult,
    cournot_simulation,
)
from sim.strategies.strategies import (  # type: ignore[import-not-found]
    RandomWalk,
    Static,
    TitForTat,
)

logging.getLogger("sim.validation.economic_validation").setLevel(logging.ERROR)
logging.getLogger("sim.games.cournot").setLevel(logging.ERROR)
logging.getLogger("sim.games.bertrand").setLevel(logging.ERROR)

app = Flask(__name__)


@app.route("/")
def dashboard() -> str:
    """Render the main dashboard interface."""
    return render_template("dashboard.html")


@app.route("/api/simulation/cournot")
def cournot_endpoint() -> Response:
    """Execute a Cournot simulation and return time series data."""
    a, b = 100.0, 1.0
    costs = [20.0, 25.0, 30.0]
    bounds = (0.0, 50.0)

    nash_quantities = [
        (a - costs[i]) / (b * (len(costs) + 1)) for i in range(len(costs))
    ]

    strategies = [
        Static(value=nash_quantities[0]),
        TitForTat(),
        RandomWalk(
            step=2.0, min_bound=10.0, max_bound=30.0, seed=random.randint(1, 10000)
        ),
    ]

    firm_histories: List[List[CournotResult]] = [[] for _ in strategies]
    history: Dict[str, List[Any]] = {
        "quantities": [],
        "prices": [],
        "profits": [],
        "rounds": [],
    }

    for round_num in range(50):
        actions = []
        for firm_idx, strategy in enumerate(strategies):
            rival_histories = [
                firm_histories[i] for i in range(len(strategies)) if i != firm_idx
            ]
            action = strategy.next_action(
                round_num=round_num,
                my_history=firm_histories[firm_idx],
                rival_histories=rival_histories,
                bounds=bounds,
                market_params={"a": a, "b": b},
            )
            # Ensure quantity is within reasonable bounds
            action = max(0.0, min(action, bounds[1]))
            actions.append(action)

        result = cournot_simulation(a, b, costs, actions)

        for firm_idx in range(len(strategies)):
            firm_result = CournotResult(
                price=result.price,
                quantities=[result.quantities[firm_idx]],
                profits=[result.profits[firm_idx]],
            )
            firm_histories[firm_idx].append(firm_result)

        history["rounds"].append(round_num)
        history["quantities"].append(result.quantities)
        history["prices"].append(result.price)
        history["profits"].append(result.profits)

    return jsonify(history)


@app.route("/api/simulation/bertrand")
def bertrand_endpoint() -> Response:
    """Execute a Bertrand simulation and return time series data.

    Uses narrower price ranges and cost-aware bounds to create realistic
    competition without extreme outcomes or firms pricing below cost.
    """
    alpha, beta = 200.0, 1.0  # Large market to demonstrate capacity constraints
    costs = [20.0, 25.0, 30.0]
    bounds = (33.0, 40.0)  # Narrow bounds keep prices clustered

    strategies = [
        Static(value=35.0),
        TitForTat(),  # Starts at midpoint (33+40)/2 = 36.5
        RandomWalk(
            step=0.8, min_bound=35.0, max_bound=39.0, seed=random.randint(1, 10000)
        ),
    ]

    firm_histories: List[List[BertrandResult]] = [[] for _ in strategies]
    history: Dict[str, List[Any]] = {
        "prices": [],
        "quantities": [],
        "profits": [],
        "rounds": [],
    }

    for round_num in range(50):
        actions = []
        for firm_idx, strategy in enumerate(strategies):
            rival_histories = [
                firm_histories[i] for i in range(len(strategies)) if i != firm_idx
            ]
            action = strategy.next_action(
                round_num=round_num,
                my_history=firm_histories[firm_idx],
                rival_histories=rival_histories,
                bounds=bounds,
                market_params={"alpha": alpha, "beta": beta},
            )
            # Enforce cost floor: never price below your own marginal cost
            action = max(action, costs[firm_idx] * 1.1)
            actions.append(action)

        result = bertrand_simulation(alpha, beta, costs, actions)

        for firm_idx in range(len(strategies)):
            firm_result = BertrandResult(
                total_demand=result.total_demand,
                prices=[result.prices[firm_idx]],
                quantities=[result.quantities[firm_idx]],
                profits=[result.profits[firm_idx]],
            )
            firm_histories[firm_idx].append(firm_result)

        history["rounds"].append(round_num)
        history["prices"].append(result.prices)
        history["quantities"].append(result.quantities)
        history["profits"].append(result.profits)

    return jsonify(history)


@app.route("/api/metrics")
def metrics_endpoint() -> Response:
    """Return aggregated market metrics from actual simulations.

    Runs a fresh Cournot simulation and returns average outcomes
    across the last 10 rounds to show realized (not theoretical) metrics.
    """
    a, b = 100.0, 1.0
    costs = [20.0, 25.0, 30.0]
    bounds = (0.0, 50.0)
    n = len(costs)

    # Run a simulation to get actual outcomes
    nash_quantities = [(a - costs[i]) / (b * (n + 1)) for i in range(n)]

    strategies = [
        Static(value=nash_quantities[0]),
        TitForTat(),
        RandomWalk(
            step=2.0, min_bound=10.0, max_bound=30.0, seed=random.randint(1, 10000)
        ),
    ]

    firm_histories: List[List[CournotResult]] = [[] for _ in strategies]

    # Run 30 rounds and average the last 10 for stable metrics
    for round_num in range(30):
        actions = []
        for firm_idx, strategy in enumerate(strategies):
            rival_histories = [
                firm_histories[i] for i in range(len(strategies)) if i != firm_idx
            ]
            action = strategy.next_action(
                round_num=round_num,
                my_history=firm_histories[firm_idx],
                rival_histories=rival_histories,
                bounds=bounds,
                market_params={"a": a, "b": b},
            )
            action = max(0.0, min(action, bounds[1]))
            actions.append(action)

        result = cournot_simulation(a, b, costs, actions)

        for firm_idx in range(len(strategies)):
            firm_result = CournotResult(
                price=result.price,
                quantities=[result.quantities[firm_idx]],
                profits=[result.profits[firm_idx]],
            )
            firm_histories[firm_idx].append(firm_result)

    # Average over last 10 rounds for stable metrics
    avg_quantities = [
        sum(firm_histories[i][-10:][j].quantities[0] for j in range(10)) / 10
        for i in range(n)
    ]
    avg_profits = [
        sum(firm_histories[i][-10:][j].profits[0] for j in range(10)) / 10
        for i in range(n)
    ]
    avg_price = (
        sum(
            a - b * sum(firm_histories[i][j].quantities[0] for i in range(n))
            for j in range(-10, 0)
        )
        / 10
    )

    total_q = sum(avg_quantities)
    hhi = sum((q / total_q * 100) ** 2 for q in avg_quantities) if total_q > 0 else 0

    return jsonify(
        {
            "nash_quantities": avg_quantities,
            "nash_price": avg_price,
            "nash_profits": avg_profits,
            "total_surplus": sum(avg_profits),
            "hhi": hhi,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
