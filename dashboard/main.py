"""Flask application serving the oligopoly simulation dashboard.

Provides endpoints for simulation data visualization and real-time metrics.
"""

import logging
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
        RandomWalk(step=2.0, min_bound=10.0, max_bound=30.0, seed=42),
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
    alpha, beta = 100.0, 1.0
    costs = [20.0, 25.0, 30.0]
    bounds = (0.0, 100.0)

    starting_prices = [35.0, 38.0, 42.0]

    strategies = [
        Static(value=starting_prices[0]),
        TitForTat(),
        RandomWalk(step=1.5, min_bound=35.0, max_bound=50.0, seed=42),
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
    """Return aggregated market metrics."""
    a, b = 100.0, 1.0
    costs = [20.0, 25.0, 30.0]
    n = len(costs)

    nash_q = [(a - costs[i]) / (b * (n + 1)) for i in range(n)]
    total_q = sum(nash_q)
    nash_price = max(0.0, a - b * total_q)
    nash_profits = [(nash_price - costs[i]) * nash_q[i] for i in range(n)]

    return jsonify(
        {
            "nash_quantities": nash_q,
            "nash_price": nash_price,
            "nash_profits": nash_profits,
            "total_surplus": sum(nash_profits),
            "hhi": sum((q / total_q * 100) ** 2 for q in nash_q) if total_q > 0 else 0,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
