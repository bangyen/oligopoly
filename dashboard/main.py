"""Flask application serving the oligopoly simulation dashboard.

Provides endpoints for simulation data visualization and real-time metrics.
"""

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request

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


def parse_list_param(param_str: Optional[str], default: List[float]) -> List[float]:
    """Parse a comma-separated string into a list of floats."""
    if not param_str:
        return default
    try:
        return [float(x.strip()) for x in param_str.split(",") if x.strip()]
    except ValueError:
        return default


def get_strategies(
    n_firms: int, nash_value: float, bounds: tuple[float, float]
) -> List[Any]:
    """Generate strategies for N firms."""
    strategies = []

    # Firm 1: Static (Nash Equilibrium)
    strategies.append(Static(value=nash_value))

    if n_firms > 1:
        # Firm 2: TitForTat
        strategies.append(TitForTat())

    # Remaining Firms: RandomWalk
    for i in range(2, n_firms):
        # Vary seeds and steps slightly
        strategies.append(
            RandomWalk(
                step=1.0 + (i * 0.2),
                min_bound=bounds[0],
                max_bound=bounds[1],
                seed=random.randint(1, 10000),
            )
        )

    return strategies


@app.route("/")
def dashboard() -> str:
    """Render the main dashboard interface."""
    return render_template("dashboard.html")


@app.route("/api/simulation/cournot")
def cournot_endpoint() -> Response:
    """Execute a Cournot simulation and return time series data."""
    a = float(request.args.get("a", 100.0))
    b = float(request.args.get("b", 1.0))
    costs = parse_list_param(request.args.get("costs"), [20.0, 25.0, 30.0])

    bounds = (0.0, a / b if b > 0 else 100.0)

    nash_quantities = [
        (a - costs[i]) / (b * (len(costs) + 1)) for i in range(len(costs))
    ]

    strategies = get_strategies(
        len(costs),
        nash_quantities[0] if nash_quantities else 0.0,
        (max(0.0, min(nash_quantities) * 0.5), max(nash_quantities) * 1.5),
    )

    firm_histories: List[List[CournotResult]] = [[] for _ in strategies]
    history: Dict[str, Any] = {
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

    # Calculate summary statistics (average of last 10 rounds)
    last_10 = slice(-10, None)
    avg_quantities = [
        sum(
            history["quantities"][i][j]
            for i in range(len(history["quantities"][last_10]))
        )
        / 10
        for j in range(len(costs))
    ]
    avg_profits = [
        sum(history["profits"][i][j] for i in range(len(history["profits"][last_10])))
        / 10
        for j in range(len(costs))
    ]
    avg_price = sum(history["prices"][last_10]) / 10
    total_q = sum(avg_quantities)
    hhi = sum((q / total_q * 100) ** 2 for q in avg_quantities) if total_q > 0 else 0

    history["summary"] = {
        "avg_quantities": avg_quantities,
        "avg_price": avg_price,
        "avg_profits": avg_profits,
        "total_surplus": sum(avg_profits),  # Simplified surplus calculation
        "hhi": hhi,
    }

    return jsonify(history)


@app.route("/api/simulation/bertrand")
def bertrand_endpoint() -> Response:
    """Execute a Bertrand simulation and return time series data.

    Uses narrower price ranges and cost-aware bounds to create realistic
    competition without extreme outcomes or firms pricing below cost.
    """
    alpha = float(request.args.get("alpha", 200.0))
    beta = float(request.args.get("beta", 1.0))
    costs = parse_list_param(request.args.get("costs"), [20.0, 25.0, 30.0])

    # Calculate a rough monopoly price for bounds reference
    avg_cost = sum(costs) / len(costs) if costs else 0
    monopoly_price = (alpha / beta + avg_cost) / 2
    bounds = (min(costs) * 1.1, monopoly_price * 1.1)

    # Use a simpler strategy set for Bertrand to avoid instability in dynamic setting
    strategies = []
    nash_price = (
        min(costs) * 1.05
    )  # Approximate competitive price slightly above min cost

    strategies = get_strategies(len(costs), nash_price, bounds)

    firm_histories: List[List[BertrandResult]] = [[] for _ in strategies]
    history: Dict[str, Any] = {
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
    """Return theoretical Nash equilibrium for comparison.

    Pure game theory calculation with no simulation.
    """
    a = float(request.args.get("a", 100.0))
    b = float(request.args.get("b", 1.0))
    costs = parse_list_param(request.args.get("costs"), [20.0, 25.0, 30.0])

    n = len(costs)

    # Calculate theoretical Nash equilibrium
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
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5050)
