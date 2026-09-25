"""FastAPI application serving the oligopoly simulation dashboard.

Provides endpoints for simulation data visualization and real-time metrics.
"""

import logging
import random
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from sim.api.runs import get_run, get_run_events  # type: ignore[import-not-found]
from sim.api.schemas import SimulationRequest  # type: ignore[import-not-found]
from sim.api.simulate import simulate  # type: ignore[import-not-found]
from sim.games.bertrand import (  # type: ignore[import-not-found]
    BertrandResult,
    bertrand_simulation,
)
from sim.games.cournot import (  # type: ignore[import-not-found]
    CournotResult,
    cournot_simulation,
)
from sim.models.models import (  # type: ignore[import-not-found]
    Base,
    CollusionEvent,
    Event,
    Result,
    Round,
    Run,
)
from sim.strategies.nash_strategies import (  # type: ignore[import-not-found]
    cournot_nash_equilibrium,
)
from sim.strategies.strategies import (  # type: ignore[import-not-found]
    RandomWalk,
    Static,
    TitForTat,
)

# Configure logging
logging.getLogger("sim.validation.economic_validation").setLevel(logging.ERROR)
logging.getLogger("sim.games.cournot").setLevel(logging.ERROR)
logging.getLogger("sim.games.bertrand").setLevel(logging.ERROR)

app = FastAPI(title="Oligopoly Simulation Dashboard")

# Setup static files and templates
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Scenario runs use the real simulation engine against a private in-memory
# database; each run is deleted once its results have been read back.
_engine = create_engine(
    "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
Base.metadata.create_all(_engine)
_session_factory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def parse_list_param(param_str: str | None, default: list[float]) -> list[float]:
    """Parse a comma-separated string into a list of floats."""
    if not param_str:
        return default
    try:
        return [float(x.strip()) for x in param_str.split(",") if x.strip()]
    except ValueError:
        return default


def get_strategies(
    n_firms: int, nash_value: float, bounds: tuple[float, float]
) -> list[Any]:
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


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the main dashboard interface."""
    # Request-first form; the legacy (name, {"request": ...}) order is deprecated
    # and misdispatches when two starlette copies are importable at once.
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/api/simulation/cournot")
async def cournot_endpoint(
    a: float = Query(100.0), b: float = Query(1.0), costs: str | None = Query(None)
):
    """Execute a Cournot simulation and return time series data."""
    parsed_costs = parse_list_param(costs, [20.0, 25.0, 30.0])

    bounds = (0.0, a / b if b > 0 else 100.0)

    nash_quantities = cournot_nash_equilibrium(a, b, parsed_costs)[0]

    strategies = get_strategies(
        len(parsed_costs),
        nash_quantities[0] if nash_quantities else 0.0,
        (max(0.0, min(nash_quantities) * 0.5), max(nash_quantities) * 1.5),
    )

    firm_histories: list[list[CournotResult]] = [[] for _ in strategies]
    history: dict[str, Any] = {
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

        result = cournot_simulation(a, b, parsed_costs, actions)

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
        for j in range(len(parsed_costs))
    ]
    avg_profits = [
        sum(history["profits"][i][j] for i in range(len(history["profits"][last_10])))
        / 10
        for j in range(len(parsed_costs))
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

    return history


@app.get("/api/simulation/bertrand")
async def bertrand_endpoint(
    alpha: float = Query(200.0),
    beta: float = Query(1.0),
    costs: str | None = Query(None),
):
    """Execute a Bertrand simulation and return time series data.

    Uses narrower price ranges and cost-aware bounds to create realistic
    competition without extreme outcomes or firms pricing below cost.
    """
    parsed_costs = parse_list_param(costs, [20.0, 25.0, 30.0])

    # Calculate a rough monopoly price for bounds reference
    avg_cost = sum(parsed_costs) / len(parsed_costs) if parsed_costs else 0
    monopoly_price = (alpha / beta + avg_cost) / 2
    bounds = (min(parsed_costs) * 1.1, monopoly_price * 1.1)

    # Use a simpler strategy set for Bertrand to avoid instability in dynamic setting
    nash_price = (
        min(parsed_costs) * 1.05
    )  # Approximate competitive price slightly above min cost

    strategies = get_strategies(len(parsed_costs), nash_price, bounds)

    firm_histories: list[list[BertrandResult]] = [[] for _ in strategies]
    history: dict[str, Any] = {
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
            action = max(action, parsed_costs[firm_idx] * 1.1)
            actions.append(action)

        result = bertrand_simulation(alpha, beta, parsed_costs, actions)

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

    return history


@app.get("/api/metrics")
async def metrics_endpoint(
    a: float = Query(100.0), b: float = Query(1.0), costs: str | None = Query(None)
):
    """Return theoretical Nash equilibrium for comparison.

    Pure game theory calculation with no simulation.
    """
    parsed_costs = parse_list_param(costs, [20.0, 25.0, 30.0])

    # Theoretical Nash equilibrium (asymmetric costs, with exit)
    nash_q, nash_price, nash_profits = cournot_nash_equilibrium(a, b, parsed_costs)

    return {
        "nash_quantities": nash_q,
        "nash_price": nash_price,
        "nash_profits": nash_profits,
        "total_surplus": sum(nash_profits),
    }


@app.post("/api/scenario")
async def scenario_endpoint(request: SimulationRequest) -> dict[str, Any]:
    """Run a full simulation (any demand system, strategy mix or market
    evolution) with the same engine and validation as ``POST /simulate``.

    Returns the run's time series and metrics plus its event log.
    """
    db = _session_factory()
    try:
        created = await simulate(request, db)
        try:
            run = await get_run(created.run_id, db)
            events = await get_run_events(created.run_id, db)
        finally:
            _delete_run(db, created.run_id)
        return {"run": run, "events": events.model_dump()["events"]}
    finally:
        db.close()


def _delete_run(db: Any, run_id: str) -> None:
    """Remove a scenario run so the in-memory database does not grow."""
    for model in (Event, CollusionEvent, Result, Round):
        db.query(model).filter(model.run_id == run_id).delete()
    db.query(Run).filter(Run.id == run_id).delete()
    db.commit()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
