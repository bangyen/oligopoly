"""Scenario Lab: a dashboard for running and inspecting oligopoly simulations.

Every scenario runs the real engine (the API's simulate/get_run/get_run_events)
against a private in-memory SQLite database, so the dashboard supports exactly
what the REST API does.
"""

import logging
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from sim.api.runs import get_run, get_run_events
from sim.api.schemas import SimulationRequest
from sim.api.simulate import simulate
from sim.models.models import Base, Event, Result, Round, Run

# Per-round economic sanity warnings are noise in an interactive tool
logging.getLogger("sim.validation.economic_validation").setLevel(logging.ERROR)
logging.getLogger("sim.games.cournot").setLevel(logging.ERROR)
logging.getLogger("sim.games.bertrand").setLevel(logging.ERROR)

app = FastAPI(title="Oligopoly Scenario Lab")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Each run is deleted once its results have been read back.
_engine = create_engine(
    "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
Base.metadata.create_all(_engine)
_session_factory = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Render the Scenario Lab."""
    # Request-first form; the legacy (name, {"request": ...}) order is deprecated
    # and misdispatches when two starlette copies are importable at once.
    return templates.TemplateResponse(request, "dashboard.html")


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


def _delete_run(db: Session, run_id: str) -> None:
    """Remove a scenario run so the in-memory database does not grow."""
    for model in (Event, Result, Round):
        db.query(model).filter(model.run_id == run_id).delete()
    db.query(Run).filter(Run.id == run_id).delete()
    db.commit()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
