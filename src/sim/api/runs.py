"""Endpoints for inspecting persisted simulation runs."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from sim.database import get_db
from sim.events.replay import ReplaySystem
from sim.models.models import Event, Run
from sim.runners.runner import get_run_results

from ._common import round_metrics
from .schemas import (
    EventItem,
    EventsResponse,
    ReplayFrame,
    ReplayResponse,
    RunDetail,
    RunSummary,
)

router = APIRouter()


@router.get("/runs/{run_id}")
async def get_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Retrieve time-series results for a simulation run.

    Returns detailed results including market prices, quantities, profits,
    HHI, and consumer surplus for each round and firm in the simulation.

    Raises:
        HTTPException: If run_id is not found or data retrieval fails
    """
    try:
        results = get_run_results(run_id, db)

        model = results.get("model", "cournot")
        rounds_data = results.get("results", {})
        # Use persisted params; fall back to defaults only if params was never stored
        stored_params = results.get("params") or {}

        metrics = {}
        for round_idx, round_firms in rounds_data.items():
            firms_data = list(round_firms.values())
            if not firms_data:
                continue
            metrics[int(round_idx)] = {
                **round_metrics(model, firms_data, stored_params, int(round_idx)),
                "num_firms": len(firms_data),
            }

        results["metrics"] = metrics
        return dict(results)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(db: Session = Depends(get_db)) -> list[RunSummary]:
    """List simulation runs."""
    try:
        runs = db.query(Run).order_by(Run.created_at.desc()).all()
        return [
            RunSummary(
                id=str(run.id),
                model=str(run.model),
                rounds=int(run.rounds),
                created_at=run.created_at.isoformat(),
                status="completed",
            )
            for run in runs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")


@router.get("/runs/{run_id}/detail", response_model=RunDetail)
async def get_run_detail(run_id: str, db: Session = Depends(get_db)) -> RunDetail:
    """Get detailed information about a simulation run."""
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        results = get_run_results(run_id, db)
        return RunDetail(
            id=str(run.id),
            model=str(run.model),
            rounds=int(run.rounds),
            created_at=run.created_at.isoformat(),
            updated_at=run.updated_at.isoformat(),
            results=results,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get run detail: {str(e)}"
        )


@router.get("/runs/{run_id}/events", response_model=EventsResponse)
async def get_run_events(run_id: str, db: Session = Depends(get_db)) -> EventsResponse:
    """Retrieve all events for a simulation run, ordered by round.

    Includes collusion events, policy interventions, and market dynamics.

    Raises:
        HTTPException: If run_id is not found or data retrieval fails
    """
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        events = (
            db.query(Event)
            .filter(Event.run_id == run_id)
            .order_by(Event.round_idx, Event.created_at)
            .all()
        )

        event_items = [
            EventItem(
                id=int(event.id),
                round_idx=int(event.round_idx),
                event_type=str(event.event_type),
                firm_id=int(event.firm_id) if event.firm_id is not None else None,
                description=str(event.description),
                event_data=(
                    dict(event.event_data) if event.event_data is not None else None
                ),
                created_at=event.created_at.isoformat(),
            )
            for event in events
        ]

        return EventsResponse(
            run_id=run_id,
            total_events=len(event_items),
            events=event_items,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/runs/{run_id}/replay", response_model=ReplayResponse)
async def get_run_replay(run_id: str, db: Session = Depends(get_db)) -> ReplayResponse:
    """Retrieve frame-by-frame replay data for a simulation run.

    Raises:
        HTTPException: If run_id is not found or data retrieval fails
    """
    try:
        replay_system = ReplaySystem(run_id, db)

        frames = replay_system.get_all_frames()
        frames_with_events = replay_system.get_frames_with_events()
        event_rounds = replay_system.get_event_rounds()

        replay_frames = [
            ReplayFrame(
                round_idx=frame.round_idx,
                timestamp=frame.timestamp.isoformat(),
                market_price=frame.market_price,
                total_quantity=frame.total_quantity,
                total_profit=frame.total_profit,
                hhi=frame.hhi,
                consumer_surplus=frame.consumer_surplus,
                num_firms=frame.num_firms,
                firm_data=frame.firm_data,
                events=frame.events,
                annotations=frame.annotations,
            )
            for frame in frames
        ]

        return ReplayResponse(
            run_id=run_id,
            total_frames=len(replay_frames),
            frames_with_events=len(frames_with_events),
            event_rounds=event_rounds,
            frames=replay_frames,
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
