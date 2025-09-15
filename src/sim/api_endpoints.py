"""Advanced API endpoints for oligopoly simulation.

This module provides additional API endpoints for detailed analysis,
metrics, and replay functionality.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sim.logging import get_logger, log_execution_time
from sim.models.models import Event, Run
from sim.monitoring import get_metrics_summary

from .database import get_db

logger = get_logger(__name__)

router = APIRouter(tags=["advanced"])


class RunSummary(BaseModel):
    """Summary of a simulation run."""

    id: str
    model: str
    rounds: int
    created_at: str
    status: str


class RunDetail(BaseModel):
    """Detailed information about a simulation run."""

    id: str
    model: str
    rounds: int
    created_at: str
    updated_at: str
    results: Optional[Dict[str, Any]] = None


@router.get("/runs", response_model=List[RunSummary])
async def list_runs(db: Session = Depends(get_db)) -> List[RunSummary]:
    """List simulation runs."""
    with log_execution_time(logger, "list runs"):
        try:
            # Get all runs, ordered by creation date (newest first)
            runs = db.query(Run).order_by(Run.created_at.desc()).all()

            # Convert to response format
            run_summaries = [
                RunSummary(
                    id=run.id,
                    model=run.model,
                    rounds=run.rounds,
                    created_at=run.created_at.isoformat(),
                    status="completed",  # All runs in DB are completed
                )
                for run in runs
            ]

            return run_summaries

        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to list runs: {str(e)}"
            )


@router.get("/runs/{run_id}/detail", response_model=RunDetail)
async def get_run_detail(run_id: str, db: Session = Depends(get_db)) -> RunDetail:
    """Get detailed information about a simulation run."""
    with log_execution_time(logger, f"get run detail {run_id}"):
        try:
            # Get run
            run = db.query(Run).filter(Run.id == run_id).first()
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")

            # Get results
            from sim.runners.runner import get_run_results

            results = get_run_results(run_id, db)

            return RunDetail(
                id=run.id,
                model=run.model,
                rounds=run.rounds,
                created_at=run.created_at.isoformat(),
                updated_at=run.updated_at.isoformat(),
                results=results,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get run detail {run_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get run detail: {str(e)}"
            )


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(
    run_id: str, db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """Get calculated metrics for a simulation run."""
    with log_execution_time(logger, f"get run metrics {run_id}"):
        try:
            # Get run
            run = db.query(Run).filter(Run.id == run_id).first()
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")

            # Get results
            from sim.runners.runner import get_run_results

            results = get_run_results(run_id, db)
            if not results:
                raise HTTPException(status_code=404, detail="Run results not found")

            # Calculate metrics
            from sim.models.metrics import (
                calculate_consumer_surplus,
                calculate_hhi,
            )

            # Extract data for metrics calculation
            rounds_data = results.get("rounds_data", [])
            firms_data = results.get("firms_data", [])

            metrics = []

            for round_idx, round_data in enumerate(rounds_data):
                quantities = [
                    firm_data["quantities"][round_idx] for firm_data in firms_data
                ]
                profits = [firm_data["profits"][round_idx] for firm_data in firms_data]
                price = round_data["price"]

                # Calculate metrics for this round
                hhi = calculate_hhi(quantities)
                cs = calculate_consumer_surplus(quantities, price)

                metrics.append(
                    {
                        "round": round_idx,
                        "total_quantity": sum(quantities),
                        "total_profit": sum(profits),
                        "hhi": hhi,
                        "consumer_surplus": cs,
                        "num_firms": len(firms_data),
                    }
                )

            return metrics

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get run metrics {run_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get run metrics: {str(e)}"
            )


@router.get("/runs/{run_id}/replay")
async def get_run_replay(run_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get replay data for a simulation run."""
    with log_execution_time(logger, f"get run replay {run_id}"):
        try:
            # Get run
            run = db.query(Run).filter(Run.id == run_id).first()
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")

            # Get results
            from sim.runners.runner import get_run_results

            results = get_run_results(run_id, db)
            if not results:
                raise HTTPException(status_code=404, detail="Run results not found")

            # Get events
            events = db.query(Event).filter(Event.run_id == run_id).all()

            # Create replay data
            replay_data = {
                "run_id": run_id,
                "model": run.model,
                "rounds": run.rounds,
                "frames": [],
            }

            # Process each round
            rounds_data = results.get("rounds_data", [])
            firms_data = results.get("firms_data", [])

            for round_idx, round_data in enumerate(rounds_data):
                # Get events for this round
                round_events = [e for e in events if e.round_idx == round_idx]

                frame = {
                    "round": round_idx,
                    "price": round_data["price"],
                    "firms": [
                        {
                            "id": firm_idx,
                            "action": firm_data["actions"][round_idx],
                            "quantity": firm_data["quantities"][round_idx],
                            "profit": firm_data["profits"][round_idx],
                        }
                        for firm_idx, firm_data in enumerate(firms_data)
                    ],
                    "events": [
                        {
                            "type": event.event_type,
                            "round": event.round_idx,
                            "firm_id": event.firm_id,
                            "value": event.value,
                            "description": event.description,
                        }
                        for event in round_events
                    ],
                    "annotations": [],
                }

                replay_data["frames"].append(frame)

            return replay_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get run replay {run_id}: {e}")

            raise HTTPException(
                status_code=500, detail=f"Failed to get run replay: {str(e)}"
            )


@router.get("/statistics")
async def get_statistics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get application statistics."""
    with log_execution_time(logger, "get statistics"):
        try:
            # Get basic counts
            total_runs = db.query(Run).count()
            total_events = db.query(Event).count()

            # Get metrics summary
            metrics_summary = get_metrics_summary()

            return {
                "total_runs": total_runs,
                "total_events": total_events,
                "metrics": metrics_summary,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to get statistics: {str(e)}"
            )


@router.get("/health")
async def health_check(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get application health status."""
    with log_execution_time(logger, "health check"):
        try:
            from sim.monitoring import get_health_status

            health = get_health_status(db)
            return {
                "status": health.status,
                "timestamp": health.timestamp.isoformat(),
                "uptime_seconds": health.uptime_seconds,
                "version": health.version,
                "checks": health.checks,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2025-01-01T00:00:00Z",
            }


@router.post("/heatmap")
async def generate_heatmap(
    model: str,
    firm_i: int,
    firm_j: int,
    costs: List[float],
    grid_size: int = 20,
    params: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Generate profit surface heatmap for two firms."""
    with log_execution_time(logger, f"generate heatmap {model}"):
        try:
            # Validate inputs
            if len(costs) < max(firm_i, firm_j) + 1:
                raise HTTPException(
                    status_code=500, detail="Not enough firms specified"
                )

            if grid_size < 5:
                raise HTTPException(
                    status_code=422, detail="Grid size must be at least 5"
                )

            if params is None:
                params = {}

            if model.lower() == "cournot":
                if "a" not in params or "b" not in params:
                    raise HTTPException(
                        status_code=500,
                        detail="Missing required Cournot parameters: a, b",
                    )

                from sim.heatmap.cournot_heatmap import generate_cournot_heatmap

                profit_surface, action_i_grid, action_j_grid = generate_cournot_heatmap(
                    firm_i=firm_i,
                    firm_j=firm_j,
                    costs=costs,
                    grid_size=grid_size,
                    a=params["a"],
                    b=params["b"],
                )

                response_data = {
                    "profit_surface": profit_surface.tolist(),
                    "market_share_surface": None,  # Cournot doesn't have market share
                    "action_i_grid": action_i_grid,
                    "action_j_grid": action_j_grid,
                    "model": model,
                    "firm_i": firm_i,
                    "firm_j": firm_j,
                }

            elif model.lower() == "bertrand":
                if "alpha" not in params or "beta" not in params:
                    raise HTTPException(
                        status_code=500,
                        detail="Missing required Bertrand parameters: alpha, beta",
                    )

                from sim.heatmap.bertrand_heatmap import generate_bertrand_heatmap

                (
                    profit_surface,
                    market_share_surface,
                    action_i_grid,
                    action_j_grid,
                ) = generate_bertrand_heatmap(
                    firm_i=firm_i,
                    firm_j=firm_j,
                    costs=costs,
                    grid_size=grid_size,
                    alpha=params["alpha"],
                    beta=params["beta"],
                )

                response_data = {
                    "profit_surface": profit_surface.tolist(),
                    "market_share_surface": market_share_surface.tolist(),
                    "action_i_grid": action_i_grid,
                    "action_j_grid": action_j_grid,
                    "model": model,
                    "firm_i": firm_i,
                    "firm_j": firm_j,
                }

            else:
                raise HTTPException(
                    status_code=422, detail="Model must be 'cournot' or 'bertrand'"
                )

            return response_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to generate heatmap: {str(e)}"
            )
