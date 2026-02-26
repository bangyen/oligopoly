"""FastAPI application for oligopoly simulation.

This module provides the main FastAPI application with endpoints
for health checks and simulation management.
"""

import math
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.sim.events.replay import ReplaySystem
from src.sim.heatmap.bertrand_heatmap import (
    compute_bertrand_heatmap,
    compute_bertrand_segmented_heatmap,
    create_price_grid,
)
from src.sim.heatmap.cournot_heatmap import (
    compute_cournot_heatmap,
    compute_cournot_segmented_heatmap,
    create_quantity_grid,
)
from src.sim.models.market_evolution import (
    MarketEvolutionConfig,
)
from src.sim.models.metrics import (
    calculate_round_metrics_bertrand,
    calculate_round_metrics_cournot,
)
from src.sim.models.models import Base, DemandSegment, Event, Run, SegmentedDemand
from src.sim.policy.policy_shocks import PolicyEvent, PolicyType
from src.sim.runners.runner import get_run_results, run_game

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://user:password@localhost/oligopoly"
)

# Create database engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables (only if not in test mode)
if not os.getenv("TESTING"):
    try:
        Base.metadata.create_all(bind=engine)
    except Exception:
        # Database not available, skip table creation
        pass


# Pydantic models for API
class DemandSegmentConfig(BaseModel):
    """Configuration for a single demand segment."""

    alpha: float = Field(
        ..., gt=0, description="Intercept parameter for segment demand curve"
    )
    beta: float = Field(
        ..., gt=0, description="Slope parameter for segment demand curve"
    )
    weight: float = Field(
        ..., gt=0, le=1, description="Market share weight for this segment"
    )


class ProductCharacteristicsConfig(BaseModel):
    """Simplified configuration for product characteristics."""

    quality: float = Field(default=1.0, gt=0, description="Product quality level")
    # Removed complex parameters: location, brand_strength, innovation_level


class FirmConfig(BaseModel):
    """Simplified configuration for a single firm in the simulation."""

    cost: float = Field(..., gt=0, description="Marginal cost of production")
    fixed_cost: float = Field(default=0.0, ge=0, description="Fixed cost per period")
    # Removed complex parameters: capacity_limit, economies_of_scale, product_characteristics


class PolicyEventRequest(BaseModel):
    """Request model for a policy event."""

    round_idx: int = Field(
        ..., ge=0, description="Round index when to apply the policy"
    )
    policy_type: PolicyType = Field(..., description="Type of policy intervention")
    value: float = Field(
        ..., ge=0, description="Policy value (tax rate, subsidy per unit, or price cap)"
    )


class AdvancedStrategyConfig(BaseModel):
    """Simplified configuration for learning strategies."""

    strategy_type: str = Field(
        ...,
        pattern="^(fictitious_play|q_learning)$",
        description="Type of learning strategy (removed complex options)",
    )
    learning_rate: float = Field(default=0.1, gt=0, le=1, description="Learning rate")
    memory_length: int = Field(
        default=10, gt=0, description="Memory length for learning (reduced from 20)"
    )


class EnhancedDemandConfig(BaseModel):
    """Simplified configuration for demand functions."""

    demand_type: str = Field(
        default="linear",
        pattern="^(linear|ces)$",
        description="Type of demand function (simplified to essential options)",
    )
    elasticity: float = Field(
        default=2.0, gt=1, description="Elasticity of substitution (CES only)"
    )


class CournotParams(BaseModel):
    """Typed demand parameters for the Cournot competition model."""

    a: float = Field(100.0, gt=0, description="Demand intercept P = a - b*Q")
    b: float = Field(
        1.0, gt=0, description="Demand slope (price sensitivity to quantity)"
    )


class BertrandParams(BaseModel):
    """Typed demand parameters for the Bertrand competition model."""

    alpha: float = Field(
        100.0, gt=0, description="Demand intercept Q(p) = alpha - beta*p"
    )
    beta: float = Field(1.0, gt=0, description="Demand slope")


class SimulationRequest(BaseModel):
    """Request model for simulation endpoint."""

    model: str = Field(
        ...,
        pattern="^(cournot|bertrand)$",
        description="Competition model type",
    )
    rounds: int = Field(..., gt=0, le=1000, description="Number of simulation rounds")
    params: Union[CournotParams, BertrandParams, None] = Field(
        default=None,
        description="Typed market demand parameters. Use CournotParams (a, b) for "
        "cournot and BertrandParams (alpha, beta) for bertrand.",
    )
    firms: List[FirmConfig] = Field(
        ..., min_length=1, max_length=10, description="Firm configurations"
    )
    segments: Optional[List[DemandSegmentConfig]] = Field(
        None,
        description="Segmented demand configuration (overrides single-segment params)",
    )
    demand_type: str = Field(
        default="linear",
        pattern="^(linear|isoelastic)$",
        description="Type of demand function",
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    events: Optional[List[PolicyEventRequest]] = Field(
        default_factory=list, description="Policy events to apply during simulation"
    )
    advanced_strategies: Optional[List[AdvancedStrategyConfig]] = Field(
        default=None, description="Simplified learning strategies for firms"
    )
    market_evolution: Optional[MarketEvolutionConfig] = Field(
        default=None, description="Market evolution configuration"
    )
    enhanced_demand: Optional[EnhancedDemandConfig] = Field(
        default=None, description="Enhanced demand function configuration"
    )


class SimulationResponse(BaseModel):
    """Response model for simulation endpoint."""

    run_id: str = Field(..., description="Unique identifier for the simulation run")


class ComparisonRequest(BaseModel):
    """Request model for comparison endpoint."""

    left_config: SimulationRequest = Field(
        ..., description="Left scenario configuration"
    )
    right_config: SimulationRequest = Field(
        ..., description="Right scenario configuration"
    )


class ComparisonResponse(BaseModel):
    """Response model for comparison endpoint."""

    left_run_id: str = Field(
        ..., description="Unique identifier for the left scenario run"
    )
    right_run_id: str = Field(
        ..., description="Unique identifier for the right scenario run"
    )


class ComparisonResults(BaseModel):
    """Response model for comparison results endpoint."""

    left_run_id: str = Field(..., description="Left scenario run ID")
    right_run_id: str = Field(..., description="Right scenario run ID")
    rounds: int = Field(..., description="Number of rounds (should be same for both)")
    left_metrics: Dict[str, List[float]] = Field(
        ..., description="Left scenario metrics arrays"
    )
    right_metrics: Dict[str, List[float]] = Field(
        ..., description="Right scenario metrics arrays"
    )
    deltas: Dict[str, List[float]] = Field(
        ..., description="Delta arrays (right - left)"
    )


class EventItem(BaseModel):
    """Individual event item for API responses."""

    id: int = Field(..., description="Unique event identifier")
    round_idx: int = Field(..., description="Round index when event occurred")
    event_type: str = Field(..., description="Type of event")
    firm_id: Optional[int] = Field(None, description="Firm involved (if applicable)")
    description: str = Field(..., description="Human-readable event description")
    event_data: Optional[Dict[str, Any]] = Field(
        None, description="Additional event data"
    )
    created_at: str = Field(..., description="Event timestamp")


class EventsResponse(BaseModel):
    """Response model for events endpoint."""

    run_id: str = Field(..., description="Simulation run ID")
    total_events: int = Field(..., description="Total number of events")
    events: List[EventItem] = Field(..., description="Ordered list of events")


class ReplayFrame(BaseModel):
    """Single frame in simulation replay."""

    round_idx: int = Field(..., description="Round index")
    timestamp: str = Field(..., description="Frame timestamp")
    market_price: float = Field(..., description="Market price")
    total_quantity: float = Field(..., description="Total quantity")
    total_profit: float = Field(..., description="Total profit")
    hhi: float = Field(..., description="Herfindahl-Hirschman Index")
    consumer_surplus: float = Field(..., description="Consumer surplus")
    num_firms: int = Field(..., description="Number of firms")
    firm_data: Dict[int, Dict[str, float]] = Field(
        ..., description="Firm-specific data"
    )
    events: List[Dict[str, Any]] = Field(..., description="Events in this round")
    annotations: List[str] = Field(..., description="Human-readable annotations")


class ReplayResponse(BaseModel):
    """Response model for replay endpoint."""

    run_id: str = Field(..., description="Simulation run ID")
    total_frames: int = Field(..., description="Total number of frames")
    frames_with_events: int = Field(..., description="Number of frames with events")
    event_rounds: List[int] = Field(..., description="Rounds containing events")
    frames: List[ReplayFrame] = Field(..., description="All replay frames")


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


class HeatmapRequest(BaseModel):
    """Request model for heatmap endpoint."""

    model: str = Field(
        ..., pattern="^(cournot|bertrand)$", description="Competition model type"
    )
    firm_i: int = Field(..., ge=0, description="Index of firm to compute surface for")
    firm_j: int = Field(..., ge=0, description="Index of second firm in heatmap")
    grid_size: int = Field(
        ..., ge=5, le=50, description="Number of grid points per dimension"
    )
    action_range: Tuple[float, float] = Field(
        ..., description="Min and max values for action grid (quantity or price)"
    )
    other_actions: List[float] = Field(
        ..., description="Fixed actions for all other firms"
    )
    params: Union[CournotParams, BertrandParams, None] = Field(
        default=None,
        description="Typed demand parameters. Use CournotParams for cournot, BertrandParams for bertrand.",
    )
    firms: List[FirmConfig] = Field(
        ..., min_length=2, max_length=10, description="Firm configurations"
    )
    segments: Optional[List[DemandSegmentConfig]] = Field(
        None,
        description="Segmented demand configuration (overrides single-segment params)",
    )


class HeatmapResponse(BaseModel):
    """Response model for heatmap endpoint."""

    model: str = Field(..., description="Competition model type")
    firm_i: int = Field(..., description="Index of firm surface computed for")
    firm_j: int = Field(..., description="Index of second firm in heatmap")
    profit_surface: List[List[float]] = Field(
        ..., description="2D array of profits for firm_i"
    )
    market_share_surface: Optional[List[List[float]]] = Field(
        None, description="2D array of market shares for firm_i (Bertrand only)"
    )
    action_i_grid: List[float] = Field(
        ..., description="Grid values for firm_i actions (quantities or prices)"
    )
    action_j_grid: List[float] = Field(
        ..., description="Grid values for firm_j actions (quantities or prices)"
    )
    computation_time_ms: float = Field(
        ..., description="Computation time in milliseconds"
    )


# Database dependency
def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# FastAPI app
app = FastAPI(
    title="Oligopoly Simulation",
    description="Market competition simulation for industrial organization research",
    version="0.1.0",
)


@app.get("/healthz")
async def health_check() -> JSONResponse:
    """Health check endpoint for monitoring and load balancers.

    Returns a simple status indicating the service is running.
    This endpoint is used by Kubernetes health checks and monitoring systems.
    """
    return JSONResponse(content={"ok": True})


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with basic API information."""
    return {"message": "Oligopoly Simulation API", "version": "0.1.0", "docs": "/docs"}


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(
    request: SimulationRequest, db: Session = Depends(get_db)
) -> SimulationResponse:
    """Run a multi-round oligopoly simulation.

    Executes the specified number of rounds of either Cournot or Bertrand
    competition and persists all results to the database.

    Args:
        request: Simulation configuration including model, rounds, parameters, and firms
        db: Database session for persistence

    Returns:
        SimulationResponse containing the unique run_id

    Raises:
        HTTPException: If simulation fails or configuration is invalid
    """
    try:
        # Validate firm costs are economically reasonable
        costs = [firm.cost for firm in request.firms]
        fixed_costs = [firm.fixed_cost for firm in request.firms]

        if any(cost <= 0 for cost in costs):
            raise HTTPException(
                status_code=400,
                detail="All firm costs must be positive",
            )

        if any(fc < 0 for fc in fixed_costs):
            raise HTTPException(
                status_code=400,
                detail="All fixed costs must be non-negative",
            )

        # Build typed params dict from the request
        raw_params = request.params
        if request.model == "cournot":
            # Cournot model: need a, b
            if raw_params is not None and (
                hasattr(raw_params, "a") and hasattr(raw_params, "b")
            ):
                params = (
                    raw_params.model_dump()
                    if hasattr(raw_params, "model_dump")
                    else dict(raw_params)
                )
            elif raw_params is None:
                params = {"a": 100.0, "b": 1.0}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Cournot model requires CournotParams (fields: a, b)",
                )
        elif request.model == "bertrand":
            # Bertrand model: need alpha, beta
            if raw_params is not None and (
                hasattr(raw_params, "alpha") and hasattr(raw_params, "beta")
            ):
                params = (
                    raw_params.model_dump()
                    if hasattr(raw_params, "model_dump")
                    else dict(raw_params)
                )
            elif raw_params is None:
                params = {"alpha": 100.0, "beta": 1.0}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Bertrand model requires BertrandParams (fields: alpha, beta)",
                )
        else:
            # Should not happen due to Pydantic validation on `model` field
            raise HTTPException(
                status_code=400, detail=f"Unknown model type: {request.model}"
            )

        # Parameter extraction logic above handles type-specific validation
        # (params variable is now a validated dict or defaults)

        # Cross-validate params vs model
        if request.model == "cournot":
            a = params.get("a", 100.0)
            b = params.get("b", 1.0)

            if any(cost >= a for cost in costs):
                raise HTTPException(
                    status_code=400,
                    detail=f"Firm costs cannot exceed demand intercept (a={a}). Firms with costs >= {a} would never be profitable.",
                )
            if b < 0.1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Demand slope (b={b}) is too flat. Use b >= 0.1.",
                )

        elif request.model == "bertrand":
            alpha = params.get("alpha", 100.0)
            beta = params.get("beta", 1.0)

            if any(cost >= alpha for cost in costs):
                raise HTTPException(
                    status_code=400,
                    detail=f"Firm costs cannot exceed demand intercept (alpha={alpha}). Firms with costs >= {alpha} would never be profitable.",
                )
            if beta < 0.1:
                raise HTTPException(
                    status_code=400,
                    detail=f"Demand slope (beta={beta}) is too flat. Use beta >= 0.1.",
                )

        # Convert Pydantic models to dict format expected by run_game
        # Use standardized params dict for simulation config
        config: Dict[str, Any] = {
            "params": params,
            "firms": [
                {"cost": f.cost, "fixed_cost": f.fixed_cost} for f in request.firms
            ],
            "seed": request.seed,
            "events": (
                [
                    PolicyEvent(
                        round_idx=event.round_idx,
                        policy_type=event.policy_type,
                        value=event.value,
                    )
                    for event in request.events
                ]
                if request.events
                else []
            ),
        }

        # Add segments configuration if provided
        if request.segments:
            # Validate that segment weights sum to 1
            total_weight = sum(segment.weight for segment in request.segments)
            if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
                raise HTTPException(
                    status_code=400,
                    detail=f"Segment weights must sum to 1.0, got {total_weight:.6f}",
                )

            # Validate segment parameters are economically reasonable
            for i, segment in enumerate(request.segments):
                if segment.alpha <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Segment {i} alpha parameter must be positive, got {segment.alpha}",
                    )
                if segment.beta <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Segment {i} beta parameter must be positive, got {segment.beta}",
                    )
                if segment.weight <= 0 or segment.weight > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Segment {i} weight must be in (0, 1], got {segment.weight}",
                    )

            # Check for unrealistic elasticity
            max_elasticity_ratio = 2.0
            for i, segment in enumerate(request.segments):
                if segment.beta / segment.alpha > max_elasticity_ratio:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Segment {i} has unrealistic elasticity: beta/alpha = {segment.beta / segment.alpha:.3f} > {max_elasticity_ratio}",
                    )

            config["params"]["segments"] = [
                {"alpha": segment.alpha, "beta": segment.beta, "weight": segment.weight}
                for segment in request.segments
            ]

        # Run the simulation
        run_id = run_game(
            model=request.model, rounds=request.rounds, config=config, db=db
        )

        return SimulationResponse(run_id=run_id)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# The /differentiated-bertrand endpoint has been intentionally removed.
# The underlying calculate_differentiated_nash_equilibrium function always
# returned a stub response (rounds=1, no time-series data) that was
# indistinguishable from a real simulation response. Remove until the feature
# is complete. To run a Bertrand competition, use POST /simulate with
# model="bertrand".


@app.get("/runs/{run_id}")
async def get_run(run_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve time-series results for a simulation run.

    Returns detailed results including market prices, quantities, profits,
    HHI, and consumer surplus for each round and firm in the simulation.

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        Dictionary containing time-series data with arrays of equal length
        and calculated metrics (HHI, consumer surplus)

    Raises:
        HTTPException: If run_id is not found or data retrieval fails
    """
    try:
        results = get_run_results(run_id, db)

        model = results.get("model", "cournot")
        rounds_data = results.get("results", {})
        # Use persisted params; fall back to narrow defaults only if params was never stored
        stored_params = results.get("params") or {}

        metrics = {}
        for round_idx, round_firms in rounds_data.items():
            round_idx = int(round_idx)
            firms_data = list(round_firms.values())
            if not firms_data:
                continue

            quantities = [firm["quantity"] for firm in firms_data]
            prices = [firm["price"] for firm in firms_data]
            profits = [firm["profit"] for firm in firms_data]

            if model == "cournot":
                market_price = prices[0] if prices else 0.0
                demand_a = float(stored_params.get("a", 100.0))
                hhi, cs = calculate_round_metrics_cournot(
                    quantities, market_price, demand_a
                )
            else:  # bertrand
                total_demand = sum(quantities)
                demand_alpha = float(stored_params.get("alpha", 100.0))
                hhi, cs = calculate_round_metrics_bertrand(
                    prices, quantities, total_demand, demand_alpha
                )

            metrics[round_idx] = {
                "hhi": hhi,
                "consumer_surplus": cs,
                "market_price": (
                    market_price
                    if model == "cournot"
                    else min(prices) if prices else 0.0
                ),
                "total_quantity": sum(quantities),
                "total_profit": sum(profits),
                "num_firms": len(firms_data),
            }

        results["metrics"] = metrics
        return results

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/runs", response_model=List[RunSummary])
async def list_runs(db: Session = Depends(get_db)) -> List[RunSummary]:
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


@app.get("/runs/{run_id}/detail", response_model=RunDetail)
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


@app.post("/compare", response_model=ComparisonResponse)
async def compare_scenarios(
    request: ComparisonRequest, db: Session = Depends(get_db)
) -> ComparisonResponse:
    """Run two simulation scenarios for comparison.

    Executes both left and right scenario simulations and returns their run IDs.
    Both simulations must have the same number of rounds for valid comparison.

    Args:
        request: Comparison configuration with left and right scenarios
        db: Database session for persistence

    Returns:
        ComparisonResponse containing both run IDs

    Raises:
        HTTPException: If simulations fail or configurations are invalid
    """
    try:
        # Validate that both scenarios have the same number of rounds
        if request.left_config.rounds != request.right_config.rounds:
            raise HTTPException(
                status_code=400,
                detail=f"Both scenarios must have the same number of rounds. "
                f"Left: {request.left_config.rounds}, Right: {request.right_config.rounds}",
            )

        # Normalise params to plain dicts (may come in as CournotParams/BertrandParams objects)
        def _params_to_dict(
            params: Optional[Union[CournotParams, BertrandParams]],
        ) -> dict:
            if params is None:
                return {}
            return (
                params.model_dump() if hasattr(params, "model_dump") else dict(params)
            )

        # Run left scenario
        left_config: Dict[str, Any] = {
            "params": _params_to_dict(request.left_config.params),
            "firms": [{"cost": firm.cost} for firm in request.left_config.firms],
            "seed": request.left_config.seed,
            "events": (
                [
                    PolicyEvent(
                        round_idx=event.round_idx,
                        policy_type=event.policy_type,
                        value=event.value,
                    )
                    for event in request.left_config.events
                ]
                if request.left_config.events
                else []
            ),
        }

        # Add segments configuration if provided for left scenario
        if request.left_config.segments:
            total_weight = sum(
                segment.weight for segment in request.left_config.segments
            )
            if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
                raise HTTPException(
                    status_code=400,
                    detail=f"Left scenario segment weights must sum to 1.0, got {total_weight:.6f}",
                )
            left_config["params"]["segments"] = [
                {"alpha": segment.alpha, "beta": segment.beta, "weight": segment.weight}
                for segment in request.left_config.segments
            ]

        # Run right scenario
        right_config: Dict[str, Any] = {
            "params": _params_to_dict(request.right_config.params),
            "firms": [{"cost": firm.cost} for firm in request.right_config.firms],
            "seed": request.right_config.seed,
            "events": (
                [
                    PolicyEvent(
                        round_idx=event.round_idx,
                        policy_type=event.policy_type,
                        value=event.value,
                    )
                    for event in request.right_config.events
                ]
                if request.right_config.events
                else []
            ),
        }

        # Add segments configuration if provided for right scenario
        if request.right_config.segments:
            total_weight = sum(
                segment.weight for segment in request.right_config.segments
            )
            if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
                raise HTTPException(
                    status_code=400,
                    detail=f"Right scenario segment weights must sum to 1.0, got {total_weight:.6f}",
                )
            right_config["params"]["segments"] = [
                {"alpha": segment.alpha, "beta": segment.beta, "weight": segment.weight}
                for segment in request.right_config.segments
            ]

        # Run both simulations
        left_run_id = run_game(
            model=request.left_config.model,
            rounds=request.left_config.rounds,
            config=left_config,
            db=db,
        )

        right_run_id = run_game(
            model=request.right_config.model,
            rounds=request.right_config.rounds,
            config=right_config,
            db=db,
        )

        return ComparisonResponse(left_run_id=left_run_id, right_run_id=right_run_id)

    except HTTPException:
        # Re-raise HTTPExceptions (like validation errors) as-is
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/compare/{left_run_id}/{right_run_id}", response_model=ComparisonResults)
async def get_comparison_results(
    left_run_id: str, right_run_id: str, db: Session = Depends(get_db)
) -> ComparisonResults:
    """Retrieve aligned comparison results for two simulation runs.

    Returns time-series metrics for both runs aligned by round, along with
    calculated deltas (right - left) for each metric.

    Args:
        left_run_id: Unique identifier for the left scenario run
        right_run_id: Unique identifier for the right scenario run
        db: Database session

    Returns:
        ComparisonResults containing aligned metrics arrays and deltas

    Raises:
        HTTPException: If either run_id is not found or data retrieval fails
    """
    try:
        # Get results for both runs
        left_results = get_run_results(left_run_id, db)
        right_results = get_run_results(right_run_id, db)

        left_results_dict = dict(left_results)
        right_results_dict = dict(right_results)

        # Validate that both runs have the same number of rounds
        left_rounds = left_results_dict.get("rounds", 0)
        right_rounds = right_results_dict.get("rounds", 0)
        if left_rounds != right_rounds:
            raise HTTPException(
                status_code=400,
                detail=f"Runs must have the same number of rounds. "
                f"Left: {left_rounds}, Right: {right_rounds}",
            )

        # Calculate metrics for both runs
        left_metrics = _calculate_comparison_metrics(left_results_dict)
        right_metrics = _calculate_comparison_metrics(right_results_dict)

        # Calculate deltas (right - left)
        deltas = {}
        for metric_name in left_metrics:
            if metric_name in right_metrics:
                left_values = left_metrics[metric_name]
                right_values = right_metrics[metric_name]
                # Ensure both arrays have the same length
                min_length = min(len(left_values), len(right_values))
                deltas[metric_name] = [
                    right_values[i] - left_values[i] for i in range(min_length)
                ]

        return ComparisonResults(
            left_run_id=left_run_id,
            right_run_id=right_run_id,
            rounds=left_rounds,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            deltas=deltas,
        )

    except HTTPException:
        # Re-raise HTTPExceptions (like validation errors) as-is
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def _calculate_comparison_metrics(run_data: Dict[str, Any]) -> Dict[str, List[float]]:
    """Calculate metrics arrays for comparison from run data.

    Expects the canonical nested-dict format from get_run_results:
    ``results[round_idx][firm_id] = {action, price, quantity, profit}``

    Args:
        run_data: Dictionary containing run results, params, and metadata

    Returns:
        Dictionary with metric names as keys and arrays of values as values
    """
    rounds_data = run_data.get("results", {})
    model = run_data.get("model", "cournot")
    stored_params = run_data.get("params") or {}

    metrics: Dict[str, List[float]] = {
        "market_price": [],
        "total_quantity": [],
        "total_profit": [],
        "hhi": [],
        "consumer_surplus": [],
    }

    for round_idx in sorted(rounds_data.keys(), key=int):
        round_firms = rounds_data[round_idx]
        firms_data = list(round_firms.values())
        if not firms_data:
            continue

        quantities = [firm["quantity"] for firm in firms_data]
        prices = [firm["price"] for firm in firms_data]
        profits = [firm["profit"] for firm in firms_data]

        if model == "cournot":
            market_price = prices[0] if prices else 0.0
            demand_a = float(stored_params.get("a", 100.0))
            hhi, cs = calculate_round_metrics_cournot(
                quantities, market_price, demand_a
            )
        else:  # bertrand
            total_demand = sum(quantities)
            demand_alpha = float(stored_params.get("alpha", 100.0))
            market_price = min(prices) if prices else 0.0
            hhi, cs = calculate_round_metrics_bertrand(
                prices, quantities, total_demand, demand_alpha
            )

        metrics["market_price"].append(market_price)
        metrics["total_quantity"].append(sum(quantities))
        metrics["total_profit"].append(sum(profits))
        metrics["hhi"].append(hhi)
        metrics["consumer_surplus"].append(cs)

    return metrics


@app.get("/runs/{run_id}/events", response_model=EventsResponse)
async def get_run_events(run_id: str, db: Session = Depends(get_db)) -> EventsResponse:
    """Retrieve all events for a simulation run.

    Returns an ordered list of all events that occurred during the simulation,
    including collusion events, policy interventions, and market dynamics.

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        EventsResponse containing ordered list of events

    Raises:
        HTTPException: If run_id is not found or data retrieval fails
    """
    try:
        # Verify run exists
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        # Get all events for this run
        events = (
            db.query(Event)
            .filter(Event.run_id == run_id)
            .order_by(Event.round_idx, Event.created_at)
            .all()
        )

        # Convert to API format
        event_items = []
        for event in events:
            event_item = EventItem(
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
            event_items.append(event_item)

        return EventsResponse(
            run_id=run_id,
            total_events=len(event_items),
            events=event_items,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.get("/runs/{run_id}/replay", response_model=ReplayResponse)
async def get_run_replay(run_id: str, db: Session = Depends(get_db)) -> ReplayResponse:
    """Retrieve replay data for a simulation run.

    Returns frame-by-frame replay data including market metrics, firm actions,
    and event annotations for comprehensive simulation playback.

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        ReplayResponse containing all replay frames

    Raises:
        HTTPException: If run_id is not found or data retrieval fails
    """
    try:
        # Initialize replay system
        replay_system = ReplaySystem(run_id, db)

        # Get all frames
        frames = replay_system.get_all_frames()
        frames_with_events = replay_system.get_frames_with_events()
        event_rounds = replay_system.get_event_rounds()

        # Convert frames to API format
        replay_frames = []
        for frame in frames:
            replay_frame = ReplayFrame(
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
            replay_frames.append(replay_frame)

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


@app.post("/heatmap", response_model=HeatmapResponse)
async def compute_heatmap(request: HeatmapRequest) -> HeatmapResponse:
    """Compute 2D heatmap for strategy/action spaces.

    Generates profit surfaces by sweeping over action grids for two firms while
    holding other firms' actions fixed. Supports both Cournot (quantity) and
    Bertrand (price) competition models.

    Args:
        request: Heatmap configuration including model, firms, grid parameters

    Returns:
        HeatmapResponse containing 2D profit surface and market share surface
        (for Bertrand), along with action grids and computation timing

    Raises:
        HTTPException: If configuration is invalid or computation fails
    """
    start_time = time.time()

    try:
        # Validate firm indices
        if request.firm_i >= len(request.firms):
            raise HTTPException(
                status_code=400,
                detail=f"firm_i ({request.firm_i}) must be less than number of firms ({len(request.firms)})",
            )
        if request.firm_j >= len(request.firms):
            raise HTTPException(
                status_code=400,
                detail=f"firm_j ({request.firm_j}) must be less than number of firms ({len(request.firms)})",
            )
        if request.firm_i == request.firm_j:
            raise HTTPException(
                status_code=400, detail="firm_i and firm_j must be different"
            )

        # Validate other_actions length
        expected_other_length = len(request.firms) - 2
        if len(request.other_actions) != expected_other_length:
            raise HTTPException(
                status_code=400,
                detail=f"other_actions length ({len(request.other_actions)}) must equal "
                f"number of firms - 2 ({expected_other_length})",
            )

        # Extract costs
        costs = [firm.cost for firm in request.firms]

        # Create action grids
        min_action, max_action = request.action_range
        if request.model == "cournot":
            action_i_grid = create_quantity_grid(
                min_action, max_action, request.grid_size
            )
            action_j_grid = create_quantity_grid(
                min_action, max_action, request.grid_size
            )
        else:  # bertrand
            action_i_grid = create_price_grid(min_action, max_action, request.grid_size)
            action_j_grid = create_price_grid(min_action, max_action, request.grid_size)

        # Compute heatmap based on model type
        if request.model == "cournot":
            # Resolve demand parameters from typed CournotParams or defaults
            # Use hasattr check to be robust against class instance mismatches from re-imports
            if request.params is not None and (
                hasattr(request.params, "a") and hasattr(request.params, "b")
            ):
                a = getattr(request.params, "a")
                b = getattr(request.params, "b")
            elif request.params is None:
                a, b = 100.0, 1.0
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Cournot model requires CournotParams (fields: a, b)",
                )

            if request.segments:
                # Segmented demand
                segments = [
                    DemandSegment(
                        alpha=segment.alpha, beta=segment.beta, weight=segment.weight
                    )
                    for segment in request.segments
                ]
                segmented_demand = SegmentedDemand(segments=segments)

                profit_matrix, _, _ = compute_cournot_segmented_heatmap(
                    segmented_demand,
                    costs,
                    request.firm_i,
                    request.firm_j,
                    action_i_grid,
                    action_j_grid,
                    request.other_actions,
                )
                market_share_surface = None
            else:
                # Single-segment demand
                profit_matrix, _, _ = compute_cournot_heatmap(
                    a,
                    b,
                    costs,
                    request.firm_i,
                    request.firm_j,
                    action_i_grid,
                    action_j_grid,
                    request.other_actions,
                )
                market_share_surface = None

        else:  # bertrand
            # Resolve demand parameters from typed BertrandParams or defaults
            # Use hasattr check to be robust against class instance mismatches from re-imports
            if request.params is not None and (
                hasattr(request.params, "alpha") and hasattr(request.params, "beta")
            ):
                alpha = getattr(request.params, "alpha")
                beta = getattr(request.params, "beta")
            elif request.params is None:
                alpha, beta = 100.0, 1.0
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Bertrand model requires BertrandParams (fields: alpha, beta)",
                )

            if request.segments:
                # Segmented demand
                segments = [
                    DemandSegment(
                        alpha=segment.alpha, beta=segment.beta, weight=segment.weight
                    )
                    for segment in request.segments
                ]
                segmented_demand = SegmentedDemand(segments=segments)

                profit_matrix, market_share_matrix, _, _ = (
                    compute_bertrand_segmented_heatmap(
                        segmented_demand,
                        costs,
                        request.firm_i,
                        request.firm_j,
                        action_i_grid,
                        action_j_grid,
                        request.other_actions,
                    )
                )
                market_share_surface = market_share_matrix.tolist()
            else:
                # Single-segment demand
                profit_matrix, market_share_matrix, _, _ = compute_bertrand_heatmap(
                    alpha,
                    beta,
                    costs,
                    request.firm_i,
                    request.firm_j,
                    action_i_grid,
                    action_j_grid,
                    request.other_actions,
                )
                market_share_surface = market_share_matrix.tolist()

        # Convert numpy arrays to lists for JSON serialization
        profit_surface = profit_matrix.tolist()

        computation_time_ms = (time.time() - start_time) * 1000

        return HeatmapResponse(
            model=request.model,
            firm_i=request.firm_i,
            firm_j=request.firm_j,
            profit_surface=profit_surface,
            market_share_surface=market_share_surface,
            action_i_grid=action_i_grid,
            action_j_grid=action_j_grid,
            computation_time_ms=computation_time_ms,
        )

    except HTTPException:
        # Re-raise HTTPExceptions (like validation errors) as-is
        raise
    except ValueError as e:
        # Check if it's a validation error (starts with error code)
        error_msg = str(e)
        if error_msg.startswith("400:"):
            # Extract the actual error message after the status code
            actual_error = error_msg.split(":", 1)[1].strip()
            raise HTTPException(status_code=400, detail=actual_error)
        else:
            # Handle ValueError from heatmap computation functions
            raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
