"""FastAPI application for oligopoly simulation.

This module provides the main FastAPI application with endpoints
for health checks and simulation management.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from sim.models import Base
from sim.runner import get_run_results, run_game

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
class FirmConfig(BaseModel):
    """Configuration for a single firm in the simulation."""

    cost: float = Field(..., gt=0, description="Marginal cost of production")


class SimulationRequest(BaseModel):
    """Request model for simulation endpoint."""

    model: str = Field(
        ..., pattern="^(cournot|bertrand)$", description="Competition model type"
    )
    rounds: int = Field(..., gt=0, le=1000, description="Number of simulation rounds")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Market parameters"
    )
    firms: List[FirmConfig] = Field(
        ..., min_length=1, max_length=10, description="Firm configurations"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SimulationResponse(BaseModel):
    """Response model for simulation endpoint."""

    run_id: str = Field(..., description="Unique identifier for the simulation run")


# Database dependency
def get_db() -> Session:
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
        # Convert Pydantic models to dict format expected by run_game
        config = {
            "params": request.params,
            "firms": [{"cost": firm.cost} for firm in request.firms],
            "seed": request.seed,
        }

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


@app.get("/runs/{run_id}")
async def get_run(run_id: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Retrieve time-series results for a simulation run.

    Returns detailed results including market prices, quantities, and profits
    for each round and firm in the simulation.

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        Dictionary containing time-series data with arrays of equal length

    Raises:
        HTTPException: If run_id is not found or data retrieval fails
    """
    try:
        results = get_run_results(run_id, db)
        return dict(results)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
