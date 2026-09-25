"""FastAPI application assembly and server entry point."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from sim.database import get_engine
from sim.models.models import Base

from . import runs, simulate

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create tables on startup unless running under tests."""
    if not os.getenv("TESTING"):
        try:
            Base.metadata.create_all(bind=get_engine())
        except Exception:
            logger.warning(
                "Database unavailable; skipping table creation", exc_info=True
            )
    yield


app = FastAPI(
    title="Oligopoly Simulation",
    description="Market competition simulation for industrial organization research",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(simulate.router)
app.include_router(runs.router)


@app.get("/healthz")
async def health_check() -> JSONResponse:
    """Health check endpoint for monitoring and load balancers."""
    return JSONResponse(content={"ok": True})


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with basic API information."""
    return {"message": "Oligopoly Simulation API", "version": "0.1.0", "docs": "/docs"}


def serve() -> None:
    """Run the API server. Entry point for the ``oligopoly`` console script."""
    import uvicorn

    uvicorn.run(
        "sim.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
