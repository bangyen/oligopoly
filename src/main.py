"""FastAPI application for oligopoly simulation.

This module provides the main FastAPI application with endpoints
for health checks and simulation management.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sim.models import Base

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/oligopoly")

# Create database engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI app
app = FastAPI(
    title="Oligopoly Simulation",
    description="Market competition simulation for industrial organization research",
    version="0.1.0"
)


@app.get("/healthz")
async def health_check() -> JSONResponse:
    """Health check endpoint for monitoring and load balancers.
    
    Returns a simple status indicating the service is running.
    This endpoint is used by Kubernetes health checks and monitoring systems.
    """
    return JSONResponse(content={"ok": True})


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Oligopoly Simulation API",
        "version": "0.1.0",
        "docs": "/docs"
    }
