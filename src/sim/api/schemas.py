"""Request and response schemas for the REST API."""

from typing import Any

from pydantic import BaseModel, Field

from sim.models.market_evolution import (
    MarketEvolutionConfig,
)
from sim.policy.policy_shocks import PolicyType


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
    params: CournotParams | BertrandParams | None = Field(
        default=None,
        description="Typed market demand parameters. Use CournotParams (a, b) for "
        "cournot and BertrandParams (alpha, beta) for bertrand.",
    )
    firms: list[FirmConfig] = Field(
        ..., min_length=1, max_length=10, description="Firm configurations"
    )
    segments: list[DemandSegmentConfig] | None = Field(
        None,
        description="Segmented demand configuration (overrides single-segment params)",
    )
    demand_type: str = Field(
        default="linear",
        pattern="^(linear|isoelastic)$",
        description="Type of demand function",
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")
    events: list[PolicyEventRequest] | None = Field(
        default_factory=list, description="Policy events to apply during simulation"
    )
    advanced_strategies: list[AdvancedStrategyConfig] | None = Field(
        default=None, description="Simplified learning strategies for firms"
    )
    market_evolution: MarketEvolutionConfig | None = Field(
        default=None, description="Market evolution configuration"
    )
    enhanced_demand: EnhancedDemandConfig | None = Field(
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
    left_metrics: dict[str, list[float]] = Field(
        ..., description="Left scenario metrics arrays"
    )
    right_metrics: dict[str, list[float]] = Field(
        ..., description="Right scenario metrics arrays"
    )
    deltas: dict[str, list[float]] = Field(
        ..., description="Delta arrays (right - left)"
    )


class EventItem(BaseModel):
    """Individual event item for API responses."""

    id: int = Field(..., description="Unique event identifier")
    round_idx: int = Field(..., description="Round index when event occurred")
    event_type: str = Field(..., description="Type of event")
    firm_id: int | None = Field(None, description="Firm involved (if applicable)")
    description: str = Field(..., description="Human-readable event description")
    event_data: dict[str, Any] | None = Field(None, description="Additional event data")
    created_at: str = Field(..., description="Event timestamp")


class EventsResponse(BaseModel):
    """Response model for events endpoint."""

    run_id: str = Field(..., description="Simulation run ID")
    total_events: int = Field(..., description="Total number of events")
    events: list[EventItem] = Field(..., description="Ordered list of events")


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
    firm_data: dict[int, dict[str, float]] = Field(
        ..., description="Firm-specific data"
    )
    events: list[dict[str, Any]] = Field(..., description="Events in this round")
    annotations: list[str] = Field(..., description="Human-readable annotations")


class ReplayResponse(BaseModel):
    """Response model for replay endpoint."""

    run_id: str = Field(..., description="Simulation run ID")
    total_frames: int = Field(..., description="Total number of frames")
    frames_with_events: int = Field(..., description="Number of frames with events")
    event_rounds: list[int] = Field(..., description="Rounds containing events")
    frames: list[ReplayFrame] = Field(..., description="All replay frames")


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
    results: dict[str, Any] | None = None


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
    action_range: tuple[float, float] = Field(
        ..., description="Min and max values for action grid (quantity or price)"
    )
    other_actions: list[float] = Field(
        ..., description="Fixed actions for all other firms"
    )
    params: CournotParams | BertrandParams | None = Field(
        default=None,
        description="Typed demand parameters. Use CournotParams for cournot, BertrandParams for bertrand.",
    )
    firms: list[FirmConfig] = Field(
        ..., min_length=2, max_length=10, description="Firm configurations"
    )
    segments: list[DemandSegmentConfig] | None = Field(
        None,
        description="Segmented demand configuration (overrides single-segment params)",
    )


class HeatmapResponse(BaseModel):
    """Response model for heatmap endpoint."""

    model: str = Field(..., description="Competition model type")
    firm_i: int = Field(..., description="Index of firm surface computed for")
    firm_j: int = Field(..., description="Index of second firm in heatmap")
    profit_surface: list[list[float]] = Field(
        ..., description="2D array of profits for firm_i"
    )
    market_share_surface: list[list[float]] | None = Field(
        None, description="2D array of market shares for firm_i (Bertrand only)"
    )
    action_i_grid: list[float] = Field(
        ..., description="Grid values for firm_i actions (quantities or prices)"
    )
    action_j_grid: list[float] = Field(
        ..., description="Grid values for firm_j actions (quantities or prices)"
    )
    computation_time_ms: float = Field(
        ..., description="Computation time in milliseconds"
    )
