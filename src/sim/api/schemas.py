"""Request and response schemas for the REST API."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

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


class CournotParams(BaseModel):
    """Typed linear demand parameters for the Cournot competition model."""

    model_config = ConfigDict(extra="forbid")

    a: float = Field(100.0, gt=0, description="Demand intercept P = a - b*Q")
    b: float = Field(
        1.0, gt=0, description="Demand slope (price sensitivity to quantity)"
    )


class BertrandParams(BaseModel):
    """Typed linear demand parameters for the Bertrand competition model."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(
        100.0, gt=0, description="Demand intercept Q(p) = alpha - beta*p"
    )
    beta: float = Field(1.0, gt=0, description="Demand slope")


class IsoelasticParams(BaseModel):
    """Isoelastic demand Q(P) = (A / P)^elasticity, i.e. P(Q) = A * Q^(-1/elasticity).

    Used with ``demand_type="isoelastic"`` for either competition model.
    """

    model_config = ConfigDict(extra="forbid")

    A: float = Field(100.0, gt=0, description="Demand scale parameter")
    elasticity: float = Field(
        2.0, gt=1, description="Constant price elasticity of demand (> 1)"
    )


class EnhancedDemandConfig(BaseModel):
    """Differentiated-products demand for Bertrand competition.

    ``ces``: CES demand with elasticity of substitution ``elasticity``, total
    consumer expenditure ``market_size`` and optional per-firm ``qualities``.
    ``linear`` is the default homogeneous-good demand (same as omitting this).
    """

    model_config = ConfigDict(extra="forbid")

    demand_type: str = Field(
        default="linear",
        pattern="^(linear|ces)$",
        description="Demand function type",
    )
    elasticity: float = Field(
        default=2.0, gt=1, description="Elasticity of substitution (CES only)"
    )
    market_size: float = Field(
        default=100.0, gt=0, description="Total consumer expenditure (CES only)"
    )
    qualities: list[float] | None = Field(
        default=None,
        description="Per-firm product quality (CES only; defaults to 1 for all)",
    )


class AdvancedStrategyConfig(BaseModel):
    """A learning or collusion strategy for one firm.

    Firms without one follow adaptive Nash play.
    """

    model_config = ConfigDict(extra="forbid")

    firm_id: int = Field(..., ge=0, description="Index of the firm in `firms`")
    strategy_type: str = Field(
        ...,
        pattern=(
            "^(fictitious_play|q_learning|deep_q_learning|behavioral"
            "|cartel|collusive|opportunistic)$"
        ),
        description="Learning strategy, or a collusion strategy: two or more "
        "cartel/collusive/opportunistic firms form a cartel at their joint-profit "
        "maximising action",
    )
    learning_rate: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Learning rate (q_learning, deep_q_learning, behavioral)",
    )
    memory_length: int | None = Field(
        default=None, gt=0, description="Rounds of rival history kept (fictitious_play)"
    )
    exploration_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Exploration probability (fictitious_play, q_learning)",
    )


class MarketEvolutionRequest(BaseModel):
    """Market dynamics applied between rounds."""

    model_config = ConfigDict(extra="forbid")

    growth_rate: float = Field(
        default=0.02, gt=-1, le=1, description="Demand growth per round"
    )
    entry_cost: float = Field(
        default=100.0, gt=0, description="Cost a potential entrant must recoup"
    )
    exit_threshold: float = Field(
        default=-50.0, description="Per-round profit below which firms may exit"
    )
    innovation_rate: float = Field(
        default=0.1, ge=0, le=1, description="Probability scale for innovation"
    )


class SimulationRequest(BaseModel):
    """Request model for simulation endpoint.

    Unknown fields are rejected (422) rather than silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        ...,
        pattern="^(cournot|bertrand)$",
        description="Competition model type",
    )
    rounds: int = Field(..., gt=0, le=1000, description="Number of simulation rounds")
    params: CournotParams | BertrandParams | IsoelasticParams | None = Field(
        default=None,
        description="Typed market demand parameters. Use CournotParams (a, b) for "
        "cournot and BertrandParams (alpha, beta) for bertrand with linear demand, "
        "or IsoelasticParams (A, elasticity) with demand_type='isoelastic'.",
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
        description="Homogeneous-good demand curve",
    )
    seed: int | None = Field(None, description="Random seed for reproducibility")
    events: list[PolicyEventRequest] | None = Field(
        default_factory=list, description="Policy events to apply during simulation"
    )
    advanced_strategies: list[AdvancedStrategyConfig] | None = Field(
        default=None, description="Learning strategies for individual firms"
    )
    market_evolution: MarketEvolutionRequest | None = Field(
        default=None,
        description="Enable market growth, innovation and entry/exit between rounds",
    )
    enhanced_demand: EnhancedDemandConfig | None = Field(
        default=None,
        description="Differentiated-products demand (CES) for bertrand",
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
    left_metrics: dict[str, list[float | None]] = Field(
        ...,
        description="Left scenario metrics arrays (consumer_surplus is null for "
        "CES demand, which has no money-metric surplus)",
    )
    right_metrics: dict[str, list[float | None]] = Field(
        ..., description="Right scenario metrics arrays"
    )
    deltas: dict[str, list[float | None]] = Field(
        ..., description="Delta arrays (right - left); null where either side is"
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
    consumer_surplus: float | None = Field(
        ..., description="Consumer surplus (null for CES demand)"
    )
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
