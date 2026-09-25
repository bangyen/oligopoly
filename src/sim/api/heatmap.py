"""Profit-surface heatmap endpoint."""

import time

from fastapi import APIRouter, HTTPException

from sim.heatmap.bertrand_heatmap import (
    compute_bertrand_heatmap,
    compute_bertrand_segmented_heatmap,
    create_price_grid,
)
from sim.heatmap.cournot_heatmap import (
    compute_cournot_heatmap,
    compute_cournot_segmented_heatmap,
    create_quantity_grid,
)
from sim.models.models import DemandSegment, SegmentedDemand

from .schemas import HeatmapRequest, HeatmapResponse

router = APIRouter()


@router.post("/heatmap", response_model=HeatmapResponse)
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
