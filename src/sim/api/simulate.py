"""Simulation and scenario-comparison endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from sim.database import get_db
from sim.runners.runner import get_run_results, run_game

from ._common import (
    build_policy_events,
    extended_options,
    params_to_dict,
    round_metrics,
    segments_to_config,
)
from .schemas import (
    ComparisonRequest,
    ComparisonResponse,
    ComparisonResults,
    SimulationRequest,
    SimulationResponse,
)

router = APIRouter()

_DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "cournot": {"a": 100.0, "b": 1.0},
    "bertrand": {"alpha": 100.0, "beta": 1.0},
}
_PARAM_FIELDS = {"cournot": ("a", "b"), "bertrand": ("alpha", "beta")}
_PARAM_TYPES = {"cournot": "CournotParams", "bertrand": "BertrandParams"}


def _linear_params(request: SimulationRequest, costs: list[float]) -> dict[str, Any]:
    """Resolve and cross-validate linear demand params for ``request``."""
    # The hasattr check is robust against class instance mismatches from re-imports.
    intercept_name, slope_name = _PARAM_FIELDS[request.model]
    raw_params = request.params
    if raw_params is None:
        params: dict[str, Any] = dict(_DEFAULT_PARAMS[request.model])
    elif hasattr(raw_params, intercept_name) and hasattr(raw_params, slope_name):
        params = params_to_dict(raw_params)
    else:
        model_name = request.model.capitalize()
        raise HTTPException(
            status_code=400,
            detail=f"{model_name} model requires {_PARAM_TYPES[request.model]} "
            f"(fields: {intercept_name}, {slope_name})",
        )

    # Cross-validate params vs costs
    intercept = params.get(intercept_name, 100.0)
    slope = params.get(slope_name, 1.0)
    if any(cost >= intercept for cost in costs):
        raise HTTPException(
            status_code=400,
            detail=f"Firm costs cannot exceed demand intercept ({intercept_name}={intercept}). Firms with costs >= {intercept} would never be profitable.",
        )
    if slope < 0.1:
        raise HTTPException(
            status_code=400,
            detail=f"Demand slope ({slope_name}={slope}) is too flat. Use {slope_name} >= 0.1.",
        )
    return params


@router.post("/simulate", response_model=SimulationResponse)
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
        # Firm cost bounds are enforced by FirmConfig (cost > 0, fixed_cost >= 0),
        # which rejects invalid input with a 422 before this handler runs.
        # Costs are still needed below to cross-validate against demand params.
        costs = [firm.cost for firm in request.firms]

        if request.model not in _PARAM_FIELDS:
            # Should not happen due to Pydantic validation on `model` field
            raise HTTPException(
                status_code=400, detail=f"Unknown model type: {request.model}"
            )

        nonlinear_params, extra_config = extended_options(request)
        if nonlinear_params is not None:
            params: dict[str, Any] = nonlinear_params
        else:
            params = _linear_params(request, costs)

        config: dict[str, Any] = {
            "params": params,
            "firms": [
                {"cost": f.cost, "fixed_cost": f.fixed_cost} for f in request.firms
            ],
            "seed": request.seed,
            "events": build_policy_events(request.events),
            **extra_config,
        }
        if request.capacity_constraints is not None:
            config["params"]["capacity_constraints"] = request.capacity_constraints

        if request.segments:
            segments = segments_to_config(request.segments)

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

            config["params"]["segments"] = segments

        run_id = run_game(
            model=request.model, rounds=request.rounds, config=config, db=db
        )
        return SimulationResponse(run_id=run_id)

    except HTTPException:
        # Client errors raised above already carry the right status code.
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def _scenario_config(scenario: SimulationRequest, label: str) -> dict[str, Any]:
    """Build the run_game config for one side of a comparison."""
    config: dict[str, Any] = {
        "params": params_to_dict(scenario.params),
        "firms": [{"cost": firm.cost} for firm in scenario.firms],
        "seed": scenario.seed,
        "events": build_policy_events(scenario.events),
    }
    if scenario.segments:
        config["params"]["segments"] = segments_to_config(
            scenario.segments, f"{label} scenario segment"
        )
    nonlinear_params, extra_config = extended_options(scenario, label)
    if nonlinear_params is not None:
        config["params"] = nonlinear_params
    if scenario.capacity_constraints is not None:
        config["params"]["capacity_constraints"] = scenario.capacity_constraints
    config.update(extra_config)
    return config


@router.post("/compare", response_model=ComparisonResponse)
async def compare_scenarios(
    request: ComparisonRequest, db: Session = Depends(get_db)
) -> ComparisonResponse:
    """Run two simulation scenarios for comparison.

    Executes both left and right scenario simulations and returns their run IDs.
    Both simulations must have the same number of rounds for valid comparison.

    Raises:
        HTTPException: If simulations fail or configurations are invalid
    """
    try:
        if request.left_config.rounds != request.right_config.rounds:
            raise HTTPException(
                status_code=400,
                detail=f"Both scenarios must have the same number of rounds. "
                f"Left: {request.left_config.rounds}, Right: {request.right_config.rounds}",
            )

        left_config = _scenario_config(request.left_config, "Left")
        right_config = _scenario_config(request.right_config, "Right")

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
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/compare/{left_run_id}/{right_run_id}", response_model=ComparisonResults)
async def get_comparison_results(
    left_run_id: str, right_run_id: str, db: Session = Depends(get_db)
) -> ComparisonResults:
    """Retrieve aligned comparison results for two simulation runs.

    Returns time-series metrics for both runs aligned by round, along with
    calculated deltas (right - left) for each metric.

    Raises:
        HTTPException: If either run_id is not found or data retrieval fails
    """
    try:
        left_results = dict(get_run_results(left_run_id, db))
        right_results = dict(get_run_results(right_run_id, db))

        left_rounds = left_results.get("rounds", 0)
        right_rounds = right_results.get("rounds", 0)
        if left_rounds != right_rounds:
            raise HTTPException(
                status_code=400,
                detail=f"Runs must have the same number of rounds. "
                f"Left: {left_rounds}, Right: {right_rounds}",
            )

        left_metrics = _calculate_comparison_metrics(left_results)
        right_metrics = _calculate_comparison_metrics(right_results)

        # Deltas are right - left, truncated to the shorter series
        deltas = {
            name: [
                None
                if left_value is None or right_value is None
                else right_value - left_value
                for left_value, right_value in zip(
                    left_metrics[name], right_metrics[name]
                )
            ]
            for name in left_metrics
            if name in right_metrics
        }

        return ComparisonResults(
            left_run_id=left_run_id,
            right_run_id=right_run_id,
            rounds=left_rounds,
            left_metrics=left_metrics,
            right_metrics=right_metrics,
            deltas=deltas,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def _calculate_comparison_metrics(
    run_data: dict[str, Any],
) -> dict[str, list[float | None]]:
    """Calculate per-round metric arrays for comparison from run data.

    Expects the canonical nested-dict format from get_run_results:
    ``results[round_idx][firm_id] = {action, price, quantity, profit}``
    """
    rounds_data = run_data.get("results", {})
    model = run_data.get("model", "cournot")
    stored_params = run_data.get("params") or {}

    metrics: dict[str, list[float | None]] = {
        "market_price": [],
        "total_quantity": [],
        "total_profit": [],
        "hhi": [],
        "consumer_surplus": [],
    }

    for round_idx in sorted(rounds_data.keys(), key=int):
        firms_data = list(rounds_data[round_idx].values())
        if not firms_data:
            continue
        round_values = round_metrics(model, firms_data, stored_params, int(round_idx))
        for name, values in metrics.items():
            values.append(round_values[name])

    return metrics
