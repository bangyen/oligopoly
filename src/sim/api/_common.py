"""Helpers shared by the API route modules."""

import math
from typing import Any

from fastapi import HTTPException

from sim.markets import market_metrics
from sim.policy.policy_shocks import PolicyEvent

from .schemas import (
    BertrandParams,
    CournotParams,
    DemandSegmentConfig,
    IsoelasticParams,
    PolicyEventRequest,
    SimulationRequest,
)


def params_to_dict(
    params: CournotParams | BertrandParams | IsoelasticParams | None,
) -> dict[str, Any]:
    """Normalise typed demand params to a plain dict."""
    if params is None:
        return {}
    return params.model_dump() if hasattr(params, "model_dump") else dict(params)


def build_policy_events(events: list[PolicyEventRequest] | None) -> list[PolicyEvent]:
    """Convert request policy events to simulation policy events."""
    return [
        PolicyEvent(
            round_idx=event.round_idx,
            policy_type=event.policy_type,
            value=event.value,
        )
        for event in events or []
    ]


def segments_to_config(
    segments: list[DemandSegmentConfig], label: str = "Segment"
) -> list[dict[str, float]]:
    """Validate that segment weights sum to 1 and convert them to dicts."""
    total_weight = sum(segment.weight for segment in segments)
    if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
        raise HTTPException(
            status_code=400,
            detail=f"{label} weights must sum to 1.0, got {total_weight:.6f}",
        )
    return [
        {"alpha": segment.alpha, "beta": segment.beta, "weight": segment.weight}
        for segment in segments
    ]


def extended_options(
    request: SimulationRequest, label: str = ""
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate non-linear demand, learning strategies and market evolution.

    Returns ``(params, extra_config)``: ``params`` replaces the linear demand
    params when a non-linear demand system is requested (else None), and
    ``extra_config`` is merged into the run_game config.
    """
    prefix = f"{label} scenario: " if label else ""

    def fail(detail: str) -> HTTPException:
        return HTTPException(status_code=400, detail=prefix + detail)

    params: dict[str, Any] | None = None
    raw = request.params
    is_iso_params = raw is not None and hasattr(raw, "elasticity")
    ces = request.enhanced_demand
    if ces is not None and ces.demand_type != "ces":
        ces = None

    if request.demand_type == "isoelastic":
        if ces is not None:
            raise fail("Use either demand_type='isoelastic' or CES enhanced_demand")
        if request.segments:
            raise fail("Segmented demand is only supported with linear demand")
        if raw is not None and not is_iso_params:
            raise fail(
                "demand_type='isoelastic' requires IsoelasticParams "
                "(fields: A, elasticity)"
            )
        iso = raw if is_iso_params else IsoelasticParams(A=100.0, elasticity=2.0)
        params = {
            "demand_type": "isoelastic",
            "A": getattr(iso, "A"),
            "elasticity": getattr(iso, "elasticity"),
        }
    elif is_iso_params:
        raise fail("IsoelasticParams require demand_type='isoelastic'")

    if ces is not None:
        if request.model != "bertrand":
            raise fail("CES demand (enhanced_demand) requires model='bertrand'")
        if raw is not None:
            raise fail(
                "CES demand takes its parameters from enhanced_demand; omit params"
            )
        if request.segments:
            raise fail("Segmented demand is only supported with linear demand")
        if len(request.firms) < 2:
            raise fail("CES demand needs at least two firms")
        if ces.qualities is not None:
            if len(ces.qualities) != len(request.firms):
                raise fail(
                    f"enhanced_demand.qualities has {len(ces.qualities)} entries "
                    f"for {len(request.firms)} firms"
                )
            if any(q <= 0 for q in ces.qualities):
                raise fail("enhanced_demand.qualities must be positive")
        params = {
            "demand_type": "ces",
            "elasticity": ces.elasticity,
            "market_size": ces.market_size,
        }
        if ces.qualities is not None:
            params["qualities"] = list(ces.qualities)

    extra: dict[str, Any] = {}
    if request.advanced_strategies:
        firm_ids = [spec.firm_id for spec in request.advanced_strategies]
        if len(set(firm_ids)) != len(firm_ids):
            raise fail("advanced_strategies has more than one entry for a firm")
        for firm_id in firm_ids:
            if firm_id >= len(request.firms):
                raise fail(
                    f"advanced_strategies firm_id {firm_id} is out of range for "
                    f"{len(request.firms)} firms"
                )
        extra["advanced_strategies"] = [
            spec.model_dump(exclude_none=True) for spec in request.advanced_strategies
        ]
    if request.market_evolution is not None:
        extra["market_evolution"] = request.market_evolution.model_dump()

    return params, extra


def round_metrics(
    model: str, firms_data: list[dict[str, Any]], params: dict[str, Any]
) -> dict[str, float | None]:
    """Compute market-level metrics for one round of stored firm results."""
    quantities = [firm["quantity"] for firm in firms_data]
    prices = [firm["price"] for firm in firms_data]
    profits = [firm["profit"] for firm in firms_data]
    market_price, hhi, cs = market_metrics(model, params, prices, quantities)
    return {
        "hhi": hhi,
        "consumer_surplus": cs,
        "market_price": market_price,
        "total_quantity": sum(quantities),
        "total_profit": sum(profits),
    }
