"""Helpers shared by the API route modules."""

import math
from typing import Any

from fastapi import HTTPException

from sim.models.metrics import (
    calculate_round_metrics_bertrand,
    calculate_round_metrics_cournot,
)
from sim.policy.policy_shocks import PolicyEvent

from .schemas import (
    BertrandParams,
    CournotParams,
    DemandSegmentConfig,
    PolicyEventRequest,
)


def params_to_dict(params: CournotParams | BertrandParams | None) -> dict[str, Any]:
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


def round_metrics(
    model: str, firms_data: list[dict[str, Any]], params: dict[str, Any]
) -> dict[str, float]:
    """Compute market-level metrics for one round of stored firm results."""
    quantities = [firm["quantity"] for firm in firms_data]
    prices = [firm["price"] for firm in firms_data]
    profits = [firm["profit"] for firm in firms_data]

    if model == "cournot":
        market_price = prices[0] if prices else 0.0
        hhi, cs = calculate_round_metrics_cournot(
            quantities, market_price, float(params.get("a", 100.0))
        )
    else:  # bertrand
        market_price = min(prices) if prices else 0.0
        hhi, cs = calculate_round_metrics_bertrand(
            prices, quantities, sum(quantities), float(params.get("alpha", 100.0))
        )

    return {
        "hhi": hhi,
        "consumer_surplus": cs,
        "market_price": market_price,
        "total_quantity": sum(quantities),
        "total_profit": sum(profits),
    }
