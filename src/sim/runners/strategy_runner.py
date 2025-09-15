"""Strategy-based simulation runner for oligopoly models.

This module provides a strategy-based alternative to the adaptive simulation runner,
allowing firms to use explicit strategies (Static, TitForTat, RandomWalk) instead of
the built-in profit-based adaptation.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sqlalchemy.orm import Session

from ..events.event_logger import EventLogger
from ..events.event_types import EventType
from ..games.bertrand import BertrandResult, bertrand_simulation
from ..games.cournot import CournotResult, cournot_simulation
from ..models.models import Result, Round, Run
from ..policy.policy_shocks import PolicyEvent, PolicyType, apply_policy_shock
from ..strategies.strategies import Strategy


def run_strategy_game(
    model: str,
    rounds: int,
    strategies: List[Strategy],
    costs: List[float],
    params: Dict[str, Any],
    bounds: Tuple[float, float],
    db: Session,
    seed: Optional[int] = None,
    events: Optional[List[PolicyEvent]] = None,
) -> str:
    """Run a multi-round oligopoly simulation using explicit strategies.

    Args:
        model: Type of competition model ("cournot" or "bertrand")
        rounds: Number of rounds to simulate
        strategies: List of strategy instances for each firm
        costs: List of marginal costs for each firm
        params: Market parameters (a, b for Cournot; alpha, beta for Bertrand)
        bounds: Tuple of (min, max) action bounds
        db: Database session for persistence
        seed: Optional random seed for reproducibility

    Returns:
        run_id: Unique identifier for this simulation run

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If database operations fail
    """
    # Validate inputs
    if model not in ["cournot", "bertrand"]:
        raise ValueError(f"Model must be 'cournot' or 'bertrand', got '{model}'")

    if rounds <= 0:
        raise ValueError(f"Rounds must be positive, got {rounds}")

    if len(strategies) != len(costs):
        raise ValueError(
            f"Number of strategies ({len(strategies)}) must match number of firms ({len(costs)})"
        )

    if not strategies:
        raise ValueError("At least one strategy is required")

    # Create run record
    run = Run(model=model, rounds=rounds)
    db.add(run)
    db.flush()  # Get the run_id

    # Initialize event logger
    event_logger = EventLogger(str(run.id), db)

    try:
        # Initialize firm histories
        firm_histories: List[List[Union[CournotResult, BertrandResult]]] = [
            [] for _ in strategies
        ]

        # Run simulation rounds
        for round_num in range(rounds):
            # Create round record
            round_record = Round(run_id=run.id, idx=round_num)
            db.add(round_record)
            db.flush()

            # Get actions from strategies
            actions = []
            for firm_idx, strategy in enumerate(strategies):
                # Build rival histories (exclude current firm)
                rival_histories: List[
                    Sequence[Union[CournotResult, BertrandResult]]
                ] = [firm_histories[i] for i in range(len(strategies)) if i != firm_idx]

                action = strategy.next_action(
                    round_num=round_num,
                    my_history=firm_histories[firm_idx],
                    rival_histories=rival_histories,
                    bounds=bounds,
                    market_params=params,
                )
                actions.append(action)

            # Run simulation round
            if model == "cournot":
                result: Union[CournotResult, BertrandResult] = _run_cournot_round(
                    params, costs, actions
                )
            else:  # bertrand
                result = _run_bertrand_round(params, costs, actions)

            # Apply policy shocks if any
            if events:
                for event in events:
                    if event.round_idx == round_num:
                        result = apply_policy_shock(result, event, costs)

                        # Log policy event
                        policy_event_mapping = {
                            PolicyType.TAX: EventType.TAX_APPLIED,
                            PolicyType.SUBSIDY: EventType.SUBSIDY_APPLIED,
                            PolicyType.PRICE_CAP: EventType.PRICE_CAP_APPLIED,
                        }

                        policy_event_type = policy_event_mapping.get(
                            event.policy_type, EventType.TAX_APPLIED
                        )
                        event_logger.log_policy_event(
                            event_type=policy_event_type,
                            round_idx=round_num,
                            policy_value=event.value,
                            policy_details={
                                "policy_type": event.policy_type.value,
                                "original_value": event.value,
                            },
                        )

            # Store results in database
            for firm_idx, (action, cost) in enumerate(zip(actions, costs)):
                if model == "cournot":
                    # Type guard: result is CournotResult
                    assert isinstance(result, CournotResult)
                    profit = (result.price - cost) * action
                    qty = action
                    price = result.price
                else:  # bertrand
                    # Type guard: result is BertrandResult
                    assert isinstance(result, BertrandResult)
                    profit = (action - cost) * result.quantities[firm_idx]
                    qty = result.quantities[firm_idx]
                    price = action

                result_record = Result(
                    run_id=run.id,
                    round_id=round_record.id,
                    round_idx=round_num,
                    firm_id=firm_idx,
                    action=action,
                    price=price,
                    qty=qty,
                    profit=profit,
                )
                db.add(result_record)

            # Update firm histories
            for firm_idx in range(len(strategies)):
                firm_histories[firm_idx].append(result)

        db.commit()
        return str(run.id)

    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Simulation failed: {e}")


def _run_cournot_round(
    params: Dict[str, Any], costs: List[float], quantities: List[float]
) -> CournotResult:
    """Run a single Cournot round."""
    a = params.get("a", 100.0)
    b = params.get("b", 1.0)
    return cournot_simulation(a, b, costs, quantities)


def _run_bertrand_round(
    params: Dict[str, Any], costs: List[float], prices: List[float]
) -> BertrandResult:
    """Run a single Bertrand round."""
    alpha = params.get("alpha", 100.0)
    beta = params.get("beta", 1.0)
    return bertrand_simulation(alpha, beta, costs, prices)


def get_strategy_run_results(run_id: str, db: Session) -> Dict[str, Any]:
    """Get results for a strategy-based simulation run.

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        Dictionary containing run results and metadata

    Raises:
        ValueError: If run_id is not found
    """
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise ValueError(f"Run {run_id} not found")

    # Get all results for this run
    results = (
        db.query(Result)
        .filter(Result.run_id == run_id)
        .order_by(Result.round_idx, Result.firm_id)
        .all()
    )

    # Organize results by round and firm
    rounds_data: Dict[int, Dict[int, Dict[str, float]]] = {}
    for result in results:
        round_idx = int(result.round_idx)
        firm_id = int(result.firm_id)

        if round_idx not in rounds_data:
            rounds_data[round_idx] = {}

        rounds_data[round_idx][firm_id] = {
            "action": float(result.action),
            "price": float(result.price),
            "quantity": float(result.qty),
            "profit": float(result.profit),
        }

    return {
        "run_id": run_id,
        "model": run.model,
        "rounds": run.rounds,
        "created_at": run.created_at,
        "results": rounds_data,
    }
