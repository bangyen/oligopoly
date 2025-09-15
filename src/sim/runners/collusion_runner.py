"""Enhanced strategy runner with collusion and regulator dynamics.

This module extends the basic strategy runner to include collusion detection,
defection mechanisms, and regulatory interventions. It integrates with the
collusion manager to track cartel behavior and regulatory responses.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sqlalchemy.orm import Session

from ..collusion import CollusionEventType, CollusionManager, RegulatorState
from ..events.event_logger import EventLogger
from ..events.event_types import EventType
from ..games.bertrand import BertrandResult, bertrand_simulation
from ..games.cournot import CournotResult, cournot_simulation
from ..models.models import CollusionEvent, Result, Round, Run
from ..strategies.collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
)
from ..strategies.strategies import Strategy


def run_collusion_game(
    model: str,
    rounds: int,
    strategies: List[Strategy],
    costs: List[float],
    params: Dict[str, Any],
    bounds: Tuple[float, float],
    db: Session,
    collusion_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> str:
    """Run a multi-round oligopoly simulation with collusion dynamics.

    Args:
        model: Type of competition model ("cournot" or "bertrand")
        rounds: Number of rounds to simulate
        strategies: List of strategy instances for each firm
        costs: List of marginal costs for each firm
        params: Market parameters (a, b for Cournot; alpha, beta for Bertrand)
        bounds: Tuple of (min, max) action bounds
        db: Database session for persistence
        collusion_config: Configuration for collusion dynamics
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

    # Initialize collusion manager
    regulator_state = None
    if collusion_config:
        regulator_state = RegulatorState(
            hhi_threshold=collusion_config.get("hhi_threshold", 0.8),
            price_threshold_multiplier=collusion_config.get(
                "price_threshold_multiplier", 1.5
            ),
            baseline_price=collusion_config.get("baseline_price", 0.0),
            intervention_probability=collusion_config.get(
                "intervention_probability", 0.8
            ),
            penalty_amount=collusion_config.get("penalty_amount", 100.0),
            price_cap_multiplier=collusion_config.get("price_cap_multiplier", 0.9),
        )

    collusion_manager = CollusionManager(regulator_state)

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

        # Add model type to params for strategies
        params_with_model = params.copy()
        params_with_model["model_type"] = model

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

                # Handle collusion-aware strategies
                if isinstance(
                    strategy, (CollusiveStrategy, CartelStrategy, OpportunisticStrategy)
                ):
                    action = strategy.next_action(
                        round_num=round_num,
                        my_history=firm_histories[firm_idx],
                        rival_histories=rival_histories,
                        bounds=bounds,
                        market_params=params_with_model,
                        collusion_manager=collusion_manager,
                        my_cost=(
                            costs[firm_idx]
                            if isinstance(strategy, OpportunisticStrategy)
                            else None
                        ),
                    )
                else:
                    # Standard strategy
                    action = strategy.next_action(
                        round_num=round_num,
                        my_history=firm_histories[firm_idx],
                        rival_histories=rival_histories,
                        bounds=bounds,
                        market_params=params_with_model,
                    )
                actions.append(action)

            # Run simulation round
            if model == "cournot":
                result: Union[CournotResult, BertrandResult] = _run_cournot_round(
                    params, costs, actions
                )
            else:  # bertrand
                result = _run_bertrand_round(params, costs, actions)

            # Check for defections if cartel is active
            if (
                collusion_manager.is_cartel_active()
                and collusion_manager.current_cartel
            ):
                cartel = collusion_manager.current_cartel
                for firm_idx, action in enumerate(actions):
                    if firm_idx in cartel.participating_firms:
                        if model == "bertrand":
                            collusion_manager.detect_defection(
                                round_idx=round_num,
                                firm_id=firm_idx,
                                firm_price=action,
                                firm_quantity=result.quantities[firm_idx],
                                cartel_price=cartel.collusive_price,
                                cartel_quantity=cartel.collusive_quantity,
                            )
                        else:  # cournot
                            # Type guard: result is CournotResult
                            assert isinstance(result, CournotResult)
                            collusion_manager.detect_defection(
                                round_idx=round_num,
                                firm_id=firm_idx,
                                firm_price=result.price,
                                firm_quantity=action,
                                cartel_price=cartel.collusive_price,
                                cartel_quantity=cartel.collusive_quantity,
                            )

            # Calculate market shares for HHI
            total_quantity = (
                sum(result.quantities) if hasattr(result, "quantities") else 0
            )
            market_shares = [
                q / total_quantity if total_quantity > 0 else 0
                for q in result.quantities
            ]

            # Check for regulator intervention
            prices = (
                result.prices
                if hasattr(result, "prices")
                else [result.price] * len(costs)
            )
            quantities = result.quantities

            should_intervene, intervention_type, intervention_value = (
                collusion_manager.check_regulator_intervention(
                    round_idx=round_num,
                    market_shares=market_shares,
                    prices=prices,
                    quantities=quantities,
                )
            )

            # Apply regulator intervention if needed
            profits = result.profits.copy()
            if (
                should_intervene
                and intervention_type
                and intervention_value is not None
            ):
                profits = collusion_manager.apply_regulator_intervention(
                    round_idx=round_num,
                    intervention_type=intervention_type,
                    intervention_value=intervention_value,
                    firm_profits=profits,
                )

            # Store results in database
            for firm_idx, (action, cost) in enumerate(zip(actions, costs)):
                if model == "cournot":
                    # Type guard: result is CournotResult
                    assert isinstance(result, CournotResult)
                    profit = profits[firm_idx]
                    qty = action
                    price = result.price
                else:  # bertrand
                    # Type guard: result is BertrandResult
                    assert isinstance(result, BertrandResult)
                    profit = profits[firm_idx]
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

            # Store collusion events using new event system
            for event in collusion_manager.get_events_for_round(round_num):
                # Map old collusion event types to new event types
                event_type_mapping = {
                    CollusionEventType.CARTEL_FORMED: EventType.CARTEL_FORMED,
                    CollusionEventType.FIRM_DEFECTED: EventType.DEFECTION_DETECTED,
                    CollusionEventType.REGULATOR_INTERVENED: EventType.REGULATOR_INTERVENTION,
                    CollusionEventType.PENALTY_IMPOSED: EventType.PENALTY_IMPOSED,
                    CollusionEventType.PRICE_CAP_IMPOSED: EventType.PRICE_CAP_IMPOSED,
                }

                new_event_type = event_type_mapping.get(
                    event.event_type, EventType.REGULATOR_INTERVENTION
                )

                # Log using new event system
                event_logger.log_collusion_event(
                    event_type=new_event_type,
                    round_idx=event.round_idx,
                    firm_id=event.firm_id,
                    cartel_data=event.data,
                )

                # Also maintain legacy collusion events table for backward compatibility
                event_record = CollusionEvent(
                    run_id=run.id,
                    round_idx=event.round_idx,
                    event_type=event.event_type.value,
                    firm_id=event.firm_id,
                    description=event.description,
                    event_data=event.data,
                )
                db.add(event_record)

            # Update firm histories
            for firm_idx in range(len(strategies)):
                firm_histories[firm_idx].append(result)

            # Form cartel if conditions are met (simplified logic)
            if (
                round_num > 0
                and round_num % 5 == 0
                and not collusion_manager.is_cartel_active()
                and collusion_config
                and collusion_config.get("auto_form_cartel", False)
            ):

                # Simple cartel formation logic - can be made more sophisticated
                avg_price = sum(prices) / len(prices)
                avg_quantity = sum(quantities) / len(quantities)

                collusion_manager.form_cartel(
                    round_idx=round_num,
                    collusive_price=avg_price * 1.2,  # 20% above current average
                    collusive_quantity=avg_quantity * 0.8,  # 20% below current average
                    participating_firms=list(range(len(strategies))),
                )

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


def get_collusion_run_results(run_id: str, db: Session) -> Dict[str, Any]:
    """Get results for a collusion simulation run including events.

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        Dictionary containing run results, events, and metadata

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

    # Get all collusion events for this run
    events = (
        db.query(CollusionEvent)
        .filter(CollusionEvent.run_id == run_id)
        .order_by(CollusionEvent.round_idx, CollusionEvent.id)
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

    # Organize events by round
    events_data: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        round_idx = int(event.round_idx)
        if round_idx not in events_data:
            events_data[round_idx] = []

        events_data[round_idx].append(
            {
                "event_type": event.event_type,
                "firm_id": event.firm_id,
                "description": event.description,
                "data": event.event_data or {},
                "created_at": (
                    event.created_at.isoformat() if event.created_at else None
                ),
            }
        )

    return {
        "run_id": run_id,
        "model": run.model,
        "rounds": run.rounds,
        "created_at": run.created_at,
        "results": rounds_data,
        "events": events_data,
    }


def create_collusion_simulation_config(
    hhi_threshold: float = 0.8,
    price_threshold_multiplier: float = 1.5,
    baseline_price: float = 0.0,
    intervention_probability: float = 0.8,
    penalty_amount: float = 100.0,
    price_cap_multiplier: float = 0.9,
    auto_form_cartel: bool = False,
) -> Dict[str, Any]:
    """Create a configuration dictionary for collusion simulation.

    Args:
        hhi_threshold: HHI threshold for regulatory intervention
        price_threshold_multiplier: Price threshold as multiple of baseline
        baseline_price: Baseline competitive price
        intervention_probability: Probability of intervention when thresholds exceeded
        penalty_amount: Fixed penalty amount
        price_cap_multiplier: Price cap as fraction of detected price
        auto_form_cartel: Whether to automatically form cartels

    Returns:
        Configuration dictionary for collusion simulation
    """
    return {
        "hhi_threshold": hhi_threshold,
        "price_threshold_multiplier": price_threshold_multiplier,
        "baseline_price": baseline_price,
        "intervention_probability": intervention_probability,
        "penalty_amount": penalty_amount,
        "price_cap_multiplier": price_cap_multiplier,
        "auto_form_cartel": auto_form_cartel,
    }
