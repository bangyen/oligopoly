"""Multi-round simulation runner with persistence.

This module implements the core functionality for running multi-round
oligopoly simulations and persisting results to the database.
"""

import random
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.sim.games.bertrand import bertrand_segmented_simulation, bertrand_simulation
from src.sim.games.cournot import cournot_segmented_simulation, cournot_simulation
from src.sim.models.models import DemandSegment, Result, Round, Run, SegmentedDemand
from src.sim.policy.policy_shocks import apply_policy_shock, validate_policy_events
from src.sim.strategies.nash_strategies import (
    adaptive_nash_strategy,
    validate_market_clearing,
)


def run_game(model: str, rounds: int, config: Dict[str, Any], db: Session) -> str:
    """Run a multi-round oligopoly simulation with persistence.

    Executes the specified number of rounds of either Cournot or Bertrand
    competition, persisting each round's results to the database.

    Args:
        model: Type of competition model ("cournot" or "bertrand")
        rounds: Number of rounds to simulate
        config: Configuration dictionary containing:
            - params: Market parameters (a, b for Cournot; alpha, beta for Bertrand)
            - firms: List of firm configurations with costs
            - seed: Optional random seed for reproducibility
        db: Database session for persistence

    Returns:
        run_id: Unique identifier for this simulation run

    Raises:
        ValueError: If model is invalid or config is malformed
        RuntimeError: If database operations fail
    """
    # Validate model
    if model not in ["cournot", "bertrand"]:
        raise ValueError(f"Model must be 'cournot' or 'bertrand', got '{model}'")

    # Validate rounds
    if rounds <= 0:
        raise ValueError(f"Rounds must be positive, got {rounds}")

    # Extract configuration
    params = config.get("params", {})
    firms = config.get("firms", [])
    seed = config.get("seed")
    events = config.get("events", [])

    if not firms:
        raise ValueError("Config must contain 'firms' list")

    # Validate policy events
    if events:
        validate_policy_events(events, rounds)

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Create run record
    run = Run(model=model, rounds=rounds)
    db.add(run)
    db.flush()  # Get the run_id

    try:
        # Extract firm costs
        costs = [firm["cost"] for firm in firms]
        num_firms = len(costs)

        # Initialize firm actions using Nash equilibrium as starting point
        if model == "cournot":
            # Start near Nash equilibrium quantities
            from src.sim.strategies.nash_strategies import cournot_nash_equilibrium

            # Check if segmented demand
            segments_config = params.get("segments")
            if segments_config:
                # For segmented demand, calculate Nash equilibrium using effective parameters
                weighted_alpha = sum(
                    segment["weight"] * segment["alpha"] for segment in segments_config
                )
                weighted_beta = sum(
                    segment["weight"] * segment["beta"] for segment in segments_config
                )
                nash_quantities, _, _ = cournot_nash_equilibrium(
                    weighted_alpha, weighted_beta, costs
                )
            else:
                # Standard demand
                a = params.get("a", 100.0)
                b = params.get("b", 1.0)
                nash_quantities, _, _ = cournot_nash_equilibrium(a, b, costs)

            # Add some randomness around Nash equilibrium
            actions = [qty + random.uniform(-2, 2) for qty in nash_quantities]
        else:  # bertrand
            # Start near Nash equilibrium prices
            alpha = params.get("alpha", 100.0)
            beta = params.get("beta", 1.0)
            from src.sim.strategies.nash_strategies import bertrand_nash_equilibrium

            nash_prices, _, _, _ = bertrand_nash_equilibrium(alpha, beta, costs)
            # Add some randomness around Nash equilibrium
            actions = [price + random.uniform(-1, 1) for price in nash_prices]

        # Run simulation rounds
        for round_idx in range(rounds):
            # Create round record
            round_record = Round(run_id=run.id, idx=round_idx)
            db.add(round_record)

            # Run simulation for this round
            if model == "cournot":
                result = _run_cournot_round(params, costs, actions)
            else:  # bertrand
                result = _run_bertrand_round(params, costs, actions)

            # Apply policy shocks for this round
            for event in events:
                if event.round_idx == round_idx:
                    result = apply_policy_shock(result, event, costs)

            # Extract results after policy shocks
            if model == "cournot":
                market_price = result.price
                quantities = result.quantities
                profits = result.profits
            else:  # bertrand
                # In Bertrand, each firm can have different prices, but we store the market-clearing price
                # For simplicity, we'll use the minimum price as the market price
                market_price = min(result.prices) if result.prices else 0.0
                quantities = result.quantities
                profits = result.profits

            # Persist results for each firm
            for firm_id in range(num_firms):
                result_record = Result(
                    run_id=run.id,
                    round_id=round_record.id,
                    round_idx=round_idx,
                    firm_id=firm_id,
                    action=actions[firm_id],
                    price=market_price,
                    qty=quantities[firm_id],
                    profit=profits[firm_id],
                )
                db.add(result_record)

            # Update actions for next round using Nash equilibrium strategy
            actions = adaptive_nash_strategy(
                model, actions, profits, costs, params, round_idx, rounds
            )

            # Validate market clearing conditions
            actions = validate_market_clearing(model, actions, costs, params)

        # Commit all changes
        db.commit()
        return str(run.id)

    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Simulation failed: {e}")


def _run_cournot_round(
    params: Dict[str, Any], costs: List[float], quantities: List[float]
) -> Any:
    """Run a single Cournot round."""
    # Check if segmented demand is configured
    segments_config = params.get("segments")
    if segments_config:
        # Create segmented demand
        segments = []
        for segment_config in segments_config:
            segment = DemandSegment(
                alpha=float(segment_config["alpha"]),
                beta=float(segment_config["beta"]),
                weight=float(segment_config["weight"]),
            )
            segments.append(segment)

        segmented_demand = SegmentedDemand(segments=segments)
        return cournot_segmented_simulation(segmented_demand, costs, quantities)
    else:
        # Use traditional single-segment demand
        a = params.get("a", 100.0)
        b = params.get("b", 1.0)
        return cournot_simulation(a, b, costs, quantities)


def _run_bertrand_round(
    params: Dict[str, Any], costs: List[float], prices: List[float]
) -> Any:
    """Run a single Bertrand round."""
    # Check if segmented demand is configured
    segments_config = params.get("segments")
    if segments_config:
        # Create segmented demand
        segments = []
        for segment_config in segments_config:
            segment = DemandSegment(
                alpha=float(segment_config["alpha"]),
                beta=float(segment_config["beta"]),
                weight=float(segment_config["weight"]),
            )
            segments.append(segment)

        segmented_demand = SegmentedDemand(segments=segments)
        return bertrand_segmented_simulation(segmented_demand, costs, prices)
    else:
        # Use traditional single-segment demand
        alpha = params.get("alpha", 100.0)
        beta = params.get("beta", 1.0)
        return bertrand_simulation(alpha, beta, costs, prices)


def get_run_results(run_id: str, db: Session) -> Dict[str, Any]:
    """Retrieve time-series results for a simulation run.

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        Dictionary containing time-series data with arrays of equal length

    Raises:
        ValueError: If run_id is not found
    """
    # Get run metadata
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise ValueError(f"Run {run_id} not found")

    # Get all results for this run, ordered by round and firm
    results = (
        db.query(Result)
        .filter(Result.run_id == run_id)
        .order_by(Result.round_idx, Result.firm_id)
        .all()
    )

    if not results:
        return {
            "run_id": run_id,
            "model": run.model,
            "rounds": run.rounds,
            "created_at": run.created_at.isoformat(),
            "rounds_data": [],
            "firms_data": [],
        }

    # Group results by round
    rounds_data = []
    num_firms = len(set(r.firm_id for r in results))

    for round_idx in range(run.rounds):
        round_results = [r for r in results if r.round_idx == round_idx]
        if round_results:
            round_data = {
                "round": round_idx,
                "price": round_results[0].price,  # Same price for all firms in a round
                "total_qty": sum(r.qty for r in round_results),
                "total_profit": sum(r.profit for r in round_results),
            }
            rounds_data.append(round_data)

    # Group results by firm
    firms_data = []
    for firm_id in range(num_firms):
        firm_results = [r for r in results if r.firm_id == firm_id]
        firm_data = {
            "firm_id": firm_id,
            "actions": [r.action for r in firm_results],
            "quantities": [r.qty for r in firm_results],
            "profits": [r.profit for r in firm_results],
        }
        firms_data.append(firm_data)

    return {
        "run_id": run_id,
        "model": run.model,
        "rounds": run.rounds,
        "created_at": run.created_at.isoformat(),
        "rounds_data": rounds_data,
        "firms_data": firms_data,
    }
