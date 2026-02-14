"""Multi-round simulation runner with persistence.

This module implements the core functionality for running multi-round
oligopoly simulations and persisting results to the database.
"""

import random
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from src.sim.collusion import CollusionManager, RegulatorState
from src.sim.games.enhanced_simulation import (
    BertrandResult,
    CournotResult,
    bertrand_simulation,
    cournot_simulation,
)
from src.sim.models.metrics import (
    calculate_hhi,
    calculate_market_shares_bertrand,
    calculate_market_shares_cournot,
)
from src.sim.models.models import (
    Demand,
    DemandSegment,
    Result,
    Round,
    Run,
    SegmentedDemand,
)
from src.sim.policy.policy_shocks import apply_policy_shock, validate_policy_events
from src.sim.strategies.collusion_strategies import create_collusion_strategy
from src.sim.strategies.nash_strategies import (
    adaptive_nash_strategy,
    cournot_nash_equilibrium,
    validate_economic_parameters,
    validate_market_clearing,
)
from src.sim.validation import validate_simulation_config


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

    # Validate simulation configuration for economic consistency
    try:
        validate_simulation_config(config)
    except Exception as e:
        raise ValueError(f"Invalid simulation configuration: {e}")

    # Validate policy events
    if events:
        validate_policy_events(events, rounds)

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Create run record
    run = Run(model=model, rounds=rounds)
    db.add(run)
    db.flush()

    # Extract firm costs and initialize collusion manager
    costs = [firm["cost"] for firm in firms]
    fixed_costs = [firm.get("fixed_cost", 0.0) for firm in firms]
    num_firms = len(costs)

    # Setup strategies and collusion manager
    collusion_manager = CollusionManager()
    firm_strategies = []
    firm_histories = [[] for _ in range(num_firms)] # List of results for each firm
    
    for i, firm_config in enumerate(firms):
        strategy_type = firm_config.get("strategy_type", "nash")
        if strategy_type == "nash":
            firm_strategies.append(None)  # Use baseline adaptive Nash
        else:
            # Create specific strategy (collusive, etc.)
            # Need to avoid passing strategy_type twice if it's in firm_config
            strategy_kwargs = firm_config.copy()
            if "strategy_type" in strategy_kwargs:
                strategy_kwargs.pop("strategy_type")
            strategy = create_collusion_strategy(strategy_type, seed=seed, **strategy_kwargs)
            firm_strategies.append(strategy)

    try:
        # Validate economic parameters
        validate_economic_parameters(model, params, costs)

        # Initialize firm actions
        if model == "cournot":
            # Standard or segmented demand
            segments_config = params.get("segments")
            if segments_config:
                weighted_alpha = sum(s["alpha"] * s["weight"] for s in segments_config)
                weighted_beta = sum(s["beta"] * s["weight"] for s in segments_config)
                nash_quantities, _, _ = cournot_nash_equilibrium(
                    weighted_alpha, weighted_beta, costs, fixed_costs
                )
            else:
                a = params.get("a", 100.0)
                b = params.get("b", 1.0)
                nash_quantities, _, _ = cournot_nash_equilibrium(a, b, costs, fixed_costs)
            actions = [max(0.1, qty + random.uniform(-1, 1)) for qty in nash_quantities]
        else:  # bertrand
            alpha = params.get("alpha", 100.0)
            beta = params.get("beta", 1.0)
            from src.sim.strategies.nash_strategies import bertrand_nash_equilibrium
            nash_prices, _, _, _ = bertrand_nash_equilibrium(alpha, beta, costs)
            actions = [max(costs[i] + 0.1, p + random.uniform(-1, 1)) for i, p in enumerate(nash_prices)]

        # Run simulation rounds
        for round_idx in range(rounds):
            round_record = Round(run_id=run.id, idx=round_idx)
            db.add(round_record)

            # 1. Get Actions for this round
            # (First round uses initialized actions, subsequent rounds use updated ones)
            
            # 2. Run simulation for this round
            if model == "cournot":
                result = _run_cournot_round(params, costs, actions, fixed_costs)
            else:
                result = _run_bertrand_round(params, costs, actions, fixed_costs)

            # 3. Apply policy shocks
            for event in events:
                if event.round_idx == round_idx:
                    result = apply_policy_shock(result, event, costs)

            # 4. Detect collusion and update manager
            market_shares = []
            if model == "cournot":
                market_shares = calculate_market_shares_cournot(result.quantities)
            else:
                market_shares = calculate_market_shares_bertrand(result.prices, result.quantities)
            
            # Check for defections if a cartel exists
            if collusion_manager.is_cartel_active():
                cartel = collusion_manager.current_cartel
                for firm_id in range(num_firms):
                    collusion_manager.detect_defection(
                        round_idx,
                        firm_id,
                        result.price if model == "cournot" else result.prices[firm_id],
                        result.quantities[firm_id],
                        cartel.collusive_price,
                        cartel.collusive_quantity / len(cartel.participating_firms) # Approximate
                    )
            
            # Check if any firm is playing a collusive strategy and try to form cartel if not active
            colluding_firms = [i for i, s in enumerate(firm_strategies) if s is not None and hasattr(s, 'is_colluding') and s.is_colluding]
            if colluding_firms and not collusion_manager.is_cartel_active():
                # Form a cartel agreement based on current collusive intent
                # In a real scenario, this would be negotiated. Here we use the target of the first colluding firm.
                first_colluding_idx = colluding_firms[0]
                strat = firm_strategies[first_colluding_idx]
                if hasattr(strat, 'target_price') and hasattr(strat, 'target_quantity'):
                    collusion_manager.form_cartel(
                        round_idx,
                        strat.target_price,
                        strat.target_quantity,
                        colluding_firms
                    )

            # 5. Persist results and update histories
            for firm_id in range(num_firms):
                firm_price = result.price if model == "cournot" else result.prices[firm_id]
                
                # Update firm history
                if model == "cournot":
                    res = CournotResult(price=result.price, quantities=result.quantities, profits=result.profits)
                else:
                    res = BertrandResult(total_demand=result.total_demand, prices=result.prices, quantities=result.quantities, profits=result.profits)
                firm_histories[firm_id].append(res)

                result_record = Result(
                    run_id=run.id,
                    round_id=round_record.id,
                    round_idx=round_idx,
                    firm_id=firm_id,
                    action=actions[firm_id],
                    price=firm_price,
                    qty=result.quantities[firm_id],
                    profit=result.profits[firm_id],
                )
                db.add(result_record)

            # 6. Update actions for next round
            new_actions = []
            for i, strategy in enumerate(firm_strategies):
                if strategy is None:
                    # Adaptive Nash logic (handled below for simplicity or per-firm)
                    # For now, we'll store a placeholder and update later
                    new_actions.append(None)
                else:
                    # Build rival histories for this firm
                    rival_histories = [firm_histories[j] for j in range(num_firms) if j != i]
                    
                    # Determine bounds (could be expanded)
                    if model == "cournot":
                        a = params.get("a", 100.0)
                        b = params.get("b", 1.0)
                        bounds = (0.1, a / b)
                    else:
                        costs_i = costs[i]
                        bounds = (costs_i + 0.1, 1000.0)
                    
                    market_params = {**params, "model_type": model}
                    
                    try:
                        action = strategy.next_action(
                            round_idx, 
                            firm_histories[i], 
                            rival_histories, 
                            bounds, 
                            market_params
                        )
                        new_actions.append(action)
                    except Exception as e:
                        print(f"Strategy error for firm {i}: {e}. Falling back to Nash.")
                        new_actions.append(None)
            
            # Handle Nash competitors and merge
            # First, get a baseline update for everyone
            nash_actions = adaptive_nash_strategy(
                model, actions, result.profits, costs, params, round_idx, rounds
            )
            
            # Use specific strategy action if available, else use Nash
            final_actions = []
            for i, action in enumerate(new_actions):
                if action is not None:
                    final_actions.append(action)
                else:
                    final_actions.append(nash_actions[i])
            
            actions = validate_market_clearing(model, final_actions, costs, params)

        db.commit()
        return str(run.id)

    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Simulation failed: {e}")


def _run_cournot_round(
    params: Dict[str, Any],
    costs: List[float],
    quantities: List[float],
    fixed_costs: List[float],
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
        return cournot_segmented_simulation(
            segmented_demand, costs, quantities, fixed_costs
        )
    else:
        # Use traditional single-segment demand
        a = params.get("a", 100.0)
        b = params.get("b", 1.0)
        return cournot_simulation(a, b, costs, quantities, fixed_costs)


def _run_bertrand_round(
    params: Dict[str, Any],
    costs: List[float],
    prices: List[float],
    fixed_costs: List[float],
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
        return bertrand_segmented_simulation(
            segmented_demand, costs, prices, fixed_costs
        )
    else:
        # Use traditional single-segment demand
        alpha = params.get("alpha", 100.0)
        beta = params.get("beta", 1.0)
        return bertrand_simulation(
            alpha, beta, costs, prices, fixed_costs, use_capacity_constraints=True
        )


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
