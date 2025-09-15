"""Multi-round simulation runner with persistence.

This module implements the core functionality for running multi-round
oligopoly simulations and persisting results to the database.
"""

from typing import Dict, Any, List, Tuple, Union
import random
import math
from sqlalchemy.orm import Session
from sim.models import Run, Round, Result, Base
from sim.cournot import cournot_simulation
from sim.bertrand import bertrand_simulation


def run_game(
    model: str,
    rounds: int,
    config: Dict[str, Any],
    db: Session
) -> str:
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
    
    if not firms:
        raise ValueError("Config must contain 'firms' list")
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Create run record
    run = Run(
        model=model,
        rounds=rounds
    )
    db.add(run)
    db.flush()  # Get the run_id
    
    try:
        # Extract firm costs
        costs = [firm["cost"] for firm in firms]
        num_firms = len(costs)
        
        # Initialize firm actions (quantities for Cournot, prices for Bertrand)
        if model == "cournot":
            # Start with equal quantities plus some randomness
            total_quantity = params.get("initial_total_qty", 100.0)
            base_qty = total_quantity / num_firms
            actions = [base_qty + random.uniform(-5, 5) for _ in range(num_firms)]
        else:  # bertrand
            # Start with equal prices plus some randomness
            initial_price = params.get("initial_price", 50.0)
            actions = [initial_price + random.uniform(-2, 2) for _ in range(num_firms)]
        
        # Run simulation rounds
        for round_idx in range(rounds):
            # Create round record
            round_record = Round(
                run_id=run.id,
                idx=round_idx
            )
            db.add(round_record)
            
            # Run simulation for this round
            if model == "cournot":
                result = _run_cournot_round(params, costs, actions)
                market_price = result.price
                quantities = result.quantities
                profits = result.profits
            else:  # bertrand
                result = _run_bertrand_round(params, costs, actions)
                market_price = result.prices[0]  # All firms have same price in Bertrand
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
                    profit=profits[firm_id]
                )
                db.add(result_record)
            
            # Update actions for next round (simple adaptive strategy)
            actions = _update_actions(model, actions, profits, costs, params)
        
        # Commit all changes
        db.commit()
        return run.id
        
    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Simulation failed: {e}")


def _run_cournot_round(params: Dict[str, Any], costs: List[float], quantities: List[float]) -> Any:
    """Run a single Cournot round."""
    a = params.get("a", 100.0)
    b = params.get("b", 1.0)
    return cournot_simulation(a, b, costs, quantities)


def _run_bertrand_round(params: Dict[str, Any], costs: List[float], prices: List[float]) -> Any:
    """Run a single Bertrand round."""
    alpha = params.get("alpha", 100.0)
    beta = params.get("beta", 1.0)
    return bertrand_simulation(alpha, beta, costs, prices)


def _update_actions(
    model: str,
    current_actions: List[float],
    profits: List[float],
    costs: List[float],
    params: Dict[str, Any]
) -> List[float]:
    """Update firm actions for the next round using adaptive strategies.
    
    Implements simple profit-based adaptation where firms adjust their
    actions based on recent performance.
    """
    new_actions = []
    
    for i, (action, profit, cost) in enumerate(zip(current_actions, profits, costs)):
        if model == "cournot":
            # Cournot: adjust quantity based on profit
            if profit > 0:
                # Increase quantity if profitable
                adjustment = params.get("quantity_adjustment", 0.1) * action
                new_action = action + adjustment
            else:
                # Decrease quantity if unprofitable
                adjustment = params.get("quantity_adjustment", 0.1) * action
                new_action = max(0.1, action - adjustment)
        else:  # bertrand
            # Bertrand: adjust price based on profit
            if profit > 0:
                # Decrease price to gain market share
                adjustment = params.get("price_adjustment", 0.05) * action
                new_action = max(cost + 0.1, action - adjustment)
            else:
                # Increase price if unprofitable
                adjustment = params.get("price_adjustment", 0.05) * action
                new_action = action + adjustment
        
        # Add some randomness
        noise_factor = params.get("noise_factor", 0.02)
        noise = random.uniform(-noise_factor, noise_factor) * new_action
        new_action = max(0.1, new_action + noise)
        
        new_actions.append(new_action)
    
    return new_actions


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
    results = db.query(Result).filter(Result.run_id == run_id).order_by(
        Result.round_idx, Result.firm_id
    ).all()
    
    if not results:
        return {
            "run_id": run_id,
            "model": run.model,
            "rounds": run.rounds,
            "created_at": run.created_at.isoformat(),
            "rounds_data": [],
            "firms_data": []
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
                "total_profit": sum(r.profit for r in round_results)
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
            "profits": [r.profit for r in firm_results]
        }
        firms_data.append(firm_data)
    
    return {
        "run_id": run_id,
        "model": run.model,
        "rounds": run.rounds,
        "created_at": run.created_at.isoformat(),
        "rounds_data": rounds_data,
        "firms_data": firms_data
    }
