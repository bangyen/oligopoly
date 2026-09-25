"""Multi-round simulation runner with persistence.

This module implements the core functionality for running multi-round
oligopoly simulations and persisting results to the database.
"""

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from sim.collusion import CollusionManager
from sim.games.bertrand import (
    BertrandResult,
)
from sim.games.cournot import (
    CournotResult,
)
from sim.markets import Market, build_market
from sim.models.market_evolution import MarketEvolutionConfig, MarketEvolutionEngine
from sim.models.metrics import (
    calculate_market_shares_bertrand,
    calculate_market_shares_cournot,
)
from sim.models.models import (
    Event,
    Result,
    Round,
    Run,
)
from sim.policy.policy_shocks import apply_policy_shock, validate_policy_events
from sim.strategies.collusion_strategies import (
    CartelStrategy,
    CollusiveStrategy,
    OpportunisticStrategy,
    create_collusion_strategy,
)
from sim.strategies.learning import create_learning_strategy
from sim.strategies.nash_strategies import validate_economic_parameters
from sim.validation import validate_simulation_config

logger = logging.getLogger(__name__)

_INNOVATION_EVENT_THRESHOLD = 0.03


COLLUSION_STRATEGY_TYPES = ("cartel", "collusive", "opportunistic")
_COLLUSION_CLASSES = (CartelStrategy, CollusiveStrategy, OpportunisticStrategy)
# Rounds a cartel waits after breaking down before members try again
CARTEL_REFORM_DELAY = 5


def _collusion_strategy(
    strategy_type: str,
    seed: int | None,
    manager: CollusionManager,
    **kwargs: Any,
) -> CartelStrategy | CollusiveStrategy | OpportunisticStrategy:
    kwargs.pop("seed", None)
    strategy = create_collusion_strategy(strategy_type, seed=seed, **kwargs)
    strategy.collusion_manager = manager
    return strategy


def _members(firm_ids: list[int], strategies: list[Any]) -> list[int]:
    """Firm ids currently playing a collusion strategy."""
    return [
        fid
        for fid, strat in zip(firm_ids, strategies)
        if isinstance(strat, _COLLUSION_CLASSES)
    ]


def _update_cartel(
    manager: CollusionManager,
    market: Market,
    result: CournotResult | BertrandResult,
    round_idx: int,
    firm_ids: list[int],
    strategies: list[Any],
    actions: list[float],
    costs: list[float],
    reform_after: int,
) -> int:
    """Detect defections and (re)form the cartel; return the next reform round.

    A member defects if it undercuts the cartel price (Bertrand) or
    overproduces the cartel quantity (Cournot) by more than 5%; the cartel then
    breaks down and members may try again after CARTEL_REFORM_DELAY rounds.
    Cartel targets maximise the members' joint profit in the current market,
    taking non-members' current actions as given.
    """
    position = {fid: pos for pos, fid in enumerate(firm_ids)}
    cartel = manager.current_cartel
    if cartel is not None:
        defected = False
        for fid in cartel.participating_firms:
            pos = position.get(fid)
            if pos is None:
                continue
            if market.model == "cournot":
                # Cournot firms share one market price; only quantity is theirs
                defected |= manager.detect_defection(
                    round_idx,
                    fid,
                    cartel.collusive_price,
                    result.quantities[pos],
                    cartel.collusive_price,
                    cartel.collusive_quantity,
                )
            else:
                defected |= manager.detect_defection(
                    round_idx,
                    fid,
                    result.prices[pos],
                    0.0,
                    cartel.collusive_price,
                    cartel.collusive_quantity,
                )
        if defected:
            manager.dissolve_cartel(round_idx)
            return round_idx + CARTEL_REFORM_DELAY
        return reform_after

    members = _members(firm_ids, strategies)
    if len(members) < 2 or round_idx < reform_after:
        return reform_after

    member_pos = [position[fid] for fid in members]
    target = market.collusive_action(member_pos, actions, costs)
    trial = list(actions)
    for pos in member_pos:
        trial[pos] = target
    outcome = market.play(trial, costs, [0.0] * len(costs))
    if market.model == "cournot":
        price, quantity = outcome.price, target
    else:
        price = target
        quantity = sum(outcome.quantities[p] for p in member_pos) / len(member_pos)
    manager.form_cartel(round_idx, price, quantity, members)
    return reform_after


def _cartel_profit_fn(
    market: Market,
    firm: int,
    actions: list[float],
    costs: list[float],
    manager: CollusionManager,
    firm_ids: list[int],
) -> Any:
    """Map this firm's action to its profit when the other members comply."""
    cartel = manager.current_cartel
    trial = list(actions)
    if cartel is not None:
        target = (
            cartel.collusive_quantity
            if market.model == "cournot"
            else cartel.collusive_price
        )
        for pos, fid in enumerate(firm_ids):
            if fid in cartel.participating_firms:
                trial[pos] = target

    def evaluate(action: float) -> float:
        return float(market.profit(firm, action, trial, costs))

    return evaluate


def _project(
    result: CournotResult | BertrandResult, index: int
) -> CournotResult | BertrandResult:
    """One firm's view of a round: its own price, quantity and profit at index 0."""
    if isinstance(result, CournotResult):
        return CournotResult(
            price=result.price,
            quantities=[result.quantities[index]],
            profits=[result.profits[index]],
        )
    return BertrandResult(
        total_demand=result.total_demand,
        prices=[result.prices[index]],
        quantities=[result.quantities[index]],
        profits=[result.profits[index]],
    )


def _market_shares(model: str, result: CournotResult | BertrandResult) -> list[float]:
    n = len(result.quantities)
    if sum(result.quantities) <= 0:
        return [1.0 / n] * n
    if model == "cournot":
        return list(calculate_market_shares_cournot(result.quantities))
    return list(calculate_market_shares_bertrand(result.prices, result.quantities))


def run_game(model: str, rounds: int, config: dict[str, Any], db: Session) -> str:
    """Run a multi-round oligopoly simulation with persistence.

    Executes the specified number of rounds of either Cournot or Bertrand
    competition, persisting each round's results to the database.

    Args:
        model: Type of competition model ("cournot" or "bertrand")
        rounds: Number of rounds to simulate
        config: Configuration dictionary containing:
            - params: Market parameters. ``demand_type`` selects the demand
              system ("linear" default, "isoelastic", "ces"); see
              :func:`sim.markets.build_market`.
            - firms: List of firm configurations with costs
            - seed: Optional random seed for reproducibility
            - events: Optional policy events
            - advanced_strategies: Optional list of
              ``{"firm_id", "strategy_type", ...}`` learning strategies
            - market_evolution: Optional MarketEvolutionConfig fields enabling
              growth, entry/exit and innovation between rounds
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
    learning_specs = config.get("advanced_strategies") or []
    evolution_config = config.get("market_evolution")

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

    # Normalise params to a plain serializable dict for DB persistence.
    # The caller may pass a Pydantic model (CournotParams / BertrandParams),
    # a plain dict, or None.
    params_dict: dict | None = None
    if params is not None:
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump()
        elif isinstance(params, dict):
            params_dict = params
        # else: unsupported type — leave as None
    params = params_dict or {}

    costs = [firm["cost"] for firm in firms]
    fixed_costs = [firm.get("fixed_cost", 0.0) for firm in firms]
    num_firms = len(costs)

    market = build_market(model, params, num_firms)
    is_linear = market.demand_type == "linear"

    # Persist params so metrics can be recomputed faithfully later. Linear runs
    # keep the caller's params exactly; other demand systems store the
    # normalised demand spec (including demand_type).
    persisted: dict[str, Any] | None = params_dict if is_linear else market.params()
    if learning_specs or evolution_config is not None:
        persisted = dict(persisted or {})
        if learning_specs:
            persisted["advanced_strategies"] = list(learning_specs)
        if evolution_config is not None:
            persisted["market_evolution"] = dict(evolution_config)

    # Create run record
    run = Run(model=model, rounds=rounds, params=persisted)
    db.add(run)
    db.flush()

    # Setup strategies and collusion manager
    collusion_manager = CollusionManager()
    firm_strategies: list[Any] = []
    firm_histories: list[list[CournotResult | BertrandResult]] = [
        [] for _ in range(num_firms)
    ]  # Per-firm results (index 0 = that firm)

    for firm_config in firms:
        strategy_type = firm_config.get("strategy_type", "nash")
        if strategy_type == "nash":
            firm_strategies.append(None)  # Use baseline adaptive Nash
        else:
            strategy_kwargs = {
                k: v for k, v in firm_config.items() if k != "strategy_type"
            }
            firm_strategies.append(
                _collusion_strategy(
                    strategy_type, seed, collusion_manager, **strategy_kwargs
                )
            )

    nash_actions_initial = market.nash(costs)
    for spec in learning_specs:
        firm_idx = spec["firm_id"]
        if not 0 <= firm_idx < num_firms:
            raise ValueError(
                f"advanced_strategies firm_id {firm_idx} is out of range for "
                f"{num_firms} firms"
            )
        if firm_strategies[firm_idx] is not None:
            raise ValueError(f"Firm {firm_idx} already has a strategy")
        if spec["strategy_type"] in COLLUSION_STRATEGY_TYPES:
            firm_strategies[firm_idx] = _collusion_strategy(
                spec["strategy_type"],
                None if seed is None else seed + firm_idx,
                collusion_manager,
            )
            continue
        firm_strategies[firm_idx] = create_learning_strategy(
            spec["strategy_type"],
            bounds=market.action_bounds(costs[firm_idx], costs),
            reference_action=nash_actions_initial[firm_idx],
            seed=None if seed is None else seed + firm_idx,
            learning_rate=spec.get("learning_rate"),
            memory_length=spec.get("memory_length"),
            exploration_rate=spec.get("exploration_rate"),
        )

    firm_ids = list(range(num_firms))
    qualities = list(params.get("qualities") or [1.0] * num_firms)
    evolution = None
    if evolution_config is not None:
        evolution = MarketEvolutionEngine(
            MarketEvolutionConfig(**evolution_config), seed=seed
        )
        evolution.state.num_firms = num_firms

    try:
        # Validate economic parameters
        if is_linear:
            validate_economic_parameters(model, params, costs)

        # Initialize firm actions
        actions = market.initial_actions(costs)
        reform_after = 0  # earliest round a (new) cartel may form
        logged_events = 0

        # Run simulation rounds
        for round_idx in range(rounds):
            round_record = Round(run_id=run.id, idx=round_idx)
            db.add(round_record)

            # 1. Run simulation for this round
            result = market.play(actions, costs, fixed_costs)

            # 2. Apply policy shocks
            for event in events:
                if event.round_idx == round_idx:
                    result = apply_policy_shock(result, event, costs)

            # 3. Cartel dynamics: detect defections, (re)form the cartel
            reform_after = _update_cartel(
                collusion_manager,
                market,
                result,
                round_idx,
                firm_ids,
                firm_strategies,
                actions,
                costs,
                reform_after,
            )
            for collusion_event in collusion_manager.events[logged_events:]:
                db.add(
                    Event(
                        run_id=run.id,
                        round_idx=collusion_event.round_idx,
                        event_type=collusion_event.event_type.value,
                        firm_id=collusion_event.firm_id,
                        description=collusion_event.description,
                        event_data=collusion_event.data,
                    )
                )
            logged_events = len(collusion_manager.events)

            # 4. Persist results and update per-firm histories
            for pos, firm_id in enumerate(firm_ids):
                firm_price = result.price if model == "cournot" else result.prices[pos]
                firm_histories[pos].append(_project(result, pos))
                db.add(
                    Result(
                        run_id=run.id,
                        round_id=round_record.id,
                        round_idx=round_idx,
                        firm_id=firm_id,
                        action=actions[pos],
                        price=firm_price,
                        qty=result.quantities[pos],
                        profit=result.profits[pos],
                    )
                )

            # 5. Update actions for next round
            base_params = market.strategy_params()
            new_actions: list[float | None] = []
            for i, strat_opt in enumerate(firm_strategies):
                if strat_opt is None:
                    new_actions.append(None)
                    continue
                rival_histories: list[Sequence[CournotResult | BertrandResult]] = [
                    firm_histories[j] for j in range(len(firm_ids)) if j != i
                ]
                market_params = {
                    **base_params,
                    "my_cost": costs[i],
                    "best_response": _best_response_fn(market, i, actions, costs),
                    "evaluate": _cartel_profit_fn(
                        market, i, actions, costs, collusion_manager, firm_ids
                    ),
                }
                extra = (
                    {"my_cost": costs[i]}
                    if isinstance(strat_opt, _COLLUSION_CLASSES)
                    else {}
                )
                try:
                    action = strat_opt.next_action(
                        round_idx,
                        firm_histories[i],
                        rival_histories,
                        market.action_bounds(costs[i], costs),
                        market_params,
                        **extra,
                    )
                    new_actions.append(action)
                except Exception as e:
                    logger.warning(
                        "Strategy error for firm %s: %s. Falling back to Nash.",
                        firm_ids[i],
                        e,
                    )
                    new_actions.append(None)

            # Firms without their own strategy adapt towards Nash
            nash_actions = market.adapt(
                actions, result.profits, costs, round_idx, rounds
            )
            final_actions = [
                float(a) if a is not None else float(nash_actions[i])
                for i, a in enumerate(new_actions)
            ]
            actions = market.clear(final_actions, costs)

            # 6. Market evolution between rounds: growth, innovation, entry/exit
            if evolution is not None and round_idx < rounds - 1:
                state = _evolve(
                    evolution,
                    market,
                    model,
                    result,
                    round_idx,
                    firm_ids,
                    costs,
                    fixed_costs,
                    qualities,
                    actions,
                    firm_strategies,
                    firm_histories,
                )
                for event_type, event_firm, description, data in state.events:
                    db.add(
                        Event(
                            run_id=run.id,
                            round_idx=round_idx,
                            event_type=event_type,
                            firm_id=event_firm,
                            description=description,
                            event_data=data,
                        )
                    )
                firm_ids, costs, fixed_costs, qualities = (
                    state.firm_ids,
                    state.costs,
                    state.fixed_costs,
                    state.qualities,
                )
                cartel = collusion_manager.current_cartel
                if cartel is not None:
                    cartel.participating_firms = [
                        f for f in cartel.participating_firms if f in firm_ids
                    ]
                    if len(cartel.participating_firms) < 2:
                        collusion_manager.dissolve_cartel(round_idx)
                actions, firm_strategies, firm_histories = (
                    state.actions,
                    state.strategies,
                    state.histories,
                )
                if not firm_ids:
                    break

        db.commit()
        return str(run.id)

    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Simulation failed: {e}")


def _best_response_fn(
    market: Market, firm: int, actions: list[float], costs: list[float]
) -> Any:
    """Map rivals' actions (rival order) to ``firm``'s best response."""

    def best_response(rival_actions: list[float]) -> float:
        full = list(actions)
        rivals = [j for j in range(len(actions)) if j != firm]
        for j, value in zip(rivals, rival_actions):
            full[j] = value
        return float(market.best_response(firm, full, costs))

    return best_response


@dataclass
class _EvolvedState:
    firm_ids: list[int]
    costs: list[float]
    fixed_costs: list[float]
    qualities: list[float]
    actions: list[float]
    strategies: list[Any]
    histories: list[list[CournotResult | BertrandResult]]
    events: list[tuple[str, int | None, str, dict[str, Any]]]


def _evolve(
    engine: MarketEvolutionEngine,
    market: Market,
    model: str,
    result: CournotResult | BertrandResult,
    round_idx: int,
    firm_ids: list[int],
    costs: list[float],
    fixed_costs: list[float],
    qualities: list[float],
    actions: list[float],
    strategies: list[Any],
    histories: list[list[CournotResult | BertrandResult]],
) -> _EvolvedState:
    """Apply one round of market evolution and realign per-firm state by id."""
    growth = 1.0 + engine.config.growth_rate
    # The engine mutates the cost/quality lists it is given; pass copies.
    new_ids, new_costs, new_qualities, _ = engine.evolve_market(
        list(firm_ids),
        list(result.profits),
        _market_shares(model, result),
        list(costs),
        list(qualities),
        {},
    )
    market.scale_demand(growth)

    by_id = {fid: pos for pos, fid in enumerate(firm_ids)}
    mean_fixed = sum(fixed_costs) / len(fixed_costs) if fixed_costs else 0.0
    events: list[tuple[str, int | None, str, dict[str, Any]]] = []

    for fid in firm_ids:
        if fid not in new_ids:
            events.append(
                (
                    "firm_exit",
                    fid,
                    f"Firm {fid} exited the market",
                    {"icon": "🚪", "profit": result.profits[by_id[fid]]},
                )
            )

    new_fixed, new_actions, new_strategies, new_histories = [], [], [], []
    if new_ids:
        market.set_qualities(new_qualities)
        entrant_nash = market.nash(new_costs)
    for pos, fid in enumerate(new_ids):
        if fid in by_id:
            old = by_id[fid]
            new_fixed.append(fixed_costs[old])
            new_actions.append(actions[old])
            new_strategies.append(strategies[old])
            new_histories.append(histories[old])
            # Log firm-level innovations (5% cuts), not the ~1% industry-wide
            # technology spillover every firm receives
            if new_costs[pos] < costs[old] * (1 - _INNOVATION_EVENT_THRESHOLD):
                events.append(
                    (
                        "innovation",
                        fid,
                        f"Firm {fid} cut marginal cost to {new_costs[pos]:.2f}",
                        {
                            "icon": "💡",
                            "old_cost": costs[old],
                            "new_cost": new_costs[pos],
                        },
                    )
                )
        else:
            new_fixed.append(mean_fixed)
            new_actions.append(entrant_nash[pos])
            new_strategies.append(None)
            new_histories.append([])
            events.append(
                (
                    "firm_entry",
                    fid,
                    f"Firm {fid} entered with marginal cost {new_costs[pos]:.2f}",
                    {"icon": "🏭", "cost": new_costs[pos]},
                )
            )

    if new_ids:
        new_actions = market.clear(new_actions, new_costs)
    else:
        events.append(
            ("market_collapse", None, "All firms exited the market", {"icon": "⚠️"})
        )

    return _EvolvedState(
        firm_ids=list(new_ids),
        costs=list(new_costs),
        fixed_costs=new_fixed,
        qualities=list(new_qualities),
        actions=new_actions,
        strategies=new_strategies,
        histories=new_histories,
        events=events,
    )


def get_run_results(run_id: str, db: Session) -> dict[str, Any]:
    """Retrieve time-series results for a simulation run.

    Returns results in the canonical nested-dict format:
    ``results[round_idx][firm_id] = {action, price, quantity, profit}``

    Args:
        run_id: Unique identifier for the simulation run
        db: Database session

    Returns:
        Dictionary with keys: run_id, model, rounds, created_at, params, results

    Raises:
        ValueError: If run_id is not found
    """
    # Get run metadata
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise ValueError(f"Run {run_id} not found")

    # Get all results ordered by round then firm
    db_results = (
        db.query(Result)
        .filter(Result.run_id == run_id)
        .order_by(Result.round_idx, Result.firm_id)
        .all()
    )

    # Build canonical nested dict: results[round_idx][firm_id] = {...}
    nested: dict[str, dict[str, dict[str, float]]] = {}
    for r in db_results:
        ridx = str(r.round_idx)
        fid = f"firm_{r.firm_id}"
        nested.setdefault(ridx, {})[fid] = {
            "action": float(r.action),
            "price": float(r.price),
            "quantity": float(r.qty),
            "profit": float(r.profit),
        }

    return {
        "run_id": run_id,
        "model": str(run.model),
        "rounds": int(run.rounds),
        "created_at": run.created_at.isoformat(),
        "params": run.params or {},
        "results": nested,
    }
