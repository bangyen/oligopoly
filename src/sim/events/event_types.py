"""Event type definitions for oligopoly simulation.

This module defines all possible event types that can occur during
simulation runs, enabling comprehensive event tracking and analysis.
"""

from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Comprehensive event types for simulation tracking.

    Covers all major events that can occur during oligopoly simulations
    including collusion dynamics, policy interventions, and market shocks.
    """

    # Collusion Events
    CARTEL_FORMED = "cartel_formed"
    CARTEL_DISSOLVED = "cartel_dissolved"
    DEFECTION_DETECTED = "defection_detected"
    PUNISHMENT_TRIGGERED = "punishment_triggered"

    # Regulatory Events
    REGULATOR_INTERVENTION = "regulator_intervention"
    PENALTY_IMPOSED = "penalty_imposed"
    PRICE_CAP_IMPOSED = "price_cap_imposed"
    INVESTIGATION_STARTED = "investigation_started"

    # Policy Events
    TAX_APPLIED = "tax_applied"
    SUBSIDY_APPLIED = "subsidy_applied"
    PRICE_CAP_APPLIED = "price_cap_applied"

    # Market Events
    FIRM_ENTRY = "firm_entry"
    FIRM_EXIT = "firm_exit"
    DEMAND_SHOCK = "demand_shock"
    COST_SHOCK = "cost_shock"
    TECHNOLOGY_SHOCK = "technology_shock"

    # Strategy Events
    STRATEGY_CHANGE = "strategy_change"
    LEARNING_UPDATE = "learning_update"
    Q_VALUE_UPDATE = "q_value_update"

    # Market Dynamics
    EQUILIBRIUM_REACHED = "equilibrium_reached"
    CONVERGENCE_ACHIEVED = "convergence_achieved"
    INSTABILITY_DETECTED = "instability_detected"


def get_event_category(event_type: EventType) -> str:
    """Get the category for an event type.

    Args:
        event_type: The event type to categorize

    Returns:
        Category string for grouping events
    """
    category_map = {
        EventType.CARTEL_FORMED: "collusion",
        EventType.CARTEL_DISSOLVED: "collusion",
        EventType.DEFECTION_DETECTED: "collusion",
        EventType.PUNISHMENT_TRIGGERED: "collusion",
        EventType.REGULATOR_INTERVENTION: "regulatory",
        EventType.PENALTY_IMPOSED: "regulatory",
        EventType.PRICE_CAP_IMPOSED: "regulatory",
        EventType.INVESTIGATION_STARTED: "regulatory",
        EventType.TAX_APPLIED: "policy",
        EventType.SUBSIDY_APPLIED: "policy",
        EventType.PRICE_CAP_APPLIED: "policy",
        EventType.FIRM_ENTRY: "market",
        EventType.FIRM_EXIT: "market",
        EventType.DEMAND_SHOCK: "market",
        EventType.COST_SHOCK: "market",
        EventType.TECHNOLOGY_SHOCK: "market",
        EventType.STRATEGY_CHANGE: "strategy",
        EventType.LEARNING_UPDATE: "strategy",
        EventType.Q_VALUE_UPDATE: "strategy",
        EventType.EQUILIBRIUM_REACHED: "dynamics",
        EventType.CONVERGENCE_ACHIEVED: "dynamics",
        EventType.INSTABILITY_DETECTED: "dynamics",
    }

    return category_map.get(event_type, "other")


def get_event_description(event_type: EventType, **kwargs: Any) -> str:
    """Generate a human-readable description for an event.

    Args:
        event_type: The type of event
        **kwargs: Additional context for description generation

    Returns:
        Human-readable event description
    """
    descriptions = {
        EventType.CARTEL_FORMED: f"Cartel formed with {kwargs.get('participating_firms', 0)} firms",
        EventType.CARTEL_DISSOLVED: "Cartel dissolved due to defection or external pressure",
        EventType.DEFECTION_DETECTED: f"Firm {kwargs.get('firm_id', 'unknown')} defected from cartel",
        EventType.PUNISHMENT_TRIGGERED: f"Punishment phase triggered for {kwargs.get('duration', 0)} rounds",
        EventType.REGULATOR_INTERVENTION: "Regulator intervened in market",
        EventType.PENALTY_IMPOSED: f"Penalty of ${kwargs.get('amount', 0):.2f} imposed",
        EventType.PRICE_CAP_IMPOSED: f"Price cap of ${kwargs.get('price_cap', 0):.2f} imposed",
        EventType.INVESTIGATION_STARTED: "Regulatory investigation started",
        EventType.TAX_APPLIED: f"Tax of {kwargs.get('rate', 0):.1%} applied to profits",
        EventType.SUBSIDY_APPLIED: f"Subsidy of ${kwargs.get('amount', 0):.2f} per unit applied",
        EventType.PRICE_CAP_APPLIED: f"Price cap of ${kwargs.get('price_cap', 0):.2f} applied",
        EventType.FIRM_ENTRY: f"New firm entered market with cost ${kwargs.get('cost', 0):.2f}",
        EventType.FIRM_EXIT: f"Firm {kwargs.get('firm_id', 'unknown')} exited market",
        EventType.DEMAND_SHOCK: f"Demand shock: {kwargs.get('magnitude', 0):.1%} change",
        EventType.COST_SHOCK: f"Cost shock: {kwargs.get('magnitude', 0):.1%} change",
        EventType.TECHNOLOGY_SHOCK: f"Technology shock: {kwargs.get('description', 'unknown')}",
        EventType.STRATEGY_CHANGE: f"Firm {kwargs.get('firm_id', 'unknown')} changed strategy",
        EventType.LEARNING_UPDATE: f"Learning algorithm updated for firm {kwargs.get('firm_id', 'unknown')}",
        EventType.Q_VALUE_UPDATE: f"Q-values updated for firm {kwargs.get('firm_id', 'unknown')}",
        EventType.EQUILIBRIUM_REACHED: "Market reached equilibrium",
        EventType.CONVERGENCE_ACHIEVED: "Strategy convergence achieved",
        EventType.INSTABILITY_DETECTED: "Market instability detected",
    }

    return descriptions.get(event_type, f"Event of type {event_type.value}")


def get_event_icon(event_type: EventType) -> str:
    """Get an icon/emoji for an event type.

    Args:
        event_type: The event type

    Returns:
        Icon/emoji string for the event
    """
    icon_map = {
        EventType.CARTEL_FORMED: "🤝",
        EventType.CARTEL_DISSOLVED: "💔",
        EventType.DEFECTION_DETECTED: "⚔️",
        EventType.PUNISHMENT_TRIGGERED: "🔨",
        EventType.REGULATOR_INTERVENTION: "🏛️",
        EventType.PENALTY_IMPOSED: "💰",
        EventType.PRICE_CAP_IMPOSED: "📊",
        EventType.INVESTIGATION_STARTED: "🔍",
        EventType.TAX_APPLIED: "📈",
        EventType.SUBSIDY_APPLIED: "📉",
        EventType.PRICE_CAP_APPLIED: "📊",
        EventType.FIRM_ENTRY: "🚀",
        EventType.FIRM_EXIT: "💸",
        EventType.DEMAND_SHOCK: "📊",
        EventType.COST_SHOCK: "⚡",
        EventType.TECHNOLOGY_SHOCK: "🔬",
        EventType.STRATEGY_CHANGE: "🔄",
        EventType.LEARNING_UPDATE: "🧠",
        EventType.Q_VALUE_UPDATE: "📚",
        EventType.EQUILIBRIUM_REACHED: "⚖️",
        EventType.CONVERGENCE_ACHIEVED: "🎯",
        EventType.INSTABILITY_DETECTED: "⚠️",
    }

    return icon_map.get(event_type, "📝")
