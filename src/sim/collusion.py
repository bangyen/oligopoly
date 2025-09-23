"""Collusion and defection dynamics for oligopoly simulation.

This module implements cartel behavior where firms can collude to set high prices/output,
and individual firms can defect by undercutting to gain higher profits. It also includes
event logging for tracking collusion, defection, and regulatory interventions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CollusionEventType(Enum):
    """Types of events that can occur in collusion dynamics."""

    CARTEL_FORMED = "cartel_formed"
    FIRM_DEFECTED = "firm_defected"
    REGULATOR_INTERVENED = "regulator_intervened"
    PENALTY_IMPOSED = "penalty_imposed"
    PRICE_CAP_IMPOSED = "price_cap_imposed"


@dataclass
class CollusionEvent:
    """Represents an event in the collusion dynamics.

    Events track important moments like cartel formation, defections,
    and regulatory interventions with timestamps and relevant data.
    """

    event_type: CollusionEventType
    round_idx: int
    firm_id: Optional[int] = None  # None for market-wide events
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation for logging and debugging."""
        if self.firm_id is not None:
            return f"Round {self.round_idx}: {self.description}"
        else:
            return f"Round {self.round_idx}: {self.description}"


@dataclass
class CartelAgreement:
    """Represents a cartel agreement between firms.

    Defines the collusive price/output level and tracks which firms
    are participating in the cartel.
    """

    collusive_price: float
    collusive_quantity: float
    participating_firms: List[int]
    formed_round: int

    def __post_init__(self) -> None:
        """Validate cartel agreement parameters."""
        if self.collusive_price < 0:
            raise ValueError(
                f"Collusive price {self.collusive_price} must be non-negative"
            )
        if self.collusive_quantity < 0:
            raise ValueError(
                f"Collusive quantity {self.collusive_quantity} must be non-negative"
            )
        if not self.participating_firms:
            raise ValueError("Cartel must have at least one participating firm")


@dataclass
class RegulatorState:
    """Tracks regulator monitoring and intervention state.

    Monitors market concentration (HHI) and average prices to detect
    collusion and determine when to intervene.
    """

    hhi_threshold: float = 0.8  # HHI threshold for intervention
    price_threshold_multiplier: float = 1.5  # Price threshold as multiple of baseline
    baseline_price: float = 0.0  # Baseline competitive price
    intervention_probability: float = (
        0.8  # Probability of intervention when thresholds exceeded
    )
    penalty_amount: float = 100.0  # Fixed penalty amount
    price_cap_multiplier: float = 0.9  # Price cap as fraction of detected price

    def __post_init__(self) -> None:
        """Validate regulator parameters."""
        if not 0 <= self.hhi_threshold <= 1:
            raise ValueError(f"HHI threshold {self.hhi_threshold} must be in [0, 1]")
        if self.price_threshold_multiplier <= 1:
            raise ValueError(
                f"Price threshold multiplier {self.price_threshold_multiplier} must be > 1"
            )
        if not 0 <= self.intervention_probability <= 1:
            raise ValueError(
                f"Intervention probability {self.intervention_probability} must be in [0, 1]"
            )
        if self.penalty_amount < 0:
            raise ValueError(
                f"Penalty amount {self.penalty_amount} must be non-negative"
            )
        if not 0 < self.price_cap_multiplier <= 1:
            raise ValueError(
                f"Price cap multiplier {self.price_cap_multiplier} must be in (0, 1]"
            )


class CollusionManager:
    """Manages collusion dynamics and event tracking.

    Handles cartel formation, defection detection, regulator monitoring,
    and event logging for the oligopoly simulation.
    """

    def __init__(self, regulator_state: Optional[RegulatorState] = None):
        """Initialize collusion manager.

        Args:
            regulator_state: Regulator configuration, uses defaults if None
        """
        self.regulator_state = regulator_state or RegulatorState()
        self.current_cartel: Optional[CartelAgreement] = None
        self.events: List[CollusionEvent] = []
        self.firm_defection_history: Dict[
            int, List[int]
        ] = {}  # firm_id -> list of rounds when defected

    def calculate_hhi(self, market_shares: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index (HHI) for market concentration.

        Args:
            market_shares: List of market shares for each firm (should sum to 1.0)

        Returns:
            HHI value between 0 and 1
        """
        if not market_shares:
            return 0.0

        # Normalize shares to sum to 1.0
        total_share = sum(market_shares)
        if total_share == 0:
            return 0.0

        normalized_shares = [share / total_share for share in market_shares]
        return sum(share**2 for share in normalized_shares)

    def calculate_average_price(
        self, prices: List[float], quantities: List[float]
    ) -> float:
        """Calculate quantity-weighted average price.

        Args:
            prices: List of prices set by each firm
            quantities: List of quantities sold by each firm

        Returns:
            Weighted average price
        """
        if not prices or not quantities or len(prices) != len(quantities):
            return 0.0

        total_quantity = sum(quantities)
        if total_quantity == 0:
            return 0.0

        weighted_price = sum(p * q for p, q in zip(prices, quantities)) / total_quantity
        return weighted_price

    def detect_defection(
        self,
        round_idx: int,
        firm_id: int,
        firm_price: float,
        firm_quantity: float,
        cartel_price: float,
        cartel_quantity: float,
        tolerance: float = 0.05,
    ) -> bool:
        """Detect if a firm has defected from the cartel.

        Args:
            round_idx: Current round index
            firm_id: ID of the firm to check
            firm_price: Price set by the firm
            firm_quantity: Quantity produced by the firm
            cartel_price: Agreed cartel price
            cartel_quantity: Agreed cartel quantity per firm
            tolerance: Tolerance for deviation from cartel agreement

        Returns:
            True if firm has defected, False otherwise
        """
        if (
            not self.current_cartel
            or firm_id not in self.current_cartel.participating_firms
        ):
            return False

        # Check for price undercutting (significant deviation below cartel price)
        price_defection = firm_price < cartel_price * (1 - tolerance)

        # Check for quantity overproduction (significant deviation above cartel quantity)
        quantity_defection = firm_quantity > cartel_quantity * (1 + tolerance)

        defected = price_defection or quantity_defection

        if defected:
            # Record defection
            if firm_id not in self.firm_defection_history:
                self.firm_defection_history[firm_id] = []
            self.firm_defection_history[firm_id].append(round_idx)

            # Log defection event
            event = CollusionEvent(
                event_type=CollusionEventType.FIRM_DEFECTED,
                round_idx=round_idx,
                firm_id=firm_id,
                description=f"Firm {firm_id} defects",
                data={
                    "firm_price": firm_price,
                    "cartel_price": cartel_price,
                    "firm_quantity": firm_quantity,
                    "cartel_quantity": cartel_quantity,
                    "price_defection": price_defection,
                    "quantity_defection": quantity_defection,
                },
            )
            self.events.append(event)

        return defected

    def form_cartel(
        self,
        round_idx: int,
        collusive_price: float,
        collusive_quantity: float,
        participating_firms: List[int],
    ) -> None:
        """Form a new cartel agreement.

        Args:
            round_idx: Round when cartel is formed
            collusive_price: Agreed cartel price
            collusive_quantity: Agreed quantity per firm
            participating_firms: List of firm IDs participating in cartel
        """
        self.current_cartel = CartelAgreement(
            collusive_price=collusive_price,
            collusive_quantity=collusive_quantity,
            participating_firms=participating_firms,
            formed_round=round_idx,
        )

        # Log cartel formation event
        event = CollusionEvent(
            event_type=CollusionEventType.CARTEL_FORMED,
            round_idx=round_idx,
            description=f"Cartel formed with {len(participating_firms)} firms",
            data={
                "collusive_price": collusive_price,
                "collusive_quantity": collusive_quantity,
                "participating_firms": participating_firms,
            },
        )
        self.events.append(event)

    def check_regulator_intervention(
        self,
        round_idx: int,
        market_shares: List[float],
        prices: List[float],
        quantities: List[float],
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check if regulator should intervene based on HHI and price thresholds.

        Args:
            round_idx: Current round index
            market_shares: Market shares for each firm
            prices: Prices set by each firm
            quantities: Quantities sold by each firm

        Returns:
            Tuple of (should_intervene, intervention_type, intervention_value)
            intervention_type can be "penalty" or "price_cap"
            intervention_value is penalty amount or price cap level
        """
        # Calculate HHI
        hhi = self.calculate_hhi(market_shares)

        # Calculate average price
        avg_price = self.calculate_average_price(prices, quantities)

        # Check thresholds
        hhi_exceeded = hhi > self.regulator_state.hhi_threshold
        price_exceeded = (
            avg_price
            > self.regulator_state.baseline_price
            * self.regulator_state.price_threshold_multiplier
        )

        should_intervene = hhi_exceeded and price_exceeded

        if should_intervene:
            # Determine intervention type (penalty vs price cap)
            # Use penalty if HHI is very high, price cap otherwise
            if hhi > 0.9:
                intervention_type = "penalty"
                intervention_value = self.regulator_state.penalty_amount
            else:
                intervention_type = "price_cap"
                intervention_value = (
                    avg_price * self.regulator_state.price_cap_multiplier
                )

            # Log regulator intervention event
            event = CollusionEvent(
                event_type=CollusionEventType.REGULATOR_INTERVENED,
                round_idx=round_idx,
                description="Regulator intervenes",
                data={
                    "hhi": hhi,
                    "avg_price": avg_price,
                    "intervention_type": intervention_type,
                    "intervention_value": intervention_value,
                    "hhi_threshold": self.regulator_state.hhi_threshold,
                    "price_threshold": self.regulator_state.baseline_price
                    * self.regulator_state.price_threshold_multiplier,
                },
            )
            self.events.append(event)

            return True, intervention_type, intervention_value

        return False, None, None

    def apply_regulator_intervention(
        self,
        round_idx: int,
        intervention_type: str,
        intervention_value: float,
        firm_profits: List[float],
    ) -> List[float]:
        """Apply regulator intervention to firm profits.

        Args:
            round_idx: Current round index
            intervention_type: Type of intervention ("penalty" or "price_cap")
            intervention_value: Value of intervention (penalty amount or price cap)
            firm_profits: Current firm profits

        Returns:
            Modified firm profits after intervention
        """
        modified_profits = firm_profits.copy()

        if intervention_type == "penalty":
            # Apply penalty to all firms (reduces profits)
            for i in range(len(modified_profits)):
                modified_profits[i] = max(0, modified_profits[i] - intervention_value)

            # Log penalty event
            event = CollusionEvent(
                event_type=CollusionEventType.PENALTY_IMPOSED,
                round_idx=round_idx,
                description=f"Penalty of {intervention_value} imposed on all firms",
                data={"penalty_amount": intervention_value},
            )
            self.events.append(event)

        elif intervention_type == "price_cap":
            # Price cap affects future rounds, not current profits
            # Log price cap event
            event = CollusionEvent(
                event_type=CollusionEventType.PRICE_CAP_IMPOSED,
                round_idx=round_idx,
                description=f"Price cap of {intervention_value} imposed",
                data={"price_cap": intervention_value},
            )
            self.events.append(event)

        return modified_profits

    def get_events_for_round(self, round_idx: int) -> List[CollusionEvent]:
        """Get all events that occurred in a specific round.

        Args:
            round_idx: Round index to filter events

        Returns:
            List of events for the specified round
        """
        return [event for event in self.events if event.round_idx == round_idx]

    def get_firm_defection_count(self, firm_id: int) -> int:
        """Get the number of times a firm has defected.

        Args:
            firm_id: Firm ID to check

        Returns:
            Number of defections by the firm
        """
        return len(self.firm_defection_history.get(firm_id, []))

    def is_cartel_active(self) -> bool:
        """Check if there is an active cartel agreement.

        Returns:
            True if cartel is active, False otherwise
        """
        return self.current_cartel is not None

    def dissolve_cartel(self, round_idx: int) -> None:
        """Dissolve the current cartel agreement.

        Args:
            round_idx: Round when cartel is dissolved
        """
        if self.current_cartel:
            # Log cartel dissolution (could add specific event type)
            event = CollusionEvent(
                event_type=CollusionEventType.FIRM_DEFECTED,  # Reuse existing type
                round_idx=round_idx,
                description="Cartel dissolved",
                data={"participating_firms": self.current_cartel.participating_firms},
            )
            self.events.append(event)

            self.current_cartel = None
