"""Collusion and defection dynamics for oligopoly simulation.

Cartels agree a collusive price and per-firm quantity; members may defect by
undercutting (Bertrand) or overproducing (Cournot). The manager records cartel
formation, defections and dissolution as events.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CollusionEventType(Enum):
    """Types of events that can occur in collusion dynamics."""

    CARTEL_FORMED = "cartel_formed"
    FIRM_DEFECTED = "firm_defected"
    CARTEL_DISSOLVED = "cartel_dissolved"


@dataclass
class CollusionEvent:
    """Represents an event in the collusion dynamics.

    Events track important moments like cartel formation, defections,
    and dissolution with the relevant data.
    """

    event_type: CollusionEventType
    round_idx: int
    firm_id: int | None = None  # None for market-wide events
    description: str = ""
    data: dict[str, Any] = field(default_factory=dict)

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
    participating_firms: list[int]
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


class CollusionManager:
    """Manages collusion dynamics and event tracking.

    Handles cartel formation, defection detection,
    and event logging for the oligopoly simulation.
    """

    def __init__(self) -> None:
        """Initialize collusion manager."""
        self.current_cartel: CartelAgreement | None = None
        self.events: list[CollusionEvent] = []
        self.firm_defection_history: dict[
            int, list[int]
        ] = {}  # firm_id -> list of rounds when defected

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
        participating_firms: list[int],
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
                event_type=CollusionEventType.CARTEL_DISSOLVED,
                round_idx=round_idx,
                description="Cartel dissolved",
                data={"participating_firms": self.current_cartel.participating_firms},
            )
            self.events.append(event)

            self.current_cartel = None
