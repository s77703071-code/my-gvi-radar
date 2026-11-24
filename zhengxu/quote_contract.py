"""Public quote metadata shared by research modules."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum


class QuoteKind(str, Enum):
    CHART_BAR = "chart_bar"
    OFFICIAL_CLOSE = "official_close"


class QuoteStatus(str, Enum):
    AVAILABLE = "available"
    LAST_VALID = "last_valid"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    source: str
    kind: QuoteKind
    status: QuoteStatus
    price: Decimal | None = None
    as_of: date | datetime | None = None
    currency: str | None = None
    retrieved_at: datetime | None = None

    def __post_init__(self):
        if self.status is QuoteStatus.UNAVAILABLE:
            if self.price is not None or self.as_of is not None or self.retrieved_at is not None:
                raise ValueError("An unavailable quote cannot have observation values")
        elif (self.price is None or not self.price.is_finite() or self.price <= 0
              or self.as_of is None):
            raise ValueError("An available quote needs a positive price and observation time")
