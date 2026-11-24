"""Shared read-only valuation of the holdings in one account's ledger."""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext


@dataclass(frozen=True)
class Valuation:
    subtotal: Decimal | None
    priced: int
    count: int

    @property
    def incomplete(self):
        return self.priced < self.count


@dataclass(frozen=True)
class AssetState:
    positions: tuple[Decimal | None, ...]
    total: Valuation
    stock: Valuation
    cash: Valuation


def ledger_number(raw):
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _position_value(row):
    quantity, price, fx = (ledger_number(row.get(field)) for field in
                           ("quantity", "price", "fx_to_twd"))
    currency = row.get("currency")
    if (quantity is None or quantity < 0 or price is None or price <= 0
            or fx is None or fx <= 0 or not isinstance(currency, str) or not currency.strip()):
        return None
    return quantity * price * fx


def _summarize(values):
    priced = [value for value in values if value is not None]
    return Valuation(sum(priced, Decimal(0)) if priced else None, len(priced), len(values))


def build_asset_state(positions):
    grouped = defaultdict(list)
    values = []
    with localcontext() as ctx:
        ctx.prec = 96
        for row in positions:
            value = _position_value(row)
            values.append(value)
            grouped[row.get("asset_type")].append(value)
        return AssetState(
            tuple(values), _summarize(values), _summarize(grouped["股票"]), _summarize(grouped["現金"]),
        )
