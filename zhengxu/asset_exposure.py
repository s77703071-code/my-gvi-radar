"""Group ledger valuations for read-only exposure review."""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext

from .asset_state import Valuation


@dataclass(frozen=True)
class ExposureGroup:
    label: str
    valuation: Valuation


@dataclass(frozen=True)
class AssetExposure:
    classes: tuple[ExposureGroup, ...]
    symbols: tuple[ExposureGroup, ...]
    unidentified: int
    denominator: Decimal | None

    def share(self, valuation):
        if self.denominator is None or valuation.subtotal is None:
            return None
        return valuation.subtotal / self.denominator


def _groups(grouped, *, symbol_labels=False):
    output = []
    for label, values in grouped.items():
        known = [value for value in values if value is not None]
        amount = sum(known, Decimal(0)) if known else None
        name = " · ".join(label) if symbol_labels else label
        output.append(ExposureGroup(name, Valuation(amount, len(known), len(values))))
    return tuple(sorted(output, key=lambda item: (
        item.valuation.subtotal is None, -(item.valuation.subtotal or Decimal(0)), item.label
    )))


def build_asset_exposure(rows, assets):
    classes = defaultdict(list)
    symbols = defaultdict(list)
    unidentified = 0
    for row, value in zip(rows, assets.positions, strict=True):
        kind = row.get("asset_type") or "未分類"
        classes[kind].append(value)
        if kind == "現金":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        currency = str(row.get("currency") or "").strip().upper()
        if not symbol or not currency:
            unidentified += 1
            continue
        symbols[symbol, currency].append(value)

    total = assets.total
    denominator = total.subtotal if not total.incomplete and total.subtotal else None
    with localcontext() as context:
        context.prec = 96
        return AssetExposure(_groups(classes), _groups(symbols, symbol_labels=True), unidentified, denominator)
