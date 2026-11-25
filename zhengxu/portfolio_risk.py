"""Read-only portfolio risk and cash budget using recorded ledger data."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from .asset_state import ledger_number


ALLOCATION_TARGETS = {
    "股票": "target_stock_pct",
    "債券": "target_bond_pct",
    "現金": "target_cash_pct",
}
DRAWDOWN_KEYS = ("drawdown_warning_pct", "drawdown_review_pct", "drawdown_limit_pct")
TAIPEI = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class CashBudget:
    floor: Decimal | None
    available: Decimal | None
    shortfall: Decimal | None

    def covers(self, amount):
        value = ledger_number(amount)
        if value is None or value < 0 or self.available is None:
            return None
        return value <= self.available and not self.shortfall


def cash_budget(assets, settings):
    floor = ledger_number(settings.get("minimum_cash_twd"))
    if floor is None or floor < 0:
        return CashBudget(None, None, None)
    cash = assets.cash
    if not cash.count or cash.incomplete or cash.subtotal is None:
        return CashBudget(floor, None, None)
    difference = cash.subtotal - floor
    return CashBudget(floor, max(difference, Decimal(0)), max(-difference, Decimal(0)))


def allocation_check(exposure, settings):
    targets = {name: ledger_number(settings.get(key)) for name, key in ALLOCATION_TARGETS.items()}
    threshold = ledger_number(settings.get("rebalance_threshold_pct"))
    if (exposure.denominator is None or threshold is None or threshold <= 0 or threshold > 100
            or any(value is None or not 0 <= value <= 100 for value in targets.values())
            or abs(sum(targets.values()) - 100) > Decimal("0.01")):
        return ()
    groups = {item.label: item.valuation for item in exposure.classes}
    comparisons = []
    for name, target in targets.items():
        actual = exposure.share(groups[name]) * 100 if name in groups else Decimal(0)
        gap = actual - target
        comparisons.append((name, actual, target, gap, abs(gap) >= threshold))
    return tuple(comparisons)


@dataclass(frozen=True)
class SnapshotRisk:
    count: int
    latest_at: datetime
    latest_value: Decimal
    latest_drawdown: Decimal
    deepest_drawdown: Decimal


def snapshot_risk(snapshots):
    if snapshots is None:
        raise ValueError("Snapshot source unavailable")
    series = {}
    for row in snapshots:
        stamp = datetime.fromisoformat(row["snapshot_date"])
        stamp = stamp.replace(tzinfo=TAIPEI) if stamp.tzinfo is None else stamp.astimezone(TAIPEI)
        value = ledger_number(row.get("total_value_twd"))
        if stamp in series or value is None or value < 0:
            raise ValueError("Invalid snapshot history")
        series[stamp] = value
    if len(series) < 2:
        return None

    peak = Decimal(0)
    deepest = Decimal(0)
    current = None
    for stamp, value in sorted(series.items()):
        peak = max(peak, value)
        if peak:
            current = (peak - value) / peak * 100
            deepest = max(deepest, current)
    if current is None:
        return None
    return SnapshotRisk(len(series), stamp, value, current, deepest)


def drawdown_status(percent, settings):
    warning, review, limit = (ledger_number(settings.get(key)) for key in DRAWDOWN_KEYS)
    if any(value is None for value in (warning, review, limit)) or not 0 < warning < review < limit <= 100:
        return None
    if percent >= limit:
        return "達風險審查門檻"
    if percent >= review:
        return "達檢視門檻"
    if percent >= warning:
        return "達提醒門檻"
    return "低於已存提醒門檻"
