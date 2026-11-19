"""Read-only risk findings shared by stock diagnosis and radar screening."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
import time

import pandas as pd


@dataclass(frozen=True)
class RiskFinding:
    category: str
    reason: str
    source: str


def _known(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def evaluate_radar_risk(history, *, timeframe, cache_updated, cache_ttl, roe=None, roe_source="",
                        free_cash_flow=None, fcf_source="", gvi=None, momentum=None,
                        fcf_yield=None, peg=None, now=None):
    findings = []
    valid = pd.DataFrame()
    quote_source = f"yfinance｜{timeframe} K 棒"
    if isinstance(history.index, pd.DatetimeIndex) and history.columns.is_unique and "Close" in history:
        closes = pd.to_numeric(history["Close"], errors="coerce")
        valid = history.loc[history.index.notna() & (closes > 0) & (closes < float("inf"))].sort_index()
        valid = valid.loc[~valid.index.duplicated(keep="last")]
        if not valid.empty:
            quote_source += f"｜最後有效 K 棒 {valid.index[-1]:%Y-%m-%d %H:%M}"
    if valid.empty:
        findings.append(RiskFinding("資料不足", "未取得可核對日期的有效行情。", quote_source))

    recent = valid.tail(5)
    volumes = pd.to_numeric(recent["Volume"], errors="coerce") if "Volume" in recent else ()
    if len(volumes) < 5 or not all(_known(value) and value >= 0 for value in volumes):
        findings.append(RiskFinding("資料不足", "成交量資料不足，無法核對近期是否持續無量。", quote_source))
    elif all(value == 0 for value in volumes):
        findings.append(RiskFinding("風險提醒", "最近 5 根有效 K 棒的成交量均為 0，請核對流動性與資料來源。", quote_source))

    current = time.time() if now is None else now
    if not _known(cache_updated) or not 0 < cache_updated <= current:
        findings.append(RiskFinding("資料不足", "無法核對基本面快取的更新時間。", "正序行情快取更新時間"))
    elif current - cache_updated >= cache_ttl:
        cache_date = pd.Timestamp(cache_updated, unit="s", tz="UTC").tz_convert("Asia/Taipei")
        findings.append(RiskFinding("風險提醒", "基本面快取已超過原有更新期限。", f"正序行情快取｜{cache_date:%Y-%m-%d %H:%M} 台北時間"))

    if _known(roe) and roe < 0:
        source = roe_source.strip() if isinstance(roe_source, str) else ""
        findings.append(RiskFinding("風險提醒", f"股東權益報酬率為負值（{roe:.2%}）。", source or "基本面快取的 ROE 欄位"))
    if _known(free_cash_flow) and free_cash_flow < 0:
        findings.append(RiskFinding("風險提醒", "自由現金流為負值。", fcf_source or "自由現金流欄位"))
    for available, label, basis in (
        (_known(gvi), "GVI", "基本面快取的淨值與 ROE、yfinance 股價"),
        (_known(momentum), "動能", quote_source),
        (_known(fcf_yield) and _known(peg), "QARP", "自由現金流、市值與 PEG"),
    ):
        if not available:
            findings.append(RiskFinding("資料不足", f"{label} 資料不足，無法完整判讀。", basis))
    return tuple(findings)


def risk_status(findings):
    labels = {finding.category for finding in findings}
    return "／".join(label for label in ("風險提醒", "資料不足") if label in labels) or "未見本次規則提示"
