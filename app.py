# ==============================================================================
# 【機構級三核心策略雷達 3.12 網格複利對比(CAGR/MDD/稅費)與專業畫線全能版】 - app.py
# ==============================================================================
import sys, os, re, streamlit as st, yfinance as yf, pandas as pd, numpy as np, json, sqlite3, io, time, requests
import google.generativeai as genai
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


# ============================================================================
# 配息複利策略模組（內嵌於主程式，評分與原策略分開）
# ============================================================================
import io
import re
import time
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


SCORE_WEIGHTS = {
    "dividend_history": 25.0,
    "eps_stability": 20.0,
    "cash_flow": 20.0,
    "payout": 15.0,
    "safety": 10.0,
    "yield_valuation": 10.0,
}

DEFAULT_RULES = {
    "dividend_history": True,
    "eps_stability": True,
    "cash_flow": True,
    "payout": True,
    "safety": True,
    "yield_valuation": True,
    "anomaly": True,
    "min_dividend_years": 5,
    "min_positive_eps_years": 4,
    "min_positive_cashflow_years": 3,
    "max_payout_ratio": 0.80,
    "max_debt_equity": 1.50,
    "min_interest_coverage": 2.0,
    "max_yield_pct": 12.0,
    "max_pe": 25.0,
    "anomaly_yield_pct": 8.0,
    "anomaly_drawdown_pct": 30.0,
}

_TWSE_DIVIDEND_CACHE: tuple[float, list[dict[str, Any]]] = (0.0, [])
_TWSE_TR_CACHE: tuple[float, pd.DataFrame] = (0.0, pd.DataFrame())


def annual_cash_dividends(dividends: Optional[pd.Series]) -> dict[int, float]:
    """Return cash dividends per share by calendar year; absent years stay absent."""
    if dividends is None or len(dividends) == 0:
        return {}
    series = pd.to_numeric(dividends, errors="coerce").dropna()
    if series.empty:
        return {}
    idx = pd.to_datetime(series.index, errors="coerce")
    valid = ~idx.isna()
    series = series.loc[valid].copy()
    series.index = idx[valid]
    return {int(year): float(value) for year, value in series.groupby(series.index.year).sum().items()}


def trailing_dividend_yield(dividends: Optional[pd.Series], price: Optional[float], as_of: Any = None) -> Optional[float]:
    """Trailing twelve-month cash dividend yield as a percentage, not total return."""
    if dividends is None or len(dividends) == 0 or price is None or not np.isfinite(price) or price <= 0:
        return None
    series = pd.to_numeric(dividends, errors="coerce").dropna()
    if series.empty:
        return None
    idx = pd.to_datetime(series.index, errors="coerce")
    valid = ~idx.isna()
    series = series.loc[valid].copy()
    series.index = idx[valid]
    end = pd.Timestamp(as_of) if as_of is not None else series.index.max()
    start = end - pd.Timedelta(days=365)
    return float(series.loc[(series.index > start) & (series.index <= end)].sum() / price * 100.0)


def parse_twse_dividend_records(records: list[Mapping[str, Any]], symbol: str) -> tuple[dict[int, float], Optional[str]]:
    """Parse TWSE official distribution records, de-duplicating revisions by year/period."""
    code = str(symbol).replace(".TWO", "").replace(".TW", "").strip()
    prepared: dict[tuple[int, str], tuple[int, str, float]] = {}
    official_name = None
    for record in records:
        if str(record.get("公司代號", "")).strip() != code:
            continue
        official_name = str(record.get("公司名稱") or official_name or "").strip() or official_name
        year_raw = str(record.get("股利年度", "")).strip()
        try:
            year = int(year_raw)
            if year < 1911:
                year += 1911
        except ValueError:
            continue
        period = str(record.get("股利所屬期間") or record.get("股利所屬年(季)度") or "").strip()
        status = str(record.get("決議（擬議）進度") or "")
        # Prefer shareholder-confirmed rows over board proposals when the same
        # year/period was published more than once.
        priority = 2 if "股東會確認" in status else 1 if "董事會決議" in status else 0
        parts = [
            record.get("股東配發-盈餘分配之現金股利(元/股)"),
            record.get("股東配發-法定盈餘公積發放之現金(元/股)"),
            record.get("股東配發-資本公積發放之現金(元/股)"),
        ]
        amount = sum(value for value in (_finite_number(part) for part in parts) if value is not None)
        key = (year, period)
        prior = prepared.get(key)
        if prior is None or priority > prior[0]:
            prepared[key] = (priority, status, amount)
    annual: dict[int, float] = {}
    for (year, _period), (_priority, _status, amount) in prepared.items():
        annual[year] = annual.get(year, 0.0) + amount
    return annual, official_name


def fetch_twse_official_dividends(symbol: str) -> tuple[dict[int, float], Optional[str]]:
    """Fetch the cached TWSE OpenAPI dividend-distribution table for listed stocks."""
    global _TWSE_DIVIDEND_CACHE
    if not str(symbol).endswith(".TW") or str(symbol).endswith(".TWO"):
        return {}, None
    now = time.time()
    cached_at, records = _TWSE_DIVIDEND_CACHE
    if now - cached_at > 3600 or not records:
        try:
            import requests
            response = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap45_L",
                                    headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                records = payload
                _TWSE_DIVIDEND_CACHE = (now, records)
        except Exception:
            # Keep the existing Yahoo action history available as a fallback.
            return {}, None
    return parse_twse_dividend_records(records, symbol)


def _parse_twse_date(value: Any) -> Optional[pd.Timestamp]:
    text = str(value).strip()
    match = re.fullmatch(r"(\d{2,4})[/\-](\d{1,2})[/\-](\d{1,2})", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        if year < 1911:
            year += 1911
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            return None
    match = re.fullmatch(r"(\d{3})(\d{2})(\d{2})", text)
    if match:
        year, month, day = (int(part) for part in match.groups())
        try:
            return pd.Timestamp(year=year + 1911, month=month, day=day)
        except ValueError:
            return None
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _parse_twse_tr_csv(content: bytes) -> pd.DataFrame:
    errors = []
    for encoding in ("big5", "utf-8-sig", "utf-8"):
        for skipped in range(0, 4):
            try:
                frame = pd.read_csv(io.BytesIO(content), encoding=encoding, skiprows=skipped)
                normalized = {column: str(column).strip().replace(" ", "") for column in frame.columns}
                date_col = next((column for column, name in normalized.items() if name in ("日期", "日付", "Date") or "日期" in name), None)
                value_col = next((column for column, name in normalized.items() if "發行量加權股價報酬指數" in name), None)
                if date_col is None or value_col is None:
                    continue
                dates = frame[date_col].map(_parse_twse_date)
                values = pd.to_numeric(frame[value_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
                result = pd.DataFrame({"index_level": values.to_numpy()}, index=pd.DatetimeIndex(dates))
                result = result.loc[~result.index.isna()].dropna().sort_index()
                if not result.empty:
                    return result[~result.index.duplicated(keep="last")]
            except Exception as exc:
                errors.append(str(exc))
    raise ValueError("無法辨識證交所報酬指數 CSV 欄位" + (f"：{errors[-1]}" if errors else "。"))


def fetch_twse_total_return_index(start_date: Any, end_date: Any) -> Optional[pd.Series]:
    """Return official TAIEX total-return index levels; data.gov.tw lists a daily CSV."""
    global _TWSE_TR_CACHE
    cached_at, cached_frame = _TWSE_TR_CACHE
    if cached_frame.empty or time.time() - cached_at > 3600:
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.twse.com.tw/"}
            response = requests.get("https://www.twse.com.tw/indicesReport/MFI94U?response=open_data",
                                    headers=headers, timeout=25)
            response.raise_for_status()
            parsed = _parse_twse_tr_csv(response.content)
            _TWSE_TR_CACHE = (time.time(), parsed)
            cached_frame = parsed
        except Exception:
            return None
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start.tzinfo is not None:
        start = start.tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    series = cached_frame.loc[(cached_frame.index >= start) & (cached_frame.index <= end), "index_level"]
    return series if len(series) >= 2 else None


def _statement_values(statement: Optional[pd.DataFrame], aliases: tuple[str, ...]) -> list[float]:
    if statement is None or not isinstance(statement, pd.DataFrame) or statement.empty:
        return []
    for label in statement.index:
        normalized = str(label).strip().lower().replace("_", " ")
        if any(alias in normalized for alias in aliases):
            values = pd.to_numeric(statement.loc[label], errors="coerce").dropna()
            if not values.empty:
                try:
                    values.index = pd.to_datetime(values.index, errors="coerce")
                    values = values.loc[~values.index.isna()].sort_index()
                except Exception:
                    pass
                return [float(value) for value in values.tolist()]
    return []


def extract_financial_metrics(
    income_statement: Optional[pd.DataFrame],
    cash_flow: Optional[pd.DataFrame],
    balance_sheet: Optional[pd.DataFrame],
    info: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Extract available annual Yahoo statement values without filling gaps with zero."""
    info = info or {}
    eps = _statement_values(income_statement, ("basic eps", "diluted eps", "basic earnings per share", "diluted earnings per share"))
    if not eps and _finite_number(info.get("trailingEps")) is not None:
        eps = [float(info["trailingEps"])]
    ocf = _statement_values(cash_flow, ("operating cash flow", "cash flow from continuing operating activities", "total cash from operating activities"))
    capex = _statement_values(cash_flow, ("capital expenditure", "capital expenditures", "purchase of ppe"))
    fcf = [ocf[i] + capex[i] for i in range(min(len(ocf), len(capex)))] if ocf and capex else []

    total_debt = _statement_values(balance_sheet, ("total debt",))
    equity = _statement_values(balance_sheet, ("stockholders equity", "total equity gross minority interest", "total stockholder equity"))
    debt_equity = None
    if total_debt and equity and equity[-1] > 0:
        debt_equity = max(0.0, total_debt[-1]) / equity[-1]

    ebit = _statement_values(income_statement, ("ebit", "earnings before interest and taxes"))
    interest = _statement_values(income_statement, ("interest expense",))
    interest_coverage = None
    if ebit and interest and abs(interest[-1]) > 0:
        interest_coverage = ebit[-1] / abs(interest[-1])

    return {
        "eps_years": eps[-5:],
        "operating_cashflow_years": ocf[-5:],
        "free_cashflow_years": fcf[-5:],
        "debt_equity": debt_equity,
        "interest_coverage": interest_coverage,
        "pe": _finite_number(info.get("trailingPE")) or _finite_number(info.get("forwardPE")),
    }


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def score_dividend_quality(snapshot: Mapping[str, Any], rules: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Return an independent 0-100 score, data coverage, screening reasons and risks."""
    config = dict(DEFAULT_RULES)
    if rules:
        config.update(rules)
    annual_dividends = snapshot.get("annual_dividends") or {}
    recent_years = sorted(int(year) for year in annual_dividends)[-5:]
    dividends_complete = len(recent_years) >= 5 and all(annual_dividends.get(year) is not None for year in recent_years)
    dividend_years = sum(float(annual_dividends.get(year, 0.0)) > 0 for year in recent_years) if dividends_complete else None

    eps_values = snapshot.get("eps_years") or []
    eps_known = [float(v) for v in eps_values if _finite_number(v) is not None]
    eps_positive_years = sum(v > 0 for v in eps_known) if len(eps_known) >= 3 else None
    eps_growth = None
    if len(eps_known) >= 2 and eps_known[0] > 0 and eps_known[-1] > 0:
        eps_growth = (eps_known[-1] / eps_known[0]) ** (1 / (len(eps_known) - 1)) - 1

    ocf_values = [float(v) for v in (snapshot.get("operating_cashflow_years") or []) if _finite_number(v) is not None]
    fcf_values = [float(v) for v in (snapshot.get("free_cashflow_years") or []) if _finite_number(v) is not None]
    positive_cashflow_years = sum((ocf_values[i] > 0 and fcf_values[i] > 0) for i in range(min(len(ocf_values), len(fcf_values)))) if min(len(ocf_values), len(fcf_values)) >= 3 else None

    price = _finite_number(snapshot.get("price"))
    ttm_dividend = _finite_number(snapshot.get("ttm_dividend"))
    yield_pct = (ttm_dividend / price * 100.0) if price and price > 0 and ttm_dividend is not None else None
    latest_eps = eps_known[-1] if eps_known else _finite_number(snapshot.get("trailing_eps"))
    payout_ratio = _finite_number(snapshot.get("payout_ratio"))
    if payout_ratio is None and ttm_dividend is not None and latest_eps and latest_eps > 0:
        payout_ratio = ttm_dividend / latest_eps
    debt_equity = _finite_number(snapshot.get("debt_equity"))
    interest_coverage = _finite_number(snapshot.get("interest_coverage"))
    drawdown_from_52w_high = _finite_number(snapshot.get("drawdown_52w_pct"))
    pe = _finite_number(snapshot.get("pe"))

    component_scores: dict[str, Optional[float]] = {
        "dividend_history": (SCORE_WEIGHTS["dividend_history"] * min(dividend_years / 5, 1.0)) if dividend_years is not None else None,
        "eps_stability": (SCORE_WEIGHTS["eps_stability"] * min(eps_positive_years / 5, 1.0)) if eps_positive_years is not None else None,
        "cash_flow": (SCORE_WEIGHTS["cash_flow"] * min(positive_cashflow_years / 5, 1.0)) if positive_cashflow_years is not None else None,
        "payout": (SCORE_WEIGHTS["payout"] * max(0.0, 1.0 - payout_ratio / max(float(config["max_payout_ratio"]), 0.01))) if payout_ratio is not None else None,
        "safety": None,
        "yield_valuation": None,
    }
    component_coverage = {key: weight for key, weight in SCORE_WEIGHTS.items()}
    if debt_equity is not None or interest_coverage is not None:
        safety_parts = []
        if debt_equity is not None:
            safety_parts.append(max(0.0, min(1.0, 1.0 - debt_equity / max(float(config["max_debt_equity"]), 0.01))))
        if interest_coverage is not None:
            safety_parts.append(max(0.0, min(1.0, interest_coverage / max(float(config["min_interest_coverage"]), 0.01))))
        component_scores["safety"] = SCORE_WEIGHTS["safety"] * float(np.mean(safety_parts))
        component_coverage["safety"] = SCORE_WEIGHTS["safety"] * len(safety_parts) / 2.0
    if yield_pct is not None or pe is not None:
        valuation_parts = []
        if yield_pct is not None:
            valuation_parts.append(1.0 if yield_pct <= float(config["max_yield_pct"]) else max(0.0, float(config["max_yield_pct"]) / yield_pct))
        if pe is not None and pe > 0:
            valuation_parts.append(1.0 if pe <= float(config["max_pe"]) else max(0.0, float(config["max_pe"]) / pe))
        component_scores["yield_valuation"] = SCORE_WEIGHTS["yield_valuation"] * float(np.mean(valuation_parts)) if valuation_parts else None
        component_coverage["yield_valuation"] = SCORE_WEIGHTS["yield_valuation"] * len(valuation_parts) / 2.0

    available_weight = sum(component_coverage[k] for k, score in component_scores.items() if score is not None)
    score_points = sum(float(score) * component_coverage[k] / SCORE_WEIGHTS[k]
                       for k, score in component_scores.items() if score is not None)
    score = (score_points / available_weight * 100.0) if available_weight else None
    coverage = available_weight
    failures: list[str] = []
    data_gaps: list[str] = []
    risks: list[str] = []

    def assess(enabled_key: str, value: Any, label: str, predicate, missing_message: str):
        if not config.get(enabled_key):
            return
        if value is None:
            data_gaps.append(missing_message)
        elif not predicate(value):
            failures.append(label)

    assess("dividend_history", dividend_years, "近五年現金股利未達門檻", lambda v: v >= int(config["min_dividend_years"]), "缺少完整近五年現金股利歷史")
    assess("eps_stability", eps_positive_years, "EPS 正值年度不足", lambda v: v >= int(config["min_positive_eps_years"]), "缺少至少三年可比 EPS 資料")
    assess("cash_flow", positive_cashflow_years, "營業現金流與自由現金流正值年度不足", lambda v: v >= int(config["min_positive_cashflow_years"]), "缺少至少三年營業現金流／自由現金流資料")
    assess("payout", payout_ratio, "現金股利發放率高於上限", lambda v: v <= float(config["max_payout_ratio"]), "無法以現有 EPS 與股利資料計算發放率")
    if config.get("safety"):
        if debt_equity is None:
            data_gaps.append("缺少負債權益比資料")
        elif debt_equity > float(config["max_debt_equity"]):
            failures.append("負債權益比高於上限")
        if interest_coverage is None:
            data_gaps.append("缺少利息保障倍數資料")
        elif interest_coverage < float(config["min_interest_coverage"]):
            failures.append("利息保障倍數低於門檻")
    if config.get("yield_valuation"):
        if yield_pct is None:
            data_gaps.append("缺少現金殖利率資料")
        elif yield_pct > float(config["max_yield_pct"]):
            risks.append("殖利率高於設定上限，需確認股利可持續性")
        if pe is None:
            data_gaps.append("缺少本益比估值資料")
        elif pe > float(config["max_pe"]):
            failures.append("本益比高於上限")
    if config.get("anomaly"):
        if yield_pct is not None and yield_pct >= float(config["anomaly_yield_pct"]):
            if drawdown_from_52w_high is None:
                data_gaps.append("高殖利率異常檢查缺少 52 週高點資料")
            elif drawdown_from_52w_high >= float(config["anomaly_drawdown_pct"]):
                failures.append("疑似股價大幅下跌造成異常高殖利率")
                risks.append("殖利率可能受股價急跌扭曲")
    if eps_growth is not None and eps_growth < 0:
        risks.append("EPS 長期年化成長率為負")
    if payout_ratio is not None and payout_ratio > 1:
        risks.append("股利高於最新可得 EPS，可能由資本公積或過去盈餘支應")

    return {
        "quality_score": round(float(np.clip(score, 0, 100)), 1) if score is not None else None,
        "coverage_pct": round(float(coverage), 1),
        "passed": not failures and not data_gaps,
        "dividend_years": dividend_years,
        "yield_pct": yield_pct,
        "eps_positive_years": eps_positive_years,
        "eps_growth": eps_growth,
        "positive_cashflow_years": positive_cashflow_years,
        "payout_ratio": payout_ratio,
        "debt_equity": debt_equity,
        "interest_coverage": interest_coverage,
        "failures": failures,
        "data_gaps": data_gaps,
        "risks": risks,
    }


def build_snapshot(
    symbol: str,
    name: str,
    history: pd.DataFrame,
    income_statement: Optional[pd.DataFrame] = None,
    cash_flow: Optional[pd.DataFrame] = None,
    balance_sheet: Optional[pd.DataFrame] = None,
    info: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build a dividend/fundamental snapshot from unadjusted OHLC and action columns."""
    info = info or {}
    if history is None or history.empty or "Close" not in history:
        raise ValueError("無可用歷史收盤價")
    frame = history.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.loc[~frame.index.isna()].sort_index()
    prices = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if prices.empty:
        raise ValueError("歷史收盤價欄位沒有有效資料")
    as_of = prices.index[-1]
    dividends = pd.to_numeric(frame.get("Dividends", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    div_events = dividends[dividends > 0]
    annual = annual_cash_dividends(div_events)
    official_annual, official_name = fetch_twse_official_dividends(symbol)
    if official_annual:
        annual = official_annual
    financial = extract_financial_metrics(income_statement, cash_flow, balance_sheet, info)
    ttm_dividend = float(div_events.loc[div_events.index > as_of - pd.Timedelta(days=365)].sum())
    peak = float(prices.tail(252).max()) if not prices.empty else None
    current = float(prices.iloc[-1])
    drawdown = ((peak - current) / peak * 100.0) if peak and peak > 0 else None
    latest_eps = _finite_number(info.get("trailingEps"))
    if latest_eps is None and financial["eps_years"]:
        latest_eps = financial["eps_years"][-1]
    payout_ratio = (ttm_dividend / latest_eps) if latest_eps and latest_eps > 0 else None
    return {
        "symbol": symbol,
        "name": official_name or name or symbol,
        "price": current,
        "as_of": as_of,
        "annual_dividends": annual,
        "dividend_history_source": "證交所公開資料" if official_annual else "Yahoo Finance 配息事件",
        "dividend_years_display": ", ".join(f"{year}:{annual[year]:.2f}" for year in sorted(annual)[-5:]),
        "ttm_dividend": ttm_dividend,
        "payout_ratio": payout_ratio,
        "drawdown_52w_pct": drawdown,
        "trailing_eps": _finite_number(info.get("trailingEps")),
        **financial,
    }


def download_symbol_snapshot(symbol: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fetch a current quality snapshot and action-aware daily price history via yfinance."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    history = ticker.history(period="6y", auto_adjust=False, actions=True)
    if history is None or history.empty:
        raise ValueError("Yahoo Finance 無回傳六年股價與配息歷史")
    if isinstance(history.index, pd.DatetimeIndex) and history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    def safe_statement(name: str):
        try:
            return getattr(ticker, name, None)
        except Exception:
            return None
    snapshot = build_snapshot(
        symbol,
        str(info.get("shortName") or info.get("longName") or symbol),
        history,
        safe_statement("income_stmt"),
        safe_statement("cashflow"),
        safe_statement("balance_sheet"),
        info,
    )
    return snapshot, history


def _normalize_events(events: Optional[pd.DataFrame], column: str) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=["ticker", "value"], index=pd.DatetimeIndex([], name="date"))
    frame = events.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date")
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.loc[~frame.index.isna()]
    if "ticker" not in frame:
        frame["ticker"] = column
    if "value" not in frame:
        value_col = "Dividends" if "Dividends" in frame else "Stock Splits" if "Stock Splits" in frame else None
        if value_col is None:
            return pd.DataFrame(columns=["ticker", "value"], index=pd.DatetimeIndex([], name="date"))
        frame["value"] = frame[value_col]
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame[["ticker", "value"]].dropna(subset=["value"]).sort_index()


def _buy(symbol: str, amount: float, price: float, commission: float) -> tuple[float, float]:
    if amount <= 0 or price <= 0:
        return 0.0, 0.0
    shares = amount / (price * (1.0 + commission))
    return shares, amount - shares * price


def _portfolio_value(positions: Mapping[str, float], prices: Mapping[str, float], cash: float) -> float:
    return float(cash + sum(positions.get(t, 0.0) * prices.get(t, 0.0) for t in positions))


def simulate_portfolio(
    prices: pd.DataFrame,
    dividends: Optional[pd.DataFrame] = None,
    splits: Optional[pd.DataFrame] = None,
    initial_capital: float = 1_000_000.0,
    monthly_contribution: float = 0.0,
    reinvest_dividends: bool = True,
    max_holdings: Optional[int] = None,
    rebalance: str = "不再平衡",
    commission_rate: float = 0.001425,
    sell_tax_rate: float = 0.003,
    dividend_tax_rate: float = 0.0,
) -> dict[str, Any]:
    """Simulate an equal-weight portfolio using split-adjusted closes and cash dividends.

    Dividend events are credited on the ex-dividend date as a practical proxy;
    fractional shares are allowed. Contributions are invested on each month's
    first available session. Commission, sell tax, and dividend tax are editable.
    Optional split events are for callers supplying unsplit raw closes; do not
    pass them together with yfinance Close, which is already split-adjusted.
    """
    if prices is None or prices.empty:
        raise ValueError("沒有可用價格資料")
    frame = prices.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame = frame.loc[~frame.index.isna()].sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    frame = frame.loc[:, frame.notna().any(axis=0)]
    if frame.empty or initial_capital < 0 or monthly_contribution < 0:
        raise ValueError("價格資料或投入金額無效")
    symbols = [str(s) for s in frame.columns]
    if max_holdings is not None and int(max_holdings) > 0:
        symbols = symbols[: int(max_holdings)]
        frame = frame[symbols]
    if not symbols:
        raise ValueError("沒有可模擬股票")
    div_events = _normalize_events(dividends, "ticker")
    split_events = _normalize_events(splits, "ticker")
    if not div_events.empty:
        div_events["ticker"] = div_events["ticker"].astype(str)
    if not split_events.empty:
        split_events["ticker"] = split_events["ticker"].astype(str)

    commission_rate = max(0.0, float(commission_rate))
    sell_tax_rate = max(0.0, float(sell_tax_rate))
    dividend_tax_rate = max(0.0, min(float(dividend_tax_rate), 1.0))
    positions = {ticker: 0.0 for ticker in symbols}
    cash = 0.0
    total_dividend_gross = 0.0
    total_dividend_net = 0.0
    total_reinvested = 0.0
    total_contributed = 0.0
    total_fees = 0.0
    history_rows = []
    month_keys = frame.index.to_period("M")
    first_sessions = set(frame.groupby(month_keys, sort=True).head(1).index)
    last_rebalance_key = None
    rebalance_months = {"不再平衡": None, "每月": 1, "每季": 3, "每半年": 6, "每年": 12}.get(rebalance, None)
    dividend_groups = {(pd.Timestamp(idx).normalize(), str(row.ticker)): float(row.value) for idx, row in div_events.iterrows() for row in [row]}
    split_groups = {(pd.Timestamp(idx).normalize(), str(row.ticker)): float(row.value) for idx, row in split_events.iterrows() for row in [row]}

    for index, row in frame.iterrows():
        day = pd.Timestamp(index).normalize()
        day_prices = {ticker: _finite_number(row.get(ticker)) for ticker in symbols}
        day_prices = {ticker: value for ticker, value in day_prices.items() if value is not None and value > 0}
        if not day_prices:
            continue
        for ticker in symbols:
            split = split_groups.get((day, ticker))
            if split is not None and split > 0:
                positions[ticker] *= split

        contribution = 0.0
        if index == frame.index[0]:
            contribution = float(initial_capital)
        elif index in first_sessions and monthly_contribution > 0:
            contribution = float(monthly_contribution)
        if contribution:
            cash += contribution
            total_contributed += contribution

        # Dividend cash is accounted for once, from raw Close plus event data.
        for ticker in symbols:
            per_share = dividend_groups.get((day, ticker))
            if per_share is not None and per_share > 0 and ticker in day_prices:
                gross = positions[ticker] * per_share
                net = gross * (1.0 - dividend_tax_rate)
                total_dividend_gross += gross
                total_dividend_net += net
                cash += net
                if reinvest_dividends and net > 0:
                    shares, fee = _buy(ticker, net, day_prices[ticker], commission_rate)
                    positions[ticker] += shares
                    cash -= net
                    total_fees += fee
                    total_reinvested += net - fee

        month_number = index.year * 12 + index.month
        due_rebalance = rebalance_months and index in first_sessions and (
            last_rebalance_key is None or month_number - last_rebalance_key >= rebalance_months
        )
        if index == frame.index[0] or contribution > 0 or due_rebalance:
            active = [ticker for ticker in symbols if ticker in day_prices]
            if active:
                current_value = _portfolio_value(positions, day_prices, cash)
                target = current_value / len(active)
                for ticker in active:
                    current = positions[ticker] * day_prices[ticker]
                    if current > target and current > 0:
                        proceeds = min(current - target, current)
                        tax = proceeds * sell_tax_rate
                        fee = proceeds * commission_rate
                        shares_sold = proceeds / day_prices[ticker]
                        positions[ticker] -= shares_sold
                        cash += proceeds - tax - fee
                        total_fees += tax + fee
                current_value = _portfolio_value(positions, day_prices, cash)
                target = current_value / len(active)
                for ticker in active:
                    current = positions[ticker] * day_prices[ticker]
                    if current < target and cash > 0:
                        desired = min(target - current, cash)
                        shares, fee = _buy(ticker, desired, day_prices[ticker], commission_rate)
                        positions[ticker] += shares
                        cash -= desired
                        total_fees += fee
                last_rebalance_key = month_number if due_rebalance else last_rebalance_key

        value = _portfolio_value(positions, day_prices, cash)
        history_rows.append({"date": index, "portfolio_value": value, "cash": cash,
                             "contribution": contribution, "total_dividend_gross": total_dividend_gross,
                             "total_dividend_net": total_dividend_net, "total_reinvested": total_reinvested,
                             **{f"shares:{ticker}": positions[ticker] for ticker in symbols}})

    result = pd.DataFrame(history_rows).set_index("date")
    if result.empty:
        raise ValueError("價格期間沒有有效交易日")
    denominator = result["portfolio_value"].shift(1) + result["contribution"]
    daily_returns = result["portfolio_value"].div(denominator.replace(0, np.nan)).sub(1).fillna(0.0)
    result["time_weighted_return"] = daily_returns
    result["growth_index"] = (1.0 + daily_returns).cumprod()
    growth = result["growth_index"]
    total_return = float(growth.iloc[-1] - 1.0)
    years = max((result.index[-1] - result.index[0]).days / 365.25, 1.0 / 365.25)
    cagr = float(growth.iloc[-1] ** (1.0 / years) - 1.0) if growth.iloc[-1] >= 0 else None
    drawdown = growth / growth.cummax() - 1.0
    volatility = float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    sharpe = float(daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0 else None
    final_shares = {ticker: float(result[f"shares:{ticker}"].iloc[-1]) for ticker in symbols}
    metrics = {
        "cagr": cagr,
        "mdd": float(drawdown.min()),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "cumulative_total_return": total_return,
        "ending_value": float(result["portfolio_value"].iloc[-1]),
        "total_contributed": float(total_contributed),
        "gross_dividends": float(total_dividend_gross),
        "net_dividends": float(total_dividend_net),
        "reinvested_dividends": float(total_reinvested),
        "fees_and_taxes": float(total_fees),
        "ending_cash": float(result["cash"].iloc[-1]),
        "final_shares": final_shares,
    }
    return {"history": result, "metrics": metrics}


def split_in_sample_oos(history: pd.DataFrame, fraction: float = 0.70) -> dict[str, Any]:
    """Chronologically split an existing return path; future observations never enter training."""
    if history is None or len(history) < 10:
        return {"in_sample": None, "out_of_sample": None, "split_date": None, "error": "有效回測日數不足，至少需要 10 個交易日。"}
    boundary = max(1, min(len(history) - 1, int(len(history) * fraction)))
    return {
        "in_sample": history.iloc[:boundary].copy(),
        "out_of_sample": history.iloc[boundary:].copy(),
        "split_date": history.index[boundary],
        "error": None,
    }


def metrics_for_return_path(history: pd.DataFrame) -> dict[str, Optional[float]]:
    """Compute comparable TWR metrics for a sample segment already simulated."""
    if history is None or history.empty or "time_weighted_return" not in history:
        return {"cagr": None, "mdd": None, "annualized_volatility": None, "sharpe": None, "cumulative_total_return": None}
    returns = pd.to_numeric(history["time_weighted_return"], errors="coerce").fillna(0.0)
    growth = (1.0 + returns).cumprod()
    years = max((history.index[-1] - history.index[0]).days / 365.25, 1.0 / 365.25)
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return {
        "cagr": float(growth.iloc[-1] ** (1.0 / years) - 1.0) if growth.iloc[-1] >= 0 else None,
        "mdd": float((growth / growth.cummax() - 1.0).min()),
        "annualized_volatility": float(std * np.sqrt(252)),
        "sharpe": float(returns.mean() / std * np.sqrt(252)) if std and np.isfinite(std) else None,
        "cumulative_total_return": float(growth.iloc[-1] - 1.0),
    }


def yearly_scenario_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Classify calendar-year time-weighted returns as rising, flat, or falling."""
    if history is None or history.empty or "time_weighted_return" not in history:
        return pd.DataFrame(columns=["年度", "年度含息報酬(%)", "市場情境"])
    returns = pd.to_numeric(history["time_weighted_return"], errors="coerce").fillna(0.0)
    annual = (1.0 + returns).groupby(history.index.year).prod() - 1.0
    rows = []
    for year, value in annual.items():
        scenario = "多頭" if value >= 0.10 else "空頭" if value <= -0.10 else "盤整"
        rows.append({"年度": int(year), "年度含息報酬(%)": float(value * 100), "市場情境": scenario})
    return pd.DataFrame(rows)


def market_scenario_summary(strategy_history: pd.DataFrame, benchmark_history: pd.DataFrame) -> pd.DataFrame:
    """Compare annual strategy returns inside market regimes classified by a total-return proxy."""
    if strategy_history is None or benchmark_history is None or strategy_history.empty or benchmark_history.empty:
        return pd.DataFrame(columns=["年度", "市場情境", "基準含息報酬(%)", "策略含息報酬(%)"])
    strategy = (1.0 + pd.to_numeric(strategy_history["time_weighted_return"], errors="coerce").fillna(0.0)).groupby(strategy_history.index.year).prod() - 1.0
    benchmark = (1.0 + pd.to_numeric(benchmark_history["time_weighted_return"], errors="coerce").fillna(0.0)).groupby(benchmark_history.index.year).prod() - 1.0
    common_years = sorted(set(strategy.index) & set(benchmark.index))
    rows = []
    for year in common_years:
        market_return = float(benchmark.loc[year])
        regime = "多頭" if market_return >= 0.10 else "空頭" if market_return <= -0.10 else "盤整"
        rows.append({"年度": int(year), "市場情境": regime, "基準含息報酬(%)": market_return * 100,
                     "策略含息報酬(%)": float(strategy.loc[year]) * 100})
    return pd.DataFrame(rows)


def _render_controls(st, prefix: str = "div") -> dict[str, Any]:
    c1, c2, c3 = st.columns(3)
    with c1:
        initial = st.number_input("初始投入本金", min_value=0.0, value=1_000_000.0, step=100_000.0, key=f"{prefix}_initial")
        monthly = st.number_input("每月追加投入", min_value=0.0, value=10_000.0, step=1_000.0, key=f"{prefix}_monthly")
    with c2:
        period = st.slider("回測期間（年）", min_value=1, max_value=15, value=5, key=f"{prefix}_years")
        max_holdings = st.number_input("最多持股數", min_value=1, max_value=50, value=10, step=1, key=f"{prefix}_max_holdings")
    with c3:
        reinvest = st.checkbox("股息再投入", value=True, key=f"{prefix}_reinvest")
        rebalance = st.selectbox("定期再平衡", ["不再平衡", "每月", "每季", "每半年", "每年"], index=2, key=f"{prefix}_rebalance")
    c4, c5, c6 = st.columns(3)
    with c4:
        commission = st.number_input("單邊手續費率 (%)", min_value=0.0, max_value=2.0, value=0.1425, step=0.01, format="%.4f", key=f"{prefix}_commission") / 100
    with c5:
        sell_tax = st.number_input("賣出交易稅 (%)", min_value=0.0, max_value=5.0, value=0.30, step=0.05, format="%.2f", key=f"{prefix}_sell_tax") / 100
    with c6:
        div_tax = st.number_input("股利稅費估算 (%)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, format="%.1f", key=f"{prefix}_div_tax") / 100
    return {"initial_capital": initial, "monthly_contribution": monthly, "years": period,
            "max_holdings": max_holdings, "reinvest_dividends": reinvest, "rebalance": rebalance,
            "commission_rate": commission, "sell_tax_rate": sell_tax, "dividend_tax_rate": div_tax}


def _render_screen(st, tickers: list[str], rules: dict[str, Any], label: str = "dividend") -> list[dict[str, Any]]:
    cache_key = f"{label}_snapshots"
    if st.button("🔎 取得資料並執行配息品質篩選", type="primary", key=f"{label}_screen_button"):
        output = []
        errors = []
        for symbol in tickers[:50]:
            try:
                snapshot, _ = download_symbol_snapshot(symbol)
                output.append(snapshot)
            except Exception as exc:
                errors.append({"symbol": symbol, "error": str(exc)})
        st.session_state[cache_key] = output
        st.session_state[f"{cache_key}_errors"] = errors
    output = [
        {**row, **score_dividend_quality(row, rules)}
        for row in st.session_state.get(cache_key, [])
        if row.get("symbol") in tickers
    ]
    errors = [error for error in st.session_state.get(f"{cache_key}_errors", []) if error.get("symbol") in tickers]
    if errors:
        st.warning("部分代碼無法取得資料：" + "；".join(f"{x['symbol']} ({x['error']})" for x in errors[:8]))
    if output:
        rows = []
        for item in output:
            rows.append({
                "代號": item.get("symbol"), "名稱": item.get("name"), "資料日期": str(item.get("as_of", ""))[:10],
                "股利資料來源": item.get("dividend_history_source"),
                "現金殖利率(%)": item.get("yield_pct"), "近五年股利(元/股)": item.get("dividend_years_display"),
                "EPS 正值年數": item.get("eps_positive_years"), "EPS 年化成長(%)": item.get("eps_growth") * 100 if item.get("eps_growth") is not None else None,
                "現金流正值年數": item.get("positive_cashflow_years"), "股利發放率(%)": item.get("payout_ratio") * 100 if item.get("payout_ratio") is not None else None,
                "最新年度EPS(元)": item.get("eps_years", [None])[-1] if item.get("eps_years") else item.get("trailing_eps"),
                "營業現金流(最新年)": item.get("operating_cashflow_years", [None])[-1] if item.get("operating_cashflow_years") else None,
                "自由現金流(最新年)": item.get("free_cashflow_years", [None])[-1] if item.get("free_cashflow_years") else None,
                "本益比": item.get("pe"), "負債/權益": item.get("debt_equity"), "利息保障倍數": item.get("interest_coverage"),
                "品質分數": item.get("quality_score"), "資料覆蓋(%)": item.get("coverage_pct"),
                "篩選狀態": "通過" if item.get("passed") else "未通過/資料不足",
                "不合格原因": "；".join(item.get("failures", [])),
                "資料缺漏": "；".join(item.get("data_gaps", [])),
                "風險警示": "；".join(item.get("risks", [])),
            })
        view = pd.DataFrame(rows)
        only_passed = st.checkbox("只顯示通過條件的股票", value=False, key=f"{label}_passed_only")
        if only_passed:
            view = view[view["篩選狀態"] == "通過"]
        st.dataframe(view, use_container_width=True, hide_index=True)
    return output


def _render_simulation(st, snapshots: list[dict[str, Any]], controls: dict[str, Any], label: str = "dividend",
                       include_benchmark: bool = False, benchmark_symbol: Optional[str] = None,
                       only_passed: bool = True) -> None:
    if not st.button("📈 執行股息再投入與歷史回測", key=f"{label}_simulate_button"):
        return
    if only_passed:
        snapshots = [row for row in snapshots if row.get("passed")]
    if not snapshots:
        st.warning("沒有符合目前設定的股票。請先執行配息品質篩選，或取消『僅採用通過條件股票』。")
        return
    selected = sorted(snapshots, key=lambda row: (row.get("quality_score") is not None, row.get("quality_score") or -1), reverse=True)[: int(controls["max_holdings"])]
    symbols = [row["symbol"] for row in selected]
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=int(controls["years"]))
    price_series = {}
    dividend_rows = []
    errors = []
    for symbol in symbols:
        try:
            _, history = download_symbol_snapshot(symbol)
            history = history.loc[history.index >= start_date]
            price_series[symbol] = pd.to_numeric(history["Close"], errors="coerce")
            # yfinance Close is already split-adjusted even with auto_adjust=False.
            # We pass dividend events only, avoiding applying splits twice.
            if "Dividends" in history:
                actions = pd.to_numeric(history["Dividends"], errors="coerce").dropna()
                for when, value in actions[actions > 0].items():
                    dividend_rows.append({"date": when, "ticker": symbol, "value": float(value)})
        except Exception as exc:
            errors.append(f"{symbol}: {exc}")
    if errors:
        st.warning("回測資料有缺漏，未能模擬：" + "；".join(errors))
    if not price_series:
        st.error("沒有可回測的歷史價格資料。")
        return
    price_frame = pd.concat(price_series, axis=1).sort_index()
    price_frame = price_frame.loc[price_frame.index >= start_date]
    try:
        result = simulate_portfolio(
            price_frame,
            pd.DataFrame(dividend_rows),
            None,
            initial_capital=controls["initial_capital"],
            monthly_contribution=controls["monthly_contribution"],
            reinvest_dividends=controls["reinvest_dividends"],
            max_holdings=controls["max_holdings"],
            rebalance=controls["rebalance"], commission_rate=controls["commission_rate"],
            sell_tax_rate=controls["sell_tax_rate"], dividend_tax_rate=controls["dividend_tax_rate"],
        )
    except Exception as exc:
        st.error(f"回測失敗：{exc}")
        return
    history, metrics = result["history"], result["metrics"]
    split = split_in_sample_oos(history)
    percent_keys = ("cagr", "mdd", "annualized_volatility", "cumulative_total_return")
    cols = st.columns(4)
    labels = {"cagr": "年化報酬 CAGR", "mdd": "最大回撤 MDD", "annualized_volatility": "年化波動率", "sharpe": "Sharpe Ratio",
              "cumulative_total_return": "累積含息總報酬", "gross_dividends": "股息收入（稅前）", "reinvested_dividends": "股息再投入金額", "ending_value": "期末資產"}
    display_metrics = ["cagr", "mdd", "annualized_volatility", "sharpe", "cumulative_total_return", "gross_dividends", "reinvested_dividends", "ending_value"]
    for idx, key in enumerate(display_metrics):
        value = metrics.get(key)
        if key in percent_keys:
            shown = f"{value:.2%}" if value is not None else "資料不足"
        elif key == "sharpe":
            shown = f"{value:.2f}" if value is not None else "資料不足"
        else:
            shown = f"{value:,.0f}" if value is not None else "資料不足"
        with cols[idx % 4]:
            st.metric(labels[key], shown)
    st.line_chart(history[["portfolio_value"]], use_container_width=True)
    st.caption(f"股價報酬、股息收入與含息總報酬已分開計算；價格使用拆股調整後收盤價，股息於除息日入帳。Sharpe 以 0 無風險利率估算。期末股數：{metrics['final_shares']}。")
    if split.get("error"):
        st.warning(split["error"])
    else:
        in_metrics = metrics_for_return_path(split["in_sample"])
        out_metrics = metrics_for_return_path(split["out_of_sample"])
        st.write(f"樣本內至 {split['split_date']:%Y-%m-%d}：CAGR {in_metrics['cagr']:.2%}；樣本外：CAGR {out_metrics['cagr']:.2%}。")
    benchmark_history_path = None
    benchmark_metrics = None
    benchmark_label = benchmark_symbol or ""
    if benchmark_symbol:
        if benchmark_symbol == "TWSE_TAIEX_TR":
            index_series = fetch_twse_total_return_index(start_date, end_date)
            if index_series is not None:
                benchmark_label = "發行量加權股價報酬指數（證交所）"
                benchmark_prices = pd.DataFrame({"TAIEX_TR": index_series})
                try:
                    benchmark_result = simulate_portfolio(
                        benchmark_prices,
                        initial_capital=controls["initial_capital"], monthly_contribution=controls["monthly_contribution"],
                        reinvest_dividends=True, max_holdings=1, rebalance="不再平衡",
                        commission_rate=0.0, sell_tax_rate=0.0, dividend_tax_rate=0.0,
                    )
                    benchmark_metrics = benchmark_result["metrics"]
                    benchmark_history_path = benchmark_result["history"]
                except Exception as exc:
                    if include_benchmark:
                        st.warning(f"證交所報酬指數資料無法計算：{exc}")
            else:
                benchmark_symbol = "0050.TW"
                benchmark_label = "0050.TW（官方報酬指數無法取得時的代理）"
                if include_benchmark:
                    st.warning("目前未能讀取證交所報酬指數，基準比較改用 0050.TW ETF 的股價與股利事件。")
        if benchmark_history_path is None and (benchmark_symbol != "TWSE_TAIEX_TR" or benchmark_label.startswith("0050")):
            try:
                import yfinance as yf
                benchmark_history = yf.Ticker(benchmark_symbol).history(period=f"{int(controls['years'])}y", auto_adjust=False, actions=True)
                if isinstance(benchmark_history.index, pd.DatetimeIndex) and benchmark_history.index.tz is not None:
                    benchmark_history.index = benchmark_history.index.tz_localize(None)
                benchmark_history = benchmark_history.loc[benchmark_history.index >= start_date]
                benchmark_prices = pd.DataFrame({benchmark_symbol: pd.to_numeric(benchmark_history["Close"], errors="coerce")})
                benchmark_dividend_series = pd.to_numeric(benchmark_history.get("Dividends", pd.Series(0.0, index=benchmark_history.index)), errors="coerce").fillna(0)
                benchmark_dividends = pd.DataFrame([{"date": when, "ticker": benchmark_symbol, "value": float(value)}
                                                    for when, value in benchmark_dividend_series.items()
                                                    if value > 0])
                benchmark_result = simulate_portfolio(
                    benchmark_prices, benchmark_dividends,
                    initial_capital=controls["initial_capital"], monthly_contribution=controls["monthly_contribution"],
                    reinvest_dividends=True, max_holdings=1, rebalance="不再平衡",
                    commission_rate=controls["commission_rate"], sell_tax_rate=controls["sell_tax_rate"],
                    dividend_tax_rate=controls["dividend_tax_rate"],
                )
                benchmark_metrics = benchmark_result["metrics"]
                benchmark_history_path = benchmark_result["history"]
            except Exception as exc:
                if include_benchmark:
                    st.warning(f"基準資料目前無法取得（{benchmark_label}）：{exc}")
    scenarios = market_scenario_summary(history, benchmark_history_path) if benchmark_history_path is not None else yearly_scenario_summary(history)
    if not scenarios.empty:
        st.markdown("##### 不同市場情境分析（情境以基準含息報酬分類）")
        formats = {"年度含息報酬(%)": "{:+.2f}%"} if "年度含息報酬(%)" in scenarios.columns else {"基準含息報酬(%)": "{:+.2f}%", "策略含息報酬(%)": "{:+.2f}%"}
        st.dataframe(scenarios.style.format(formats), use_container_width=True, hide_index=True)

    if include_benchmark and benchmark_symbol and benchmark_metrics is not None:
        st.markdown(f"##### 基準比較：{benchmark_label}（同投入額、含息報酬）")
        compare = pd.DataFrame([
            {"組合": "配息品質策略", "CAGR(%)": metrics["cagr"] * 100 if metrics["cagr"] is not None else None,
             "累積含息報酬(%)": metrics["cumulative_total_return"] * 100, "MDD(%)": metrics["mdd"] * 100,
             "期末資產": metrics["ending_value"]},
            {"組合": benchmark_label, "CAGR(%)": benchmark_metrics["cagr"] * 100 if benchmark_metrics["cagr"] is not None else None,
             "累積含息報酬(%)": benchmark_metrics["cumulative_total_return"] * 100, "MDD(%)": benchmark_metrics["mdd"] * 100,
             "期末資產": benchmark_metrics["ending_value"]},
        ])
        st.dataframe(compare.style.format({"CAGR(%)": "{:+.2f}%", "累積含息報酬(%)": "{:+.2f}%", "MDD(%)": "{:.2f}%", "期末資產": "{:,.0f}"}),
                     use_container_width=True, hide_index=True)
    st.warning("資料限制：Yahoo Finance 歷史財報通常不含逐期公告日版本，因此此回測以目前設定的持股固定回放價格與股息，不代表當時可得資訊下的歷史選股；結果存在存活者偏誤與資料修訂風險。下市股票若未列入輸入清單或來源已無歷史資料，也無法完整還原。台股個人股利稅費與交易費率依使用者設定估算，不是個人稅務計算。")


def render_dividend_tab(st, default_ticker: str = "2330.TW", strategy_scores: Optional[Mapping[str, Any]] = None) -> None:
    """Render an additive, independent dividend strategy panel."""
    st.markdown("### 💸 配息複利投資策略（獨立模組）")
    st.caption("配息評分獨立計算，不讀寫原系統的 GVI、動能、QARP 或綜合評分。殖利率是近 12 個月現金股利／現價；股利發放率以近 12 個月股利與最新可得 EPS 估算。總報酬將價格變化與股息再投入分開處理。")
    mode = st.radio("策略檢視模式", ["只使用原有策略", "只使用配息策略", "比較策略"], horizontal=True, key="dividend_strategy_mode")
    if mode == "只使用原有策略":
        st.info("目前檢視原有選股策略；此配息模組不會改寫或執行原有評分。原有策略表格仍在本頁下方原位置顯示。")
        return
    raw = st.text_input("配息策略觀察名單（最多 50 檔，逗號分隔）", value=default_ticker, key="dividend_universe")
    tickers = list(dict.fromkeys(x.strip().upper() for x in raw.split(",") if x.strip()))[:50]
    if not tickers:
        st.warning("請輸入至少一個股票代碼。")
        return
    st.markdown("#### 配息品質條件（各項可獨立啟用）")
    columns = st.columns(4)
    toggles = {}
    for i, (key, title) in enumerate((("dividend_history", "五年現金股利"), ("eps_stability", "EPS 穩定"), ("cash_flow", "營業／自由現金流"),
                                     ("payout", "股利發放率"), ("safety", "負債與利息安全"), ("yield_valuation", "殖利率與估值"), ("anomaly", "異常高殖利率排除"))):
        with columns[i % 4]:
            toggles[key] = st.checkbox(title, value=bool(DEFAULT_RULES[key]), key=f"dividend_rule_{key}")
    a, b, c = st.columns(3)
    with a:
        min_dividend_years = st.number_input("五年中至少配息年數", 1, 5, 5, key="dividend_min_years")
        min_positive_eps_years = st.number_input("EPS 正值年數至少", 1, 5, 4, key="dividend_min_eps")
    with b:
        min_cash_years = st.number_input("現金流正值年數至少", 1, 5, 3, key="dividend_min_cash")
        max_payout = st.number_input("最高股利發放率 (%)", 10.0, 200.0, 80.0, step=5.0, key="dividend_max_payout") / 100
    with c:
        max_debt = st.number_input("最高負債／權益", 0.1, 10.0, 1.5, step=0.1, key="dividend_max_debt")
        max_yield = st.number_input("殖利率參考上限 (%)", 1.0, 50.0, 12.0, step=1.0, key="dividend_max_yield")
    rules = {**toggles, "min_dividend_years": min_dividend_years, "min_positive_eps_years": min_positive_eps_years,
             "min_positive_cashflow_years": min_cash_years, "max_payout_ratio": max_payout,
             "max_debt_equity": max_debt, "max_yield_pct": max_yield}
    snapshots = _render_screen(st, tickers, rules)
    st.markdown("#### 股息再投入模擬與歷史績效")
    only_passed = st.checkbox("回測只採用通過所有啟用條件的股票", value=True, key="dividend_sim_passed_only")
    controls = _render_controls(st)
    taiwan_flags = [symbol.endswith((".TW", ".TWO")) for symbol in tickers]
    if any(taiwan_flags) and not all(taiwan_flags):
        st.warning("台股與美股幣別不同，請分開執行品質篩選與回測，避免把新台幣與美元資產直接相加。")
        return
    all_taiwan = all(taiwan_flags)
    all_twse_listed = all(symbol.endswith(".TW") and not symbol.endswith(".TWO") for symbol in tickers)
    benchmark_symbol = "TWSE_TAIEX_TR" if all_twse_listed else "0050.TW" if all_taiwan else "SPY"
    _render_simulation(st, snapshots, controls, include_benchmark=mode == "比較策略", benchmark_symbol=benchmark_symbol,
                       only_passed=only_passed)
    if mode == "比較策略":
        benchmark_scope = "TWSE 上市市場官方含息報酬指數；官方資料失效時退回 0050" if benchmark_symbol == "TWSE_TAIEX_TR" else "0050 臺灣50大型股含息代理" if benchmark_symbol == "0050.TW" else "SPY S&P 500 含息代理"
        st.info(f"比較模式以相同投入額比較配息組合與 {benchmark_symbol}（{benchmark_scope}）。舊系統未保存逐日歷史成分，故無法重建舊策略的歷史績效。")

st.set_page_config(page_title="機構級三核心策略雷達 3.12", layout="wide", page_icon="📈")

# 📱 行動端手機螢幕顯示優化 CSS
st.markdown("""
    <style>
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
        .js-plotly-plot .plotly .main-svg { border-radius: 8px; }
        @media (max-width: 768px) {
            .stMetric { padding: 4px !important; }
            .block-container { padding-top: 1rem !important; }
        }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "market_cache.db")
CACHE_TTL = 86400  # 快取有效期限：24 小時 (86400 秒)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS market_data 
                 (ticker TEXT PRIMARY KEY, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, 
                  gvi REAL, roe REAL, mcap REAL, shares REAL, book_value REAL, 
                  roe_source TEXT, bv_source TEXT, updated_at INTEGER)''')
    
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(market_data)").fetchall()]
    cols_to_add = {
        "book_value": "REAL",
        "roe_source": "TEXT",
        "bv_source": "TEXT",
        "updated_at": "INTEGER"
    }
    for col, col_type in cols_to_add.items():
        if col not in existing_cols:
            try:
                c.execute(f"ALTER TABLE market_data ADD COLUMN {col} {col_type}")
            except Exception:
                pass
    conn.commit()
    conn.close()

init_db()

def parse_roe(raw_roe):
    if raw_roe is None or np.isnan(raw_roe) or not np.isfinite(raw_roe):
        return None
    val = float(raw_roe)
    if abs(val) > 3.0:
        val = val / 100.0
    return val

def is_cache_valid(updated_at):
    if not updated_at or np.isnan(updated_at):
        return False
    return (time.time() - float(updated_at)) < CACHE_TTL

def generate_content_with_retry(prompt_text, api_key_val):
    genai.configure(api_key=api_key_val)
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash']
    last_exception = None
    
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt_text)
                return response.text.strip()
            except Exception as e:
                last_exception = e
                if "503" in str(e) or "high demand" in str(e).lower() or "busy" in str(e).lower():
                    sleep_time = 2 ** (attempt + 1)
                    time.sleep(sleep_time)
                    continue
                break
    raise last_exception

def fetch_tw_official_pb_pe(ticker):
    clean_code = ticker.replace(".TW", "").replace(".TWO", "")
    is_tpex = ".TWO" in ticker
    try:
        if is_tpex:
            url = "https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php?l=zh-tw&response=json"
            res = requests.get(url, timeout=4)
            data = res.json()
            for row in data.get('aaData', []):
                if row[0] == clean_code:
                    pb = float(row[5]) if len(row) > 5 and row[5] not in ['N/A', '-', ''] else None
                    pe = float(row[2]) if len(row) > 2 and row[2] not in ['N/A', '-', ''] else None
                    return pb, pe
        else:
            url = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?response=json"
            res = requests.get(url, timeout=4)
            data = res.json()
            for row in data.get('data', []):
                if row[0] == clean_code:
                    pb = float(row[5]) if len(row) > 5 and row[5] not in ['N/A', '-', ''] else None
                    pe = float(row[4]) if len(row) > 4 and row[4] not in ['N/A', '-', ''] else None
                    return pb, pe
    except Exception as e:
        st.sidebar.caption(f"TWSE/TPEx API 擷取提醒 ({ticker}): {e}")
    return None, None

def _to_number(value):
    """Parse TWSE/TPEx comma-formatted numbers without turning missing values into zero."""
    if value is None:
        return None
    text = str(value).strip().replace(',', '').replace(' ', '')
    if text in ('', '-', '--', 'N/A', 'nan', 'None'):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_institutional_market_data():
    """Fetch official latest daily institutional trades and issued share counts for TWSE/TPEx."""
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.twse.com.tw/'}
    result = {}
    errors = []
    # TWSE T86 is date based. Walk back over weekends/holidays to the latest report.
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo('Asia/Taipei')).date()
    twse_payload = None
    twse_date = None
    for offset in range(8):
        query_date = (today - timedelta(days=offset)).strftime('%Y%m%d')
        try:
            response = requests.get('https://www.twse.com.tw/rwd/zh/fund/T86',
                                    params={'date': query_date, 'selectType': 'ALLBUT0999', 'response': 'json'},
                                    headers=headers, timeout=12)
            response.raise_for_status()
            payload = response.json()
            if payload.get('data') and payload.get('fields'):
                twse_payload, twse_date = payload, query_date
                break
        except Exception as exc:
            errors.append(f'TWSE: {exc}')
    if twse_payload:
        fields = twse_payload.get('fields', [])
        for row in twse_payload.get('data', []):
            values = dict(zip(fields, row))
            code = str(values.get('證券代號', '')).strip()
            if not code:
                continue
            def pick(prefix):
                for key, val in values.items():
                    if prefix in key:
                        return _to_number(val)
                return None
            foreign_buy = pick('外陸資買進股數(不含外資自營商)')
            foreign_sell = pick('外陸資賣出股數(不含外資自營商)')
            trust_buy, trust_sell = pick('投信買進股數'), pick('投信賣出股數')
            # Domestic dealer buy/sell includes both proprietary and hedge categories.
            dealer_buys = [pick('自營商買進股數(自行買賣)'), pick('自營商買進股數(避險)')]
            dealer_sells = [pick('自營商賣出股數(自行買賣)'), pick('自營商賣出股數(避險)')]
            result[code] = {'日期': twse_date, '市場': '上市', '股票名稱': str(values.get('證券名稱', '')).strip(),
                '外資買進股數': foreign_buy, '外資賣出股數': foreign_sell,
                '外資買賣超股數': (foreign_buy - foreign_sell) if None not in (foreign_buy, foreign_sell) else None,
                '投信買進股數': trust_buy, '投信賣出股數': trust_sell,
                '投信買賣超股數': (trust_buy - trust_sell) if None not in (trust_buy, trust_sell) else None,
                '自營商買進股數': sum(x for x in dealer_buys if x is not None) if any(x is not None for x in dealer_buys) else None,
                '自營商賣出股數': sum(x for x in dealer_sells if x is not None) if any(x is not None for x in dealer_sells) else None}
            if result[code]['自營商買進股數'] is not None and result[code]['自營商賣出股數'] is not None:
                result[code]['自營商買賣超股數'] = result[code]['自營商買進股數'] - result[code]['自營商賣出股數']
            else:
                result[code]['自營商買賣超股數'] = None
    else:
        errors.append('TWSE 最近 8 日沒有可用的 T86 資料。')

    try:
        response = requests.get('https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading',
                                headers=headers, timeout=12)
        response.raise_for_status()
        payload = response.json()
        for values in payload if isinstance(payload, list) else []:
            code = str(values.get('SecuritiesCompanyCode', '')).strip()
            if not code:
                continue
            def find_value(*needles):
                for key, val in values.items():
                    if all(n.lower() in key.lower() for n in needles):
                        return _to_number(val)
                return None
            fb, fs = find_value('Foreign Investors', 'Total Buy'), find_value('Foreign Investors', 'Total Sell')
            tb, ts = find_value('SecuritiesInvestmentTrustCompanies', 'TotalBuy'), find_value('SecuritiesInvestmentTrustCompanies', 'TotalSell')
            db, ds = find_value('Dealers', 'TotalBuy'), find_value('Dealers', 'TotalSell')
            # If endpoint labels use spaces in place of camel case, retry common variants.
            if tb is None: tb = find_value('Securities Investment Trust Companies', 'Total Buy')
            if ts is None: ts = find_value('Securities Investment Trust Companies', 'Total Sell')
            if db is None: db = find_value('Dealers', 'Total Buy')
            if ds is None: ds = find_value('Dealers', 'Total Sell')
            result[code] = {'日期': str(values.get('Date', '最新交易日')), '市場': '上櫃',
                '股票名稱': str(values.get('CompanyName') or values.get('SecuritiesCompanyName') or values.get('公司名稱') or '').strip(),
                '外資買進股數': fb, '外資賣出股數': fs, '外資買賣超股數': fb-fs if None not in (fb, fs) else None,
                '投信買進股數': tb, '投信賣出股數': ts, '投信買賣超股數': tb-ts if None not in (tb, ts) else None,
                '自營商買進股數': db, '自營商賣出股數': ds, '自營商買賣超股數': db-ds if None not in (db, ds) else None}
    except Exception as exc:
        errors.append(f'TPEx: {exc}')

    shares_by_code = {}
    names_by_code = {}
    for url in ('https://openapi.twse.com.tw/v1/opendata/t187ap03_L',
                'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O'):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            profiles = response.json()
            for profile in profiles if isinstance(profiles, list) else []:
                code = str(profile.get('公司代號') or profile.get('SecuritiesCompanyCode') or '').strip()
                shares = _to_number(profile.get('已發行普通股數或TDR原股發行股數') or profile.get('已發行普通股數') or profile.get('IssuedShares'))
                company_name = str(profile.get('公司名稱') or profile.get('CompanyName') or profile.get('公司簡稱') or '').strip()
                if code and shares and shares > 0:
                    shares_by_code[code] = shares
                if code and company_name:
                    names_by_code[code] = company_name
        except Exception as exc:
            errors.append(f'股數資料: {exc}')
    for code, trade in result.items():
        if not trade.get('股票名稱'):
            trade['股票名稱'] = names_by_code.get(code, code)
    return result, shares_by_code, errors

def build_institutional_screen(trades, shares_by_code, tickers, investor, min_net_pct):
    rows = []
    for ticker in tickers:
        code = ticker.replace('.TWO', '').replace('.TW', '').strip()
        trade = trades.get(code)
        shares = shares_by_code.get(code)
        if not trade or not shares:
            continue
        row = {'股票代碼': ticker, '股票名稱': trade.get('股票名稱') or ticker, '市場': trade['市場'], '資料日期': trade['日期'], '已發行股數': shares}
        for name in ('外資', '投信', '自營商'):
            for measure, label in (('買進股數', '買進'), ('賣出股數', '賣出'), ('買賣超股數', '買賣超')):
                value = trade.get(f'{name}{measure}')
                row[f'{name}{label}佔股本比(%)'] = value / shares * 100 if value is not None else None
        if row.get(f'{investor}買賣超佔股本比(%)') is not None and row[f'{investor}買賣超佔股本比(%)'] >= min_net_pct:
            rows.append(row)
    return pd.DataFrame(rows)

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tdcc_concentration(ticker):
    """Fetch weekly TDCC shareholding distribution; return big-holder/retail ratios and dates."""
    code = ticker.replace('.TWO', '').replace('.TW', '').strip()
    response = requests.get('https://openapi.tdcc.com.tw/v1/opendata/5-9', timeout=20,
                            headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    payload = response.json()
    rows = []
    for raw in payload if isinstance(payload, list) else []:
        raw_code = str(raw.get('證券代號') or raw.get('股票代號') or raw.get('SecuritiesCompanyCode') or raw.get('Code') or '').strip()
        if raw_code != code:
            continue
        level = str(raw.get('持股分級') or raw.get('持股級距') or raw.get('股數/單位數級距') or raw.get('Level') or '')
        date_value = str(raw.get('資料日期') or raw.get('資料年月') or raw.get('Date') or '')
        ratio = _to_number(raw.get('占集保庫存數比例%') or raw.get('占集保庫存數比例') or raw.get('持股比例') or raw.get('Ratio'))
        if ratio is None:
            continue
        digits = [int(x.replace(',', '')) for x in re.findall(r'\d[\d,]*', level)]
        if not digits:
            continue
        low = digits[0]
        high = digits[1] if len(digits) > 1 else None
        rows.append({'date': date_value, 'low': low, 'high': high, 'ratio': ratio})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    dates = sorted(frame['date'].dropna().unique(), reverse=True)
    if not dates:
        return None
    summaries = []
    for date_value in dates[:2]:
        day = frame[frame['date'] == date_value]
        retail = day[day['high'].notna() & (day['high'] <= 5000)]['ratio'].sum()
        big = day[(day['low'] >= 400001) | (day['high'].isna() & (day['low'] >= 400001))]['ratio'].sum()
        summaries.append({'date': date_value, 'retail_pct': float(retail), 'big_pct': float(big)})
    latest = summaries[0]
    previous = summaries[1] if len(summaries) > 1 else None
    return {'date': latest['date'], 'retail_pct': latest['retail_pct'], 'big_pct': latest['big_pct'],
            'big_change_pp': latest['big_pct'] - previous['big_pct'] if previous else None,
            'retail_change_pp': latest['retail_pct'] - previous['retail_pct'] if previous else None,
            'previous_date': previous['date'] if previous else None}

def fetch_mops_insider_holding(ticker):
    """Read latest MOPS monthly director/insider holding table for a single Taiwan company."""
    code = ticker.replace('.TWO', '').replace('.TW', '').strip()
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('Asia/Taipei'))
    headers = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/x-www-form-urlencoded'}
    url = 'https://mopsov.twse.com.tw/mops/web/ajax_stapap1'
    for month_offset in range(0, 4):
        month_date = now.replace(day=1) - timedelta(days=month_offset * 28)
        roc_year = month_date.year - 1911
        payload = {'encodeURIComponent':'1','step':'1','firstin':'1','off':'1','keyword4':'','code1':'','TYPEK2':'',
                   'checkbtn':'1','queryName':'co_id','inpuType':'co_id','TYPEK':'all','isnew':'false',
                   'co_id':code,'year':str(roc_year),'month':f'{month_date.month:02d}'}
        try:
            response = requests.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            for table in tables:
                columns = [' '.join(str(part) for part in col if str(part) != 'nan') if isinstance(col, tuple) else str(col) for col in table.columns]
                normalized = [c.replace(' ', '').replace('\n', '') for c in columns]
                table.columns = normalized
                insider_cols = [c for c in normalized if '內部人關係人目前持股合計' in c]
                direct_cols = [c for c in normalized if ('目前持股' in c or '持有股數' in c) and '設質' not in c]
                if insider_cols:
                    amount = _to_number(table[insider_cols[0]].iloc[-1])
                elif direct_cols:
                    amount = pd.to_numeric(table[direct_cols[0]].astype(str).str.replace(',', ''), errors='coerce').sum()
                else:
                    continue
                if amount and amount > 0:
                    return {'shares': float(amount), 'date': f'{roc_year}/{month_date.month:02d}', 'source':'MOPS 董監事持股餘額'}
        except Exception:
            continue
    return None

def summarize_broker_branch_csv(uploaded_file, stock_code):
    """Summarize optional official broker-branch CSV supplied by the user."""
    if uploaded_file is None:
        return None
    try:
        frame = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except Exception:
        uploaded_file.seek(0)
        frame = pd.read_csv(uploaded_file, encoding='big5', encoding_errors='replace')
    code_col = next((c for c in frame.columns if '代號' in str(c) and ('證券' in str(c) or '股票' in str(c))), None)
    buy_col = next((c for c in frame.columns if '買進' in str(c) and ('股數' in str(c) or '數量' in str(c))), None)
    sell_col = next((c for c in frame.columns if '賣出' in str(c) and ('股數' in str(c) or '數量' in str(c))), None)
    if not all((code_col, buy_col, sell_col)):
        return {'error':'檔案需包含證券代號、買進股數、賣出股數欄位。'}
    selected = frame[frame[code_col].astype(str).str.strip() == str(stock_code)].copy()
    if selected.empty:
        return {'error':'上傳檔案找不到目前分析股票。'}
    selected['_buy'] = pd.to_numeric(selected[buy_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    selected['_sell'] = pd.to_numeric(selected[sell_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    selected['_net'] = selected['_buy'] - selected['_sell']
    buyers = int((selected['_net'] > 0).sum())
    sellers = int((selected['_net'] < 0).sum())
    return {'net_shares': float(selected['_net'].sum()), 'buyers':buyers, 'sellers':sellers,
            'count_diff':buyers-sellers, 'top_net_buy':float(selected.nlargest(15, '_net')['_net'].clip(lower=0).sum()),
            'top_net_sell':float(selected.nsmallest(15, '_net')['_net'].clip(upper=0).sum())}

def build_institutional_rankings(trades, shares_by_code, limit=10):
    rows = []
    for code, trade in trades.items():
        shares = shares_by_code.get(code)
        if not shares:
            continue
        row = {'股票代碼':code, '股票名稱':trade.get('股票名稱') or code, '市場':trade.get('市場'), '資料日期':trade.get('日期')}
        for investor in ('外資','投信','自營商'):
            for measure, label in (('買進股數','買進'),('賣出股數','賣出'),('買賣超股數','買賣超')):
                value = trade.get(f'{investor}{measure}')
                row[f'{investor}{label}佔股本比(%)'] = value / shares * 100 if value is not None else None
        rows.append(row)
    return pd.DataFrame(rows)

def signature_save_to_db(t):
    try:
        stock = yf.Ticker(t)
        info = stock.info if hasattr(stock, 'info') and stock.info else {}
        
        p_raw = info.get('currentPrice') or info.get('previousClose')
        if not p_raw:
            hist = stock.history(period="5d")
            if not hist.empty:
                p_raw = float(hist['Close'].iloc[-1])
        price = float(p_raw) if (p_raw and float(p_raw) > 0) else None

        book_value = info.get('bookValue')
        bv_source = "yfinance 最新財報" if (book_value is not None and not np.isnan(book_value) and book_value > 0) else None

        raw_roe = info.get('returnOnEquity')
        roe_val = parse_roe(raw_roe)
        roe_source = "yfinance 最新財報" if roe_val is not None else None

        # 台股回退機制
        if (".TW" in t or ".TWO" in t) and (book_value is None or roe_val is None):
            official_pb, official_pe = fetch_tw_official_pb_pe(t)
            if (book_value is None or book_value <= 0) and price and official_pb and official_pb > 0:
                book_value = price / official_pb
                bv_source = "TWSE/TPEx PB估算值"
            if roe_val is None and official_pe and official_pe > 0 and official_pb and official_pb > 0:
                roe_val = parse_roe(official_pb / official_pe)
                roe_source = "TWSE/TPEx PB/PE代理值"

        # 美股/其他市場彈性回退機制 (1Q -> 半年 -> 1年)
        if (".TW" not in t and ".TWO" not in t) and (book_value is None or roe_val is None):
            try:
                bs_q = stock.quarterly_balance_sheet
                bs_a = stock.balance_sheet
                inc_q = stock.quarterly_financials
                inc_a = stock.financials

                tot_equity = None
                bv_source_temp = ""
                # 依序嘗試：季報 -> 年報
                for bs, src_name in [(bs_q, "yfinance 最新季報遞補"), (bs_a, "yfinance 最新年報遞補")]:
                    if not bs.empty:
                        for key in ['Total Stockholder Equity', 'Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest']:
                            if key in bs.index:
                                found_eq = bs.loc[key].iloc[0]
                                if pd.notnull(found_eq):
                                    tot_equity = float(found_eq)
                                    bv_source_temp = src_name
                                    break
                    if tot_equity is not None: break
                
                if book_value is None and tot_equity is not None:
                    shares_out = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
                    if shares_out and shares_out > 0:
                        calculated_bv = tot_equity / float(shares_out)
                        if calculated_bv > 0:
                            book_value = calculated_bv
                            bv_source = bv_source_temp
                        else:
                            bv_source = "美股負股東權益(庫藏股/虧損)"

                if roe_val is None and tot_equity is not None and tot_equity > 0:
                    roe_source_temp = ""
                    net_inc = None
                    if not inc_q.empty:
                        for net_key in ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests']:
                            if net_key in inc_q.index:
                                inc_series = inc_q.loc[net_key].dropna()
                                if len(inc_series) >= 4:
                                    net_inc = float(inc_series.iloc[:4].sum())
                                    roe_source_temp = "yfinance 近四季(TTM)推算"
                                elif len(inc_series) >= 2:
                                    net_inc = float(inc_series.iloc[:2].sum()) * 2
                                    roe_source_temp = "yfinance 近半年年化推算"
                                elif len(inc_series) > 0:
                                    net_inc = float(inc_series.iloc[0]) * 4
                                    roe_source_temp = "yfinance 單季年化推算"
                                break
                    if net_inc is None and not inc_a.empty:
                        for net_key in ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests']:
                            if net_key in inc_a.index:
                                inc_series = inc_a.loc[net_key].dropna()
                                if len(inc_series) > 0:
                                    net_inc = float(inc_series.iloc[0])
                                    roe_source_temp = "yfinance 最新年報遞補"
                                break
                                
                    if net_inc is not None:
                        roe_val = parse_roe(net_inc / tot_equity)
                        roe_source = roe_source_temp
            except Exception as us_err:
                st.sidebar.caption(f"美股財報備援解析提醒 [{t}]: {us_err}")

        bv_val = float(book_value) if (book_value is not None and not np.isnan(book_value) and book_value > 0) else None
        roe_val = parse_roe(roe_val)

        gvi = (bv_val / price) * ((1 + roe_val) ** 5) if (price and price > 0 and bv_val and bv_val > 0 and roe_val is not None) else None

        mcap = info.get('marketCap')
        mcap_val = float(mcap) if (mcap and float(mcap) > 0) else None
        shares_val = (mcap_val / price) if (mcap_val and price and price > 0) else None
        updated_at = int(time.time())

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO market_data 
                     (ticker, date, open, high, low, close, volume, gvi, roe, mcap, shares, book_value, roe_source, bv_source, updated_at) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                  (t, "LATEST", price, price, price, price, 10000, 
                   gvi, roe_val, mcap_val, shares_val, bv_val, roe_source, bv_source, updated_at))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"❌ 數據寫入資料庫失敗 [{t}]: {e}")

def compute_backtest_indicators(df, ma_entry_p=20, ma_exit_p=20, rsi_p=14, bb_p=20):
    df_calc = df.copy()
    df_calc['MA_entry'] = df_calc['Close'].rolling(window=ma_entry_p).mean()
    df_calc['MA_exit'] = df_calc['Close'].rolling(window=ma_exit_p).mean()

    low_min = df_calc['Low'].rolling(window=9).min()
    high_max = df_calc['High'].rolling(window=9).max()
    rsv = np.where(high_max == low_min, 50.0, (df_calc['Close'] - low_min) / (high_max - low_min) * 100.0)

    k_list, d_list = [50.0], [50.0]
    for r in rsv[1:]:
        if np.isnan(r): r = 50.0
        k_val = (2.0/3.0) * k_list[-1] + (1.0/3.0) * r
        d_val = (2.0/3.0) * d_list[-1] + (1.0/3.0) * k_val
        k_list.append(k_val)
        d_list.append(d_val)

    df_calc['K'] = k_list
    df_calc['D'] = d_list

    ema12 = df_calc['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_calc['Close'].ewm(span=26, adjust=False).mean()
    df_calc['DIF'] = ema12 - ema26
    df_calc['DEM'] = df_calc['DIF'].ewm(span=9, adjust=False).mean()

    df_calc['VOL_MA20'] = df_calc['Volume'].rolling(window=20).mean()

    delta = df_calc['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_p).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_p).mean()
    rs = gain / (loss + 1e-9)
    df_calc['RSI'] = 100 - (100 / (1 + rs))

    bb_middle = df_calc['Close'].rolling(window=bb_p).mean()
    bb_std = df_calc['Close'].rolling(window=bb_p).std()
    df_calc['BB_Upper'] = bb_middle + (bb_std * 2)
    df_calc['BB_Lower'] = bb_middle - (bb_std * 2)

    return df_calc

def run_strategy_backtest(df_calc, entry_conds, exit_conds, date_chinese_list):
    trades = []
    position = False
    entry_price = 0.0
    entry_date = ""
    highest_price = 0.0

    closes = df_calc['Close'].to_numpy()
    highs = df_calc['High'].to_numpy()
    lows = df_calc['Low'].to_numpy()
    vols = df_calc['Volume'].to_numpy()
    vol_ma20 = df_calc['VOL_MA20'].to_numpy()

    ma_entry = df_calc['MA_entry'].to_numpy()
    ma_exit = df_calc['MA_exit'].to_numpy()
    k_arr = df_calc['K'].to_numpy()
    d_arr = df_calc['D'].to_numpy()
    dif_arr = df_calc['DIF'].to_numpy()
    dem_arr = df_calc['DEM'].to_numpy()
    rsi_arr = df_calc['RSI'].to_numpy()
    bb_upper = df_calc['BB_Upper'].to_numpy()
    bb_lower = df_calc['BB_Lower'].to_numpy()

    for i in range(1, len(df_calc)):
        if not position:
            buy_checks = []
            
            if entry_conds['ma_mode'] == '突破均線':
                buy_checks.append(not np.isnan(ma_entry[i-1]) and closes[i-1] <= ma_entry[i-1] and closes[i] > ma_entry[i])
            elif entry_conds['ma_mode'] == '站穩均線上':
                buy_checks.append(not np.isnan(ma_entry[i]) and closes[i] > ma_entry[i])

            if entry_conds['kd_mode'] == '黃金交叉':
                buy_checks.append(k_arr[i-1] <= d_arr[i-1] and k_arr[i] > d_arr[i])

            if entry_conds['macd_mode'] == '黃金交叉':
                buy_checks.append(dif_arr[i-1] <= dem_arr[i-1] and dif_arr[i] > dem_arr[i])

            if entry_conds['vol_mode'] == '爆量 (大於20日均量1.5倍)':
                buy_checks.append(not np.isnan(vol_ma20[i]) and vols[i] >= vol_ma20[i] * 1.5)
            elif entry_conds['vol_mode'] == '爆量 (大於20日均量2.0倍)':
                buy_checks.append(not np.isnan(vol_ma20[i]) and vols[i] >= vol_ma20[i] * 2.0)

            if entry_conds['rsi_mode'] == '超賣回升 (RSI向上突破30)':
                buy_checks.append(rsi_arr[i-1] <= 30 and rsi_arr[i] > 30)
            elif entry_conds['rsi_mode'] == '強勢突破 (RSI向上突破50)':
                buy_checks.append(rsi_arr[i-1] <= 50 and rsi_arr[i] > 50)

            if entry_conds['bb_mode'] == '突破布林上軌':
                buy_checks.append(closes[i-1] <= bb_upper[i-1] and closes[i] > bb_upper[i])
            elif entry_conds['bb_mode'] == '觸及布林下軌反彈':
                buy_checks.append(lows[i-1] <= bb_lower[i-1] and closes[i] > bb_lower[i])

            if buy_checks and (all(buy_checks) if entry_conds['match_mode'] == 'ALL' else any(buy_checks)):
                position = True
                entry_price = float(closes[i])
                entry_date = date_chinese_list[i]
                highest_price = float(highs[i])
        else:
            highest_price = max(highest_price, float(highs[i]))
            exit_reasons = []

            if exit_conds['stop_loss_pct'] > 0:
                sl_price = entry_price * (1.0 - exit_conds['stop_loss_pct'] / 100.0)
                if lows[i] <= sl_price or closes[i] <= sl_price:
                    exit_reasons.append(f"觸發停損 (-{exit_conds['stop_loss_pct']}%)")

            if exit_conds['trailing_stop_pct'] > 0:
                trail_price = highest_price * (1.0 - exit_conds['trailing_stop_pct'] / 100.0)
                if lows[i] <= trail_price or closes[i] <= trail_price:
                    exit_reasons.append(f"最高點回撤 (-{exit_conds['trailing_stop_pct']}%)")

            if exit_conds['ma_mode'] == '跌破均線' and not np.isnan(ma_exit[i-1]):
                if closes[i-1] >= ma_exit[i-1] and closes[i] < ma_exit[i]:
                    exit_reasons.append(f"跌破 MA{exit_conds['ma_p']}")

            if exit_conds['kd_mode'] == '死亡交叉':
                if k_arr[i-1] >= d_arr[i-1] and k_arr[i] < d_arr[i]:
                    exit_reasons.append("KD 死亡交叉")

            if exit_conds['macd_mode'] == '死亡交叉':
                if dif_arr[i-1] >= dem_arr[i-1] and dif_arr[i] < dem_arr[i]:
                    exit_reasons.append("MACD 死亡交叉")

            if exit_conds['vol_mode'] == '極端爆量倒貨 (大於20日均量2.5倍)':
                if not np.isnan(vol_ma20[i]) and vols[i] >= vol_ma20[i] * 2.5:
                    exit_reasons.append("觸發極端爆量離場")

            if exit_conds['rsi_mode'] == '超買警戒 (RSI跌破70)':
                if rsi_arr[i-1] >= 70 and rsi_arr[i] < 70:
                    exit_reasons.append("RSI 70 死亡交叉離場")

            if exit_conds['bb_mode'] == '跌破布林下軌':
                if closes[i-1] >= bb_lower[i-1] and closes[i] < bb_lower[i]:
                    exit_reasons.append("跌破布林下軌離場")
            elif exit_conds['bb_mode'] == '觸及上軌拉回':
                if highs[i-1] >= bb_upper[i-1] and closes[i] < bb_upper[i]:
                    exit_reasons.append("布林上軌受阻離場")

            if exit_reasons:
                exit_price = float(closes[i])
                exit_date = date_chinese_list[i]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append({
                    '買進日期': entry_date,
                    '買進價格 (元)': round(entry_price, 2),
                    '賣出日期': exit_date,
                    '賣出價格 (元)': round(exit_price, 2),
                    '平倉報酬率 (%)': round(pnl_pct, 2),
                    '離場觸發原因': " | ".join(exit_reasons)
                })
                position = False

    return pd.DataFrame(trades)

# 🔐 管理員操盤密碼固定為 7770
st.sidebar.markdown("### 🔒 操盤手安全密碼鎖")
input_password = st.sidebar.text_input("請輸入管理員操盤密碼：", type="password")

if input_password != "7770":
    st.title("🔒 華爾街機構級三核心策略雷達終端")
    st.warning("⚠️ 請於左側邊欄輸入正確的管理員操盤密碼以解鎖核心雷達。")
    st.stop()
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 全球個股即時診斷")
    st.sidebar.caption("💡 提示：上市請加 `.TW`，上櫃請加 `.TWO`（例如：3293.TWO）")
    selected_stock = st.sidebar.text_input("輸入台美股代碼：", value="3293.TWO").strip().upper()
    st.title("📈 機構級三核心策略雷達 3.12（網格複利對比與專業畫線全能版）")

    st.markdown("### 🌐 全球大盤即時看板")
    col1, col2, col3, col4 = st.columns(4)

    def get_market_index(ticker):
        try:
            df = yf.Ticker(ticker).history(period="5d")
            if not df.empty and len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                close_series = df['Close'].dropna().to_numpy().flatten()
                if len(close_series) >= 2:
                    close_today = float(close_series[-1])
                    change = close_today - float(close_series[-2])
                    return close_today, change, (change / float(close_series[-2])) * 100
        except Exception: 
            pass
        return None, None, None

    tw_val, tw_chg, tw_pct = get_market_index("^TWII")
    sp_val, sp_chg, sp_pct = get_market_index("^GSPC")
    dj_val, dj_chg, dj_pct = get_market_index("^DJI")
    nas_val, nas_chg, nas_pct = get_market_index("^IXIC")

    with col1: st.metric(label="🇹🇼 台灣加權指數 (^TWII)", value=f"{tw_val:,.2f}" if tw_val else "更新中...", delta=f"{tw_chg:+.2f} ({tw_pct:+.2f}%)" if tw_val else "--")
    with col2: st.metric(label="🇺🇸 標普 500 指數 (^GSPC)", value=f"{sp_val:,.2f}" if sp_val else "載入中...", delta=f"{sp_chg:+.2f} ({sp_pct:+.2f}%)" if sp_val else "--")
    with col3: st.metric(label="🇺🇸 道瓊工業指數 (^DJI)", value=f"{dj_val:,.2f}" if dj_val else "載入中...", delta=f"{dj_chg:+.2f} ({dj_pct:+.2f}%)" if dj_val else "--")
    with col4: st.metric(label="🇺🇸 那斯達克指數 (^IXIC)", value=f"{nas_val:,.2f}" if nas_val else "載入中...", delta=f"{nas_chg:+.2f} ({nas_pct:+.2f}%)" if nas_val else "--")

    st.markdown("---")

    st.sidebar.markdown("### 🧠 系統 API 金鑰設定")
    api_key_input = ""
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip() != "":
            api_key_input = st.secrets["GEMINI_API_KEY"].strip()
            st.sidebar.success("🔒 Gemini API 金鑰已從雲端安全盾自動加載")
    except Exception:
        pass

    if not api_key_input:
        api_key_input = st.sidebar.text_input("請輸入您的 Gemini API Key：", type="password")

    st.sidebar.markdown("### 🎛️ 個人自訂看盤面板")
    selected_tf = st.sidebar.selectbox("⏱️ 看盤 K 線時間軸級別", options=["1分鐘", "5分鐘", "30分鐘", "60分鐘", "1日", "1週", "1個月", "1季", "半年", "1年"], index=4)
    tf_mapping = {
        "1分鐘": {"p": "7d", "i": "1m"}, "5分鐘": {"p": "7d", "i": "5m"}, "30分鐘": {"p": "30d", "i": "30m"}, "60分鐘": {"p": "60d", "i": "60m"},
        "1日": {"p": "2y", "i": "1d"}, "1週": {"p": "5y", "i": "1wk"}, "1個月": {"p": "max", "i": "1mo"}, 
        "1季": {"p": "max", "i": "3mo"}, "半年": {"p": "max", "i": "3mo"}, "1年": {"p": "max", "i": "3mo"}
    }
    cfg = tf_mapping[selected_tf]

    with st.sidebar.expander("🛠️ 均線指標快速視窗設定", expanded=True):
        preset_ma = st.selectbox("📊 選擇均線快捷套組", options=["標準視窗 (5日 / 10日 / 20日)", "極短線 (3日 / 5日 / 10日)", "波段趨勢 (20日 / 60日 / 120日)", "自訂均線配置"], index=0)
        default_days = [5, 10, 20, 60, 240] if preset_ma == "標準視窗 (5日 / 10日 / 20日)" else ([3, 5, 10, 20, 60] if preset_ma == "極短線 (3日 / 5日 / 10日)" else [20, 60, 120, 240, 500])
        default_actives = [True, True, True, False, False]
        personal_ma_configs = []
        default_colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3"]

        for i in range(1, 6):
            col_show, col_p, col_c = st.columns([0.8, 1.4, 0.8])
            with col_show: is_active = st.checkbox(f"MA{i}", value=default_actives[i-1], key=f"ma_active_{i}")
            with col_p: ma_p = st.number_input(f"天數{i}", min_value=1, max_value=500, value=int(default_days[i-1]), label_visibility="collapsed", key=f"personal_ma_p_{i}")
            with col_c: ma_c = str(st.color_picker(f"C{i}", value=default_colors[i-1], label_visibility="collapsed", key=f"personal_ma_c_{i}"))
            if is_active: personal_ma_configs.append({"period": int(ma_p), "color": ma_c})

    AUTO_TW_UNIVERSE = [
        '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2382.TW', 
        '3008.TW', '2303.TW', '2881.TW', '2882.TW', '2891.TW', 
        '1301.TW', '1303.TW', '2002.TW', '2207.TW', '2327.TW', 
        '2357.TW', '2379.TW', '2395.TW', '2408.TW', '2603.TW',
        '3293.TWO', '6121.TWO', '8069.TWO', '5483.TWO', '8299.TWO',
        '3105.TWO', '3529.TWO', '6488.TWO', '4966.TWO', '3264.TWO'
    ]
    
    AUTO_US_UNIVERSE = [
        'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 
        'META', 'TSLA', 'TSM', 'AMD', 'AVGO', 
        'QCOM', 'INTC', 'NFLX', 'SMCI', 'ASML', 
        'COST', 'WMT', 'JPM', 'V', 'LLY'
    ]
    
    STOCK_NAME_MAP = {
        '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '2308.TW': '台達電',
        '2382.TW': '廣達', '3008.TW': '大立光', '2303.TW': '聯電', '2881.TW': '富邦金',
        '2882.TW': '國泰金', '2891.TW': '中信金', '1301.TW': '台塑', '1303.TW': '南亞',
        '2002.TW': '中鋼', '2207.TW': '和泰車', '2327.TW': '國巨', '2357.TW': '華碩',
        '2379.TW': '瑞昱', '2395.TW': '研華', '2408.TW': '南亞科', '2603.TW': '長榮',
        '3293.TWO': '鈊象', '6121.TWO': '新普', '8069.TWO': '元太', '5483.TWO': '中美晶',
        '8299.TWO': '群聯', '3105.TWO': '穩懋', '3529.TWO': '力旺', '6488.TWO': '環球晶',
        '4966.TWO': '譜瑞-KY', '3264.TWO': '欣銓',
        'AAPL': '蘋果公司', 'NVDA': '輝達', 'MSFT': '微軟', 'GOOGL': '谷歌',
        'AMZN': '亞馬遜', 'META': '臉書META', 'TSLA': '特斯拉', 'TSM': '台積電ADR',
        'AMD': '超微半導體', 'AVGO': '博通', 'QCOM': '高通', 'INTC': '英特爾',
        'NFLX': '網飛', 'SMCI': '美超微', 'ASML': '艾司摩爾', 'COST': '好市多',
        'WMT': '沃爾瑪', 'JPM': '摩根大通', 'V': 'VISA卡', 'LLY': '禮來藥廠'
    }

    def calculate_chip_and_backtest(ticker_name, df, timeframe_name):
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df.dropna(subset=['Close', 'Low', 'Volume']).copy()
        
        # 避開不必要的 Resample，防止時間軸被擠壓成單一 K 棒
        if timeframe_name == "半年": df = df.resample('6ME').last().dropna()
        elif timeframe_name == "1年": df = df.resample('12ME').last().dropna()
        
        if df.empty or len(df) < 5: return df, "盤後快取中", 0.0
        
        closes = df['Close'].astype(float).to_numpy().flatten()
        vols = df['Volume'].astype(float).to_numpy().flatten()
        vol_ma20 = pd.Series(vols).rolling(20, min_periods=1).mean().to_numpy()

        current_trend = 0.0
        sim_net = np.zeros(len(vols))
        
        for i in range(len(vols)):
            pct_change = (closes[i] - closes[i-1]) / closes[i-1] if (i > 0 and closes[i-1] > 0) else 0.0
            vol_ratio = (vols[i] / vol_ma20[i]) if vol_ma20[i] > 0 else 1.0
            signal = pct_change * min(vol_ratio, 3.0) * 100.0
            
            current_trend = 0.85 * current_trend + signal
            sim_net[i] = np.clip(current_trend, -6.0, 6.0)
            
        df['籌碼集中度'] = sim_net
        df['5日差值'] = df['籌碼集中度'].diff(1)
        diff_values = df['5日差值'].dropna().to_numpy().flatten()
        latest_val = float(df['籌碼集中度'].to_numpy().flatten()[-1])
        
        count = 0
        if len(diff_values) > 0:
            is_positive = diff_values[-1] > 0
            for val in reversed(diff_values):
                if (val > 0) == is_positive and val != 0: count += 1
                else: break
            status = f"🔥 連續 {count} 日上升" if is_positive else f"📉 連續 {count} 日下降"
        else: status = "🔄 區間震盪洗盤"
        
        return df, f"{latest_val:+.2f}% ({status})", latest_val

    tabs_created = False
    df_chart = None
    if selected_stock:
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT book_value, roe, gvi, close, bv_source, roe_source, updated_at FROM market_data WHERE ticker=?", (selected_stock,))
            db_row = c.fetchone()
            conn.close()

            needs_sync = False
            was_missing_in_db = False

            if not db_row:
                needs_sync = True
                was_missing_in_db = True
            elif not is_cache_valid(db_row[6]):
                needs_sync = True

            if needs_sync:
                with st.spinner(f"📥 資料庫未發現【{selected_stock}】或數據已逾期，正自動啟動線上下載並更新資料庫..."):
                    signature_save_to_db(selected_stock)
                
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("SELECT book_value, roe, gvi, close, bv_source, roe_source, updated_at FROM market_data WHERE ticker=?", (selected_stock,))
                db_row = c.fetchone()
                conn.close()

            df_chart = yf.download(selected_stock, period=cfg["p"], interval=cfg["i"], progress=False)
            if len(df_chart) > 0:
                if isinstance(df_chart.columns, pd.MultiIndex): df_chart.columns = df_chart.columns.get_level_values(0)
                df_chart, chip_status_text, _ = calculate_chip_and_backtest(selected_stock, df_chart, selected_tf)
                
                # 📅 日期格式適應性調整
                if "分鐘" in selected_tf:
                    date_strings = df_chart.index.strftime('%m-%d %H:%M').tolist()
                else:
                    date_strings = df_chart.index.strftime('%Y-%m-%d').tolist()

                c_name = STOCK_NAME_MAP.get(selected_stock, selected_stock)
                price_val = float(df_chart['Close'].to_numpy().flatten()[-1])
                is_tw_stock = ".TW" in selected_stock or ".TWO" in selected_stock

                bv_val = float(db_row[0]) if (db_row and db_row[0] is not None and db_row[0] > 0) else None
                roe_val = parse_roe(db_row[1]) if (db_row and db_row[1] is not None) else None
                bv_src = db_row[4] if (db_row and db_row[4]) else "無數據"
                roe_src = db_row[5] if (db_row and db_row[5]) else "無數據"
                cache_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(db_row[6])) if (db_row and db_row[6]) else "未設定"

                st.markdown("### 🎛️ 戰術面板模式切換")
                panel_mode = st.radio("選擇看盤模式：", ["📈 操盤/動能模式 (專注量價與短線動能)", "🏦 存股/價值模式 (專注基本面與估值)"], horizontal=True, label_visibility="collapsed")
                
                # 計算動能指標 (安全防呆)
                prev_close = float(df_chart['Close'].iloc[-2]) if len(df_chart) > 1 else price_val
                change = price_val - prev_close
                change_pct = (change / prev_close) * 100 if prev_close > 0 else 0.0
                
                vol_today = float(df_chart['Volume'].iloc[-1]) if len(df_chart) > 0 else 0
                vol_ma5 = float(df_chart['Volume'].rolling(5).mean().iloc[-1]) if len(df_chart) >= 5 else vol_today
                vol_ratio = (vol_today / vol_ma5) if vol_ma5 > 0 else 1.0
                
                ma5_val = float(df_chart['Close'].rolling(5).mean().iloc[-1]) if len(df_chart) >= 5 else price_val
                bias5 = ((price_val - ma5_val) / ma5_val) * 100 if ma5_val > 0 else 0.0
                
                high_today = float(df_chart['High'].iloc[-1]) if len(df_chart) > 0 else price_val
                low_today = float(df_chart['Low'].iloc[-1]) if len(df_chart) > 0 else price_val
                
                # 💡 修正：將變數計算獨立移出 UI 判斷區塊外，確保下方雷達圖程式永遠能讀取到 gvi_val
                if price_val > 0 and bv_val is not None and roe_val is not None:
                    pb_val = price_val / bv_val
                    gvi_val = (bv_val / price_val) * ((1 + roe_val) ** 5)
                    gvi_str = f"{gvi_val:.4f}"
                    roe_str = f"{roe_val * 100:.2f}%"
                    bv_str = f"{bv_val:,.2f} 元" if is_tw_stock else f"${bv_val:,.2f}"
                    pb_str = f"{pb_val:.2f} 倍"
                else:
                    pb_val = None
                    gvi_val = None
                    gvi_str = "資料不足"
                    roe_str = "資料不足"
                    bv_str = "資料不足"
                    pb_str = "資料不足"

                if "價值" in panel_mode:
                    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                    with gc1: st.metric(label=f"💰 當前現價 ({selected_stock})", value=f"{price_val:,.2f} 元" if is_tw_stock else f"${price_val:,.2f}"); st.caption(f"📢 即時報價 ({cache_time_str})")
                    
                    if bv_val is not None:
                        with gc2: st.metric(label="👑 GVI 成長價值", value=gvi_str); st.caption("📢 最新重算")
                        with gc3: st.metric(label="📊 ROE 股東權益報酬率", value=roe_str); st.caption(f"📢 {roe_src}")
                        with gc4: st.metric(label="📖 每股淨值", value=bv_str); st.caption(f"📢 {bv_src}")
                        with gc5: st.metric(label="⚖️ 股價淨值比 (PB)", value=pb_str); st.caption("📢 溢價程度")
                    else:
                        with gc2: st.info("⚠️ 缺乏財報淨值數據，系統自動隱藏 PB 與相關估值欄位。建議切換至「操盤/動能模式」觀看。")
                else:
                    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                    with gc1: st.metric(label=f"⚡ 當前現價 ({selected_stock})", value=f"{price_val:,.2f}", delta=f"{change:+.2f} ({change_pct:+.2f}%)")
                    with gc2: st.metric(label="🌊 今日成交量", value=f"{vol_today:,.0f}", delta=f"量比: {vol_ratio:.2f}x", delta_color="normal" if vol_ratio >= 1 else "off")
                    with gc3: st.metric(label="🎯 5日均線乖離率", value=f"{bias5:+.2f}%", delta="偏離大留意回檔" if abs(bias5) > 5 else "乖離正常", delta_color="inverse" if abs(bias5) > 5 else "off")
                    with gc4: st.metric(label="🔺 今日最高價", value=f"{high_today:,.2f}")
                    with gc5: st.metric(label="🔻 今日最低價", value=f"{low_today:,.2f}")

                # ======= 升級：計算三大策略所需數值與交叉分析 =======
                L = len(df_chart)
                close_today = price_val
                high_252d = close_today * 1.05 if L < 50 else float(df_chart['Close'].tail(min(252, L)).max())
                momentum_score = float((close_today / high_252d) * 100) if high_252d > 0 else 0.0

                stock_obj_main = yf.Ticker(selected_stock)
                info_obj_main = stock_obj_main.info if hasattr(stock_obj_main, 'info') and stock_obj_main.info else {}
                
                fcf_main = info_obj_main.get('freeCashflow')
                mcap_main = info_obj_main.get('marketCap')
                if (fcf_main is None or np.isnan(fcf_main)) and mcap_main and mcap_main > 0:
                    try:
                        cf_main = stock_obj_main.quarterly_cashflow
                        if cf_main.empty: cf_main = stock_obj_main.cashflow
                        if not cf_main.empty:
                            ocf, capex = None, None
                            for ocf_k in ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities']:
                                if ocf_k in cf_main.index:
                                    s = cf_main.loc[ocf_k].dropna()
                                    ocf = float(s.iloc[:4].sum()) if len(s) >= 4 else float(s.iloc[0]) * 4
                                    break
                            for cap_k in ['Capital Expenditure', 'Capital Expenditures']:
                                if cap_k in cf_main.index:
                                    s = cf_main.loc[cap_k].dropna()
                                    capex = float(s.iloc[:4].sum()) if len(s) >= 4 else float(s.iloc[0]) * 4
                                    break
                            if ocf is not None:
                                capex_val = abs(float(capex)) if capex is not None else 0.0
                                fcf_main = float(ocf) - capex_val
                    except Exception:
                        pass
                
                fcf_yield_main = (float(fcf_main) / float(mcap_main) * 100.0) if (fcf_main and mcap_main and float(mcap_main) > 0) else None
                
                peg_raw_main = info_obj_main.get('pegRatio')
                peg_val_main = float(peg_raw_main) if (peg_raw_main and not np.isnan(peg_raw_main) and peg_raw_main > 0) else None
                if peg_val_main is None:
                    try:
                        pe_val = info_obj_main.get('trailingPE') or info_obj_main.get('forwardPE')
                        growth_val = info_obj_main.get('earningsGrowth') or info_obj_main.get('earningsQuarterlyGrowth')
                        if pe_val and growth_val and float(pe_val) > 0 and float(growth_val) > 0:
                            peg_val_main = float(pe_val) / (float(growth_val) * 100.0)
                    except Exception:
                        pass

                scores = {}
                reasons = {}

                # 1. GVI 價值策略評分
                if gvi_val is not None and gvi_val >= 0.20:
                    scores['GVI 價值雷達'] = 90 if gvi_val >= 0.35 else 75
                    reasons['GVI 價值雷達'] = f"GVI={gvi_val:.4f}，具備高 ROE 與足夠的安全邊際。"
                else:
                    scores['GVI 價值雷達'] = 40
                    reasons['GVI 價值雷達'] = "估值偏貴、安全邊際不足或因庫藏股致淨值為負。"

                # 2. 動能突破策略評分
                if momentum_score >= 80:
                    scores['動能突破雷達'] = 95 if momentum_score >= 95 else 85
                    reasons['動能突破雷達'] = f"創高距離={momentum_score:.1f}%，技術面強勢突破，不受財報滯後影響。"
                else:
                    scores['動能突破雷達'] = 50
                    reasons['動能突破雷達'] = "技術面處於整理或弱勢區間。"

                # 3. QARP 現金流策略評分
                if fcf_yield_main is not None and fcf_yield_main >= 2.5 and (peg_val_main is None or peg_val_main <= 1.5):
                    scores['QARP 現金流雷達'] = 90
                    reasons['QARP 現金流雷達'] = f"自由現金流收益率={fcf_yield_main:.2f}%，現金流充沛且成長合理。"
                else:
                    scores['QARP 現金流雷達'] = 45
                    reasons['QARP 現金流雷達'] = "現金流收益率偏低或 PEG 過高。"

                # 選出最適策略
                best_strategy = max(scores, key=scores.get)
                best_score = scores[best_strategy]
                
                if best_score >= 90:
                    valuation_color = "#00cc66"
                elif best_score >= 75:
                    valuation_color = "#2baf2b"
                elif best_score >= 60:
                    valuation_color = "#ff9900"
                else:
                    valuation_color = "#cc0000"

                fin_src_display = f"BV: {bv_src} / ROE: {roe_src}"

                advice = f"🎯 **最適推薦策略：【{best_strategy}】（匹配度：{best_score}分）**<br><br>"
                advice += f"📌 **採用數據源**：{fin_src_display}<br>"
                if was_missing_in_db:
                    advice += f"📡 **備援提示**：自動啟動多層次財報遞補機制以確保診斷不中斷。<br>"
                advice += f"🔍 **交叉分析摘要**：<br>"
                for strat, sc in scores.items():
                    icon = "✅" if strat == best_strategy else "🔹"
                    advice += f"&nbsp;&nbsp;&nbsp;&nbsp;{icon} **{strat}** ({sc}分)：{reasons[strat]}<br>"
                    
                advice += f"<br>💡 **操盤手執行建議**：<br>"
                if best_strategy == 'GVI 價值雷達' and best_score >= 75:
                    advice += "&nbsp;&nbsp;&nbsp;&nbsp;當前標的具備極佳價值邊際，建議採**分批逢低布局**或使用下方 **Tab 3 網格再平衡** 鎖定長期複利。"
                elif best_strategy == '動能突破雷達' and best_score >= 80:
                    advice += "&nbsp;&nbsp;&nbsp;&nbsp;價格動能強勁，建議順勢操作！可搭配下方 **Tab 4 均線/KD/RSI 動能回測條件** 嚴守移動停損。"
                elif best_strategy == 'QARP 現金流雷達' and best_score >= 80:
                    advice += "&nbsp;&nbsp;&nbsp;&nbsp;公司現金流極度健全，適合美股巨頭或高品質成長股，建議以**波段持有**搭配現金流指標觀察。"
                else:
                    advice += "&nbsp;&nbsp;&nbsp;&nbsp;三大核心策略目前皆無強烈進場訊號，建議**暫時觀望**或縮小部位應對。"

                # ======= 視覺化圖表整合 (Radar Chart) =======
                diag_c1, diag_c2 = st.columns([1.5, 1])
                with diag_c1:
                    st.markdown(
                        f"<div style='background-color:rgba(30,30,30,0.7); padding:14px 18px; border-left:6px solid {valuation_color}; border-radius:4px; margin-bottom:15px; height: 100%;'>"
                        f"<h5 style='margin:0; color:white;'>⚖️ 華爾街三大核心策略交叉診斷（智能備援版）</h5>"
                        f"<p style='margin:10px 0 0 0; font-size:14px; color:#cccccc;'>{advice}</p>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                with diag_c2:
                    categories = ['GVI 價值雷達', '動能突破雷達', 'QARP 現金流雷達']
                    radar_scores = [scores['GVI 價值雷達'], scores['動能突破雷達'], scores['QARP 現金流雷達']]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_scores,
                        theta=categories,
                        fill='toself',
                        name=c_name,
                        line_color=valuation_color,
                        fillcolor=valuation_color.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in valuation_color else f"{valuation_color}33" 
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(128,128,128,0.2)"),
                            angularaxis=dict(gridcolor="rgba(128,128,128,0.2)")
                        ),
                        showlegend=False,
                        margin=dict(l=30, r=30, t=30, b=30),
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                st.info(f"🔮 【{c_name}】{selected_tf} 即時籌碼動能判定：{chip_status_text}")

                if st.button(f"🧠 啟動 Gemini AI 分析【{c_name}】個股綜合投資價值", use_container_width=True, type="primary"):
                    if not api_key_input: 
                        st.error("⚠️ 請先在左側邊欄輸入 Gemini API Key！")
                    else:
                        with st.spinner(f"🤖 精算 {c_name} 報告中..."):
                            try:
                                ai_prompt = f"請針對個股:{selected_stock}({c_name})，當前量化指標：GVI={gvi_str}, ROE={roe_str}, PB={pb_str}。經過三核心策略交叉分析，最適策略為【{best_strategy}】(匹配度{best_score}分)。請300字內提供繁體中文投資報告。"
                                success_report = generate_content_with_retry(ai_prompt, api_key_input)
                                st.session_state[f"ai_report_{selected_stock}"] = success_report
                            except Exception as ai_e: 
                                st.error(f"Google 雲端機房繁忙或金鑰異常: {ai_e}")
                
                if f"ai_report_{selected_stock}" in st.session_state:
                    st.success(f"📋 AI 【{c_name}】核心投資分析報告")
                    st.markdown(st.session_state[f"ai_report_{selected_stock}"])
                    st.markdown("---")
        except Exception as e:
            st.error(f"數據載入異常：{e}")

    try:
        if df_chart is not None and not df_chart.empty:
            csv_data = df_chart.to_csv().encode('utf-8-sig')
            st.download_button(
                label=f"📥 一鍵下載【{c_name}】歷史 K 線資料 (CSV)",
                data=csv_data,
                file_name=f"{selected_stock}_historical_data.csv",
                mime='text/csv',
                use_container_width=True
            )

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📊 彩色 K 線圖與成交量",
            "🧭 籌碼集中度分析",
            "🤖 網格自動生成器",
            "⚖️ 網格複利 vs 買進持有對比",
            "🧪 技術指標自訂策略回測",
            "💸 配息複利策略",
            "🚀 華爾街機構級三核心策略雷達",
            "🏦 台股三大法人獨立選股策略",
            "📊 當日全市場三大法人佔股本比 Top 10",
        ])
        tabs_created = True
        if df_chart is None or df_chart.empty:
            st.error(f"❌ 無此標的或無法取得數據：【{selected_stock}】，請檢查股票代碼是否正確。")
        else:
            
            # ==============================================================================
            # 【Tab 1: Plotly 雙子圖原生 K 線圖 + 成交量 + 完整畫線與文字工具箱】
            # ==============================================================================
            with tab1:
                col_title, col_draw_color, col_draw_width, col_draw_clear = st.columns([2.5, 1, 1, 1])
                with col_title:
                    ma_display_html = "<div style='background-color:rgba(20,20,20,0.8); padding:4px 8px; border:1px solid #444; border-radius:6px; display:inline-block; font-size:12px; color:white; vertical-align:middle;'>"
                    for ma in personal_ma_configs:
                        p, c = ma["period"], ma["color"]
                        ma_series = df_chart['Close'].rolling(window=p).mean().dropna()
                        if not ma_series.empty:
                            latest_ma_val = ma_series.to_numpy().flatten()[-1]
                            ma_display_html += f"<span style='color:{c}; font-weight:bold; margin-right:8px;'>■ {p}MA: {latest_ma_val:,.2f}</span>"
                    ma_display_html += "</div>"
                    st.markdown(f"### 📊 【{c_name}】專業 K 線圖與成交量 {ma_display_html}", unsafe_allow_html=True)
                
                with col_draw_color:
                    draw_color = st.color_picker("🎨 自訂畫線/文字顏色", value="#FF3333", key="draw_line_color_picker")
                with col_draw_width:
                    draw_line_width = st.slider("線條粗細", min_value=0.5, max_value=10.0, value=2.5, step=0.5, key=f"draw_line_width_{selected_stock}",
                                                help="設定接下來繪製的趨勢線、矩形與圓形邊框粗細。")
                with col_draw_clear:
                    clear_shapes = st.button("🧹 清除全部畫線", key=f"clear_shapes_{selected_stock}_{selected_tf}", use_container_width=True)
                if clear_shapes:
                    clear_key = f"shape_clear_revision_{selected_stock}_{selected_tf}"
                    st.session_state[clear_key] = int(st.session_state.get(clear_key, 0)) + 1
                
                # 📊 建立 2 行 1 列雙子圖
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.03, 
                    row_heights=[0.75, 0.25]
                )
                
                # 1️⃣ 上半部：主圖 Candlestick K棒
                fig.add_trace(go.Candlestick(
                    x=date_strings, 
                    open=df_chart['Open'].to_numpy().flatten().tolist(), 
                    high=df_chart['High'].to_numpy().flatten().tolist(), 
                    low=df_chart['Low'].to_numpy().flatten().tolist(), 
                    close=df_chart['Close'].to_numpy().flatten().tolist(), 
                    name="K線",
                    increasing_line_color='#ef5350', # 漲紅
                    decreasing_line_color='#26a69a', # 跌綠
                    hovertext=[f"日期：{d}" for d in date_strings]
                ), row=1, col=1)
                
                # 均線繪製
                for ma in personal_ma_configs:
                    p, c = ma["period"], ma["color"]
                    ma_series = df_chart['Close'].rolling(window=p).mean().dropna()
                    if not ma_series.empty:
                        ma_list = ma_series.to_numpy().flatten().tolist()
                        fig.add_trace(go.Scatter(
                            x=date_strings[-len(ma_list):], 
                            y=ma_list, 
                            mode='lines', 
                            name=f'{p}MA', 
                            line=dict(color=c, width=1.5)
                        ), row=1, col=1)

                # 2️⃣ 下半部：副圖 成交量 Bar 柱狀圖
                open_arr = df_chart['Open'].to_numpy().flatten()
                close_arr = df_chart['Close'].to_numpy().flatten()
                vol_colors = ['#ef5350' if c >= o else '#26a69a' for o, c in zip(open_arr, close_arr)]
                
                fig.add_trace(go.Bar(
                    x=date_strings,
                    y=df_chart['Volume'].to_numpy().flatten().tolist(),
                    name="成交量",
                    marker_color=vol_colors,
                    opacity=0.8
                ), row=2, col=1)

                # 🚀 排版設定與畫線/文字預設色彩配置
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    uirevision=f"chart-{selected_stock}-{selected_tf}",
                    editrevision=int(st.session_state.get(f"shape_clear_revision_{selected_stock}_{selected_tf}", 0)),
                    height=600, 
                    margin=dict(l=10, r=10, t=25, b=10), 
                    dragmode='pan', 
                    showlegend=False,
                    xaxis=dict(
                        type='category',
                        showgrid=True, 
                        gridcolor="rgba(128,128,128,0.2)",
                        nticks=10
                    ),
                    xaxis2=dict(
                        type='category',
                        showgrid=True,
                        gridcolor="rgba(128,128,128,0.2)",
                        nticks=10
                    ),
                    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", side="right"),
                    yaxis2=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", side="right", title="量"),
                    newshape=dict(
                        line=dict(color=draw_color, width=draw_line_width),
                        fillcolor=draw_color,
                        opacity=0.6
                    )
                )
                
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    key=f"candlestick_chart_{selected_stock}_{selected_tf}",
                    config={
                        'modeBarButtonsToAdd': [
                            'drawline',       # 直線工具
                            'drawopenpath',   # 畫筆工具
                            'drawrect',       # 矩形
                            'drawcircle',     # 圓形
                            'drawtext',       # 📝 文字標記工具
                            'eraseshape'      # 🧹 橡皮擦擦除工具
                        ],
                        'displayModeBar': True, # 強制固定顯示右上角 ModeBar 工具列
                        'displaylogo': False,
                        'editable': True,       # 開放圖上文字與形狀可點擊編輯
                        'scrollZoom': True,
                        'responsive': True
                    }
                )

            with tab6:
                render_dividend_tab(st, default_ticker=selected_stock)
            
            with tab2:
                st.markdown("### 🧭 籌碼集中度分析")
                st.caption("四項觀察：內部人持股、大戶與散戶持股、主力買賣與買賣家數差、籌碼集中度。來源更新頻率不同，請以各指標標示日期為準。")
                if selected_stock.endswith(('.TW', '.TWO')):
                    with st.spinner('讀取官方法人交易與已發行股數資料…'):
                        inst_trades, inst_shares, inst_errors = fetch_institutional_market_data()
                    stock_code = selected_stock.replace('.TWO','').replace('.TW','')
                    shares = inst_shares.get(stock_code)
                    insider = fetch_mops_insider_holding(selected_stock)
                    tdcc = None
                    tdcc_error = None
                    try:
                        with st.spinner('讀取集保每週股權分散資料…'):
                            tdcc = fetch_tdcc_concentration(selected_stock)
                    except Exception as exc:
                        tdcc_error = str(exc)

                    st.markdown("#### 1. 內部人持股（MOPS 月資料）")
                    if insider and shares:
                        insider_pct = insider['shares'] / shares * 100
                        st.metric(f"內部人及關係人持股比｜{insider['date']}", f"{insider_pct:.2f}%", help="MOPS 董監事持股餘額彙總股數 ÷ 官方已發行普通股數。資料為月頻，非即時持股。")
                    else:
                        st.info("MOPS 尚未回傳可辨識的持股餘額，或缺少已發行普通股數。可由 MOPS 個別公司頁面確認最新月份。")

                    st.markdown("#### 2. 大戶與散戶持股（集保週資料）")
                    if tdcc:
                        big_col, retail_col, big_change_col = st.columns(3)
                        with big_col: st.metric(f"大戶 ≥ 400 張｜{tdcc['date']}", f"{tdcc['big_pct']:.2f}%")
                        with retail_col: st.metric("散戶 ≤ 5 張", f"{tdcc['retail_pct']:.2f}%")
                        with big_change_col: st.metric("大戶週變化", f"{tdcc['big_change_pp']:+.2f} 個百分點" if tdcc['big_change_pp'] is not None else "缺少前週資料")
                    else:
                        st.info("目前無法取得此股集保戶股權分散資料。")
                        if tdcc_error: st.caption(f"集保來源暫時無法連線：{tdcc_error}")

                    st.markdown("#### 3. 主力買賣超與買賣家數差")
                    broker_file = st.file_uploader("上傳此股當日券商分點買賣 CSV（需含證券代號、買進股數、賣出股數）", type=['csv'], key=f'broker_csv_{selected_stock}')
                    broker_summary = summarize_broker_branch_csv(broker_file, stock_code) if broker_file else None
                    if broker_summary and broker_summary.get('error'):
                        st.warning(broker_summary['error'])
                    elif broker_summary:
                        broker_cols = st.columns(4)
                        with broker_cols[0]: st.metric("前 15 大買超分點淨買", f"{broker_summary['top_net_buy']:,.0f} 股")
                        with broker_cols[1]: st.metric("前 15 大賣超分點淨賣", f"{broker_summary['top_net_sell']:,.0f} 股")
                        with broker_cols[2]: st.metric("買超分點家數", f"{broker_summary['buyers']:,}")
                        with broker_cols[3]: st.metric("買賣分點家數差", f"{broker_summary['count_diff']:+,}")
                        if shares:
                            st.caption(f"前 15 大買超分點淨買佔股本 {broker_summary['top_net_buy']/shares*100:.4f}%；前 15 大賣超分點淨賣佔股本 {broker_summary['top_net_sell']/shares*100:.4f}%。")
                    else:
                        st.info("券商分點交易檔需另行取得後上傳；此報表不會把三大法人淨買超冒充為主力分點資料。")

                    st.markdown("#### 4. 籌碼集中度")
                    if tdcc:
                        concentration_cols = st.columns(3)
                        concentration = tdcc['big_pct'] - tdcc['retail_pct']
                        with concentration_cols[0]: st.metric("大戶－散戶持股差", f"{concentration:+.2f} 個百分點")
                        with concentration_cols[1]: st.metric("大戶持股週變化", f"{tdcc['big_change_pp']:+.2f} 個百分點" if tdcc['big_change_pp'] is not None else "缺少前週資料")
                        with concentration_cols[2]: st.metric("散戶持股週變化", f"{tdcc['retail_change_pp']:+.2f} 個百分點" if tdcc['retail_change_pp'] is not None else "缺少前週資料")
                        st.caption("集中度觀察值定義為大戶持股比減散戶持股比；同時呈現大戶與散戶週變化，供比較趨勢，不是官方發布的單一指數。")
                    else:
                        st.info("集保週資料不足，暫無法計算集中度觀察值。")

                    st.markdown("#### 三大法人買賣佔股本比（每日）")
                    inst_one = build_institutional_screen(inst_trades, inst_shares, [selected_stock], '外資', -100000)
                    if not inst_one.empty:
                        view_cols = ['資料日期', '市場', '外資買進佔股本比(%)', '外資賣出佔股本比(%)', '外資買賣超佔股本比(%)',
                                     '投信買進佔股本比(%)', '投信賣出佔股本比(%)', '投信買賣超佔股本比(%)',
                                     '自營商買進佔股本比(%)', '自營商賣出佔股本比(%)', '自營商買賣超佔股本比(%)']
                        st.dataframe(inst_one[view_cols].style.format({c: '{:+.4f}%' for c in view_cols if c.endswith('(%)')}), use_container_width=True)
                        st.caption("分類口徑：外資含外陸資、不含外資自營商；自營商買賣股數合計自行買賣與避險。")
                    else:
                        st.info("目前無此股票的法人資料或官方已發行股數；可能是非交易日、資料尚未更新，或標的非普通股。")
                    if inst_errors:
                        st.caption("部分官方資料來源暫時無法連線，畫面僅顯示可取得的市場資料。")
                else:
                    st.info("集中度分析與三大法人資料目前適用台灣上市 `.TW` 與上櫃 `.TWO` 普通股。")
            
            # ==============================================================================
            # 【Tab 3: 3.12 智慧型動態網格策略與資金自動規劃器 (含三大機制解析與複利警示)】
            # ==============================================================================
            with tab3:
                st.markdown("### 🎛️ 智慧型動態網格策略與資金自動規劃器 (3.12 操盤手教學版)")
                
                with st.expander("📚 點擊展開：網格交易 3 大資金管理機制與適用場景解析", expanded=False):
                    st.markdown("""
                    ##### **1. 固定比例再平衡 (1:1 / 自訂比例)**
                    * **核心機制**：動態維持「股票價值 : 現金」的固定的比例（如 50%:50%）。股票上漲過高即賣出補回現金；股票下跌過深即動用現金買進股票。
                    * **優點**：紀律化控制曝險，下跌時永遠有預備現金可用。
                    * **缺點**：持續單邊大漲時可能過早賣出；持續單邊大跌時會持續消耗預備現金。
                    * **適合場景**：**控制風險**、偏好長期穩定波動、不希望承受全額持股回撤的投資人。

                    ##### **2. 固定股數交易**
                    * **核心機制**：每次觸發目標價時，買進或賣出「完全相同的股數」（如每次 1,000 股）。
                    * **優點**：規則最直觀、容易進行歷史回測與預估手續費。
                    * **缺點**：股價越高時每次交易金額越大，資金壓力呈線性成長；股價極低時交易金額過小。
                    * **適合場景**：**建立簡單易執行的自動化交易規則**。

                    ##### **3. 庫存百分比管理**
                    * **核心機制**：每次買賣當前庫存的一定比例（如每次交易當前庫存的 10%）。
                    * **優點**：庫存部位較大時調整幅度自動放大；庫存縮小（低位）時調整幅度隨之減少，降低爆倉風險。
                    * **缺點**：庫存極低時賣出數量會指數型遞減，可能較難完全出清清倉。
                    * **適合場景**：讓交易規模跟隨現有庫存規模動態調節。
                    """)

                st.markdown("---")

                grid_p = float(price_val)
                is_tw = ".TW" in selected_stock or ".TWO" in selected_stock
                unit_label = "股" if not is_tw else "股 (台股預設)"

                g_top1, g_top2, g_top3 = st.columns(3)
                with g_top1:
                    total_capital = st.number_input("💵 投入總資金 (元/$)", value=100000, step=10000, key="grid_capital")
                with g_top2:
                    grid_strategy = st.selectbox(
                        "🎯 選擇網格核心策略機制", 
                        options=["1. 固定比例再平衡 (例如 1:1)", "2. 固定股數交易", "3. 庫存百分比管理"],
                        key="grid_strat_select"
                    )
                with g_top3:
                    init_stock_ratio = st.slider("⚖️ 初始股票部位佔比 (%)", min_value=10, max_value=90, value=50, step=5, key="grid_init_ratio")

                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1: 
                    input_lower = st.number_input("網格下限價格", value=round(grid_p * 0.80, 2), key="grid_lower")
                with g_col2: 
                    input_upper = st.number_input("網格上限價格", value=round(grid_p * 1.20, 2), key="grid_upper")
                with g_col3: 
                    input_num = st.number_input("規劃總格數", min_value=3, max_value=50, value=10, step=1, key="grid_num")

                init_stock_cash = total_capital * (init_stock_ratio / 100.0)
                init_shares = int(init_stock_cash // grid_p) if grid_p > 0 else 0
                actual_init_stock_val = init_shares * grid_p
                actual_init_cash = total_capital - actual_init_stock_val

                st.info(
                    f"💡 **初始建倉試算**：總資金 {total_capital:,.0f} 元 | "
                    f"建倉購買 **{init_shares:,}** {unit_label} (約 {actual_init_stock_val:,.0f} 元，佔 {actual_init_stock_val/total_capital*100:.1f}%) | "
                    f"保留預備現金 **{actual_init_cash:,.0f}** 元"
                )

                levels = np.linspace(input_lower, input_upper, input_num)
                grid_details = []

                for idx, level in enumerate(levels):
                    action = "🟢 買進掛單" if level < grid_p else ("🔴 賣出掛單" if level > grid_p else "⚪ 當前基準價")
                    
                    if "固定比例" in grid_strategy:
                        est_stock_val = init_shares * level
                        est_total = est_stock_val + actual_init_cash
                        target_stock_val = est_total * (init_stock_ratio / 100.0)
                        diff_val = target_stock_val - est_stock_val
                        
                        if diff_val > 0:
                            trade_desc = f"動用現金買進約 {abs(diff_val):,.0f} 元股票"
                        elif diff_val < 0:
                            trade_desc = f"賣出約 {abs(diff_val):,.0f} 元股票補回現金"
                        else:
                            trade_desc = "平衡狀態"

                    elif "固定股數" in grid_strategy:
                        fixed_shares = max(1, int((init_shares or 1000) / input_num))
                        trade_desc = f"固定交易 {fixed_shares:,} {unit_label} (約 {fixed_shares * level:,.0f} 元)"

                    else:
                        pct = round(100.0 / input_num, 1)
                        trade_desc = f"交易當前庫存之 {pct}% (隨庫存規模動態增減)"

                    grid_details.append({
                        "網格層級": f"Grid #{idx+1:02d}",
                        "目標觸發價": round(level, 2),
                        "偏離現價 (%)": f"{((level - grid_p) / grid_p * 100):+.2f}%",
                        "預計動作": action,
                        "資金/部位管理機制": trade_desc
                    })

                st.dataframe(pd.DataFrame(grid_details), use_container_width=True, height=280)

                st.warning(
                    "⚠️ **操盤手複利與風險特別提示：**\n"
                    "1. **網格獲利 ≠ 長期投資正期望值**：網格在「箱型震盪行情」中最能發揮低買高賣優勢；但在「強勢單邊大漲」時，會因持續賣出股票而落後於買進持有 (Buy & Hold)；在「持續單邊下跌」時，則會逐步買進並累積帳面虧損。\n"
                    "2. **交易成本與稅費影響**：頻繁再平衡會產生手續費與證券交易稅，長期運作下會侵蝕部分複利效益。\n"
                    "3. **驗證建言**：若目標是建立長期複利系統，請切換至 **「Tab 4 策略自訂回測與網格複利對比」**，檢視歷史 **年化報酬率 (CAGR)**、**最大回撤 (MDD)** 與 **扣除稅費後的淨報酬率**。"
                )

            # ==============================================================================
            # 【Tab 4: 3.12 「網格再平衡 vs 買進持有」歷史線路模擬與全指標對比】
            # ==============================================================================
            with tab4:
                st.markdown(f"### ⚖️ 【{c_name}】「動態網格 vs 買進持有」歷史對比 (3.12 版)")
                
                st.markdown("#### ⚖️ 網格三大核心策略 vs 買進持有 (Buy & Hold) 歷史資產對比與手續費精算")
                
                gc_col1, gc_col2, gc_col3 = st.columns(3)
                with gc_col1:
                    sim_capital = st.number_input("💵 模擬初始投入本金 (元/\\$)", value=100000, step=10000, key="sim_cap_input")
                with gc_col2:
                    fee_rate = st.number_input("💸 單邊交易手續費率 (%)", value=0.1425, step=0.01, format="%.4f", key="sim_fee_input") / 100.0
                with gc_col3:
                    tax_rate = st.number_input("🏛️ 賣出證券交易稅率 (%)", value=0.3000, step=0.05, format="%.4f", key="sim_tax_input") / 100.0

                g_opt1, g_opt2 = st.columns(2)
                with g_opt1:
                    grid_sim_strategy = st.selectbox(
                        "🎯 選擇網格模擬核心策略機制",
                        options=["1. 固定比例再平衡 (1:1)", "2. 固定股數交易 (每次買賣初始部位 10%)", "3. 庫存百分比管理 (每次買賣現有庫存 10%)"],
                        key="sim_grid_strat_select"
                    )
                with g_opt2:
                    trigger_pct = st.number_input("📉 網格觸發間距 (%)", min_value=1.0, max_value=50.0, value=5.0, step=1.0, key="sim_grid_trigger") / 100.0

                closes_arr = df_chart['Close'].astype(float).to_numpy()
                
                if len(closes_arr) >= 5:
                    initial_p = closes_arr[0]
                    bh_shares = (sim_capital * (1.0 - fee_rate)) / initial_p
                    bh_asset_curve = bh_shares * closes_arr
                    bh_final_val = float(bh_asset_curve[-1])
                    bh_total_ret = ((bh_final_val - sim_capital) / sim_capital) * 100.0

                    stock_val = sim_capital * 0.5 * (1.0 - fee_rate)
                    cash_val = sim_capital * 0.5
                    grid_shares = stock_val / initial_p
                    grid_asset_curve = []
                    
                    last_trade_p = initial_p
                    fixed_trade_shares = grid_shares * 0.1 # 策略2：固定交易初始持股的 10%
                    
                    for step_idx, p in enumerate(closes_arr):
                        curr_stock_val = grid_shares * p
                        total_val = curr_stock_val + cash_val
                        
                        # 網格價格波動觸發邏輯
                        if p >= last_trade_p * (1.0 + trigger_pct) or p <= last_trade_p * (1.0 - trigger_pct):
                            if "固定比例" in grid_sim_strategy:
                                target_stock_val = total_val * 0.5
                                diff = target_stock_val - curr_stock_val
                                
                                if diff > 0 and cash_val >= diff: # 買進
                                    buy_amt = diff * (1.0 - fee_rate)
                                    cash_val -= diff
                                    grid_shares += buy_amt / p
                                elif diff < 0 and grid_shares >= abs(diff)/p: # 賣出
                                    sell_amt = abs(diff) * (1.0 - fee_rate - tax_rate)
                                    cash_val += sell_amt
                                    grid_shares -= abs(diff) / p
                                    
                            elif "固定股數" in grid_sim_strategy:
                                trade_val = fixed_trade_shares * p
                                if p >= last_trade_p * (1.0 + trigger_pct): # 上漲賣出
                                    if grid_shares >= fixed_trade_shares:
                                        cash_val += trade_val * (1.0 - fee_rate - tax_rate)
                                        grid_shares -= fixed_trade_shares
                                elif p <= last_trade_p * (1.0 - trigger_pct): # 下跌買進
                                    if cash_val >= trade_val:
                                        cash_val -= trade_val
                                        grid_shares += (trade_val * (1.0 - fee_rate)) / p
                                        
                            elif "庫存百分比" in grid_sim_strategy:
                                if p >= last_trade_p * (1.0 + trigger_pct): # 上漲賣出現有 10%
                                    sell_shares = grid_shares * 0.1
                                    if sell_shares > 0:
                                        trade_val = sell_shares * p
                                        cash_val += trade_val * (1.0 - fee_rate - tax_rate)
                                        grid_shares -= sell_shares
                                elif p <= last_trade_p * (1.0 - trigger_pct): # 下跌動用現金 10% 買進
                                    use_cash = cash_val * 0.1
                                    if use_cash > 0:
                                        cash_val -= use_cash
                                        grid_shares += (use_cash * (1.0 - fee_rate)) / p
                                        
                            last_trade_p = p # 更新最新交易參考價
                                
                        grid_asset_curve.append(grid_shares * p + cash_val)

                    grid_final_val = float(grid_asset_curve[-1])
                    grid_total_ret = ((grid_final_val - sim_capital) / sim_capital) * 100.0

                    total_bars = len(closes_arr)
                    years_est = max(0.1, total_bars / 252.0) if "1日" in selected_tf else max(0.1, total_bars / 52.0)
                    
                    bh_cagr = ((bh_final_val / sim_capital) ** (1.0 / years_est) - 1.0) * 100.0 if bh_final_val > 0 else -100.0
                    grid_cagr = ((grid_final_val / sim_capital) ** (1.0 / years_est) - 1.0) * 100.0 if grid_final_val > 0 else -100.0

                    def calc_mdd(curve):
                        arr = np.array(curve)
                        peak = np.maximum.accumulate(arr)
                        drawdown = (arr - peak) / peak
                        return float(np.min(drawdown)) * 100.0

                    bh_mdd = calc_mdd(bh_asset_curve)
                    grid_mdd = calc_mdd(grid_asset_curve)

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("持有策略 最終總資產", f"{bh_final_val:,.0f} 元", delta=f"總報酬 {bh_total_ret:+.2f}%")
                    with m2:
                        st.metric("選定網格 最終總資產", f"{grid_final_val:,.0f} 元", delta=f"總報酬 {grid_total_ret:+.2f}%")
                    with m3:
                        st.metric("CAGR 年化報酬 (買進 vs 網格)", f"{bh_cagr:+.1f}% / {grid_cagr:+.1f}%")
                    with m4:
                        st.metric("MDD 最大回撤 (買進 vs 網格)", f"{bh_mdd:.1f}% / {grid_mdd:.1f}%", delta="風險控制較佳" if abs(grid_mdd) < abs(bh_mdd) else "波動較大")

                    strat_display_name = grid_sim_strategy.split(" ")[1]

                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=date_strings, y=bh_asset_curve, mode='lines', name='單純買進持有 (Buy & Hold)', line=dict(color='#00CC66', width=2)))
                    fig_compare.add_trace(go.Scatter(x=date_strings, y=grid_asset_curve, mode='lines', name=strat_display_name, line=dict(color='#FF9900', width=2, dash='dash')))
                    fig_compare.update_layout(
                        title=f"📈 【{c_name}】歷史走勢下，{strat_display_name} vs 買進持有之資產增長曲線 (扣除交易稅費)",
                        height=380,
                        xaxis=dict(type='category', showgrid=True, gridcolor="rgba(128,128,128,0.2)", nticks=12),
                        yaxis=dict(title="資產總價值 (元)", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                        margin=dict(l=10, r=20, t=40, b=10)
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

            # ==============================================================================
            # 【Tab 5: 3.12 技術指標多空策略與動態停損回測器】
            # ==============================================================================
            with tab5:
                st.markdown(f"### 🧪 【{c_name}】技術指標多空策略與動態停損回測器 (3.12 版)")
                bt_col1, bt_col2 = st.columns(2)
                
                with bt_col1:
                    st.markdown("##### 🟢 進場 (買進) 策略下拉式選單")
                    entry_ma_mode = st.selectbox("1. 均線策略 (MA)", ["停用", "突破均線", "站穩均線上"], index=1, key="bt_entry_ma_sel")
                    entry_ma_p = st.number_input("買進均線天數 (MA)", min_value=1, max_value=240, value=20, key="bt_entry_ma_p")
                    
                    entry_kd_mode = st.selectbox("2. KD 指標策略", ["停用", "黃金交叉"], index=0, key="bt_entry_kd_sel")
                    entry_macd_mode = st.selectbox("3. MACD 指標策略", ["停用", "黃金交叉"], index=0, key="bt_entry_macd_sel")
                    entry_vol_mode = st.selectbox("4. 成交量策略", ["停用", "爆量 (大於20日均量1.5倍)", "爆量 (大於20日均量2.0倍)"], index=0, key="bt_entry_vol_sel")
                    entry_rsi_mode = st.selectbox("5. RSI 指標策略", ["停用", "超賣回升 (RSI向上突破30)", "強勢突破 (RSI向上突破50)"], index=0, key="bt_entry_rsi_sel")
                    entry_bb_mode = st.selectbox("6. 布林通道策略", ["停用", "突破布林上軌", "觸及布林下軌反彈"], index=0, key="bt_entry_bb_sel")
                    
                    entry_match_mode = st.radio("買進訊號判定邏輯", options=["同時滿足所有選定條件 (ALL)", "任一滿足即買進 (ANY)"], index=0, key="bt_match")

                with bt_col2:
                    st.markdown("##### 🔴 離場 (停損/停利) 策略下拉式選單")
                    stop_loss_pct = st.number_input("固定停損幅度 (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="bt_sl")
                    trailing_stop_pct = st.number_input("最高價移動回撤幅度 (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="bt_trail")
                    
                    exit_ma_mode = st.selectbox("1. 均線平倉策略 (MA)", ["停用", "跌破均線"], index=1, key="bt_exit_ma_sel")
                    exit_ma_p = st.number_input("賣出均線天數 (MA)", min_value=1, max_value=240, value=20, key="bt_exit_ma_p")
                    
                    exit_kd_mode = st.selectbox("2. KD 離場策略", ["停用", "死亡交叉"], index=0, key="bt_exit_kd_sel")
                    exit_macd_mode = st.selectbox("3. MACD 離場策略", ["停用", "死亡交叉"], index=0, key="bt_exit_macd_sel")
                    exit_vol_mode = st.selectbox("4. 成交量離場策略", ["停用", "極端爆量倒貨 (大於20日均量2.5倍)"], index=0, key="bt_exit_vol_sel")
                    exit_rsi_mode = st.selectbox("5. RSI 離場策略", ["停用", "超買警戒 (RSI跌破70)"], index=0, key="bt_exit_rsi_sel")
                    exit_bb_mode = st.selectbox("6. 布林通道離場策略", ["停用", "跌破布林下軌", "觸及上軌拉回"], index=0, key="bt_exit_bb_sel")

                if st.button(f"🚀 開始執行【{c_name}】全指標量化回測", type="primary", use_container_width=True, key="run_bt_btn"):
                    entry_conds = {
                        'ma_mode': entry_ma_mode,
                        'ma_p': entry_ma_p,
                        'kd_mode': entry_kd_mode,
                        'macd_mode': entry_macd_mode,
                        'vol_mode': entry_vol_mode,
                        'rsi_mode': entry_rsi_mode,
                        'bb_mode': entry_bb_mode,
                        'match_mode': 'ALL' if "同時" in entry_match_mode else 'ANY'
                    }
                    
                    exit_conds = {
                        'stop_loss_pct': stop_loss_pct,
                        'trailing_stop_pct': trailing_stop_pct,
                        'ma_mode': exit_ma_mode,
                        'ma_p': exit_ma_p,
                        'kd_mode': exit_kd_mode,
                        'macd_mode': exit_macd_mode,
                        'vol_mode': exit_vol_mode,
                        'rsi_mode': exit_rsi_mode,
                        'bb_mode': exit_bb_mode
                    }
                    
                    df_calc = compute_backtest_indicators(df_chart, ma_entry_p=entry_ma_p, ma_exit_p=exit_ma_p)
                    df_trades = run_strategy_backtest(df_calc, entry_conds, exit_conds, date_strings)

                    if not df_trades.empty:
                        st.dataframe(df_trades, use_container_width=True)
                    else:
                        st.warning("⚠️ 在選定區間內未有符合條件的完整交易紀錄。")

    except Exception as ex_tab:
        st.error(f"❌ 畫面渲染異常：{ex_tab}")

    # If stock-chart loading failed before the chart area, keep the strategy tabs available.
    if not tabs_created:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📊 彩色 K 線圖與成交量",
            "🧭 籌碼集中度分析",
            "🤖 網格自動生成器",
            "⚖️ 網格複利 vs 買進持有對比",
            "🧪 技術指標自訂策略回測",
            "💸 配息複利策略",
            "🚀 華爾街機構級三核心策略雷達",
            "🏦 台股三大法人獨立選股策略",
            "📊 當日全市場三大法人佔股本比 Top 10",
        ])
        tabs_created = True

    with tab7:
        st.markdown("---")
        st.markdown("### 🚀 華爾街機構級三核心策略雷達")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            strat_gvi = st.checkbox("開啟 GVI 價值雷達", value=True, key="strat_gvi_check")
            st.caption("優點：結合 ROE 與淨值估值，重視安全邊際。\n\n限制：財報更新較慢，金融或特殊產業比較性較低。")
        with col_s2:
            strat_momentum = st.checkbox("開啟動能突破雷達", value=False, key="strat_momentum_check")
            st.caption("優點：偏向強勢趨勢，可快速反映價格動能。\n\n限制：盤整時容易反覆訊號，追高風險較高。")
        with col_s3:
            strat_qarp = st.checkbox("開啟 QARP 現金流雷達", value=False, key="strat_qarp_check")
            st.caption("優點：同時看自由現金流收益與 PEG，兼顧現金流和成長估值。\n\n限制：現金流與成長資料可能缺漏，週期性公司容易失真。")

        custom_input_pool = st.text_area("✍️ 操盤手自訂觀察代碼掃描區：", value="2330.TW, 3293.TWO, 8069.TWO, NVDA, AAPL")
        custom_scan_list = [c.strip().upper() for c in custom_input_pool.split(",") if c.strip()]

        def load_data_from_sqlite_and_render(filter_list, title, is_custom_mode=False):
            conn = sqlite3.connect(DB_FILE)
            try: df_db = pd.read_sql_query("SELECT * FROM market_data", conn)
            except Exception: df_db = pd.DataFrame()
            conn.close()
        
            for t in filter_list:
                row_matches = df_db[df_db['ticker'] == t] if not df_db.empty else pd.DataFrame()
                if row_matches.empty or not is_cache_valid(row_matches.iloc[0].get('updated_at')):
                    signature_save_to_db(t)
        
            conn = sqlite3.connect(DB_FILE)
            df_db = pd.read_sql_query("SELECT * FROM market_data", conn)
            conn.close()

            if df_db.empty: return
            
            df_filtered = df_db[df_db['ticker'].isin(filter_list)].copy()
            raw_portfolio = []
        
            for idx, row in df_filtered.iterrows():
                t = row['ticker']
                try:
                    ch_name = STOCK_NAME_MAP.get(t, t)
                    df_hist = yf.download(t, period=cfg["p"], interval=cfg["i"], progress=False)
                    if isinstance(df_hist.columns, pd.MultiIndex): df_hist.columns = df_hist.columns.get_level_values(0)
                    if df_hist.index.tz is not None: df_hist.index = df_hist.index.tz_localize(None)
                    df_hist = df_hist.dropna(subset=['Close', 'Volume']).copy()
                    L = len(df_hist)
                
                    close_today = float(row['close']) if (L == 0 or 'Close' not in df_hist.columns) else float(df_hist['Close'].to_numpy().flatten()[-1])
                    high_252d = close_today * 1.05 if L < 50 else float(df_hist['Close'].tail(min(252, L)).max())
                
                    bv_val = float(row['book_value']) if (pd.notnull(row['book_value']) and float(row['book_value']) > 0) else None
                    roe_val = parse_roe(row['roe'])

                    gvi_live = (bv_val / close_today) * ((1 + roe_val) ** 5) if (close_today > 0 and bv_val is not None and roe_val is not None) else None

                    stock_obj = yf.Ticker(t)
                    info_obj = stock_obj.info if hasattr(stock_obj, 'info') and stock_obj.info else {}
                
                    fcf = info_obj.get('freeCashflow')
                    mcap = info_obj.get('marketCap') or row['mcap']

                    if (fcf is None or np.isnan(fcf)) and mcap and mcap > 0:
                        try:
                            cf = stock_obj.quarterly_cashflow
                            if cf.empty: cf = stock_obj.cashflow
                            if not cf.empty:
                                ocf, capex = None, None
                                for ocf_k in ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities']:
                                    if ocf_k in cf.index:
                                        s = cf.loc[ocf_k].dropna()
                                        ocf = float(s.iloc[:4].sum()) if len(s) >= 4 else float(s.iloc[0]) * 4
                                        break
                                for cap_k in ['Capital Expenditure', 'Capital Expenditures']:
                                    if cap_k in cf.index:
                                        s = cf.loc[cap_k].dropna()
                                        capex = float(s.iloc[:4].sum()) if len(s) >= 4 else float(s.iloc[0]) * 4
                                        break
                                if ocf is not None:
                                    capex_val = abs(float(capex)) if capex is not None else 0.0
                                    fcf = float(ocf) - capex_val
                        except Exception:
                            pass

                    fcf_yield = (float(fcf) / float(mcap) * 100.0) if (fcf and mcap and float(mcap) > 0) else None

                    peg_raw = info_obj.get('pegRatio')
                    peg_val = float(peg_raw) if (peg_raw and not np.isnan(peg_raw) and peg_raw > 0) else None

                    if peg_val is None:
                        try:
                            pe_val = info_obj.get('trailingPE') or info_obj.get('forwardPE')
                            growth_val = info_obj.get('earningsGrowth') or info_obj.get('earningsQuarterlyGrowth')
                            if pe_val and growth_val and float(pe_val) > 0 and float(growth_val) > 0:
                                peg_val = float(pe_val) / (float(growth_val) * 100.0)
                        except Exception:
                            pass

                    df_hist, chip_status_text, _ = calculate_chip_and_backtest(t, df_hist, selected_tf)
                
                    raw_portfolio.append({
                        '股票代碼': t, '股票名稱': ch_name, '目前股價': close_today, 
                        'GVI值': gvi_live, 
                        '動能分數': float((close_today / high_252d) * 100) if high_252d > 0 else 0.0, 
                        '自由現金流收益': fcf_yield, 
                        '本益成長比(PEG)': peg_val, 
                        '⚡ 即時籌碼動能': chip_status_text, 
                        'gvi_raw': gvi_live, 
                        'mo_raw': float((close_today / high_252d) * 100) if high_252d > 0 else 0.0, 
                        'fcf_raw': fcf_yield
                    })
                except Exception as e_item: 
                    st.sidebar.caption(f"標的跳過 [{t}]: {e_item}")
                
            if raw_portfolio:
                df_res = pd.DataFrame(raw_portfolio)
            
                if strat_gvi: df_res = df_res[df_res['gvi_raw'].notnull() & (df_res['gvi_raw'] >= 0.20)]
                if strat_momentum: df_res = df_res[df_res['mo_raw'] >= 80.0]
                if strat_qarp: df_res = df_res[df_res['fcf_raw'].notnull() & (df_res['fcf_raw'] >= 2.5) & df_res['本益成長比(PEG)'].notnull() & (df_res['本益成長比(PEG)'] <= 1.5)]
            
                if df_res.empty: df_res = pd.DataFrame(raw_portfolio)
                
                df_res['GVI_Rank'] = df_res['gvi_raw'].fillna(0).rank(pct=True)
                df_res['MO_Rank'] = df_res['mo_raw'].fillna(0).rank(pct=True)
                df_res['FCF_Rank'] = df_res['fcf_raw'].fillna(0).rank(pct=True)
                df_res['綜合分數'] = 0.0
                active_strats = 0
            
                if strat_gvi: df_res['綜合分數'] += df_res['GVI_Rank']; active_strats += 1
                if strat_momentum: df_res['綜合分數'] += df_res['MO_Rank']; active_strats += 1
                if strat_qarp: df_res['綜合分數'] += df_res['FCF_Rank']; active_strats += 1
                if active_strats == 0: df_res['綜合分數'] = df_res['GVI_Rank']; active_strats = 1
            
                df_res['策略評分_num'] = ((df_res['綜合分數'] / active_strats) * 100).round(1)
                df_res = df_res.sort_values(by='綜合分數', ascending=False).head(20).reset_index(drop=True)
            
                df_res['排名'] = [f"第 {i+1} 名" for i in range(len(df_res))]
                df_res['策略評分'] = df_res['策略評分_num'].astype(str) + " 分"
                df_res['GVI價值指標'] = df_res['GVI值'].apply(lambda x: f"{x:.4f}" if (pd.notnull(x) and x is not None) else "資料不足")
                df_res['動能(創高距離)'] = df_res['動能分數'].round(1).astype(str) + "%"
                df_res['自由現金流收益率'] = df_res['自由現金流收益'].apply(lambda x: f"{x:.2f}%" if (pd.notnull(x) and x is not None) else "資料不足")
                df_res['本益成長比(PEG)'] = df_res['本益成長比(PEG)'].apply(lambda x: f"{x:.2f}" if (pd.notnull(x) and x is not None) else "資料不足")
            
                cols = ['排名', '股票代碼', '股票名稱', '目前股價', '策略評分']
                if strat_gvi: cols.append('GVI價值指標')
                if strat_momentum: cols.append('動能(創高距離)')
                if strat_qarp: cols.extend(['自由現金流收益率', '本益成長比(PEG)'])
                if not strat_gvi and not strat_momentum and not strat_qarp:
                    cols.extend(['GVI價值指標', '動能(創高距離)', '自由現金流收益率', '本益成長比(PEG)'])
                cols.append('⚡ 即時籌碼動能')
            
                df_final_view = df_res[cols]
                st.markdown(f"#### {title} (已依勾選核心策略評分動態排名 Top 20)")
                st.dataframe(df_final_view, use_container_width=True)

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🇹🇼 一鍵執行：全自動過濾全台股核心池", type="secondary", use_container_width=True):
                load_data_from_sqlite_and_render(AUTO_TW_UNIVERSE, "🇹🇼 台灣股市核心策略篩選結果")
        with col_btn2:
            if st.button("🇺🇸 一鍵執行：全自動過濾全美股核心池", type="secondary", use_container_width=True):
                load_data_from_sqlite_and_render(AUTO_US_UNIVERSE, "🇺🇸 美國股市核心策略篩選結果")
        with col_btn3:
            if st.button("🚀 執行：自訂名單多因子本地精準過濾", type="primary", use_container_width=True):
                load_data_from_sqlite_and_render(custom_scan_list, "🎯 操盤手自訂名單策略篩選結果", is_custom_mode=True)

    with tab8:
        st.markdown("---")
        st.markdown("### 🏦 台股三大法人獨立選股策略")
        st.caption("只掃描自訂觀察名單中的台股普通股。各法人買進、賣出與淨買賣超股數分別除以官方已發行普通股數，顯示為股本百分比。")
        inst_col1, inst_col2 = st.columns([1, 2])
        with inst_col1:
            inst_investor = st.selectbox("篩選法人", ['外資', '投信', '自營商'], key='inst_screen_investor')
            inst_min_net_pct = st.number_input("淨買賣超佔股本至少 (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.01, format='%.2f', key='inst_min_net_pct')
        with inst_col2:
            st.write("篩選條件：所選法人的「買賣超佔股本比」大於等於設定值。表格仍同時列出外資、投信、自營商三方買進、賣出及買賣超佔股本比。")
        if st.button("🏦 執行法人策略篩選", type='secondary', key='run_inst_screen'):
            taiwan_tickers = [t for t in custom_scan_list if t.endswith(('.TW', '.TWO'))]
            if not taiwan_tickers:
                st.warning("請先在上方自訂觀察名單加入台股代碼，例如 2330.TW、3293.TWO。")
            else:
                with st.spinner("正在取得 TWSE／TPEx 官方法人交易與股數資料…"):
                    inst_trades, inst_shares, inst_errors = fetch_institutional_market_data()
                inst_result = build_institutional_screen(inst_trades, inst_shares, taiwan_tickers, inst_investor, float(inst_min_net_pct))
                if inst_result.empty:
                    st.info("此條件下沒有符合標的，或官方資料暫時缺漏。可調低淨買超門檻或檢查代碼格式。")
                else:
                    inst_display_cols = ['股票代碼', '股票名稱', '市場', '資料日期',
                        '外資買進佔股本比(%)', '外資賣出佔股本比(%)', '外資買賣超佔股本比(%)',
                        '投信買進佔股本比(%)', '投信賣出佔股本比(%)', '投信買賣超佔股本比(%)',
                        '自營商買進佔股本比(%)', '自營商賣出佔股本比(%)', '自營商買賣超佔股本比(%)']
                    pct_cols = [c for c in inst_display_cols if c.endswith('(%)')]
                    st.dataframe(inst_result[inst_display_cols].style.format({c: '{:+.4f}%' for c in pct_cols}), use_container_width=True)
                    st.download_button("下載法人策略結果 CSV", inst_result[inst_display_cols].to_csv(index=False, encoding='utf-8-sig'),
                                       file_name='taiwan_institutional_screen.csv', mime='text/csv', key='download_inst_screen')
                if inst_errors:
                    st.caption("有官方來源未能連線：" + "；".join(inst_errors[:2]))

    with tab9:
        st.markdown("#### 當日全市場三大法人佔股本比 Top 10")
        st.caption("依買賣超佔股本比由高至低及低至高排序；另列買進與賣出佔股本比最高的前 10 檔。上市與上櫃分別採各自官方最新資料日。")
        if st.button("📊 載入全市場法人 Top 10", key='load_inst_top10') or st.session_state.get('inst_top10_loaded', False):
            st.session_state['inst_top10_loaded'] = True
            with st.spinner("讀取上市、上櫃法人資料及官方已發行股數…"):
                market_trades, market_shares, market_errors = fetch_institutional_market_data()
            ranking_data = build_institutional_rankings(market_trades, market_shares)
            if ranking_data.empty:
                st.info("目前無法取得可排序的法人資料與已發行股數。")
            else:
                ranking_data['股票名稱'] = ranking_data.apply(
                    lambda row: row.get('股票名稱') if row.get('股票名稱') and row.get('股票名稱') != row.get('股票代碼')
                    else STOCK_NAME_MAP.get(f"{row.get('股票代碼')}.TWO" if row.get('市場') == '上櫃' else f"{row.get('股票代碼')}.TW", row.get('股票代碼')),
                    axis=1)
                investor_tabs = st.tabs(['外資', '投信', '自營商'])
                for investor, investor_tab in zip(('外資', '投信', '自營商'), investor_tabs):
                    with investor_tab:
                        sort_specs = [
                            (f'{investor}買賣超佔股本比(%)', False, '買賣超佔股本比最高'),
                            (f'{investor}買賣超佔股本比(%)', True, '買賣超佔股本比最低'),
                            (f'{investor}買進佔股本比(%)', False, '買進佔股本比最高'),
                            (f'{investor}賣出佔股本比(%)', False, '賣出佔股本比最高')]
                        ranking_cols = st.columns(2)
                        for index, (column, ascending, title) in enumerate(sort_specs):
                            ranked = ranking_data.dropna(subset=[column]).sort_values(column, ascending=ascending).head(10)
                            show = ranked[['股票代碼', '股票名稱', '市場', '資料日期', column]].rename(columns={column:'佔股本比(%)'})
                            with ranking_cols[index % 2]:
                                st.markdown(f"**{title} Top 10**")
                                st.dataframe(show.style.format({'佔股本比(%)':'{:+.4f}%'}), use_container_width=True, hide_index=True)
                if market_errors:
                    st.caption("部分官方來源未能連線：" + "；".join(market_errors[:2]))
