# ==============================================================================
# 【機構級三核心策略雷達 3.12 網格複利對比(CAGR/MDD/稅費)與專業畫線全能版】 - app.py
# ==============================================================================
import sys, os, re, csv, hashlib, hmac, secrets, streamlit as st, yfinance as yf, pandas as pd, numpy as np, json, sqlite3, io, time, requests, smtplib, ssl
import streamlit.components.v1 as components
from email.message import EmailMessage
try:
    from google import genai
except ImportError:
    # Keep the dashboard usable if a hosting environment has not installed
    # the Gemini SDK; the AI action reports the missing dependency.
    genai = None
try:
    from openai import OpenAI
except ImportError:
    # Keep existing market features available even when the optional ChatGPT
    # API client is missing from a local or hosted environment.
    OpenAI = None
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
        st.caption("表格中的殖利率、發放率、自由現金流、估值與品質分數皆依目前可取得資料估算；「資料不足」或資料覆蓋低時，請勿把缺值當成零或通過。上方「指標與計算欄位怎麼看？」可查各欄位白話說明。")
        with st.expander("🔎 配息品質表格欄位說明", expanded=False):
            st.markdown(
                "- **現金殖利率**：用近 12 個月實際發放的現金股利，對照目前股價看現金回饋規模；不是固定利息。**近五年股利／正值年數**：檢查紀錄是否連續；不代表未來仍會配息。\n"
                "- **EPS 年化成長**：可得年度每股盈餘的年化變化；**現金流正值年數**：營業現金流為正的年度數。\n"
                "- **發放率**：觀察公司把近期盈餘中多少拿來配息，過高時要留意是否難以維持；**本益比**：表示目前股價相對每股盈餘的倍數；**負債/權益**：比較負債規模與股東權益；**利息保障倍數**：觀察營業獲利支付利息的緩衝。\n"
                "- **品質分數**：目前啟用條件加權的 0–100 分；**資料覆蓋**：指標可取得程度；兩者皆不是未來報酬機率。**不合格原因、資料缺漏、風險警示**分別說明未達條件、缺資料及需要留意的情形。"
            )
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
    labels = {"cagr": "年化報酬率 CAGR", "mdd": "最大回撤 MDD", "annualized_volatility": "年化波動率", "sharpe": "夏普比率",
              "cumulative_total_return": "累積含息總報酬", "gross_dividends": "股息收入（稅前）", "reinvested_dividends": "股息再投入金額", "ending_value": "期末資產"}
    metric_help = {
        "cagr": "把回測整段複利結果換算為年平均報酬，方便比較不同期間；不代表每年都會達成。",
        "mdd": "回測期間資產由高點到後續低點的最大跌幅，用來了解曾承受的最深下跌。",
        "annualized_volatility": "報酬起伏幅度的年化估算；不是虧損機率或未來跌幅上限。",
        "sharpe": "把歷史報酬與整體起伏程度一起衡量；本回測以 0% 無風險利率估算。數值越高，代表過去承受相同程度波動時換到的報酬較多，不保證未來。",
        "cumulative_total_return": "把股價變化和已納入計算的股息一起計算，顯示整段期間相對投入金額的變化。",
        "gross_dividends": "回測期間收到的股息總額，尚未扣除設定的股利稅費。",
        "reinvested_dividends": "實際用於增加持股的股息金額；若設定不再投入，這項會較低或為零。",
        "ending_value": "期末持有資產的估算市值，依回測設定計算。",
    }
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
            st.metric(labels[key], shown, help=metric_help[key])
    st.line_chart(history[["portfolio_value"]], use_container_width=True)
    st.caption(f"股價報酬、股息收入與含息總報酬已分開計算；價格使用拆股調整後收盤價，股息於除息日入帳。夏普比率（Sharpe）以 0% 無風險利率估算。期末股數：{metrics['final_shares']}。將游標移到數值旁的說明圖示可看單項解釋；更多欄位請展開上方指標說明。")
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
    st.caption("配息評分獨立計算，不讀寫原系統的 GVI、動能、QARP 或綜合評分。殖利率用近期實際股利對照目前價格，發放率則檢查盈餘被用於配息的程度；總報酬同時考慮價格變化與股息再投入。")
    st.info("🧠 **市場心理重點｜別只追求高殖利率**\n\n現金入帳容易讓人覺得投資有回報，但股價下跌或公司削減股利，可能抵銷甚至超過股息。高殖利率也可能是股價先大跌造成；先確認盈餘、現金流與配息紀錄，再看含息後的整體資產變化。")
    if st.session_state.get("dividend_strategy_mode") == "只使用原有策略":
        st.session_state["dividend_strategy_mode"] = "只使用配息策略"
    mode = st.radio("策略檢視模式", ["只使用配息策略", "比較策略"], index=0, horizontal=True, key="dividend_strategy_mode")
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

# 🎨 機構級沉浸式深色金融終端 UI 與行動端優化 CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

        :root {
            --font-main: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft JhengHei", sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --font-display: 'Plus Jakarta Sans', sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--font-main);
            -webkit-font-smoothing: antialiased;
        }

        /* 全域動態深色漸層背景 */
        .stApp {
            background: radial-gradient(circle at 10% 10%, rgba(30, 41, 69, 0.45) 0%, transparent 40%),
                        radial-gradient(circle at 90% 90%, rgba(17, 24, 39, 0.6) 0%, transparent 45%),
                        linear-gradient(180deg, #0A0E17 0%, #060910 100%) !important;
            background-attachment: fixed !important;
        }

        /* 專業極窄漸層滾動條 */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: rgba(10, 14, 23, 0.8); }
        ::-webkit-scrollbar-thumb { background: rgba(59, 130, 246, 0.35); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(59, 130, 246, 0.7); }

        .block-container {
            width: 100%;
            max-width: 100%;
            padding: 1.2rem 1.6rem 3rem !important;
        }

        /* 專業金融指標卡片 (stMetric) 玻璃擬態與 3D 浮動效果 */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(20, 28, 48, 0.7) 0%, rgba(13, 19, 33, 0.9) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            padding: 0.95rem 1.15rem !important;
            backdrop-filter: blur(12px) !important;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4) !important;
            transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        [data-testid="stMetric"]:hover {
            border-color: rgba(59, 130, 246, 0.5) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 10px 25px -3px rgba(59, 130, 246, 0.2), 0 4px 12px rgba(0, 0, 0, 0.45) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.03em !important;
            color: #94A3B8 !important;
            text-transform: uppercase !important;
        }
        [data-testid="stMetricValue"] {
            font-family: var(--font-mono) !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            color: #FFFFFF !important;
            letter-spacing: -0.02em !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: var(--font-mono) !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }

        /* 🎛️ 十大策略導航主控台 - 實體立體按鈕 (Tactile 3D Buttons) 質感設計 */
        div[data-testid="stTabs"] {
            background: transparent !important;
            margin-top: 1.2rem !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.65rem !important;
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(10, 15, 26, 0.98) 100%) !important;
            padding: 0.9rem !important;
            border-radius: 16px !important;
            border: 1.5px solid rgba(59, 130, 246, 0.3) !important;
            backdrop-filter: blur(14px) !important;
            box-shadow: 0 12px 35px -5px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
            justify-content: flex-start !important;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            flex: 1 1 calc(20% - 0.65rem) !important;
            min-width: 175px !important;
            min-height: 54px !important;
            border-radius: 11px !important;
            padding: 0.65rem 0.9rem !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            letter-spacing: 0.01em !important;
            color: #CBD5E1 !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.45), 0 2px 4px -1px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.12) !important;
            transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            cursor: pointer !important;
            white-space: nowrap !important;
            position: relative !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            color: #FFFFFF !important;
            background: linear-gradient(180deg, #334155 0%, #1E293B 100%) !important;
            border-color: rgba(96, 165, 250, 0.85) !important;
            transform: translateY(-2.5px) !important;
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:active {
            transform: translateY(1.5px) scale(0.98) !important;
            box-shadow: inset 0 3px 6px rgba(0, 0, 0, 0.6) !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #2563EB 0%, #4F46E5 50%, #7C3AED 100%) !important;
            border: 1.5px solid #93C5FD !important;
            color: #FFFFFF !important;
            font-weight: 700 !important;
            transform: translateY(-1.5px) !important;
            box-shadow: 0 0 24px rgba(59, 130, 246, 0.7), 0 4px 14px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
        div[data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* 側邊欄深色層次與邊界線條 */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #090D15 0%, #05080E 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.07) !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding: 1.2rem 1rem 3rem !important;
        }

        /* 現代電光按鈕設計 (stButton, stDownloadButton) */
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%) !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 9px !important;
            padding: 0.55rem 1.15rem !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.22s ease !important;
        }
        div[data-testid="stButton"] button:hover,
        div[data-testid="stDownloadButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1.5px) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
            filter: brightness(1.12) !important;
        }
        div[data-testid="stButton"] button:active,
        div[data-testid="stDownloadButton"] button:active,
        div[data-testid="stFormSubmitButton"] button:active {
            transform: translateY(0) !important;
            filter: brightness(0.95) !important;
        }

        /* 表單控制項、輸入框與下拉選單 */
        input, textarea, div[data-baseweb="select"] > div {
            background-color: rgba(15, 23, 42, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
            transition: all 0.2s ease !important;
        }
        input:focus, textarea:focus, div[data-baseweb="select"] > div:focus-within {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 2.5px rgba(59, 130, 246, 0.3) !important;
        }

        /* 折疊展開面板 (stExpander) */
        div[data-testid="stExpander"] {
            background: rgba(17, 24, 39, 0.5) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
            backdrop-filter: blur(8px) !important;
            margin-bottom: 0.6rem !important;
        }

        /* Plotly 原生圖表圓角美化 */
        .js-plotly-plot .plotly .main-svg {
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* 即時訊號呼吸燈微動畫 */
        @keyframes radar-pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.15); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 0.75rem 0.65rem 4rem !important;
            }
            section.main > div {
                max-width: 100% !important;
            }
            h1 { font-size: 1.55rem !important; line-height: 1.25 !important; }
            h2 { font-size: 1.3rem !important; line-height: 1.3 !important; }
            h3 { font-size: 1.1rem !important; line-height: 1.35 !important; }
            .stMetric { padding: 0.45rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.15rem !important; }

            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.55rem !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                min-width: 0 !important;
                flex: 1 1 calc(50% - 0.55rem) !important;
                width: calc(50% - 0.55rem) !important;
                min-width: calc(50% - 0.55rem) !important;
            }
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]),
            div[data-testid="stHorizontalBlock"]:has(.js-plotly-plot) {
                align-items: stretch !important;
            }
            div[data-testid="stHorizontalBlock"]:has(div[data-testid="stPlotlyChart"]) > div[data-testid="column"],
            div[data-testid="stHorizontalBlock"]:has(.js-plotly-plot) > div[data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                display: flex !important;
                flex-wrap: wrap !important;
                gap: 0.45rem !important;
                padding: 0.6rem !important;
            }
            div[data-testid="stTabs"] button[role="tab"] {
                flex: 1 1 calc(50% - 0.45rem) !important;
                min-width: 140px !important;
                min-height: 48px !important;
                padding: 0.45rem 0.6rem !important;
                font-size: 0.84rem !important;
                white-space: normal !important;
                line-height: 1.2 !important;
            }

            div[data-testid="stButton"] button,
            div[data-testid="stDownloadButton"] button,
            div[data-testid="stFormSubmitButton"] button {
                min-height: 44px !important;
                width: 100% !important;
                touch-action: manipulation;
            }
            input, textarea,
            div[data-baseweb="select"] > div {
                min-height: 44px !important;
                font-size: 16px !important;
            }
            div[data-testid="stRadio"] [role="radiogroup"] {
                flex-wrap: wrap !important;
                gap: 0.35rem 0.75rem !important;
            }
            div[data-testid="stDataFrame"], div[data-testid="stTable"] {
                max-width: 100% !important;
            }
            div[data-testid="stPlotlyChart"],
            .stPlotlyChart,
            .js-plotly-plot,
            .js-plotly-plot .plot-container,
            .js-plotly-plot .svg-container {
                width: 100% !important;
                max-width: 100% !important;
                min-width: 0 !important;
            }
            .modebar-btn {
                min-width: 42px !important;
                min-height: 42px !important;
                padding: 10px !important;
                touch-action: manipulation !important;
            }
            .modebar-container {
                opacity: 1 !important;
                visibility: visible !important;
            }
            .modebar {
                max-width: calc(100vw - 1.5rem) !important;
                overflow-x: auto !important;
                white-space: nowrap !important;
                -webkit-overflow-scrolling: touch;
            }
            .modebar-group {
                display: inline-flex !important;
                flex-wrap: nowrap !important;
            }
            div[data-testid="stPlotlyChart"] {
                max-width: 100% !important;
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch;
            }
            div[data-testid="stPlotlyChart"] .js-plotly-plot {
                min-width: 0 !important;
                max-width: 100% !important;
            }
        }
        @media (max-width: 420px) {
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex-basis: 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }
            div[data-testid="stTabs"] button[role="tab"] {
                padding: 0.45rem 0.58rem !important;
                font-size: 0.8rem !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
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
    # Public daily institutional observations are shared market data, not account-specific portfolio data.
    c.execute('''CREATE TABLE IF NOT EXISTS chip_flow_observations (
        symbol TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        market TEXT NOT NULL DEFAULT '',
        foreign_net REAL,
        trust_net REAL,
        dealer_net REAL,
        issued_shares REAL,
        volume_shares REAL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(symbol, trade_date)
    )''')
    chip_cols = [row[1] for row in c.execute("PRAGMA table_info(chip_flow_observations)").fetchall()]
    if "volume_shares" not in chip_cols:
        c.execute("ALTER TABLE chip_flow_observations ADD COLUMN volume_shares REAL")
    
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
    # 投資組合理財：獨立資料表，沿用既有 market_cache.db，不影響原有 market_data。
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symbol TEXT NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        asset_type TEXT NOT NULL,
        currency TEXT NOT NULL DEFAULT 'TWD',
        quantity REAL NOT NULL DEFAULT 0,
        price REAL NOT NULL DEFAULT 0,
        fx_to_twd REAL NOT NULL DEFAULT 1,
        cost_basis_twd REAL NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_settings (
        setting_key TEXT PRIMARY KEY,
        setting_value REAL NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        snapshot_date TEXT NOT NULL,
        total_value_twd REAL NOT NULL,
        stock_value_twd REAL NOT NULL,
        bond_value_twd REAL NOT NULL,
        cash_value_twd REAL NOT NULL,
        note TEXT DEFAULT ''
    )''')
    # Account credentials use salted PBKDF2 hashes. Portfolio rows are scoped
    # by user_id; legacy rows remain unassigned until the first account registers.
    c.execute('''CREATE TABLE IF NOT EXISTS app_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        email TEXT,
        password_salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_user_settings (
        user_id INTEGER NOT NULL,
        setting_key TEXT NOT NULL,
        setting_value REAL NOT NULL,
        PRIMARY KEY(user_id, setting_key),
        FOREIGN KEY(user_id) REFERENCES app_users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS account_email_challenges (
        user_id INTEGER NOT NULL,
        purpose TEXT NOT NULL,
        pending_email TEXT,
        code_salt TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        expires_at REAL NOT NULL,
        sent_at REAL NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(user_id, purpose),
        FOREIGN KEY(user_id) REFERENCES app_users(id)
    )''')
    user_cols = [row[1] for row in c.execute("PRAGMA table_info(app_users)").fetchall()]
    if "email" not in user_cols:
        c.execute("ALTER TABLE app_users ADD COLUMN email TEXT")
    for table in ("portfolio_positions", "portfolio_snapshots"):
        table_cols = [row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()]
        if "user_id" not in table_cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")
    defaults = {
        "target_stock_pct": 40.0,
        "target_bond_pct": 50.0,
        "target_cash_pct": 10.0,
        "rebalance_threshold_pct": 5.0,
        "drawdown_warning_pct": 8.0,
        "drawdown_review_pct": 12.0,
        "drawdown_limit_pct": 15.0,
        "minimum_cash_twd": 0.0,
    }
    for key, value in defaults.items():
        c.execute("INSERT OR IGNORE INTO portfolio_settings(setting_key, setting_value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

init_db()

def _password_digest(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 310_000
    ).hex()


def _register_account(username, password, email):
    salt = os.urandom(16).hex()
    digest = _password_digest(password, salt)
    defaults = {
        "target_stock_pct": 40.0, "target_bond_pct": 50.0,
        "target_cash_pct": 10.0, "rebalance_threshold_pct": 5.0,
        "drawdown_warning_pct": 8.0, "drawdown_review_pct": 12.0,
        "drawdown_limit_pct": 15.0, "minimum_cash_twd": 0.0,
    }
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("BEGIN IMMEDIATE")
        account_count = conn.execute("SELECT COUNT(*) FROM app_users").fetchone()[0]
        cur = conn.execute(
            "INSERT INTO app_users(username,email,password_salt,password_hash,created_at) VALUES(?,?,?,?,?)",
            (username, email, salt, digest, pd.Timestamp.now().isoformat(timespec="seconds")),
        )
        user_id = int(cur.lastrowid)
        if account_count == 0:
            # Preserve the existing shared portfolio under the first account.
            conn.execute("UPDATE portfolio_positions SET user_id=? WHERE user_id IS NULL", (user_id,))
            conn.execute("UPDATE portfolio_snapshots SET user_id=? WHERE user_id IS NULL", (user_id,))
            for key, value in conn.execute(
                "SELECT setting_key,setting_value FROM portfolio_settings"
            ).fetchall():
                conn.execute(
                    "INSERT OR IGNORE INTO portfolio_user_settings(user_id,setting_key,setting_value) VALUES(?,?,?)",
                    (user_id, key, value),
                )
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO portfolio_user_settings(user_id,setting_key,setting_value) VALUES(?,?,?)",
                (user_id, key, value),
            )
        conn.commit()
    return user_id, account_count == 0


def _verify_account(username, password):
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute(
            "SELECT id,username,password_salt,password_hash FROM app_users WHERE username=? COLLATE NOCASE",
            (username,),
        ).fetchone()
    if not row:
        return None
    candidate = _password_digest(password, row[2])
    if hmac.compare_digest(candidate, row[3]):
        return {"id": int(row[0]), "username": row[1]}
    return None


def _request_account_code(user_id, purpose, email=None):
    """Send a short-lived one-time code for password reset or email binding."""
    user_id = int(user_id)
    now = time.time()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        row = conn.execute("SELECT email FROM app_users WHERE id=?", (user_id,)).fetchone()
        if not row:
            return False
        target_email = str(email or row[0] or "").strip().lower()
        if not target_email:
            return False
        previous = conn.execute(
            "SELECT sent_at FROM account_email_challenges WHERE user_id=? AND purpose=?",
            (user_id, purpose),
        ).fetchone()
        if previous and now - float(previous[0]) < 60:
            return False
    code = f"{secrets.randbelow(1_000_000):06d}"
    salt = os.urandom(16).hex()
    digest = _password_digest(code, salt)
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute(
            "INSERT INTO account_email_challenges(user_id,purpose,pending_email,code_salt,code_hash,expires_at,sent_at,attempts) "
            "VALUES(?,?,?,?,?,?,?,0) ON CONFLICT(user_id,purpose) DO UPDATE SET "
            "pending_email=excluded.pending_email,code_salt=excluded.code_salt,code_hash=excluded.code_hash,"
            "expires_at=excluded.expires_at,sent_at=excluded.sent_at,attempts=0",
            (user_id, purpose, target_email if purpose == "bind_email" else None,
             salt, digest, now + 600, now),
        )
        conn.commit()
    try:
        sender, app_password = get_gmail_secret_values(st.secrets)
        if not sender or not app_password:
            raise RuntimeError("尚未設定 Gmail 寄件資料。請在 Streamlit Secrets 設定 [gmail] sender_email 與 app_password。")
        purpose_text = "密碼重設" if purpose == "password_reset" else "信箱驗證"
        send_gmail_report(
            target_email, f"操盤終端{purpose_text}驗證碼",
            f"您的驗證碼是：{code}\n\n驗證碼 10 分鐘內有效，請勿提供給他人。若非本人操作，請忽略此信。",
            sender, app_password,
        )
    except Exception:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM account_email_challenges WHERE user_id=? AND purpose=?", (user_id, purpose))
            conn.commit()
        raise
    return True


def _check_email_code(user_id, purpose, code):
    now = time.time()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        row = conn.execute(
            "SELECT pending_email,code_salt,code_hash,expires_at,attempts FROM account_email_challenges "
            "WHERE user_id=? AND purpose=?", (int(user_id), purpose),
        ).fetchone()
        if not row:
            return None
        pending_email, salt, expected, expires_at, attempts = row
        if float(expires_at) < now or int(attempts) >= 5:
            conn.execute("DELETE FROM account_email_challenges WHERE user_id=? AND purpose=?", (int(user_id), purpose))
            conn.commit()
            return None
        if not hmac.compare_digest(_password_digest(str(code).strip(), salt), expected):
            conn.execute(
                "UPDATE account_email_challenges SET attempts=attempts+1 WHERE user_id=? AND purpose=?",
                (int(user_id), purpose),
            )
            conn.commit()
            return None
        conn.execute("DELETE FROM account_email_challenges WHERE user_id=? AND purpose=?", (int(user_id), purpose))
        conn.commit()
    return pending_email if purpose == "bind_email" else True


def _reset_account_password(username, code, new_password):
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        row = conn.execute("SELECT id FROM app_users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
    if not row or not _check_email_code(row[0], "password_reset", code):
        return False
    salt = os.urandom(16).hex()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute(
            "UPDATE app_users SET password_salt=?,password_hash=? WHERE id=?",
            (salt, _password_digest(new_password, salt), int(row[0])),
        )
        conn.commit()
    return True


def _confirm_recovery_email(user_id, code):
    email = _check_email_code(user_id, "bind_email", code)
    if not email:
        return False
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("UPDATE app_users SET email=? WHERE id=?", (email, int(user_id)))
        conn.commit()
    return True

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
    if genai is None:
        raise RuntimeError(
            "找不到 google-genai 套件。請將 google-genai>=2.3.0 加入部署環境的 requirements.txt，"
            "重新部署或重啟後再使用 Gemini AI 分析。"
        )
    client = genai.Client(api_key=api_key_val)
    for attempt in range(3):
        try:
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=prompt_text,
            )
            response_text = getattr(interaction, "output_text", None)
            if not response_text or not response_text.strip():
                raise RuntimeError("Gemini API 已回應，但沒有提供文字內容。")
            return response_text.strip()
        except Exception as error:
            message = str(error).lower()
            retryable = any(
                marker in message
                for marker in (
                    "429", "500", "502", "503", "504",
                    "resource_exhausted", "rate limit", "unavailable",
                    "high demand", "busy", "deadline exceeded",
                )
            )
            if not retryable or attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))

def get_openai_secret_values(secret_store):
    """Read optional ChatGPT API settings without displaying stored credentials."""
    try:
        settings = secret_store.get("openai", {})
        api_key = str(settings.get("api_key", "") or secret_store.get("OPENAI_API_KEY", "")).strip()
        model = str(settings.get("model", "") or secret_store.get("OPENAI_MODEL", "gpt-5.5")).strip()
        return api_key, model or "gpt-5.5"
    except Exception:
        return "", "gpt-5.5"

def generate_openai_research(prompt_text, api_key_val, model_name="gpt-5.5"):
    """Generate a web-researched report with clickable source links."""
    if OpenAI is None:
        raise RuntimeError(
            "找不到 openai 套件。請將 openai 套件加入部署環境的 requirements.txt，重新部署後再使用 ChatGPT 分析。"
        )
    client = OpenAI(api_key=api_key_val, timeout=120.0, max_retries=2)
    response = client.responses.create(
        model=model_name or "gpt-5.5",
        tools=[{"type": "web_search"}],
        tool_choice="required",
        input=prompt_text,
    )
    response_text = str(getattr(response, "output_text", "") or "").strip()
    if not response_text:
        raise RuntimeError("ChatGPT API 已回應，但沒有提供文字內容。")

    sources = []
    for output_item in getattr(response, "output", []) or []:
        content_items = output_item.get("content", []) if isinstance(output_item, dict) else getattr(output_item, "content", [])
        for content_item in content_items or []:
            annotations = content_item.get("annotations", []) if isinstance(content_item, dict) else getattr(content_item, "annotations", [])
            for annotation in annotations or []:
                if isinstance(annotation, dict):
                    url = annotation.get("url")
                    title = annotation.get("title")
                    nested = annotation.get("url_citation", {})
                else:
                    url = getattr(annotation, "url", None)
                    title = getattr(annotation, "title", None)
                    nested = getattr(annotation, "url_citation", None)
                if not url and nested:
                    url = nested.get("url") if isinstance(nested, dict) else getattr(nested, "url", None)
                    title = title or (nested.get("title") if isinstance(nested, dict) else getattr(nested, "title", None))
                if url and all(existing[1] != url for existing in sources):
                    sources.append((str(title or url), str(url)))
    if sources:
        response_text += "\n\n#### 網路研究來源\n" + "\n".join(
            f"- [{title}]({url})" for title, url in sources[:12]
        )
    return response_text, sources

def collect_stock_research_context(stock_obj, info_obj, market_snapshot):
    """Build an evidence-only context from the quote provider for either AI."""
    info_obj = info_obj or {}
    fields = {
        "產業": info_obj.get("industry"),
        "產業分類": info_obj.get("sector"),
        "公司簡介": info_obj.get("longBusinessSummary"),
        "市值": info_obj.get("marketCap"),
        "本益比": info_obj.get("trailingPE") or info_obj.get("forwardPE"),
        "營收成長率": info_obj.get("revenueGrowth"),
        "盈餘成長率": info_obj.get("earningsGrowth") or info_obj.get("earningsQuarterlyGrowth"),
        "毛利率": info_obj.get("grossMargins"),
        "營業利益率": info_obj.get("operatingMargins"),
        "自由現金流": info_obj.get("freeCashflow"),
    }
    lines = [f"資料擷取時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}（台灣時間）"]
    for label, value in fields.items():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            continue
        if label.endswith("率") and isinstance(value, (int, float, np.number)):
            value = f"{float(value) * 100:.2f}%"
        elif label in ("市值", "自由現金流") and isinstance(value, (int, float, np.number)):
            value = f"{float(value):,.0f}（來源原幣別）"
        lines.append(f"{label}：{str(value)[:900]}")
    if market_snapshot:
        lines.append("市場指數快照（數值可能延遲，依本次畫面資料）：" + "；".join(market_snapshot))

    headlines = []
    try:
        news_items = stock_obj.news or []
    except Exception:
        news_items = []
    for item in news_items[:8]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        if not isinstance(content, dict):
            continue
        title = content.get("title") or item.get("title")
        provider = content.get("provider") or item.get("publisher") or {}
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj or "")
        pub_date = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(pub_date, (int, float)):
            try:
                pub_date = pd.Timestamp.utcfromtimestamp(pub_date).strftime("%Y-%m-%d")
            except Exception:
                pub_date = ""
        if title:
            headlines.append(f"- {str(title)[:250]}（{publisher or 'Yahoo Finance'}；{pub_date or '日期未提供'}）{('｜' + url) if url else ''}")
    if headlines:
        lines.append("Yahoo Finance 近期新聞標題（僅供檢索線索，不代表已查證）：\n" + "\n".join(headlines))
    else:
        lines.append("近期新聞：資料來源沒有提供可用新聞；不可自行補造新聞或題材。")
    return "\n".join(lines)

def run_ai_candidate_backtests(history, is_taiwan_stock):
    """Evaluate a small fixed set of existing-system strategies, split chronologically."""
    if history is None or history.empty:
        return {"table": pd.DataFrame(), "best": None, "text": "無法下載可用日線歷史資料，因此沒有進行策略回測。"}
    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in required):
        return {"table": pd.DataFrame(), "best": None, "text": "日線資料缺少必要的開高低收或成交量欄位，沒有進行策略回測。"}
    frame = frame[required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).sort_index()
    frame = frame.loc[frame["Close"] > 0]
    if len(frame) < 504:
        return {
            "table": pd.DataFrame(), "best": None,
            "text": f"目前只有 {len(frame)} 筆有效日線（至少需要約 2 年資料），不足以分成前 70% 與後 30% 做基本樣本外比較。",
        }

    split = int(len(frame) * 0.70)
    indicator_frame = compute_backtest_indicators(frame)
    fee = 0.1425 if is_taiwan_stock else 0.0
    tax = 0.3 if is_taiwan_stock else 0.0
    slip = 0.05
    candidates = [
        {
            "name": "五線開花（MA5／10／20／60／120）",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "five_ma_mode": "五線開花", "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "five_ma_mode": "排列失序", "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用"},
        },
        {
            "name": "MACD 黃金／死亡交叉",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "黃金交叉", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "死亡交叉", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用"},
        },
        {
            "name": "KD 黃金／死亡交叉",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "黃金交叉", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "死亡交叉", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用"},
        },
        {
            "name": "RSI 超賣回升／超買回落",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "超賣回升 (RSI向上突破30)", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "超買警戒 (RSI跌破70)", "bb_mode": "停用"},
        },
        {
            "name": "布林通道下軌反彈／回到中軌",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "觸及布林下軌反彈", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "站上布林中軌"},
        },
    ]

    def score_period(part):
        dates = pd.DatetimeIndex(part.index).strftime("%Y-%m-%d").tolist()
        trades = run_strategy_backtest(part, entry, exit_rules, dates, risk_reward_ratio=3.0)
        analysis = analyze_indicator_backtest(part, trades, 100000.0, fee, tax, slip, "1日", is_taiwan_stock)
        return analysis

    rows = []
    for candidate in candidates:
        entry = candidate["entry"]
        exit_rules = candidate["exit"]
        train_part = indicator_frame.iloc[:split].copy()
        # Retain the prior candle for crossover detection; it is pre-test data.
        test_part = indicator_frame.iloc[max(0, split - 1):].copy()
        train_result = score_period(train_part)
        test_result = score_period(test_part)
        train = train_result["strategy_metrics"] if train_result else {}
        test = test_result["strategy_metrics"] if test_result else {}
        test_count = int(test.get("trade_count", 0) or 0)
        test_expectancy = _finite_number(test.get("expectancy_pct"))
        test_pf = _finite_number(test.get("profit_factor"))
        test_mdd = _finite_number(test.get("mdd"))
        passed = (
            test_count >= 5 and test_expectancy is not None and test_expectancy > 0
            and (test_pf is None or test_pf > 1.0)
            and test_mdd is not None and test_mdd >= -0.30
        )
        rows.append({
            "策略": candidate["name"],
            "前70%交易數": int(train.get("trade_count", 0) or 0),
            "前70%平均每筆損益(%)": _finite_number(train.get("expectancy_pct")),
            "後30%交易數": test_count,
            "後30%平均每筆損益(%)": test_expectancy,
            "後30%賺賠金額比": test_pf,
            "後30%最大回撤(%)": test_mdd * 100.0 if test_mdd is not None else None,
            "樣本外檢查": "通過初步門檻" if passed else "未通過／資料不足",
            "_passed": passed,
        })
    result_frame = pd.DataFrame(rows)
    passed_rows = result_frame.loc[result_frame["_passed"]].copy()
    if passed_rows.empty:
        best = None
        recommendation = "沒有策略同時通過樣本外交易數、成本後正期望、賺賠金額比及回撤門檻；本次不指定可執行策略，建議觀望或先用模擬交易。"
    else:
        best_row = passed_rows.sort_values(["後30%平均每筆損益(%)", "後30%最大回撤(%)"], ascending=[False, False]).iloc[0]
        best = str(best_row["策略"])
        best_pf = _finite_number(best_row["後30%賺賠金額比"])
        best_pf_text = f"{best_pf:.2f}" if best_pf is not None else "樣本外尚無虧損交易"
        recommendation = (
            f"初步通過門檻的策略為「{best}」；後 30% 樣本外平均每筆成本後損益 "
            f"{best_row['後30%平均每筆損益(%)']:.2f}%、交易 {int(best_row['後30%交易數'])} 筆、"
            f"賺賠金額比 {best_pf_text}、最大回撤 {best_row['後30%最大回撤(%)']:.2f}%。"
        )
    period = f"{frame.index[0]:%Y-%m-%d} 至 {frame.index[-1]:%Y-%m-%d}；前70%訓練、後30%樣本外測試"
    assumptions = (
        f"共 {len(frame)} 筆日線；成本假設：每邊手續費 {fee:.4f}%、賣出交易稅 {tax:.3f}%、每邊滑價 {slip:.2f}%；"
        "五線開花定義為 MA5＞MA10＞MA20＞MA60＞MA120 且五條線均較前一交易日上升；排列失序時出場。"
        "RSI 使用 14 日，RSI 上穿 30 進場、跌破 70 出場；布林通道使用 20 日均線與上下 2 倍標準差，下軌反彈進場、站上中軌出場。"
        "MACD 與 KD 仍保留黃金／死亡交叉比較。各策略另套用固定停損 8%、移動停損 12%、預設風險報酬 1:3。"
        "通過門檻為後段至少 5 筆交易、成本後平均單筆報酬大於 0、"
        "賺賠金額比大於 1（無虧損時不以缺值判失敗）且最大回撤不超過 30%。"
    )
    def backtest_value(value, suffix=""):
        number = _finite_number(value)
        return f"{number:+.2f}{suffix}" if number is not None else "資料不足"

    strategy_lines = []
    for _, row in result_frame.iterrows():
        pf_value = _finite_number(row.get("後30%賺賠金額比"))
        pf_text = f"{pf_value:.2f}" if pf_value is not None else "無虧損或資料不足"
        strategy_lines.append(
            f"- {row['策略']}：前段 {int(row['前70%交易數'])} 筆；後段 {int(row['後30%交易數'])} 筆；"
            f"後段平均每筆損益 {backtest_value(row.get('後30%平均每筆損益(%)'), '%')}；"
            f"賺賠金額比 {pf_text}；最大回撤 {backtest_value(row.get('後30%最大回撤(%)'), '%')}；"
            f"{row['樣本外檢查']}。"
        )
    results_text = "\n".join(strategy_lines)
    return {"table": result_frame.drop(columns=["_passed"]), "best": best,
            "text": f"回測期間：{period}\n{assumptions}\n各策略樣本外結果：\n{results_text}\n整體結論：{recommendation}",
            "period": period, "assumptions": assumptions, "recommendation": recommendation}

def build_three_core_market_psychology_notes(gvi_value, momentum_pct, fcf_yield_pct, peg_value):
    """Translate three research factors into cautious, plain-language behavior cues."""
    gvi = _finite_number(gvi_value)
    momentum = _finite_number(momentum_pct)
    fcf_yield = _finite_number(fcf_yield_pct)
    peg = _finite_number(peg_value)
    notes = {}
    checks = {}

    if gvi is None:
        notes["GVI 價值面"] = "估值資料不足，暫時無法判讀價值條件可能如何影響市場關注。"
        checks["GVI 價值面"] = None
    elif gvi >= 0.20:
        notes["GVI 價值面"] = (
            "目前達到價值研究門檻，可能吸引重視獲利能力與帳面價值的投資者進一步研究；"
            "仍要核對財報日期、產業差異及獲利是否持續，不能只因分數較高就認定便宜。"
        )
        checks["GVI 價值面"] = True
    else:
        notes["GVI 價值面"] = (
            "目前未達價值研究門檻，表示這項價值假設支持較弱；不代表市場一定高估，"
            "也可能是獲利、成長或產業風險需要進一步查證。"
        )
        checks["GVI 價值面"] = False

    if momentum is None:
        notes["動能趨勢面"] = "有效價格資料不足，暫時不能判讀趨勢延續或追價情緒。"
        checks["動能趨勢面"] = None
    elif momentum >= 80:
        notes["動能趨勢面"] = (
            "價格接近近一年可用收盤高位，表示趨勢延續較受關注，也可能吸引順勢買盤；"
            "同時要留意追價與交易過度擁擠的風險，不能把強勢直接當成後續上漲保證。"
        )
        checks["動能趨勢面"] = True
    else:
        notes["動能趨勢面"] = (
            "價格離近一年可用收盤高位較遠，順勢追價的支持相對較弱；"
            "這不等於股票便宜，可能是整理或趨勢轉弱，需搭配其他資料判讀。"
        )
        checks["動能趨勢面"] = False

    if fcf_yield is None or peg is None:
        notes["QARP 現金流與估值面"] = "現金流或成長估值資料不足，暫時不能判讀這項條件。"
        checks["QARP 現金流與估值面"] = None
    elif fcf_yield >= 2.5 and peg <= 1.5:
        notes["QARP 現金流與估值面"] = (
            "目前現金流與成長估值條件同時達標，可能吸引重視現金創造力及價格合理性的投資者；"
            "仍須確認現金流能否延續、成長估計是否可靠，避免把低估值直接當成安全。"
        )
        checks["QARP 現金流與估值面"] = True
    else:
        notes["QARP 現金流與估值面"] = (
            "目前至少一項現金流或成長估值條件未達標；這可能反映現金創造力、成長預期或價格要求需要再查證，"
            "不能單憑未達標就說市場過度悲觀或股票太貴。"
        )
        checks["QARP 現金流與估值面"] = False

    available_checks = [value for value in checks.values() if value is not None]
    passed_count = sum(value is True for value in available_checks)
    if not available_checks:
        overall = "三個研究角度的有效資料不足，先補齊資料再比較。"
    elif passed_count == len(available_checks):
        overall = (
            "目前可評估的研究條件方向一致，可能讓投資者對基本面或趨勢較有信心；"
            "一致不等於已證明有優勢，仍要檢查共同資料偏差與樣本外結果。"
        )
    elif passed_count == 0:
        overall = (
            "目前可評估的研究條件都未達門檻，代表本次篩選支持較少；"
            "不要為了得到看多結論而臨時放寬條件。"
        )
    else:
        overall = (
            "三個研究角度出現分歧，這是提醒使用者避免確認偏誤的時點："
            "不要只挑支持原有看法的指標，應先找出價值、趨勢與現金流訊號不一致的原因。"
        )
    return {"notes": notes, "checks": checks, "overall": overall}


def build_price_volume_market_psychology(change_pct, volume_ratio, timeframe_name):
    """Describe price/volume behavior without claiming to know trader identities or emotions."""
    change = _finite_number(change_pct)
    ratio = _finite_number(volume_ratio)
    interval = str(timeframe_name or "目前圖表週期")
    if change is None or ratio is None:
        return f"{interval}的價格或成交量資料不足，暫時不解讀市場參與熱度。"
    if change > 0 and ratio >= 1.0:
        return (
            f"{interval}價格上升且成交量不低於近期平均，表示這段走勢的參與度較高；"
            "可能有趨勢認同或追價行為，但成交量不能辨別買賣雙方身分。"
        )
    if change < 0 and ratio >= 1.0:
        return (
            f"{interval}價格下跌且成交量不低於近期平均，表示這段期間交易較活躍並伴隨價格走弱；"
            "可留意賣壓或風險重新評估，但不能單憑這點判定恐慌出場。"
        )
    if change > 0:
        return (
            f"{interval}價格上升但成交量低於近期平均，走勢偏強而參與度沒有同步增加；"
            "避免把少量成交的上漲直接當成市場普遍認同。"
        )
    if change < 0:
        return (
            f"{interval}價格下跌但成交量低於近期平均，表示價格走弱時成交參與相對較淡；"
            "可能是觀望或買盤不足，仍需觀察後續變化。"
        )
    return f"{interval}價格變化接近持平；可視為目前方向不明，避免只憑單一週期推斷市場情緒。"


def _render_market_psychology_guide(st, selected_module):
    """Show a concise behavior-focused reading guide for the active module."""
    guides = {
        "📊 彩色 K 線圖與成交量": "K 線和成交量呈現已發生的交易結果。價升量增可能代表參與度提高，也可能伴隨追價；單根 K 棒不能辨認投資人身分或確定情緒。",
        "📦 籌碼流向參考": "分點家數反映淨買方或淨賣方參與的分點廣度；分點不是投資人身分。再看淨額集中程度、價格和資料期間，避免直接貼上主力吸籌或散戶接盤標籤。",
        "🤖 網格自動生成器": "網格較依賴價格來回波動；人在震盪時容易想低買高賣，但遇到單邊行情可能太早賣出強勢股，或在下跌中不斷加碼。先確認資金上限和價格區間，不要把近期高低點當成必然邊界。",
        "⚖️ 網格／持有／定期定額比較": "買進持有考驗承受回撤的耐心；定期定額降低一次押時點的壓力，但不會消除虧損；網格提供規則化操作，卻可能在單邊行情落後。比較時先看自己能否承受過程中的下跌與錯過上漲。",
        "🧪 技術指標自訂策略回測": "回測容易讓人過度相信剛好適合歷史行情的參數。先看較晚期間、交易筆數、成本和最大跌幅，再確認規則能否照做；不要只挑最好看的回測結果。",
        "💸 配息複利策略": "高殖利率容易讓人只盯著現金入帳，忽略股價下跌或配息不持續。配息不是額外免費報酬；先看現金流和發放能力，再看總資產變化。",
        "🚀 華爾街機構級三核心策略雷達": "價值條件容易讓人錨定帳面價值；動能條件可能放大追漲或錯過恐懼；現金流與估值條件則要防止把便宜誤當安全。三者不一致時，先查原因，不要只挑符合原有看法的指標。",
        "🏦 台股三大法人獨立選股策略": "法人買賣超是已公布的交易流量，不等於持股變化，也不是未來方向保證。看到法人買超容易產生跟隨心理；應同時看日期、連續性、股價和成交量。",
        "📊 當日全市場三大法人佔股本比 Top 10": "排行容易讓人把『名次高』誤當成『勝率高』。單日法人流量只說明當日交易相對股本的大小；避免追逐榜單，先確認資料日期和是否連續出現。",
        "🧭 投資組合理財": "成本價容易成為心理錨點，虧損時也可能讓人不願重新評估。預先設定配置和風險檢查門檻，有助於降低情緒化決策；門檻仍須符合個人承受能力，不代表最佳配置。",
    }
    note = guides.get(selected_module)
    if note:
        st.info(f"🧠 市場心理重點｜{selected_module}\n\n{note}")
        st.caption("這些是可能的行為線索，不是對投資人心理的直接測量，也不是買賣訊號。")


def build_ai_investment_prompt(symbol, name, provider, quant_summary, company_context, backtest_summary):
    research_mode = "必須使用網路搜尋查證近期資料，並附上可點擊的來源與日期。" if provider == "ChatGPT" else "目前使用者選擇 Gemini；只能根據提供的資料與新聞線索分析，不可假裝已完成即時網路搜尋，來源不足時明確寫出資料限制。"
    return f"""你是謹慎的台灣與美國股票研究助理，請使用繁體中文寫一份重點清楚、白話、有資料依據的個股研究報告。
股票：{symbol}（{name}）
量化與策略雷達資料：
{quant_summary}

行情、公司基本資料及可取得的新聞：
{company_context}

本系統內部歷史策略回測（已扣設定的交易成本，採時間順序前70%／後30%比較）：
{backtest_summary}

研究方式：{research_mode}
來源優先順序：台股先查證交所、公開資訊觀測站、公司投資人關係及主管機關資料；美股先查公司投資人關係、SEC 與主管機關資料，再以可信財經媒體補充。每個重要市場或同業數據都要附來源與資料日期。

請先用「【先看重點】」列出最多 4 點，再依序提供：
1. **三核心條件比對與市場心理角度**：分別說明價值、價格趨勢、現金流與估值條件是否符合本次研究門檻；每項用白話說明可能吸引或削弱哪一類市場關注，以及相伴風險。資料不足就明確寫不足。三核心分數是條件比對結果，不是勝率、預期報酬或推薦排名。
2. **市場心理觀察**：只根據提供的價格、成交量、公開法人流量、券商分點摘要、新聞與基本面證據，描述可能的樂觀、觀望、追價、恐慌或風險重新評估線索。若提供分點摘要，必須直接解讀淨買超／淨賣超分點家數、淨額與集中度，並和價格方向並列；只有日期與期間對齊時才能稱為同向或背離，日期不明就明確標示期間未對齊、只能作方向參考。每項標示「觀察到的資料」與「可能的心理解讀」；不要宣稱知道投資人真實想法，不要由分點或法人資料推定主力／散戶身分。
3. **市場、產業與同業**：交代相關大盤、利率／匯率／政策、產業供需與題材；列 2–4 家真正可比同業。重要數據附來源和日期；區分事實與市場預期，資料不足就直說。
4. **公司優勢與風險**：以可查證資料說明優勢、風險，以及可能推翻目前觀點的條件。
5. **回測與風控**：說明樣本外交易數、每筆平均損益、賺賠金額比、最大回撤及通過狀態；沒有策略通過時，明確寫「目前找不到足夠證據支持正期望策略」。只把回測當歷史檢查。
6. **結尾行動重點**：列出目前資料支持的偏多／中性／偏空觀察、最重要催化因素、最重要風險，以及下一步要查證的項目。沒有通過的策略不得包裝成推薦；不得承諾獲利或提供無來源目標價。

格式要求：直接劃重點，使用短段落、清楚小標和條列；不要貼程式碼、不要列數學公式或工程符號。需要解釋計算結果時，請用日常語言說明「這個數字代表什麼、可能影響什麼、還要注意什麼」。心理解讀一律寫成可能性，不把推論寫成事實。不得虛構資料或來源；區分資料、推論與未知。"""

def render_plotly_delete_hotkey(st, chart_key, clear_revision=0):
    """Persist drawn Plotly shapes and keep erase controls working after reruns."""
    encoded_key = json.dumps(str(chart_key))
    encoded_revision = json.dumps(int(clear_revision))
    components.html(f"""
    <style>
      body {{ margin: 0; font-family: sans-serif; color: #CBD5E1; background: transparent; }}
      #shape-delete-panel {{ display: flex; align-items: center; gap: 8px; overflow-x: auto;
        padding: 8px 2px; white-space: nowrap; -webkit-overflow-scrolling: touch; }}
      #shape-delete-panel button {{ min-height: 42px; padding: 8px 12px; border-radius: 8px;
        border: 1px solid #475569; background: #172033; color: #F8FAFC; font-size: 14px;
        cursor: pointer; touch-action: manipulation; }}
      #shape-delete-panel button:disabled {{ opacity: .55; cursor: default; }}
      #shape-delete-status {{ padding: 3px 2px; font-size: 13px; color: #94A3B8; }}
    </style>
    <div id="shape-delete-status">標線刪除工具載入中…</div>
    <div id="shape-delete-panel"></div>
    <script>
    (() => {{
      try {{
        const hostWindow = window.parent;
        const hostDocument = hostWindow.document;
        const chartKey = {encoded_key};
        const clearRevision = {encoded_revision};
        hostWindow.__plotlyShapeDeleteKeys = hostWindow.__plotlyShapeDeleteKeys || {{}};
        const state = hostWindow.__plotlyShapeDeleteKeys[chartKey] || {{ overChart: false, graph: null }};
        hostWindow.__plotlyShapeDeleteKeys[chartKey] = state;
        const panel = document.getElementById('shape-delete-panel');
        const status = document.getElementById('shape-delete-status');
        const storageKey = 'codex-plotly-shapes:' + chartKey;
        const revisionKey = storageKey + ':revision';
        let storage = null;
        try {{ storage = hostWindow.localStorage; }} catch (_error) {{}}
        if (storage) {{
          try {{
            const previousRevision = storage.getItem(revisionKey);
            if (previousRevision !== String(clearRevision)) {{
              storage.removeItem(storageKey);
              storage.setItem(revisionKey, String(clearRevision));
            }}
          }} catch (_error) {{ storage = null; }}
        }}
        const findGraph = () => {{
          return Array.from(hostDocument.querySelectorAll('.js-plotly-plot'))
            .find((graph) => graph._fullLayout && graph._fullLayout.meta
              && graph._fullLayout.meta.codex_chart_key === chartKey) || null;
        }};
        const persistShapes = (graph) => {{
          if (!storage) return;
          try {{ storage.setItem(storageKey, JSON.stringify(graph.layout.shapes || [])); }} catch (_error) {{}}
        }};
        const renderShapeControls = () => {{
          if (!panel || !status) return;
          panel.replaceChildren();
          const shapes = state.graph && state.graph.layout ? (state.graph.layout.shapes || []) : [];
          status.textContent = shapes.length
            ? `目前有 ${{shapes.length}} 條標線；可直接按下方按鈕刪除，不必點選圖上細線。`
            : '目前沒有標線。';
          if (!(hostWindow.Plotly && hostWindow.Plotly.relayout)) {{
            if (shapes.length) status.textContent += ' 若按鈕無法使用，請用圖表工具列橡皮擦。';
            return;
          }}
          shapes.forEach((shape, index) => {{
            const button = document.createElement('button');
            const kind = shape.type || 'shape';
            button.textContent = `刪除第 ${{index + 1}} 條（${{kind}}）`;
            button.setAttribute('aria-label', `刪除第 ${{index + 1}} 條標線`);
            button.addEventListener('click', () => {{
              const nextShapes = (state.graph.layout.shapes || []).slice();
              nextShapes.splice(index, 1);
              hostWindow.Plotly.relayout(state.graph, {{ shapes: nextShapes }}).then(() => {{
                persistShapes(state.graph);
                renderShapeControls();
              }});
            }});
            panel.appendChild(button);
          }});
        }};
        state.renderShapeControls = renderShapeControls;
        const attachGraph = () => {{
          const graph = findGraph();
          if (!graph) {{ state.graph = null; return false; }}
          if (state.graph === graph) {{ renderShapeControls(); return true; }}
          state.graph = graph;
          if (storage) {{
            try {{
              const saved = storage.getItem(storageKey);
              if (saved !== null && hostWindow.Plotly && hostWindow.Plotly.relayout) {{
                const shapes = JSON.parse(saved);
                if (Array.isArray(shapes)) hostWindow.Plotly.relayout(graph, {{ shapes }});
              }}
            }} catch (_error) {{}}
          }}
          if (graph.on) graph.on('plotly_relayout', () => hostWindow.requestAnimationFrame(() => {{
            persistShapes(graph);
            if (state.renderShapeControls) state.renderShapeControls();
          }}));
          if (state.renderShapeControls) state.renderShapeControls();
          return true;
        }};
        if (!state.listenersAttached) {{
          state.listenersAttached = true;
          hostDocument.addEventListener('pointermove', (event) => {{
            const graph = state.graph || findGraph();
            if (!graph) {{ state.overChart = false; return; }}
            const bounds = graph.getBoundingClientRect();
            state.overChart = event.clientX >= bounds.left && event.clientX <= bounds.right
              && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
          }}, true);
          hostDocument.addEventListener('keydown', (event) => {{
            if (!['Delete', 'Backspace'].includes(event.key) || !state.overChart) return;
            const target = event.target;
            if (target && target.closest && target.closest('input, textarea, select, [contenteditable="true"]')) return;
            const graph = findGraph();
            if (graph) state.graph = graph;
            const eraseButton = graph && Array.from(graph.querySelectorAll('.modebar-btn')).find((button) => {{
              const label = [button.getAttribute('data-title'), button.getAttribute('aria-label'), button.getAttribute('title')]
                .filter(Boolean).join(' ').toLowerCase();
              return label.includes('erase') && label.includes('shape');
            }});
            if (eraseButton) {{ event.preventDefault(); eraseButton.click(); }}
          }}, true);
        }}
        if (!attachGraph()) {{
          const retry = hostWindow.setInterval(() => {{ if (attachGraph()) hostWindow.clearInterval(retry); }}, 150);
          hostWindow.setTimeout(() => hostWindow.clearInterval(retry), 10000);
        }}
      }} catch (_error) {{ /* Built-in Plotly eraser remains available if browser isolation blocks this shortcut. */ }}
    }})();
    </script>
    """, height=90, scrolling=False)

def get_gmail_secret_values(secret_store):
    """Read Gmail sender credentials without exposing them in the interface."""
    try:
        settings = secret_store["gmail"]
        sender_email = str(settings.get("sender_email", "")).strip()
        app_password = re.sub(r"\s+", "", str(settings.get("app_password", "")))
        return sender_email, app_password
    except Exception:
        return "", ""

def send_gmail_report(recipient_email, subject, report_text, sender_email, app_password):
    """Send a plain-text report through Gmail SMTP using a Google App Password."""
    required = (recipient_email, subject, report_text, sender_email, app_password)
    if any(not str(value).strip() for value in required):
        raise ValueError("收件地址、報告內容或 Gmail 寄件設定不完整。")

    message = EmailMessage()
    message["From"] = sender_email.strip()
    message["To"] = recipient_email.strip()
    message["Subject"] = subject.strip()
    message.set_content(report_text)

    stage = "連接 Gmail 安全郵件通道（SSL 465）"
    try:
        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            timeout=30,
            context=ssl.create_default_context(),
        ) as smtp:
            stage = "驗證 Gmail 寄件帳號"
            smtp.login(sender_email.strip(), re.sub(r"\s+", "", app_password))
            stage = "寄送分析郵件"
            smtp.send_message(message)
    except smtplib.SMTPAuthenticationError as error:
        raise RuntimeError(
            "Gmail 帳號驗證失敗，請確認 Secrets 的 sender_email 是完整 Gmail 地址，"
            "app_password 是兩步驟驗證產生的應用程式密碼。"
        ) from error
    except Exception as error:
        raise RuntimeError(f"{stage}時失敗：{error}") from error

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
def fetch_grid_daily_history(ticker):
    """Fetch cached daily prices used only to suggest initial grid settings."""
    try:
        frame = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def recommend_dynamic_grid(price, history):
    """Suggest a balanced grid around the current price using recent daily range."""
    price = float(price) if price is not None else 0.0
    if not np.isfinite(price) or price <= 0:
        return 0.01, 0.02, 9, "目前價格無效，請先確認股票報價。"

    atr = None
    observation_count = 0
    if isinstance(history, pd.DataFrame) and {"High", "Low", "Close"}.issubset(history.columns):
        recent = history[["High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna().tail(40)
        if len(recent) >= 10:
            previous_close = recent["Close"].shift(1)
            true_range = pd.concat([
                recent["High"] - recent["Low"],
                (recent["High"] - previous_close).abs(),
                (recent["Low"] - previous_close).abs(),
            ], axis=1).max(axis=1).dropna().tail(20)
            if len(true_range) >= 9:
                atr = float(true_range.mean())
                observation_count = len(true_range)

    if atr is None or not np.isfinite(atr) or atr <= 0:
        half_range = round(price * 0.10, 2)
        half_range = max(half_range, 0.01)
        lower = max(round(price - half_range, 2), 0.01)
        upper = round(price + half_range, 2)
        grid_count = 9
        reason = ("最近日線資料不足，先採用通用起始值：現價上下約 10%、共 9 個價位，讓買進與賣出價位分布在現價兩側。"
                  "這只是方便開始規劃的預設，不保證適合每檔股票。")
        return lower, upper, grid_count, reason

    half_range = min(max(3.0 * atr, price * 0.05), price * 0.30)
    half_range = max(round(half_range, 2), 0.01)
    lower = max(round(price - half_range, 2), 0.01)
    upper = max(round(price + half_range, 2), round(lower + 0.02, 2))
    target_spacing = max(1.25 * atr, price * 0.015)
    interval_count = int(round((upper - lower) / target_spacing))
    grid_count = max(3, min(15, interval_count + 1))
    if grid_count % 2 == 0:
        grid_count = grid_count + 1 if grid_count < 15 else grid_count - 1

    actual_spacing_pct = (upper - lower) / max(grid_count - 1, 1) / price * 100.0
    atr_pct = atr / price * 100.0
    down_pct = (price - lower) / price * 100.0
    up_pct = (upper - price) / price * 100.0
    reason = (
        f"依最近約 {observation_count} 個日線資料估算，平均單日波動約 {atr_pct:.2f}%。"
        f"上下限各距現價約 {down_pct:.1f}%／{up_pct:.1f}%：通常取約 3 倍近期波動，至少保留 5% 空間，最多放寬至 30%。"
        f"建議 {grid_count} 個價位，平均間距約 {actual_spacing_pct:.2f}%，並盡量讓現價上下都有規劃價位。"
        "此設定是兼顧交易機會與格距的起點，不代表最佳或保證獲利；可依自己的持股和風險承受度手動調整。"
    )
    return lower, upper, grid_count, reason


def calculate_money_weighted_return(cash_flows):
    """Solve annualized return for dated cash flows; return None if no valid root exists."""
    dated = {}
    for date_value, amount in cash_flows:
        date_value = pd.Timestamp(date_value)
        if date_value.tzinfo is not None:
            date_value = date_value.tz_localize(None)
        dated[date_value.normalize()] = dated.get(date_value.normalize(), 0.0) + float(amount)
    dated = sorted((day, amount) for day, amount in dated.items() if amount != 0)
    if len(dated) < 2 or not any(amount < 0 for _, amount in dated) or not any(amount > 0 for _, amount in dated):
        return None
    start = dated[0][0]
    years = np.asarray([(day - start).days / 365.25 for day, _ in dated], dtype=float)
    amounts = np.asarray([amount for _, amount in dated], dtype=float)

    def npv(rate):
        try:
            return float(np.sum(amounts / np.power(1.0 + rate, years)))
        except (OverflowError, FloatingPointError, ZeroDivisionError):
            return float("inf") if rate < 0 else float("-inf")

    low, high = -0.9999, 1.0
    low_value = npv(low)
    high_value = npv(high)
    while high_value > 0 and high < 1_000_000:
        high = high * 2.0 + 1.0
        high_value = npv(high)
    if not np.isfinite(low_value) or not np.isfinite(high_value) or low_value * high_value > 0:
        return None
    for _ in range(120):
        middle = (low + high) / 2.0
        middle_value = npv(middle)
        if abs(middle_value) < 1e-8:
            return float(middle)
        if middle_value > 0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def simulate_monthly_dca(price_series, monthly_contribution, commission_rate=0.0, exit_cost_rate=0.0):
    """Buy at the first available price each month and report cash-flow-aware returns."""
    if isinstance(price_series, pd.DataFrame):
        if "Close" not in price_series.columns:
            raise ValueError("定期定額需要收盤價資料。")
        price_series = price_series["Close"]
    prices = pd.to_numeric(price_series, errors="coerce").copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.loc[~prices.index.isna()].dropna().sort_index()
    prices = prices[prices > 0]
    monthly_contribution = float(monthly_contribution)
    commission_rate = max(0.0, float(commission_rate))
    exit_cost_rate = min(1.0, max(0.0, float(exit_cost_rate)))
    if prices.empty or monthly_contribution <= 0:
        raise ValueError("請提供有效股價與大於零的每月投入金額。")

    first_sessions = set(prices.groupby(prices.index.to_period("M"), sort=True).head(1).index)
    shares = 0.0
    total_contributed = 0.0
    total_fees = 0.0
    history_rows = []
    cash_flows = []
    time_weighted_growth = 1.0
    previous_value = 0.0
    for day, price in prices.items():
        contribution = monthly_contribution if day in first_sessions else 0.0
        if contribution > 0:
            added_shares = contribution / (float(price) * (1.0 + commission_rate))
            shares += added_shares
            total_contributed += contribution
            total_fees += contribution - added_shares * float(price)
            cash_flows.append((day, -contribution))
        portfolio_value = shares * float(price)
        denominator = previous_value + contribution
        period_growth = portfolio_value / denominator if denominator > 0 else 1.0
        time_weighted_growth *= period_growth
        history_rows.append({
            "date": day,
            "portfolio_value": portfolio_value,
            "contribution": contribution,
            "time_weighted_growth": time_weighted_growth,
        })
        previous_value = portfolio_value

    history = pd.DataFrame(history_rows).set_index("date")
    ending_value_before_exit_cost = float(history["portfolio_value"].iloc[-1])
    exit_fee = ending_value_before_exit_cost * exit_cost_rate
    if exit_fee:
        history.iloc[-1, history.columns.get_loc("portfolio_value")] = ending_value_before_exit_cost - exit_fee
        total_fees += exit_fee
    ending_value = float(history["portfolio_value"].iloc[-1])
    cash_flows.append((history.index[-1], ending_value))
    money_weighted_return = calculate_money_weighted_return(cash_flows)
    twr_curve = history["time_weighted_growth"]
    drawdown = twr_curve / twr_curve.cummax() - 1.0
    return {
        "history": history,
        "metrics": {
            "contribution_count": len(first_sessions),
            "total_contributed": total_contributed,
            "ending_value": ending_value,
            "profit_loss": ending_value - total_contributed,
            "money_weighted_return": money_weighted_return,
            "mdd": float(drawdown.min()),
            "fees": total_fees,
            "exit_fee": exit_fee,
        },
    }


def build_chip_flow_summary(price_history, institutional_trade=None, issued_shares=None, timeframe_name=None):
    """Summarize accessible trend, volume activity, and official institutional flow in plain language."""
    summary = {}
    if isinstance(price_history, pd.DataFrame) and "Close" in price_history.columns:
        close = pd.to_numeric(price_history["Close"], errors="coerce").dropna()
        if len(close) >= 20 and float(close.iloc[-1]) > 0:
            average = float(close.tail(20).mean())
            change = float(close.iloc[-1] / average - 1.0) if average > 0 else None
            recent_closes = close.tail(20)
            period_change = (
                float(recent_closes.iloc[-1] / recent_closes.iloc[0] - 1.0)
                if len(recent_closes) >= 2 and float(recent_closes.iloc[0]) > 0 else None
            )
            if change is not None:
                tone = "偏強" if change >= 0.01 else "偏弱" if change <= -0.01 else "整理"
                relation = "高於" if change > 0 else "低於" if change < 0 else "接近"
                summary["trend"] = {
                    "status": f"{tone}｜現價{relation}20日平均價",
                    "detail": f"近20筆收盤價由首筆至末筆變化 {period_change:+.2%}；現價較近20筆平均價 {change:+.2%}。這描述近期走勢，不代表未來方向。",
                    "period_change_pct": period_change,
                    "period_bars": len(recent_closes),
                }
        if isinstance(price_history, pd.DataFrame) and "Volume" in price_history.columns:
            volume = pd.to_numeric(price_history["Volume"], errors="coerce").dropna()
            if len(volume) >= 10:
                recent = float(volume.tail(5).mean())
                baseline_series = volume.iloc[-25:-5] if len(volume) >= 25 else volume.iloc[:-5]
                baseline = float(baseline_series.mean()) if not baseline_series.empty else 0.0
                if baseline > 0:
                    ratio = recent / baseline
                    tone = "交易變熱" if ratio >= 1.2 else "交易轉淡" if ratio <= 0.8 else "大致持平"
                    summary["volume"] = {
                        "status": tone,
                        "detail": f"近5筆平均成交量是前段參考平均的 {ratio:.2f} 倍；只代表成交活躍度。",
                    }
        if timeframe_name == "1日" and len(close) >= 120 and float(close.iloc[-1]) > 0:
            high_close_120 = float(close.tail(120).max())
            distance = float(close.iloc[-1] / high_close_120 - 1.0) if high_close_120 > 0 else None
            if distance is not None:
                summary["high_120"] = {
                    "status": f"距120日日線最高收盤 {distance:.2%}",
                    "detail": f"現價相對近120日日線最高收盤價變化 {distance:+.2%}；這是價格位置描述，不是突破或買賣訊號。",
                }

    net_keys = ["外資買賣超股數", "投信買賣超股數", "自營商買賣超股數"]
    if institutional_trade and issued_shares and issued_shares > 0:
        net_values = [_finite_number(institutional_trade.get(key)) for key in net_keys]
        if all(value is not None for value in net_values):
            net_pct = sum(net_values) / float(issued_shares) * 100.0
            direction = "法人當日偏買" if net_pct > 0 else "法人當日偏賣" if net_pct < 0 else "法人當日買賣接近"
            summary["institutions"] = {
                "status": direction,
                "detail": f"三大法人合計買賣超佔已發行股數 {net_pct:+.4f}%；只看當日，單日變化不等於持股集中。",
            }
    return summary


def build_broker_branch_price_breadth_reading(branch_summary, price_change_pct):
    """Pair recent price direction with net-buy/net-sell branch breadth cautiously."""
    if not isinstance(branch_summary, Mapping):
        return "分點家數資料不足，暫時無法比較價格方向與買賣參與廣度。"
    change = _finite_number(price_change_pct)
    if change is None:
        return "價格資料不足，暫時無法配對股價方向與分點家數。"

    buyers = int(branch_summary.get("buyers", 0) or 0)
    sellers = int(branch_summary.get("sellers", 0) or 0)
    unit = "分點" if branch_summary.get("count_unit") == "分點" else "資料列"
    if change > 0 and buyers > sellers:
        reading = (
            f"近20筆價格上漲 {change:+.2%}，淨買超{unit}數也多於淨賣超{unit}數；"
            "兩個方向暫時一致，可作為買方參與較廣的線索。"
        )
    elif change > 0 and sellers > buyers:
        reading = (
            f"近20筆價格上漲 {change:+.2%}，但淨賣超{unit}數較多，方向出現分歧；"
            "可能同時有賣方參與較廣與其他買盤支撐，不能只靠家數判定出貨。"
        )
    elif change < 0 and sellers > buyers:
        reading = (
            f"近20筆價格下跌 {change:+.2%}，淨賣超{unit}數也較多；"
            "可留意賣方參與較廣的線索，但不能據此斷定恐慌或預測續跌。"
        )
    elif change < 0 and buyers > sellers:
        reading = (
            f"近20筆價格下跌 {change:+.2%}，但淨買超{unit}數較多，方向出現分歧；"
            "可能有分點承接，也可能只是買進參與面較廣，價格仍偏弱，不能直接解讀成吸籌。"
        )
    else:
        reading = (
            f"近20筆價格變化為 {change:+.2%}，淨買超與淨賣超{unit}數相同；"
            "目前家數沒有顯示明確的買賣參與廣度差異。"
        )
    return reading + " 分點家數不是投資人數，也不等於交易金額。"


def _normalize_chip_trade_date(value):
    """Normalize TWSE Gregorian or TPEx ROC trade dates without guessing unknown labels."""
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r'\d{8}', text):
        return f'{text[:4]}-{text[4:6]}-{text[6:8]}'
    roc_match = re.fullmatch(r'(\d{2,3})[./年-](\d{1,2})[./月-](\d{1,2})日?', text)
    if roc_match:
        year, month, day = map(int, roc_match.groups())
        if year < 1911:
            year += 1911
        try:
            return pd.Timestamp(year=year, month=month, day=day).strftime('%Y-%m-%d')
        except ValueError:
            return None
    parsed = pd.to_datetime(text, errors='coerce')
    return parsed.strftime('%Y-%m-%d') if not pd.isna(parsed) else None


def save_chip_flow_observation(symbol, institutional_trade, issued_shares, volume_shares=None):
    """Persist one observed official daily row; missing investor values remain NULL, never zero."""
    if not institutional_trade:
        return False
    trade_date = _normalize_chip_trade_date(institutional_trade.get('日期'))
    if not trade_date:
        return False
    investor_values = (
        _finite_number(institutional_trade.get('外資買賣超股數')),
        _finite_number(institutional_trade.get('投信買賣超股數')),
        _finite_number(institutional_trade.get('自營商買賣超股數')),
    )
    if all(value is None for value in investor_values):
        return False
    try:
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            conn.execute('''INSERT INTO chip_flow_observations
                (symbol, trade_date, market, foreign_net, trust_net, dealer_net, issued_shares, volume_shares, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, trade_date) DO UPDATE SET
                    market=excluded.market,
                    foreign_net=excluded.foreign_net,
                    trust_net=excluded.trust_net,
                    dealer_net=excluded.dealer_net,
                    issued_shares=excluded.issued_shares,
                    volume_shares=COALESCE(excluded.volume_shares, chip_flow_observations.volume_shares),
                    updated_at=excluded.updated_at''',
                (str(symbol), trade_date, str(institutional_trade.get('市場') or ''),
                 *investor_values, _finite_number(issued_shares), _finite_number(volume_shares),
                 pd.Timestamp.now(tz='Asia/Taipei').isoformat()))
        return True
    except Exception:
        return False


def load_chip_flow_observations(symbol, limit=20):
    """Load only dates actually captured from official sources, newest first."""
    try:
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            return pd.read_sql_query(
                '''SELECT trade_date, market, foreign_net, trust_net, dealer_net, issued_shares, volume_shares
                   FROM chip_flow_observations WHERE symbol=?
                   ORDER BY trade_date DESC LIMIT ?''',
                conn, params=(str(symbol), int(limit)))
    except Exception:
        return pd.DataFrame()


def describe_backtest_conditions(entry_conds, exit_conds, risk_reward_ratio=None):
    """Create a concise, readable label for a manually tested indicator setup."""
    entry_labels = {
        "ma_mode": "均線", "kd_mode": "KD", "macd_mode": "MACD",
        "vol_mode": "成交量", "rsi_mode": "RSI", "bb_mode": "布林通道",
    }
    exit_labels = {
        "ma_mode": "均線離場", "kd_mode": "KD離場", "macd_mode": "MACD離場",
        "vol_mode": "成交量離場", "rsi_mode": "RSI離場", "bb_mode": "布林通道離場",
    }
    entries = []
    for key, label in entry_labels.items():
        value = entry_conds.get(key)
        if value and value != "停用":
            suffix = f"{entry_conds.get('ma_p')}日" if key == "ma_mode" else ""
            entries.append(f"{label}{suffix}:{value}")
    entry_cross_pairs = _normalize_ma_cross_pairs(entry_conds.get("ma_cross_pairs", []))
    if entry_cross_pairs:
        pairs_text = "、".join(f"{fast}/{slow}日" for fast, slow in entry_cross_pairs)
        entries.append(f"均線黃金交叉{pairs_text}")
    exits = []
    for key, label in exit_labels.items():
        value = exit_conds.get(key)
        if value and value != "停用":
            suffix = f"{exit_conds.get('ma_p')}日" if key == "ma_mode" else ""
            exits.append(f"{label}{suffix}:{value}")
    exit_cross_pairs = _normalize_ma_cross_pairs(exit_conds.get("ma_cross_pairs", []))
    if exit_cross_pairs:
        pairs_text = "、".join(f"{fast}/{slow}日" for fast, slow in exit_cross_pairs)
        exits.append(f"均線死亡交叉{pairs_text}")
    stop_loss = float(entry_conds.get("stop_loss_pct", exit_conds.get("stop_loss_pct", 0)) or 0)
    trailing = float(exit_conds.get("trailing_stop_pct", 0) or 0)
    if stop_loss > 0:
        exits.append(f"固定停損{stop_loss:g}%")
    if trailing > 0:
        exits.append(f"移動停損{trailing:g}%")
    if risk_reward_ratio is not None and (stop_loss > 0 or trailing > 0):
        exits.append(f"風險報酬1:{float(risk_reward_ratio):g}")
    elif risk_reward_ratio is not None:
        exits.append("風險報酬停利未生效（沒有價格型停損）")
    entry_text = "、".join(entries) if entries else "未啟用買進條件"
    if entries:
        entry_text += "（" + ("全部符合" if entry_conds.get("match_mode") == "ALL" else "任一符合") + "）"
    exit_text = "、".join(exits) if exits else "未啟用離場條件"
    return f"買進：{entry_text}；離場：{exit_text}"


def build_backtest_coach_advice(strategy_metrics, benchmark_metrics, dca_metrics=None,
                                grid_metrics=None, out_of_sample_metrics=None):
    """Turn historical metrics into cautious, plain-language review notes."""
    trade_count = int(strategy_metrics.get("trade_count") or 0)
    cumulative = _finite_number(strategy_metrics.get("cumulative_return"))
    win_rate = _finite_number(strategy_metrics.get("win_rate"))
    profit_factor = _finite_number(strategy_metrics.get("profit_factor"))
    expectancy = _finite_number(strategy_metrics.get("expectancy_pct"))
    notes = []
    if trade_count < 20:
        status = "暫不建議直接採用" if (
            cumulative is not None and cumulative <= 0
            or expectancy is not None and expectancy <= 0
            or profit_factor is not None and profit_factor < 1.0
        ) else "樣本不足"
        notes.append(f"目前只有 {trade_count} 筆完成交易，勝率容易被少數交易影響；先累積更多紀錄，不據此判定策略好壞。")
    elif cumulative is None or expectancy is None or cumulative <= 0 or expectancy <= 0 or (profit_factor is not None and profit_factor < 1.0):
        status = "暫不建議直接採用"
        notes.append("扣除設定費用後，整段報酬或每筆平均損益不理想；先調整條件並重新檢查較晚期間的結果。")
    elif trade_count >= 30 and profit_factor is not None and profit_factor >= 1.2 and win_rate is not None and win_rate >= 0.55:
        status = "可優先列入觀察"
        notes.append("交易筆數、勝率、整體報酬與賺賠金額比目前達到初步觀察條件；這是回測候選，不代表適合直接投入。")
    else:
        status = "繼續觀察"
        notes.append("目前回測有正向跡象，但交易筆數或賺賠金額比仍不足以列為優先候選。")

    if win_rate is not None:
        if win_rate < 0.40:
            status = "暫不建議直接採用"
            notes.append("勝率低於 40%，目前不列為優先使用條件；即使總報酬為正，也要確認賺錢交易平均幅度明顯大於賠錢交易，並以更多樣本和樣本外結果再驗證。")
        elif win_rate >= 0.55:
            if profit_factor is not None and profit_factor < 1.0:
                notes.append("勝率雖高，但賺賠金額比低於 1；少數較大的虧損可能抵銷多筆小獲利。")
            else:
                notes.append("勝率達到 55% 的參考線；仍須同時看賺賠金額比、總報酬及回撤，不能只看勝率。")
        else:
            notes.append("勝率未達 55% 參考線；請搭配賺賠金額比和每筆平均損益判讀。")

    compare_values = [("一開始買進並持有", benchmark_metrics.get("cumulative_return"))]
    if dca_metrics:
        total_contributed = _finite_number(dca_metrics.get("total_contributed"))
        ending_value = _finite_number(dca_metrics.get("ending_value"))
        dca_return = (ending_value / total_contributed - 1.0) if total_contributed and ending_value is not None else None
        compare_values.append(("定期定額", dca_return))
    if grid_metrics:
        compare_values.append(("網格（依比較頁設定）", grid_metrics.get("cumulative_return")))
    underperformed = []
    for label, value in compare_values:
        value = _finite_number(value)
        if cumulative is not None and value is not None and cumulative < value:
            underperformed.append(label)
    if underperformed:
        if status == "可優先列入觀察" and compare_values and len(underperformed) == len(compare_values):
            status = "繼續觀察"
        notes.append("同期間報酬低於「" + "、".join(underperformed) + "」；若目標是追求資產成長，先比較簡單做法並確認風險、投入時點與費用設定一致。")
    elif compare_values and cumulative is not None:
        notes.append("目前策略整段報酬沒有落後可用的同期比較項目；仍需留意最大跌幅及不同市場期間的表現。")

    oos_return = _finite_number(out_of_sample_metrics.get("cumulative_return")) if out_of_sample_metrics else None
    oos_trades = int(out_of_sample_metrics.get("trade_count") or 0) if out_of_sample_metrics else 0
    if out_of_sample_metrics:
        if oos_trades < 10:
            if status == "可優先列入觀察":
                status = "繼續觀察"
            notes.append(f"較晚 30% 資料只有 {oos_trades} 筆交易，樣本仍少，無法確認條件是否穩定。")
        elif oos_return is not None and oos_return <= 0:
            status = "暫不建議直接採用"
            notes.append("較晚 30% 資料為負報酬；整段回測的好結果可能集中在較早期間。")
        else:
            notes.append("較晚 30% 資料仍為正報酬，提供一項穩定性參考，但不代表未來能維持。")
    else:
        if status == "可優先列入觀察":
            status = "繼續觀察"
        notes.append("歷史長度不足以切出較晚期間檢查；不要只憑整段回測選用條件。")

    return {"status": status, "notes": notes}

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

def get_finmind_api_token():
    """Read provider credentials from deployment secrets or environment, never from source text."""
    token = str(os.environ.get("FINMIND_TOKEN", "")).strip()
    if token:
        return token
    try:
        return str(st.secrets.get("FINMIND_TOKEN", "")).strip()
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finmind_broker_branch(stock_code, trade_date, api_token, account_scope=""):
    """Fetch one Taiwan stock's daily branch details from FinMind's authenticated endpoint."""
    # account_scope is included in the cache key so different app users do not share cached results.
    if not api_token:
        return pd.DataFrame(), "尚未設定 FinMind API Token。"
    try:
        response = requests.get(
            "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report",
            params={"data_id": str(stock_code), "date": str(trade_date)},
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=25,
        )
        if response.status_code in (401, 402, 403):
            return pd.DataFrame(), "FinMind 拒絕此請求；請確認 Token 有效且帳號方案包含台股分點資料。"
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data", []) if isinstance(payload, dict) else []
        if not records:
            message = str(payload.get("msg") or payload.get("detail") or "該日期尚無可用分點資料。")
            return pd.DataFrame(), message
        return pd.DataFrame(records), None
    except Exception:
        return pd.DataFrame(), "自動取得分點資料失敗；請稍後重試，或使用 CSV 匯入。"


def _broker_branch_csv_column_kind(column):
    """Identify required branch CSV columns across common Chinese and API headers."""
    name = str(column).replace("\ufeff", "").strip().strip("\"'")
    compact = re.sub(r"\s+", "", name).lower()
    if compact in {"stock_id", "data_id", "stock_code", "ticker", "symbol", "股票代碼", "證券代號"}:
        return "code"
    if "代號" in compact or "代碼" in compact:
        if "證券" in compact or "股票" in compact:
            return "code"
    if compact in {"securities_trader_id", "broker_id", "branch_id"} or (
        "券商" in compact and ("代號" in compact or "代碼" in compact)
    ):
        return "branch_id"
    if compact in {"securities_trader", "broker", "broker_branch", "branch", "branch_name"}:
        return "branch"
    if "券商" in compact or "分點" in compact:
        return "branch"
    if compact in {"buy", "buy_volume", "buy_shares"}:
        return "buy"
    if "買進" in compact and any(unit in compact for unit in ("股數", "數量", "股")):
        return "buy"
    if compact in {"sell", "sell_volume", "sell_shares"}:
        return "sell"
    if "賣出" in compact and any(unit in compact for unit in ("股數", "數量", "股")):
        return "sell"
    return None


def _normalize_broker_stock_code(value):
    """Normalize Taiwan stock codes such as 3293, 3293.0, and 3293.TWO."""
    text = str(value).strip().upper()
    match = re.search(r"(?<!\d)(\d{4,6})(?!\d)", text)
    return match.group(1) if match else text.replace(".TW", "").replace(".TWO", "")


def _broker_csv_metadata_stock_code(text, delimiter, header_line):
    """Read a stock code from metadata rows like `證券代號,3293` before the header."""
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            if reader.line_num - 1 >= header_line:
                break
            if row and _broker_branch_csv_column_kind(row[0]) == "code":
                for cell in row[1:]:
                    code = _normalize_broker_stock_code(cell)
                    if code:
                        return code
    except csv.Error:
        return None
    return None


def _normalize_broker_csv_metadata(text):
    """Normalize Excel stock-code cells and harmless spaces after quoted CSV fields."""
    lines = text.splitlines(keepends=True)
    metadata_labels = ("股票代號", "股票代碼", "證券代號", "證券代碼")
    for index, line in enumerate(lines[:100]):
        if any(label in line for label in metadata_labels):
            lines[index] = re.sub(r'(?<![^,\t;|])="([^"]*)"', r"\1", line)
    text = "".join(lines)
    # Some Taiwan broker CSV exports leave a space between the final quote and
    # the delimiter/newline. Python's strict CSV parser rejects that harmless padding.
    return re.sub(r'"[ \t]+(?=[,;\t|]|\r?\n|$)', '"', text)


def _broker_branch_csv_header_groups(header_fields):
    """Find one or more side-by-side broker tables separated by blank columns."""
    sections = []
    current = []
    for index, value in enumerate(header_fields):
        if not str(value).strip():
            if current:
                sections.append(current)
                current = []
        else:
            current.append(index)
    if current:
        sections.append(current)

    def is_broker_table(indices):
        kinds = {_broker_branch_csv_column_kind(header_fields[index]) for index in indices}
        return {"buy", "sell"}.issubset(kinds) and bool({"code", "branch", "branch_id"}.intersection(kinds))

    matching_sections = [indices for indices in sections if is_broker_table(indices)]
    if matching_sections:
        return matching_sections
    all_columns = list(range(len(header_fields)))
    return [all_columns] if is_broker_table(all_columns) else []


def _combine_broker_branch_csv_tables(frame, header_fields):
    """Normalize repeated side-by-side tables into one row-oriented broker table."""
    groups = _broker_branch_csv_header_groups(header_fields)
    if not groups:
        return frame

    canonical_names = {
        "code": "stock_id",
        "branch_id": "securities_trader_id",
        "branch": "securities_trader",
        "buy": "buy_shares",
        "sell": "sell_shares",
    }
    normalized_frames = []
    for indices in groups:
        role_indices = {}
        for index in indices:
            kind = _broker_branch_csv_column_kind(header_fields[index])
            if kind in canonical_names and kind not in role_indices and index < frame.shape[1]:
                role_indices[kind] = index
        if not {"buy", "sell"}.issubset(role_indices):
            continue
        group_frame = pd.DataFrame(index=frame.index)
        for kind, index in role_indices.items():
            group_frame[canonical_names[kind]] = frame.iloc[:, index]
        normalized_frames.append(group_frame)

    if not normalized_frames:
        return frame
    return pd.concat(normalized_frames, ignore_index=True, sort=False)


def summarize_broker_branch_frame(frame, stock_code):
    """Aggregate price-level branch rows into one buy/sell total per broker branch."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {"error": "沒有可分析的分點資料。"}
    code_col = next((c for c in frame.columns if _broker_branch_csv_column_kind(c) == "code"), None)
    buy_col = next((c for c in frame.columns if _broker_branch_csv_column_kind(c) == "buy"), None)
    sell_col = next((c for c in frame.columns if _broker_branch_csv_column_kind(c) == "sell"), None)
    if not all((code_col, buy_col, sell_col)):
        return {"error": "資料需包含股票代號、買進股數、賣出股數欄位。"}
    target_code = _normalize_broker_stock_code(stock_code)
    frame_codes = frame[code_col].map(_normalize_broker_stock_code)
    selected = frame[frame_codes == target_code].copy()
    if selected.empty:
        return {"error": f"目前分析標的是 {target_code}，但這份 CSV 找不到該股票的資料。請切換至 CSV 所屬股票，或匯入 {target_code} 的分點 CSV。"}

    branch_id_col = next((c for c in selected.columns if str(c).lower() == "securities_trader_id"
                          or ('券商' in str(c) and '代號' in str(c))), None)
    branch_name_col = next((c for c in selected.columns if _broker_branch_csv_column_kind(c) == "branch"
                            or str(c).lower() == "securities_trader"), None)
    group_cols = [column for column in (branch_id_col, branch_name_col) if column]
    selected["_buy"] = pd.to_numeric(selected[buy_col].astype(str).str.replace(',', ''), errors='coerce')
    selected["_sell"] = pd.to_numeric(selected[sell_col].astype(str).str.replace(',', ''), errors='coerce')
    selected = selected.dropna(subset=["_buy", "_sell"])
    if selected.empty:
        return {"error": "買進或賣出欄位沒有可讀取的數值。"}
    if group_cols:
        grouped = selected.groupby(group_cols, dropna=False, as_index=False)[["_buy", "_sell"]].sum()
        grouped["分點"] = grouped[branch_name_col].fillna("").astype(str) if branch_name_col else ""
        if branch_id_col:
            ids = grouped[branch_id_col].fillna("").astype(str)
            grouped["分點"] = ids + grouped["分點"].map(lambda value: f"｜{value}" if value else "")
        count_unit = "分點"
    else:
        grouped = selected[["_buy", "_sell"]].copy().reset_index(drop=True)
        grouped["分點"] = [f"資料列 {i + 1}" for i in range(len(grouped))]
        count_unit = "資料列"
    grouped["買賣超"] = grouped["_buy"] - grouped["_sell"]
    buyers = int((grouped["買賣超"] > 0).sum())
    sellers = int((grouped["買賣超"] < 0).sum())
    breakdown = grouped[["分點", "_buy", "_sell", "買賣超"]].rename(
        columns={"_buy": "買進股數", "_sell": "賣出股數"}
    ).sort_values("買賣超", ascending=False).reset_index(drop=True)
    positive_net_total = float(grouped.loc[grouped["買賣超"] > 0, "買賣超"].sum())
    negative_net_total = float(-grouped.loc[grouped["買賣超"] < 0, "買賣超"].sum())
    top_net_buy = float(grouped.nlargest(15, "買賣超")["買賣超"].clip(lower=0).sum())
    top_net_sell = float(grouped.nsmallest(15, "買賣超")["買賣超"].clip(upper=0).sum())
    return {
        "net_shares": float(grouped["買賣超"].sum()),
        "buyers": buyers,
        "sellers": sellers,
        "count_unit": count_unit,
        "top_net_buy": top_net_buy,
        "top_net_sell": top_net_sell,
        "top_net_buy_share_pct": top_net_buy / positive_net_total * 100.0 if positive_net_total > 0 else None,
        "top_net_sell_share_pct": abs(top_net_sell) / negative_net_total * 100.0 if negative_net_total > 0 else None,
        "breakdown": breakdown,
    }


def _broker_branch_ai_state_key(user_id, stock_code):
    normalized_code = _normalize_broker_stock_code(stock_code)
    return f"broker_branch_ai_{str(user_id)}_{normalized_code}"


def save_broker_branch_ai_summary(session_state, user_id, stock_code, branch_summary, source_label):
    """Keep only a compact, account-scoped branch summary for this session's AI report."""
    key = _broker_branch_ai_state_key(user_id, stock_code)
    if not isinstance(branch_summary, Mapping) or branch_summary.get("error"):
        session_state.pop(key, None)
        return None
    snapshot = {
        "stock_code": _normalize_broker_stock_code(stock_code),
        "source_label": str(source_label or "使用者提供的分點資料"),
        "net_shares": _finite_number(branch_summary.get("net_shares")),
        "buyers": int(branch_summary.get("buyers", 0) or 0),
        "sellers": int(branch_summary.get("sellers", 0) or 0),
        "count_unit": str(branch_summary.get("count_unit", "分點")),
        "top_net_buy": _finite_number(branch_summary.get("top_net_buy")),
        "top_net_sell": _finite_number(branch_summary.get("top_net_sell")),
        "top_net_buy_share_pct": _finite_number(branch_summary.get("top_net_buy_share_pct")),
        "top_net_sell_share_pct": _finite_number(branch_summary.get("top_net_sell_share_pct")),
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    session_state[key] = snapshot
    return snapshot


def format_broker_branch_ai_context(snapshot, stock_code):
    """Prepare a short, cautious branch-flow context for an AI report/email."""
    if not isinstance(snapshot, Mapping):
        return ""
    target_code = _normalize_broker_stock_code(stock_code)
    if snapshot.get("stock_code") != target_code:
        return ""
    unit = "分點" if snapshot.get("count_unit") == "分點" else "資料列"
    net_shares = snapshot.get("net_shares")
    net_text = f"{net_shares:+,.0f} 股" if net_shares is not None else "資料不足"
    buy_share = snapshot.get("top_net_buy_share_pct")
    sell_share = snapshot.get("top_net_sell_share_pct")
    buy_share_text = f"前15個淨買超{unit}占全部淨買超約 {buy_share:.1f}%" if buy_share is not None else "淨買超集中度資料不足"
    sell_share_text = f"前15個淨賣超{unit}占全部淨賣超約 {sell_share:.1f}%" if sell_share is not None else "淨賣超集中度資料不足"
    saved_at = str(snapshot.get("saved_at") or "")
    return "\n".join([
        f"【分點資料摘要｜{target_code}】",
        f"資料來源：{snapshot.get('source_label', '使用者提供的分點資料')}；載入時間：{saved_at or '未提供'}。",
        f"分點淨額合計：{net_text}；淨買超{unit}：{int(snapshot.get('buyers', 0) or 0)}；淨賣超{unit}：{int(snapshot.get('sellers', 0) or 0)}。",
        f"淨流向集中度：{buy_share_text}；{sell_share_text}。",
        "資料限制：CSV 若未標示交易日期，不能假設分點統計與目前股價是同日或同期間；淨額合計接近零通常是買賣相抵，不是方向訊號。",
        "解讀原則：分點家數表示交易渠道的參與廣度，不代表投資人數或散戶／主力身分；請將家數、淨額、集中度和日期對齊後的價格變化分開觀察，不據此宣稱吸籌、出貨或預測漲跌。",
    ])


def build_broker_branch_ai_interpretation(snapshot, price_history, timeframe_name):
    """Build a deterministic, user-facing branch-flow and recent-price interpretation."""
    if not isinstance(snapshot, Mapping):
        return ""
    close = pd.to_numeric(pd.Series(price_history), errors="coerce").dropna().tail(20)
    buyers = int(snapshot.get("buyers", 0) or 0)
    sellers = int(snapshot.get("sellers", 0) or 0)
    unit = "分點" if snapshot.get("count_unit") == "分點" else "資料列"
    net_shares = _finite_number(snapshot.get("net_shares"))
    buy_share = _finite_number(snapshot.get("top_net_buy_share_pct"))
    sell_share = _finite_number(snapshot.get("top_net_sell_share_pct"))

    lines = [
        f"觀察到的分點資料：淨買超{unit} {buyers:,} 個、淨賣超{unit} {sellers:,} 個；"
        f"這份資料的分點淨額合計 {net_shares:+,.0f} 股。" if net_shares is not None
        else f"觀察到的分點資料：淨買超{unit} {buyers:,} 個、淨賣超{unit} {sellers:,} 個；淨額資料不足。"
    ]

    concentration_parts = []
    if buy_share is not None:
        top_buy = _finite_number(snapshot.get("top_net_buy"))
        amount_text = f"（合計 {top_buy:+,.0f} 股）" if top_buy is not None else ""
        concentration_parts.append(f"前 15 個淨買超{unit}{amount_text}約占全部淨買超 {buy_share:.1f}%")
    if sell_share is not None:
        top_sell = _finite_number(snapshot.get("top_net_sell"))
        amount_text = f"（合計 {top_sell:+,.0f} 股）" if top_sell is not None else ""
        concentration_parts.append(f"前 15 個淨賣超{unit}{amount_text}約占全部淨賣超 {sell_share:.1f}%")
    if concentration_parts:
        lines.append("流向集中度：" + "；".join(concentration_parts) + "。比例越高，代表該方向的淨額越集中在排名前面的分點；不代表持股集中度。")
    else:
        lines.append("流向集中度：目前無法計算前 15 個淨買超／淨賣超分點占比。")

    if len(close) >= 2 and float(close.iloc[0]) > 0:
        price_change_pct = float(close.iloc[-1] / close.iloc[0] - 1.0)
        price_direction = "上漲" if price_change_pct > 0 else "下跌" if price_change_pct < 0 else "持平"
        date_range = ""
        try:
            start_date = pd.Timestamp(close.index[0]).strftime("%Y-%m-%d")
            end_date = pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d")
            date_range = f"（{start_date} 至 {end_date}）"
        except Exception:
            pass
        lines.append(
            f"價格觀察：近 {len(close)} 筆「{timeframe_name or '目前圖表'}」收盤由 {float(close.iloc[0]):,.2f} 元"
            f"變為 {float(close.iloc[-1]):,.2f} 元，期間{price_direction} {price_change_pct:+.2%}{date_range}。"
        )
        lines.append("家數與價格搭配：" + build_broker_branch_price_breadth_reading(snapshot, price_change_pct))
    else:
        lines.append("價格觀察：可用收盤資料少於 2 筆，無法計算近 20 筆價格方向。")

    if buyers > sellers:
        breadth_note = f"淨買超{unit}比淨賣超多 {buyers - sellers:,} 個，表示淨買方參與面較廣。"
    elif sellers > buyers:
        breadth_note = f"淨賣超{unit}比淨買超多 {sellers - buyers:,} 個，表示淨賣方參與面較廣。"
    else:
        breadth_note = f"淨買超與淨賣超{unit}數相同，家數沒有顯示哪一方參與面較廣。"
    lines.append("可能的市場心理解讀：" + breadth_note + "若價格方向與買賣參與廣度一致，可視為當下行為線索；若不一致，代表訊號分歧，宜觀察後續資料，不宜直接貼上吸籌或出貨標籤。")

    source_label = str(snapshot.get("source_label") or "使用者提供的分點資料")
    source_date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", source_label)
    if source_date_match:
        date_note = f"分點資料標示日期為 {source_date_match.group(1)}；近 20 筆價格涵蓋的期間較長，不能把兩者當成完全相同期間。"
    else:
        date_note = "這份分點資料沒有可確認的交易日期；無法確認它與近 20 筆價格期間對齊，因此只能作方向參考。"
    lines.append("資料限制：" + date_note + "分點家數不是投資人數，資料也無法辨認主力或散戶身分；淨額接近零可能只是買賣相抵，不代表沒有資金流動。")
    return "\n".join(f"- {line}" for line in lines)


def summarize_broker_branch_csv(uploaded_file, stock_code):
    """Read common broker CSV encodings/delimiters and locate the actual header row."""
    if uploaded_file is None:
        return None
    try:
        raw_content = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        if isinstance(raw_content, str):
            raw_content = raw_content.encode("utf-8")
    except Exception as exc:
        return {"error": f"無法讀取上傳檔案：{exc}"}
    if not raw_content:
        return {"error": "上傳的 CSV 是空檔案。"}

    delimiters = (",", "\t", ";", "|")
    encodings = ("utf-8-sig", "utf-8", "cp950", "big5", "utf-16")
    last_parse_error = None
    for encoding in encodings:
        try:
            text = _normalize_broker_csv_metadata(raw_content.decode(encoding))
        except (UnicodeDecodeError, LookupError):
            continue

        for delimiter in delimiters:
            header_line = None
            header_fields = None
            try:
                reader = csv.reader(io.StringIO(text), delimiter=delimiter)
                for row in reader:
                    if reader.line_num > 100:
                        break
                    kinds = {_broker_branch_csv_column_kind(cell) for cell in row}
                    has_trade_columns = {"buy", "sell"}.issubset(kinds)
                    has_stock_or_branch = bool({"code", "branch", "branch_id"}.intersection(kinds))
                    if has_trade_columns and has_stock_or_branch:
                        # csv.reader.line_num is one-based; pandas skiprows is zero-based.
                        header_line = reader.line_num - 1
                        header_fields = row
                        break
            except csv.Error as exc:
                last_parse_error = str(exc)
                continue
            if header_line is None:
                continue

            try:
                frame = pd.read_csv(
                    io.StringIO(text),
                    sep=delimiter,
                    skiprows=header_line,
                    header=0,
                    engine="python",
                    dtype=str,
                    skip_blank_lines=True,
                    on_bad_lines="error",
                )
            except Exception as exc:
                last_parse_error = str(exc)
                continue
            frame = _combine_broker_branch_csv_tables(frame, header_fields or list(frame.columns))
            code_col = next((column for column in frame.columns if _broker_branch_csv_column_kind(column) == "code"), None)
            if code_col is None:
                metadata_code = _broker_csv_metadata_stock_code(text, delimiter, header_line)
                if metadata_code is None:
                    return {"error": "CSV 標題列沒有股票代號，且前方資訊列也找不到「證券代號」；無法確認資料屬於哪一檔股票。"}
                if metadata_code != _normalize_broker_stock_code(stock_code):
                    return {"error": f"CSV 內的股票代號是 {metadata_code}，目前分析標的是 {_normalize_broker_stock_code(stock_code)}。請切換至 {metadata_code}，或匯入目前標的的分點 CSV。"}
                frame["stock_id"] = metadata_code
            else:
                csv_codes = frame[code_col].map(_normalize_broker_stock_code).dropna().unique().tolist()
                target_code = _normalize_broker_stock_code(stock_code)
                if csv_codes and target_code not in csv_codes:
                    shown_codes = ", ".join(csv_codes[:5])
                    more_codes = " 等" if len(csv_codes) > 5 else ""
                    return {"error": f"這份 CSV 包含 {shown_codes}{more_codes}；目前分析標的是 {target_code}。請切換至 CSV 所屬股票，或匯入 {target_code} 的分點 CSV。"}
            return summarize_broker_branch_frame(frame, stock_code)

    if last_parse_error:
        return {
            "error": "CSV 引號或欄位分隔格式不一致，無法安全匯入。請確認每列引號成對、欄位分隔一致。"
                     f"（{last_parse_error}）"
        }
    return {
        "error": "找不到可辨識的 CSV 標題列。請確認檔案包含股票代號、買進股數與賣出股數欄位；"
                 "系統會自動搜尋前 100 列的標題，也支援逗號、Tab、分號或直線分隔。"
    }

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

def _format_optional_cell(value, format_string):
    """Format a numeric table cell without failing when the result is missing."""
    try:
        if value is None or pd.isna(value):
            return "—"
        return format_string.format(value)
    except (TypeError, ValueError, OverflowError):
        return "—"


def _render_metric_glossary(st):
    """Show short, plain-language definitions for calculated fields across panels."""
    with st.expander("📘 指標與計算欄位怎麼看？", expanded=False):
        st.caption("點開可查各面板常見數值的意思與用途；指標是分析參考，不是獲利保證或單獨的買賣訊號。")
        sections = [
            ("回測與風險", [
                "**整段／累積報酬**：期末資產相對投入本金增加或減少多少；含息欄位會把股息計入。",
                "**年化報酬 CAGR**：把整段期間的複利結果換算成一年，方便比較不同長度的期間；不代表每年都會得到這個報酬。",
                "**最大回撤 MDD／最大跌幅**：歷史資產曲線從高點到後續低點的最深跌幅，用來了解曾承受的最差下跌。",
                "**年化波動率**：報酬上下起伏的幅度估算；波動大不等於一定虧損，也不代表未來風險上限。",
                "**夏普比率 Sharpe**：比較歷史報酬與整體資金起伏。數值越高，表示過去每承受一份整體起伏所換到的報酬較多；目前以 0% 無風險利率估算，不保證未來。",
                "**索提諾比率 Sortino**：著重比較報酬與下跌時的起伏；數值較高代表過去承受下跌起伏時的報酬相對較好，不是虧損機率。",
                "**卡瑪比率 Calmar**：把平均每年報酬和歷史最深跌幅放在一起看；過去報酬相對跌幅較高，不代表未來也能維持。",
                "**勝率／賺錢交易比例**：扣除設定成本後賺錢的交易占比；高勝率仍可能被少數大虧損抵銷。",
                "**賺賠金額比／獲利因子**：比較歷史獲利交易與虧損交易的總金額；大於 1 代表過去獲利總額較高，不能單獨代表策略可靠。",
                "**每筆平均損益／期望值**：每筆已完成交易平均賺或賠多少，通常以扣除回測成本後計算；筆數太少時容易受個別交易影響。",
                "**持有時間比例／曝險**：回測中持有股票的時間占比；其餘時間通常是現金。",
                "**持有 K 棒數**：從模擬進場到出場經過的圖表資料根數；不同圖表週期下，一根代表的時間不同。",
                "**成本後報酬率**：單筆模擬交易扣除設定手續費、交易稅及滑價後的損益比例；成本仍是估算值。",
                "**風險報酬比／目標停利價**：依進場價、停損距離及設定倍數推算的目標價；是模擬規則，不是獲利機率。",
                "**回撤期間**：資產低於前次高點的連續資料根數；不同圖表週期不宜直接比較根數。",
                "**心理提醒**：漂亮的回測成績容易讓人過度自信，虧損紀錄也容易讓人只想等回本；先看較晚期間、交易成本和最深跌幅，避免只挑支持自己想法的結果。",
            ]),
            ("價格、技術與選股指標", [
                "**均線乖離率**：現價相對指定期間平均價高或低多少百分比；用來觀察短期偏離，不是必然反轉訊號。",
                "**量比**：目前成交量與近期平均量相比的倍數；高於 1 代表量比平均活躍，不能判斷買賣方向。",
                "**價量動能估算**：綜合價格漲跌與成交量相對熱度，觀察趨勢參與度；只是價量參考，不是實際籌碼集中度、法人進出或持股變化。",
                "**RSI 相對強弱指標**：依近期上漲與下跌幅度計算的 0–100 動能指標；描述近期漲跌力道，不代表單獨的買賣訊號。",
                "**KD 隨機指標**：比較收盤價在近期高低區間的位置，K、D 線是平滑後的變化；主要描述短期動能。",
                "**MACD**：兩條不同速度的指數移動平均線之差；柱狀圖顯示 MACD 與訊號線的差距，用來觀察趨勢動能變化。",
                "**布林通道**：以移動平均線和近期波動範圍形成上下界；價格靠近通道邊緣表示位置偏向區間一側，不保證反轉。",
                "**ATR 平均真實波幅**：近期價格區間的平均大小，用來估算波動幅度；不表示漲跌方向。",
                "**黃金交叉／死亡交叉**：快均線向上穿越慢均線／向下跌破慢均線；是歷史價格訊號，可能落後且會反覆失效。",
                "**GVI 成長價值指標**：綜合帳面價值、股價與獲利能力估算的研究分數；依賴財報資料與假設，不是合理價或報酬預測。",
                "**ROE 股東權益報酬率**：公司用股東投入的權益創造盈餘的比例；須留意產業差異、財報期間與一次性項目。",
                "**每股淨值／PB 股價淨值比**：每股淨值是帳面資產扣除負債後分攤到每股的金額；PB 顯示股價相對帳面價值的高低，跨產業比較要小心。",
                "**動能／距近 252 根高點比例**：觀察現價離近一年可取得的最高收盤價有多近；越接近高點代表近期趨勢較強，也要留意追價風險。歷史資料少於 50 根時不評分。",
                "**自由現金流收益率**：比較公司創造的自由現金流相對於公司市值的大小；資料期間可能落後股價，也可能受一次性現金流影響。",
                "**PEG 本益成長比**：把公司估值與預期盈餘成長放在一起觀察；成長預估不可靠時，這個數字也會失真。",
                "**策略評分／排名**：依本次股票池和已開啟條件做相對排序，不是通過機率、預期報酬或買進建議。",
                "**心理提醒**：價格靠近高點時容易引發追價，快速下跌時容易引發恐慌；指標只提供觀察線索，不能直接量出所有投資人的情緒。",
            ]),
            ("配息、網格與法人欄位", [
                "**現金殖利率**：用近 12 個月實際發放的現金股利，對照目前股價觀察回饋規模；未來股利可能變動，不能視為固定利息。",
                "**股利發放率**：觀察近期股利相對公司最新可得盈餘的高低；高於 100% 可能表示股利超過當期盈餘，需查明原因。",
                "**配息品質分數**：依配息紀錄、盈餘、現金流、發放率、安全性與估值等目前啟用條件加權整理的 0–100 分；不是配息保證或未來報酬機率。",
                "**利息保障倍數**：營業獲利相對利息支出的倍數；越低通常代表支付利息的緩衝較小，並受資料口徑影響。",
                "**資料覆蓋率**：目前可取得的必要資料占比；缺漏越多，評分越不完整，不等於公司品質分數。",
                "**網格偏離現價**：每個觸發價與目前價格的百分比差距；觸發價是規劃參考，不是券商掛單。",
                "**定期定額累計投入**：模擬期間投入金額加總；和期末資產相比才能看投入後的損益。",
                "**佔股本比**：顯示法人當日買進、賣出或淨買賣超規模相對於公司已發行股數的大小；不是法人持股比例。淨買賣超表示買進與賣出相抵後的方向和數量。",
                "**心理提醒**：高殖利率容易讓人只注意現金入帳、忽略價格與配息能否維持；法人買超也容易引發跟隨心理，兩者都需要搭配公司狀況和多日資料。",
            ]),
            ("投資組合欄位", [
                "**目前配置比例**：顯示各類資產占整個組合的比重；價格與匯率更新狀態會影響結果。",
                "**配置偏離（百分點）**：顯示目前資產比重比目標高或低多少；例如目前 45%、目標 40%，代表高出 5 個百分點。",
                "**調整金額參考**：估算回到目標配置所需的金額差；正數代表相對不足、負數代表相對超出，未扣交易成本或稅費。",
                "**未實現損益估算**：比較目前估值和記錄成本後的帳面差額；不含完整稅費、已領股息或所有外部資金流。",
                "**快照估算回撤**：手動記錄的組合市值與曾記錄高點相比的變化；未調整入金／出金，只能粗略參考。",
                "**最低保留現金**：低於設定金額時顯示提醒；系統不會因此自動買賣或更改配置建議。",
                "**心理提醒**：買入成本常成為心理錨點，虧損時可能不願重新檢視；依事前訂好的配置和風險門檻檢查，能減少臨時追高或恐慌賣出。",
            ]),
        ]
        for heading, definitions in sections:
            st.markdown(f"**{heading}**")
            st.markdown("\n".join(f"- {definition}" for definition in definitions))


def _normalize_ma_cross_pairs(pairs):
    """Return unique (fast, slow) periods where the fast average is shorter."""
    normalized = []
    for pair in pairs or []:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        try:
            fast, slow = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if fast < 1 or slow < 1 or fast >= slow:
            continue
        if (fast, slow) not in normalized:
            normalized.append((fast, slow))
    return normalized


def _count_ma_cross_events(df, pairs, direction="golden"):
    """Count actual moving-average crosses in the supplied backtest rows."""
    normalized = _normalize_ma_cross_pairs(pairs)
    if not normalized or df is None or df.empty or "Close" not in df.columns:
        return [(fast, slow, 0) for fast, slow in normalized]

    close = pd.to_numeric(df["Close"], errors="coerce")
    counts = []
    for fast_period, slow_period in normalized:
        fast_column, slow_column = f"MA_CROSS_{fast_period}", f"MA_CROSS_{slow_period}"
        fast_ma = (
            pd.to_numeric(df[fast_column], errors="coerce").to_numpy(dtype=float)
            if fast_column in df.columns else close.rolling(fast_period).mean().to_numpy(dtype=float)
        )
        slow_ma = (
            pd.to_numeric(df[slow_column], errors="coerce").to_numpy(dtype=float)
            if slow_column in df.columns else close.rolling(slow_period).mean().to_numpy(dtype=float)
        )
        events = np.zeros(len(df), dtype=bool)
        if len(df) > 1:
            valid = (
                np.isfinite(fast_ma[:-1]) & np.isfinite(slow_ma[:-1])
                & np.isfinite(fast_ma[1:]) & np.isfinite(slow_ma[1:])
            )
            if direction == "golden":
                events[1:] = valid & (fast_ma[:-1] <= slow_ma[:-1]) & (fast_ma[1:] > slow_ma[1:])
            elif direction == "death":
                events[1:] = valid & (fast_ma[:-1] >= slow_ma[:-1]) & (fast_ma[1:] < slow_ma[1:])
            else:
                raise ValueError("direction must be 'golden' or 'death'")
        counts.append((fast_period, slow_period, int(events.sum())))
    return counts


def _slice_history_for_date_range(history, start_date, end_date):
    """Return bars inside an inclusive calendar-date range."""
    if history is None or history.empty:
        return history.copy() if history is not None else pd.DataFrame()
    index = pd.DatetimeIndex(pd.to_datetime(history.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_localize(None)
    start = pd.Timestamp(start_date)
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    mask = (index >= start) & (index < end_exclusive)
    return history.iloc[np.asarray(mask)].copy()


def compute_backtest_indicators(df, ma_entry_p=20, ma_exit_p=20, rsi_p=14, bb_p=20,
                                entry_ma_cross_pairs=None, exit_ma_cross_pairs=None):
    df_calc = df.copy()
    df_calc['MA_entry'] = df_calc['Close'].rolling(window=ma_entry_p).mean()
    df_calc['MA_exit'] = df_calc['Close'].rolling(window=ma_exit_p).mean()

    cross_periods = {
        period
        for pair in (
            _normalize_ma_cross_pairs(entry_ma_cross_pairs)
            + _normalize_ma_cross_pairs(exit_ma_cross_pairs)
        )
        for period in pair
    }
    for period in cross_periods:
        df_calc[f'MA_CROSS_{period}'] = df_calc['Close'].rolling(window=period).mean()
    for period in (5, 10, 20, 60, 120):
        df_calc[f'MA_FIVE_{period}'] = df_calc['Close'].rolling(window=period).mean()

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
    df_calc['BB_Middle'] = bb_middle
    df_calc['BB_Upper'] = bb_middle + (bb_std * 2)
    df_calc['BB_Lower'] = bb_middle - (bb_std * 2)

    return df_calc

def run_strategy_backtest(df_calc, entry_conds, exit_conds, date_chinese_list,
                         risk_reward_ratio=None):
    trades = []
    position = False
    entry_price = 0.0
    entry_date = ""
    entry_index = None
    highest_price = 0.0
    initial_stop_price = None
    take_profit_price = None
    risk_ratio = _finite_number(risk_reward_ratio)

    closes = df_calc['Close'].to_numpy()
    opens = df_calc['Open'].to_numpy() if 'Open' in df_calc.columns else closes
    highs = df_calc['High'].to_numpy()
    lows = df_calc['Low'].to_numpy()
    vols = df_calc['Volume'].to_numpy()
    vol_ma20 = df_calc['VOL_MA20'].to_numpy()

    ma_entry = df_calc['MA_entry'].to_numpy()
    ma_exit = df_calc['MA_exit'].to_numpy()
    entry_ma_cross_pairs = _normalize_ma_cross_pairs(entry_conds.get('ma_cross_pairs', []))
    exit_ma_cross_pairs = _normalize_ma_cross_pairs(exit_conds.get('ma_cross_pairs', []))
    cross_periods = {
        period for pair in entry_ma_cross_pairs + exit_ma_cross_pairs for period in pair
    }
    cross_ma_values = {}
    for period in cross_periods:
        column = f'MA_CROSS_{period}'
        if column in df_calc.columns:
            cross_ma_values[period] = df_calc[column].to_numpy()
        else:
            # Keep direct callers compatible even if they did not request the
            # extra moving-average columns when computing indicators.
            cross_ma_values[period] = df_calc['Close'].rolling(window=period).mean().to_numpy()
    k_arr = df_calc['K'].to_numpy()
    d_arr = df_calc['D'].to_numpy()
    dif_arr = df_calc['DIF'].to_numpy()
    dem_arr = df_calc['DEM'].to_numpy()
    rsi_arr = df_calc['RSI'].to_numpy()
    bb_upper = df_calc['BB_Upper'].to_numpy()
    bb_lower = df_calc['BB_Lower'].to_numpy()
    bb_middle = df_calc['BB_Middle'].to_numpy() if 'BB_Middle' in df_calc.columns else np.full(len(df_calc), np.nan)
    five_ma_periods = (5, 10, 20, 60, 120)
    five_ma_arrays = {
        period: df_calc[f'MA_FIVE_{period}'].to_numpy()
        if f'MA_FIVE_{period}' in df_calc.columns
        else np.full(len(df_calc), np.nan)
        for period in five_ma_periods
    }

    def five_line_bullish(index, require_rising=False):
        if index < 0:
            return False
        values = [five_ma_arrays[period][index] for period in five_ma_periods]
        if not np.isfinite(values).all() or not all(values[i] > values[i + 1] for i in range(4)):
            return False
        if require_rising:
            if index == 0:
                return False
            previous = [five_ma_arrays[period][index - 1] for period in five_ma_periods]
            if not np.isfinite(previous).all() or not all(values[i] > previous[i] for i in range(5)):
                return False
        return True

    for i in range(1, len(df_calc)):
        if not position:
            buy_checks = []
            
            if entry_conds['ma_mode'] == '突破均線':
                buy_checks.append(not np.isnan(ma_entry[i-1]) and closes[i-1] <= ma_entry[i-1] and closes[i] > ma_entry[i])
            elif entry_conds['ma_mode'] == '站穩均線上':
                buy_checks.append(not np.isnan(ma_entry[i]) and closes[i] > ma_entry[i])

            if entry_conds.get('five_ma_mode') == '五線開花':
                buy_checks.append(five_line_bullish(i, require_rising=True))

            for fast_period, slow_period in entry_ma_cross_pairs:
                fast_ma = cross_ma_values[fast_period]
                slow_ma = cross_ma_values[slow_period]
                buy_checks.append(
                    np.isfinite(fast_ma[i-1]) and np.isfinite(slow_ma[i-1])
                    and np.isfinite(fast_ma[i]) and np.isfinite(slow_ma[i])
                    and fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]
                )

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
                entry_index = i
                # The entry is modeled at the close, so earlier highs from
                # that same candle are not part of the position's trailing peak.
                highest_price = entry_price
                # Lock the initial risk at entry. If both percentage stops are
                # enabled, use the closer price stop (the smaller loss distance).
                configured_risk_pcts = [
                    float(exit_conds.get('stop_loss_pct', 0.0)),
                    float(exit_conds.get('trailing_stop_pct', 0.0)),
                ]
                configured_risk_pcts = [value for value in configured_risk_pcts if value > 0]
                if configured_risk_pcts:
                    initial_risk_pct = min(configured_risk_pcts)
                    initial_stop_price = entry_price * (1.0 - initial_risk_pct / 100.0)
                    take_profit_price = (
                        entry_price + (entry_price - initial_stop_price) * risk_ratio
                        if risk_ratio is not None and risk_ratio > 0 else None
                    )
                else:
                    initial_stop_price = None
                    take_profit_price = None
        else:
            exit_reasons = []
            stop_hits = []

            if exit_conds['stop_loss_pct'] > 0:
                sl_price = entry_price * (1.0 - exit_conds['stop_loss_pct'] / 100.0)
                if lows[i] <= sl_price or closes[i] <= sl_price:
                    exit_reasons.append(f"觸發停損 (-{exit_conds['stop_loss_pct']}%)")
                    stop_hits.append((sl_price, exit_reasons[-1]))

            if exit_conds['trailing_stop_pct'] > 0:
                trail_price = highest_price * (1.0 - exit_conds['trailing_stop_pct'] / 100.0)
                if lows[i] <= trail_price or closes[i] <= trail_price:
                    exit_reasons.append(f"最高點回撤 (-{exit_conds['trailing_stop_pct']}%)")
                    stop_hits.append((trail_price, exit_reasons[-1]))

            if exit_conds['ma_mode'] == '跌破均線' and not np.isnan(ma_exit[i-1]):
                if closes[i-1] >= ma_exit[i-1] and closes[i] < ma_exit[i]:
                    exit_reasons.append(f"跌破 MA{exit_conds['ma_p']}")

            for fast_period, slow_period in exit_ma_cross_pairs:
                fast_ma = cross_ma_values[fast_period]
                slow_ma = cross_ma_values[slow_period]
                if (
                    np.isfinite(fast_ma[i-1]) and np.isfinite(slow_ma[i-1])
                    and np.isfinite(fast_ma[i]) and np.isfinite(slow_ma[i])
                    and fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]
                ):
                    exit_reasons.append(
                        f"均線死亡交叉 MA{fast_period}/MA{slow_period}"
                    )

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
            elif exit_conds['bb_mode'] == '站上布林中軌':
                if np.isfinite(bb_middle[i-1]) and np.isfinite(bb_middle[i]) and closes[i-1] <= bb_middle[i-1] and closes[i] > bb_middle[i]:
                    exit_reasons.append("站上布林中軌停利")

            if exit_conds.get('five_ma_mode') == '排列失序' and not five_line_bullish(i):
                exit_reasons.append("五線多頭排列失序")

            target_hit = take_profit_price is not None and (
                float(highs[i]) >= take_profit_price or float(closes[i]) >= take_profit_price
            )
            if stop_hits:
                if take_profit_price is not None:
                    # If a daily bar crosses both a stop and target, use the
                    # stop first because OHLC data cannot reveal intraday order.
                    # A gap through a stop is modeled at the opening price.
                    stop_level = max(level for level, _reason in stop_hits)
                    opening_price = _finite_number(opens[i]) or float(closes[i])
                    exit_price = min(opening_price, float(stop_level))
                else:
                    # Keep the original close-based stop execution when the
                    # risk/reward target is disabled or cannot be derived.
                    exit_price = float(closes[i])
            elif target_hit:
                # A favorable gap through the target is modeled at the open.
                opening_price = _finite_number(opens[i]) or float(closes[i])
                exit_price = max(opening_price, float(take_profit_price))
                exit_reasons = [
                    f"達成風險報酬停利目標（1:{risk_ratio:g}）"
                ]
            elif exit_reasons:
                # Indicator exits retain the existing close-based execution.
                exit_price = float(closes[i])
            else:
                # Update the trailing peak only after this bar's exit checks;
                # daily OHLC data does not reveal whether its high came first.
                highest_price = max(highest_price, float(highs[i]))
                continue

            if stop_hits and target_hit:
                exit_reasons.append("同根資料也觸及停利目標，依保守方式先計停損")
            if stop_hits or target_hit or exit_reasons:
                exit_date = date_chinese_list[i]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append({
                    '買進日期': entry_date,
                    '買進價格 (元)': round(entry_price, 2),
                    '初始停損參考價 (元)': round(initial_stop_price, 2) if initial_stop_price is not None else None,
                    '目標停利價 (元)': round(take_profit_price, 2) if take_profit_price is not None else None,
                    '設定風險報酬比': f"1:{risk_ratio:g}" if take_profit_price is not None else None,
                    '賣出日期': exit_date,
                    '賣出價格 (元)': round(exit_price, 2),
                    '平倉報酬率 (%)': round(pnl_pct, 2),
                    '離場觸發原因': " | ".join(exit_reasons),
                    '持有K棒數': max(0, i - entry_index),
                    '_entry_index': entry_index,
                    '_exit_index': i,
                    '_entry_fill_price': entry_price,
                    '_exit_fill_price': exit_price,
                })
                position = False

    # Include a still-open position at the final close so the reported result
    # and exposure do not silently omit its unrealized gain or loss.
    if position and len(df_calc) > 0 and entry_index is not None:
        exit_index = len(df_calc) - 1
        exit_price = float(closes[exit_index])
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 if entry_price else 0.0
        trades.append({
            '買進日期': entry_date,
            '買進價格 (元)': round(entry_price, 2),
            '初始停損參考價 (元)': round(initial_stop_price, 2) if initial_stop_price is not None else None,
            '目標停利價 (元)': round(take_profit_price, 2) if take_profit_price is not None else None,
            '設定風險報酬比': f"1:{risk_ratio:g}" if take_profit_price is not None else None,
            '賣出日期': date_chinese_list[exit_index],
            '賣出價格 (元)': round(exit_price, 2),
            '平倉報酬率 (%)': round(pnl_pct, 2),
            '離場觸發原因': '期末未平倉（以最後收盤價估值）',
            '持有K棒數': max(0, exit_index - entry_index),
            '_entry_index': entry_index,
            '_exit_index': exit_index,
            '_entry_fill_price': entry_price,
            '_exit_fill_price': exit_price,
        })

    return pd.DataFrame(trades)


def _annualization_bars(timeframe_name, is_taiwan_stock):
    session_minutes = 270 if is_taiwan_stock else 390
    mapping = {
        "1分鐘": 252 * session_minutes,
        "5分鐘": 252 * (session_minutes // 5),
        "30分鐘": 252 * (session_minutes // 30),
        "60分鐘": 252 * (session_minutes // 60),
        "1日": 252,
        "1週": 52,
        "1個月": 12,
        # The current data mapping fetches quarterly bars for these choices.
        "1季": 4,
        "半年": 2,
        "1年": 1,
    }
    return mapping.get(timeframe_name, 252)


def analyze_indicator_backtest(df_calc, trades, initial_capital, commission_pct,
                               sell_tax_pct, slippage_pct, timeframe_name,
                               is_taiwan_stock):
    """Build a marked-to-market equity curve and comparable buy-and-hold metrics."""
    if df_calc is None or df_calc.empty or 'Close' not in df_calc.columns:
        return None

    close = pd.to_numeric(df_calc['Close'], errors='coerce').astype(float)
    close = close.replace([np.inf, -np.inf], np.nan)
    if close.isna().any() or len(close) < 2 or (close <= 0).any():
        return None

    n = len(close)
    price = close.to_numpy()
    entry_fee_factor = max(0.0, 1.0 - (float(commission_pct) + float(slippage_pct)) / 100.0)
    exit_fee_factor = max(0.0, 1.0 - (float(commission_pct) + float(sell_tax_pct) + float(slippage_pct)) / 100.0)
    strategy_factors = np.ones(n, dtype=float)
    in_market = np.zeros(n, dtype=bool)
    trade_rows = []

    if trades is not None and not trades.empty:
        ordered = trades.sort_values('_entry_index') if '_entry_index' in trades.columns else trades
        for _, trade in ordered.iterrows():
            if '_entry_index' not in trade or '_exit_index' not in trade:
                continue
            entry_i, exit_i = int(trade['_entry_index']), int(trade['_exit_index'])
            if not (0 <= entry_i < n and entry_i <= exit_i < n):
                continue
            entry_fill = _finite_number(trade.get('_entry_fill_price')) or float(price[entry_i])
            exit_fill = _finite_number(trade.get('_exit_fill_price')) or float(price[exit_i])
            strategy_factors[entry_i] *= entry_fee_factor
            for i in range(entry_i + 1, exit_i):
                strategy_factors[i] *= price[i] / price[i - 1]
            if exit_i > entry_i:
                strategy_factors[exit_i] *= exit_fill / price[exit_i - 1]
            strategy_factors[exit_i] *= exit_fee_factor
            in_market[entry_i:exit_i + 1] = True
            gross_ratio = exit_fill / entry_fill
            net_return = entry_fee_factor * gross_ratio * exit_fee_factor - 1.0
            trade_rows.append({
                '買進日期': trade.get('買進日期'),
                '賣出日期': trade.get('賣出日期'),
                '成本後報酬率 (%)': net_return * 100.0,
                '持有K棒數': max(0, exit_i - entry_i),
                '目標停利價 (元)': trade.get('目標停利價 (元)'),
            })

    strategy_equity = pd.Series(float(initial_capital) * np.cumprod(strategy_factors), index=close.index, name='技術策略資產')
    strategy_returns = strategy_equity.pct_change().dropna()
    bars_per_year = _annualization_bars(timeframe_name, is_taiwan_stock)
    span_years = max((close.index[-1] - close.index[0]).total_seconds() / (365.25 * 24 * 60 * 60), 0.0)

    def summarize(equity, returns):
        cumulative = float(equity.iloc[-1] / float(initial_capital) - 1.0)
        drawdown = equity / equity.cummax() - 1.0
        mdd = float(drawdown.min())
        min_observations = min(20, max(2, int(round(bars_per_year * 0.5))))
        enough_history = span_years >= 0.5 and len(returns) >= min_observations
        cagr = float((equity.iloc[-1] / float(initial_capital)) ** (1.0 / span_years) - 1.0) if enough_history and span_years > 0 and equity.iloc[-1] > 0 else None
        ann_vol = float(returns.std(ddof=1) * np.sqrt(bars_per_year)) if enough_history and len(returns) > 1 else None
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(bars_per_year)) if enough_history and len(returns) > 1 and returns.std(ddof=1) > 0 else None
        downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
        downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
        sortino = float(returns.mean() / downside_dev * np.sqrt(bars_per_year)) if enough_history and downside_dev > 0 else None
        calmar = float(cagr / abs(mdd)) if cagr is not None and mdd < 0 else None
        underwater = drawdown.to_numpy() < 0
        longest_underwater = run = 0
        for below_peak in underwater:
            run = run + 1 if below_peak else 0
            longest_underwater = max(longest_underwater, run)
        return {
            'cumulative_return': cumulative, 'cagr': cagr, 'mdd': mdd,
            'annualized_volatility': ann_vol, 'sharpe': sharpe,
            'sortino': sortino, 'calmar': calmar,
            'longest_drawdown_bars': longest_underwater,
            'ending_value': float(equity.iloc[-1]), 'drawdown': drawdown,
        }

    strategy_metrics = summarize(strategy_equity, strategy_returns)
    strategy_metrics['exposure'] = float(in_market.mean()) if n else 0.0

    benchmark_factors = np.ones(n, dtype=float)
    benchmark_factors[0] *= entry_fee_factor
    for i in range(1, n):
        benchmark_factors[i] *= price[i] / price[i - 1]
    benchmark_factors[-1] *= exit_fee_factor
    benchmark_equity = pd.Series(float(initial_capital) * np.cumprod(benchmark_factors), index=close.index, name='同標的買進持有')
    benchmark_metrics = summarize(benchmark_equity, benchmark_equity.pct_change().dropna())

    trade_frame = pd.DataFrame(trade_rows)
    net_returns = trade_frame['成本後報酬率 (%)'] if not trade_frame.empty else pd.Series(dtype=float)
    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]
    # Profit factor is computed from sequential compounded dollar P&L.
    winning_amounts, losing_amounts = [], []
    capital_before = float(initial_capital)
    if not trade_frame.empty:
        for trade_return in net_returns:
            pnl_amount = capital_before * (float(trade_return) / 100.0)
            if pnl_amount > 0:
                winning_amounts.append(pnl_amount)
            elif pnl_amount < 0:
                losing_amounts.append(pnl_amount)
            capital_before += pnl_amount
    profit_factor = (sum(winning_amounts) / abs(sum(losing_amounts))) if losing_amounts else None
    consecutive_losses = longest_loss_streak = 0
    for value in net_returns:
        if value < 0:
            consecutive_losses += 1
            longest_loss_streak = max(longest_loss_streak, consecutive_losses)
        else:
            consecutive_losses = 0
    strategy_metrics.update({
        'trade_count': int(len(net_returns)),
        'win_rate': float((net_returns > 0).mean()) if len(net_returns) else None,
        'profit_factor': float(profit_factor) if profit_factor is not None else None,
        'expectancy_pct': float(net_returns.mean()) if len(net_returns) else None,
        'average_win_pct': float(wins.mean()) if len(wins) else None,
        'average_loss_pct': float(losses.mean()) if len(losses) else None,
        'payoff_ratio': float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else None,
        'average_holding_bars': float(trade_frame['持有K棒數'].mean()) if len(trade_frame) else None,
        'longest_loss_streak': longest_loss_streak,
    })

    if timeframe_name == "1季":
        period_key = close.index.to_period("Q")
        period_label, period_count, period_type = "季報酬", 4, "quarter"
    elif timeframe_name == "半年":
        period_key = pd.MultiIndex.from_arrays([close.index.year, (close.index.month - 1) // 6 + 1])
        period_label, period_count, period_type = "半年報酬", 2, "half_year"
    elif timeframe_name == "1年":
        period_key = close.index.to_period("Y")
        period_label, period_count, period_type = "年報酬", 1, "year"
    else:
        period_key = close.index.to_period("M")
        period_label, period_count, period_type = "月報酬", 12, "month"
    last_positions = pd.Series(np.arange(n), index=close.index).groupby(period_key).last().to_numpy(dtype=int)
    period_ends = strategy_equity.iloc[last_positions]
    period_returns = period_ends.pct_change()
    if len(period_returns):
        period_returns.iloc[0] = period_ends.iloc[0] / float(initial_capital) - 1.0
    period_returns.name = period_label
    return {
        'strategy_equity': strategy_equity,
        'strategy_drawdown': strategy_metrics.pop('drawdown'),
        'strategy_metrics': strategy_metrics,
        'benchmark_equity': benchmark_equity,
        'benchmark_drawdown': benchmark_metrics.pop('drawdown'),
        'benchmark_metrics': benchmark_metrics,
        'trade_analysis': trade_frame,
        'period_returns': period_returns,
        'period_label': period_label,
        'period_count': period_count,
        'period_type': period_type,
        'span_years': span_years,
        'bars_per_year': bars_per_year,
    }

# 🔐 使用者帳號登入與自助註冊
def _logout_account():
    for key in list(st.session_state.keys()):
        if key == "auth_user" or key.startswith((
            "port_", "portfolio_", "login_", "register_", "recovery_",
            "password_reset_", "reset_", "finmind_user_token_",
            "finmind_token_ack_", "finmind_shared_ack_", "branch_data_source_", "broker_branch_ai_",
        )) or key == "pending_recovery_email":
            st.session_state.pop(key, None)


def _clear_finmind_session_token(token_key, acknowledgement_key):
    """Remove a user-entered provider token and its acknowledgement from this session."""
    st.session_state.pop(token_key, None)
    st.session_state.pop(acknowledgement_key, None)


st.sidebar.markdown("### 🔒 操盤手帳號")
auth_user = st.session_state.get("auth_user")
if auth_user:
    st.sidebar.success(f"已登入：{auth_user['username']}")
    st.sidebar.button("登出", key="account_logout", on_click=_logout_account)
else:
    st.sidebar.caption("登入後可使用操盤終端，投資組合資料依帳號分開保存。")
    if not st.session_state.get("account_login_view", False):
        st.markdown("""
            <style>
            .login-hero{max-width:820px;margin:9vh auto 1rem;padding:clamp(1.5rem,5vw,3.25rem);
              border:1px solid rgba(148,163,184,.24);border-radius:24px;text-align:center;
              background:radial-gradient(ellipse at 50% 0%,rgba(37,99,235,.22),transparent 62%),
                         linear-gradient(145deg,#101a2b,#0b1220 68%,#111827);
              box-shadow:0 24px 70px rgba(0,0,0,.35);}
            .login-kicker{color:#60a5fa;font-size:.78rem;font-weight:800;letter-spacing:.16em;}
            .login-hero h1{font-size:clamp(1.8rem,5vw,2.7rem);margin:.7rem 0;color:#f8fafc;}
            .login-hero p{color:#aab7c8;font-size:1.02rem;line-height:1.7;margin:.35rem auto;max-width:570px;}
            .login-pills{display:flex;justify-content:center;flex-wrap:wrap;gap:.55rem;margin-top:1.5rem;}
            .login-pills span{border:1px solid rgba(148,163,184,.22);background:rgba(15,23,42,.68);
              border-radius:99px;padding:.42rem .8rem;color:#cbd5e1;font-size:.82rem;}
            </style>
            <section class="login-hero">
              <div class="login-kicker">SECURE INVESTMENT WORKSPACE</div>
              <h1>操盤手投資終端</h1>
              <p>整合市場診斷、策略研究與個人投資組合，為你的分析流程提供清晰、專屬的工作空間。</p>
              <div class="login-pills"><span>市場分析</span><span>策略研究</span><span>帳號專屬投資組合</span></div>
            </section>
        """, unsafe_allow_html=True)
        splash_cols = st.columns([1, 1.3, 1])
        with splash_cols[1]:
            if st.button("🔐 登入或建立帳號", type="primary", use_container_width=True, key="open_account_login"):
                st.session_state["account_login_view"] = True
                st.rerun()
        st.stop()

    st.markdown("""
        <div style="max-width:820px;margin:1.5rem auto 1rem;text-align:center;">
          <div style="color:#60a5fa;font-size:.75rem;font-weight:800;letter-spacing:.14em;">SECURE INVESTMENT WORKSPACE</div>
          <h2 style="margin:.35rem 0 .25rem;">登入操盤手終端</h2>
          <p style="color:#94a3b8;margin:0;">投資組合資料僅會顯示在登入帳號中。</p>
        </div>
    """, unsafe_allow_html=True)
    auth_columns = st.columns([1, 1.5, 1])
    with auth_columns[1]:
        login_tab, register_tab, reset_tab = st.tabs(["登入", "建立帳號", "忘記密碼"])
        with login_tab:
            with st.form("account_login_form_main"):
                login_name = st.text_input("帳號", key="login_username_main").strip()
                login_password = st.text_input("密碼", type="password", key="login_password_main")
                login_submit = st.form_submit_button("登入", use_container_width=True)
            if login_submit:
                account = _verify_account(login_name, login_password)
                if account:
                    st.session_state["auth_user"] = account
                    st.session_state.pop("account_login_view", None)
                    st.rerun()
                st.error("帳號或密碼不正確。")
        with register_tab:
            st.caption("第一個註冊帳號會接收升級前共用的投資組合資料；之後各帳號資料分開保存。")
            with st.form("account_register_form_main"):
                register_name = st.text_input("建立帳號（3–32 位英數或底線）", key="register_username_main").strip()
                register_email = st.text_input("備援 Email（用於忘記密碼）", key="register_email_main").strip().lower()
                register_password = st.text_input("建立密碼（至少 8 位）", type="password", key="register_password_main")
                register_confirm = st.text_input("再次輸入密碼", type="password", key="register_password_confirm_main")
                register_submit = st.form_submit_button("註冊並登入", use_container_width=True)
            if register_submit:
                if not re.fullmatch(r"[A-Za-z0-9_]{3,32}", register_name):
                    st.error("帳號需為 3–32 位英數字或底線。")
                elif not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", register_email):
                    st.error("請輸入有效的 Email，作為忘記密碼時的驗證信箱。")
                elif len(register_password) < 8:
                    st.error("密碼至少需要 8 位。")
                elif register_password != register_confirm:
                    st.error("兩次輸入的密碼不一致。")
                else:
                    try:
                        user_id, is_first_account = _register_account(register_name, register_password, register_email)
                        st.session_state["auth_user"] = {"id": user_id, "username": register_name}
                        st.session_state.pop("account_login_view", None)
                        if is_first_account:
                            st.session_state["account_notice"] = "已建立第一個帳號；原有共用投資組合資料已保留並歸入此帳號。"
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("這個帳號名稱已有人使用。")
        with reset_tab:
            st.caption("驗證碼會寄到帳號的備援 Email；寄件服務需先設定 Gmail Secrets。")
            with st.form("password_reset_request_form_main"):
                reset_name_input = st.text_input("帳號", key="reset_request_username_main").strip()
                request_reset = st.form_submit_button("寄送重設驗證碼", use_container_width=True)
            if request_reset:
                st.session_state["password_reset_username"] = reset_name_input
                with sqlite3.connect(DB_FILE) as _conn:
                    reset_row = _conn.execute(
                        "SELECT id FROM app_users WHERE username=? COLLATE NOCASE", (reset_name_input,)
                    ).fetchone()
                if reset_row:
                    try:
                        _request_account_code(reset_row[0], "password_reset")
                    except Exception:
                        st.error("目前無法寄送驗證信，請稍後再試或聯絡系統管理者。")
                st.info("若帳號已設定備援信箱，驗證碼會寄到該信箱；請檢查垃圾郵件匣。")
            reset_name = st.session_state.get("password_reset_username", "")
            if reset_name:
                with st.form("password_reset_complete_form_main"):
                    st.caption(f"重設帳號：{reset_name}")
                    reset_code = st.text_input("六位數驗證碼", max_chars=6, key="password_reset_code_main")
                    reset_new_password = st.text_input("新密碼（至少 8 位）", type="password", key="password_reset_new_password_main")
                    reset_confirm = st.text_input("再次輸入新密碼", type="password", key="password_reset_confirm_main")
                    complete_reset = st.form_submit_button("重設密碼", use_container_width=True)
                if complete_reset:
                    if len(reset_new_password) < 8:
                        st.error("新密碼至少需要 8 位。")
                    elif reset_new_password != reset_confirm:
                        st.error("兩次輸入的新密碼不一致。")
                    elif _reset_account_password(reset_name, reset_code, reset_new_password):
                        st.session_state.pop("password_reset_username", None)
                        st.success("密碼已重設，請使用新密碼登入。")
                    else:
                        st.error("驗證碼無效、已逾時或錯誤次數過多。")
    back_cols = st.columns([1, 1.5, 1])
    with back_cols[1]:
        if st.button("返回首頁", key="account_login_back", use_container_width=True):
            st.session_state["account_login_view"] = False
            st.rerun()
    st.stop()

if st.session_state.pop("account_notice", None):
    st.sidebar.info("原有共用投資組合資料已保留並歸入第一個建立的帳號。")

if auth_user:
    with st.sidebar.expander("設定／更新備援 Email"):
        with st.form("recovery_email_request_form"):
            recovery_email = st.text_input("備援 Email", key="recovery_email_input").strip().lower()
            request_email_code = st.form_submit_button("寄送信箱驗證碼", use_container_width=True)
        if request_email_code:
            if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", recovery_email):
                st.error("請輸入有效的 Email 地址。")
            else:
                try:
                    sent = _request_account_code(auth_user["id"], "bind_email", recovery_email)
                    if sent:
                        st.session_state["pending_recovery_email"] = recovery_email
                        st.success("驗證碼已寄出，請查看信箱。")
                    else:
                        st.warning("請稍候一分鐘後再重新寄送驗證碼。")
                except Exception:
                    st.error("目前無法寄送驗證信，請確認系統管理者已設定郵件服務。")
        if st.session_state.get("pending_recovery_email"):
            with st.form("recovery_email_verify_form"):
                email_code = st.text_input("六位數驗證碼", max_chars=6, key="recovery_email_code")
                confirm_email = st.form_submit_button("確認備援 Email", use_container_width=True)
            if confirm_email:
                if _confirm_recovery_email(auth_user["id"], email_code):
                    st.session_state.pop("pending_recovery_email", None)
                    st.success("備援 Email 已驗證並更新。")
                else:
                    st.error("驗證碼無效、已逾時或錯誤次數過多。")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 全球個股即時診斷")
    st.sidebar.caption("💡 提示：上市請加 `.TW`，上櫃請加 `.TWO`（例如：3293.TWO）")
    selected_stock = st.sidebar.text_input(
        "輸入台美股代碼：", value="3293.TWO", key="selected_stock_input"
    ).strip().upper()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 核心策略模組導航")
    MODULE_LIST = [
        "📊 彩色 K 線圖與成交量",
        "📦 籌碼流向參考",
        "🤖 網格自動生成器",
        "⚖️ 網格／持有／定期定額比較",
        "🧪 技術指標自訂策略回測",
        "💸 配息複利策略",
        "🚀 華爾街機構級三核心策略雷達",
        "🏦 台股三大法人獨立選股策略",
        "📊 當日全市場三大法人佔股本比 Top 10",
        "🧭 投資組合理財",
    ]
    # Keep the original option values stable for existing session state and
    # module dispatch conditions; show a category prefix only in the menu.
    MODULE_GROUPS = {
        "📊 彩色 K 線圖與成交量": "📈 圖表與籌碼",
        "📦 籌碼流向參考": "📈 圖表與籌碼",
        "🤖 網格自動生成器": "⚖️ 策略與回測",
        "⚖️ 網格／持有／定期定額比較": "⚖️ 策略與回測",
        "🧪 技術指標自訂策略回測": "⚖️ 策略與回測",
        "💸 配息複利策略": "⚖️ 策略與回測",
        "🚀 華爾街機構級三核心策略雷達": "🏦 選股與市場掃描",
        "🏦 台股三大法人獨立選股策略": "🏦 選股與市場掃描",
        "📊 當日全市場三大法人佔股本比 Top 10": "🏦 選股與市場掃描",
        "🧭 投資組合理財": "🧭 投資組合",
    }
    selected_module = st.sidebar.selectbox(
        "選擇要檢視的分析面板：",
        options=MODULE_LIST,
        index=0,
        format_func=lambda module: f"{MODULE_GROUPS[module]}｜{module}",
        key="sidebar_strategy_selector",
    )
    _render_market_psychology_guide(st, selected_module)

    def _jump_to_stock_diagnosis(ticker):
        st.session_state["selected_stock_input"] = str(ticker).strip().upper()
        st.session_state["sidebar_strategy_selector"] = "📊 彩色 K 線圖與成交量"

    def _queue_observer_ticker(ticker, notice_key):
        ticker = str(ticker).strip().upper()
        defaults = ["2330.TW", "3293.TWO", "8069.TWO", "NVDA", "AAPL"] + [""] * 7
        existing = [str(code).strip().upper() for code in st.session_state.get("trader_watchlist_values", defaults)]
        if ticker in existing:
            st.session_state[notice_key] = f"{ticker} 已在操盤手自訂觀察名單中。"
            return
        try:
            empty_index = existing.index("")
        except ValueError:
            st.session_state[notice_key] = "自訂觀察代碼欄位已滿，請先清出一格再加入。"
            return
        existing[empty_index] = ticker
        st.session_state["trader_watchlist_values"] = existing
        st.session_state["trader_watchlist_sync_needed"] = True
        st.session_state[notice_key] = f"已將 {ticker} 加入操盤手自訂觀察代碼。"

    def _render_stock_actions(ticker, stock_name, key_prefix):
        st.caption(f"已選取：{stock_name}（{ticker}）。選擇要開啟的功能：")
        action_cols = st.columns(2)
        with action_cols[0]:
            st.button(
                "🔍 使用個股即時診斷",
                key=f"{key_prefix}_diagnose",
                on_click=_jump_to_stock_diagnosis,
                args=(ticker,),
                use_container_width=True,
            )
        with action_cols[1]:
            notice_key = f"observer_action_notice_{key_prefix}"
            st.button(
                "➕ 加入自訂觀察代碼",
                key=f"{key_prefix}_watch",
                on_click=_queue_observer_ticker,
                args=(ticker, notice_key),
                use_container_width=True,
            )
        notice = st.session_state.pop(f"observer_action_notice_{key_prefix}", None)
        if notice:
            st.info(notice)

    st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(20, 29, 49, 0.75) 0%, rgba(11, 16, 29, 0.9) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.2rem 1.6rem;
                    margin-bottom: 1.2rem; backdrop-filter: blur(12px); box-shadow: 0 8px 30px rgba(0,0,0,0.35);
                    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
                    <span style="display:inline-block; width:9px; height:9px; border-radius:50%; background:#10B981; box-shadow:0 0 10px #10B981; animation: radar-pulse 2s infinite;"></span>
                    <span style="font-size: 0.75rem; font-weight: 700; color: #10B981; letter-spacing: 0.08em; text-transform: uppercase;">LIVE MARKET TERMINAL</span>
                    <span style="background: rgba(59, 130, 246, 0.2); color: #93C5FD; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; border: 1px solid rgba(59, 130, 246, 0.3);">v3.12 PRO</span>
                </div>
                <h1 style="font-size: 1.7rem; font-weight: 800; margin: 0;
                           background: linear-gradient(90deg, #FFFFFF 0%, #93C5FD 40%, #A78BFA 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    機構級三核心策略雷達 · 全能量化看盤終端
                </h1>
            </div>
            <div style="display: flex; gap: 0.6rem; flex-wrap: wrap;">
                <span style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 5px 12px; border-radius: 8px; font-size: 0.8rem; color: #CBD5E1;">🧠 AI 雙核心診斷</span>
                <span style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 5px 12px; border-radius: 8px; font-size: 0.8rem; color: #CBD5E1;">⚖️ 網格複利與回測</span>
                <span style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); padding: 5px 12px; border-radius: 8px; font-size: 0.8rem; color: #CBD5E1;">🏦 三大法人獨立雷達</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

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

    with col1: st.metric(label="🇹🇼 台灣加權指數 (^TWII)", value=f"{tw_val:,.2f}" if tw_val else "更新中...", delta=f"{tw_chg:+.2f} ({tw_pct:+.2f}%)" if tw_val else "--", help="顯示指數最新可取得點位；下方變化為相較前一筆資料的點數與百分比差異。")
    with col2: st.metric(label="🇺🇸 標普 500 指數 (^GSPC)", value=f"{sp_val:,.2f}" if sp_val else "載入中...", delta=f"{sp_chg:+.2f} ({sp_pct:+.2f}%)" if sp_val else "--", help="追蹤美國大型企業整體股價表現的指數；變化為相較前一筆資料。")
    with col3: st.metric(label="🇺🇸 道瓊工業指數 (^DJI)", value=f"{dj_val:,.2f}" if dj_val else "載入中...", delta=f"{dj_chg:+.2f} ({dj_pct:+.2f}%)" if dj_val else "--", help="由 30 家大型美國企業組成的價格加權指數；變化為相較前一筆資料。")
    with col4: st.metric(label="🇺🇸 那斯達克指數 (^IXIC)", value=f"{nas_val:,.2f}" if nas_val else "載入中...", delta=f"{nas_chg:+.2f} ({nas_pct:+.2f}%)" if nas_val else "--", help="以那斯達克掛牌公司為主的綜合指數；變化為相較前一筆資料。")

    _render_metric_glossary(st)

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

    openai_api_key_input, secret_openai_model = get_openai_secret_values(st.secrets)
    if openai_api_key_input:
        st.sidebar.success("🔒 ChatGPT API 金鑰已從雲端安全設定載入")
    else:
        openai_api_key_input = st.sidebar.text_input(
            "請輸入您的 ChatGPT API Key：", type="password", key="openai_api_key_input"
        )
    openai_model_input = st.sidebar.text_input(
        "ChatGPT 模型名稱", value=secret_openai_model, key="openai_model_input",
        help="預設為 gpt-5.5；若您的 API 帳號沒有此模型權限，可改成帳號可用的模型。",
    ).strip() or "gpt-5.5"
    st.sidebar.caption("ChatGPT API 使用量依 OpenAI API 帳戶計費；ChatGPT 訂閱不包含 API 額度。")

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
            
        df['價量動能估算'] = sim_net
        df['價量動能變化'] = df['價量動能估算'].diff(1)
        diff_values = df['價量動能變化'].dropna().to_numpy().flatten()
        latest_val = float(df['價量動能估算'].to_numpy().flatten()[-1])
        
        count = 0
        if len(diff_values) > 0:
            is_positive = diff_values[-1] > 0
            for val in reversed(diff_values):
                if (val > 0) == is_positive and val != 0: count += 1
                else: break
            status = f"🔥 連續 {count} 根上升" if is_positive else f"📉 連續 {count} 根下降"
        else: status = "🔄 區間震盪"
        
        return df, f"{latest_val:+.2f}% ({status})", latest_val

    tabs_created = False
    grid_advice_metrics = None
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

                stock_obj_main = yf.Ticker(selected_stock)
                try:
                    info_obj_main = stock_obj_main.info or {}
                except Exception:
                    info_obj_main = {}
                c_name = (
                    STOCK_NAME_MAP.get(selected_stock)
                    or info_obj_main.get('longName')
                    or info_obj_main.get('shortName')
                    or selected_stock
                )
                current_price_label = f"當前現價 ({c_name} {selected_stock})"
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
                    with gc1: st.metric(label=f"💰 {current_price_label}", value=f"{price_val:,.2f} 元" if is_tw_stock else f"${price_val:,.2f}", help="資料來源最近更新的價格；可能不是即時成交價，請留意下方資料時間。"); st.caption(f"📢 即時報價 ({cache_time_str})")
                    
                    if bv_val is not None:
                        with gc2: st.metric(label="👑 GVI 成長價值", value=gvi_str, help="把帳面價值、目前股價與獲利能力放在一起估算的研究指標；不是合理價或未來報酬預測。"); st.caption("📢 最新重算")
                        with gc3: st.metric(label="📊 ROE 股東權益報酬率", value=roe_str, help="公司以股東權益創造盈餘的比率；需搭配產業特性及財報期間判讀。"); st.caption(f"📢 {roe_src}")
                        with gc4: st.metric(label="📖 每股淨值", value=bv_str, help="公司帳面資產扣除負債後，平均分配到每一股的金額；不等於股票的市場價格。"); st.caption(f"📢 {bv_src}")
                        with gc5: st.metric(label="⚖️ 股價淨值比 (PB)", value=pb_str, help="顯示目前股價相對每股帳面價值的倍數；產業間差異大，跨產業比較要小心。"); st.caption("📢 溢價程度")
                    else:
                        with gc2: st.info("⚠️ 缺乏財報淨值數據，系統自動隱藏 PB 與相關估值欄位。建議切換至「操盤/動能模式」觀看。")
                else:
                    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                    with gc1: st.metric(label=f"⚡ {current_price_label}", value=f"{price_val:,.2f}", delta=f"{change:+.2f} ({change_pct:+.2f}%)", help="價格與漲跌以圖表目前選取的週期為準，最新一筆資料可能有延遲。")
                    with gc2: st.metric(label="🌊 最近一根 K 棒成交量", value=f"{vol_today:,.0f}", delta=f"量比: {vol_ratio:.2f}x", delta_color="normal" if vol_ratio >= 1 else "off", help="量比將最近一根 K 棒的成交量和近 5 根平均量比較；高於 1 表示比近期平均活躍，無法單獨辨別買賣方向。")
                    with gc3: st.metric(label="🎯 5根均線乖離率", value=f"{bias5:+.2f}%", delta="偏離大留意回檔" if abs(bias5) > 5 else "乖離正常", delta_color="inverse" if abs(bias5) > 5 else "off", help="最新收盤價相對近 5 根收盤均價的百分比差距；用來觀察短期偏離，不是反轉訊號。")
                    with gc4: st.metric(label="🔺 最近一根 K 棒最高價", value=f"{high_today:,.2f}", help="所選圖表週期中最新一根 K 棒的最高價。")
                    with gc5: st.metric(label="🔻 最近一根 K 棒最低價", value=f"{low_today:,.2f}", help="所選圖表週期中最新一根 K 棒的最低價。")

                # ======= 升級：計算三大策略所需數值與交叉分析 =======
                L = len(df_chart)
                close_today = price_val
                close_history = pd.to_numeric(df_chart['Close'], errors='coerce').dropna() if 'Close' in df_chart.columns else pd.Series(dtype=float)
                high_252d = float(close_history.tail(min(252, len(close_history))).max()) if len(close_history) >= 50 else np.nan
                momentum_score = float((close_today / high_252d) * 100) if pd.notna(high_252d) and high_252d > 0 else None

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
                if gvi_val is None:
                    scores['GVI 價值雷達'] = None
                    reasons['GVI 價值雷達'] = "基本面資料不足，未評分。"
                elif gvi_val >= 0.20:
                    scores['GVI 價值雷達'] = 90 if gvi_val >= 0.35 else 75
                    reasons['GVI 價值雷達'] = f"成長價值分數為 {gvi_val:.4f}，達到目前設定的價值研究門檻；尚未證明能帶來較佳績效。"
                else:
                    scores['GVI 價值雷達'] = 40
                    reasons['GVI 價值雷達'] = f"成長價值分數為 {gvi_val:.4f}，未達目前設定的價值研究門檻。"

                # 2. 動能突破策略評分
                if momentum_score is None:
                    scores['動能突破雷達'] = None
                    reasons['動能突破雷達'] = "有效歷史收盤資料少於 50 根，動能值缺失且不納入評分。"
                elif momentum_score >= 80:
                    scores['動能突破雷達'] = 95 if momentum_score >= 95 else 85
                    reasons['動能突破雷達'] = f"目前價格約在近一年可用最高收盤價的 {momentum_score:.1f}%，達到趨勢研究門檻。"
                else:
                    scores['動能突破雷達'] = 50
                    reasons['動能突破雷達'] = f"目前價格約在近一年可用最高收盤價的 {momentum_score:.1f}%，未達趨勢研究門檻。"

                # 3. QARP 現金流策略評分
                if fcf_yield_main is None or peg_val_main is None:
                    scores['QARP 現金流雷達'] = None
                    reasons['QARP 現金流雷達'] = "自由現金流收益率或 PEG 資料不足，未評分。"
                elif fcf_yield_main >= 2.5 and peg_val_main <= 1.5:
                    scores['QARP 現金流雷達'] = 90
                    reasons['QARP 現金流雷達'] = f"現金流收益參考為 {fcf_yield_main:.2f}%，成長估值參考為 {peg_val_main:.2f}，兩項均達目前研究門檻。"
                else:
                    scores['QARP 現金流雷達'] = 45
                    reasons['QARP 現金流雷達'] = f"現金流收益參考為 {fcf_yield_main:.2f}%，成長估值參考為 {peg_val_main:.2f}；至少一項未達目前研究門檻。"

                three_core_psychology = build_three_core_market_psychology_notes(
                    gvi_val, momentum_score, fcf_yield_main, peg_val_main
                )

                # 顯示目前門檻的符合度，不把它解讀為實證優勢或進場建議。
                valid_scores = {name: score for name, score in scores.items() if score is not None}
                best_strategy = max(valid_scores, key=valid_scores.get) if valid_scores else "資料不足"
                best_score = valid_scores.get(best_strategy, 0)
                
                if best_score >= 90:
                    valuation_color = "#00cc66"
                elif best_score >= 75:
                    valuation_color = "#2baf2b"
                elif best_score >= 60:
                    valuation_color = "#ff9900"
                else:
                    valuation_color = "#cc0000"

                fin_src_display = f"每股淨值來源：{bv_src}；ROE 來源：{roe_src}"

                advice = f"📊 **目前研究分數最高的單項條件：【{best_strategy}】（研究比對分數：{best_score}分，不是勝率）**<br><br>"
                advice += f"📌 **採用數據源**：{fin_src_display}<br>"
                if was_missing_in_db:
                    advice += f"📡 **備援提示**：自動啟動多層次財報遞補機制以確保診斷不中斷。<br>"
                advice += f"🔍 **門檻比對摘要**：<br>"
                psychology_key_by_strategy = {
                    "GVI 價值雷達": "GVI 價值面",
                    "動能突破雷達": "動能趨勢面",
                    "QARP 現金流雷達": "QARP 現金流與估值面",
                }
                for strat, sc in scores.items():
                    icon = "●" if strat == best_strategy else "·"
                    score_label = f"研究比對分數 {sc} 分" if sc is not None else "資料不足，未評分"
                    psych_label = three_core_psychology["notes"].get(psychology_key_by_strategy.get(strat), "")
                    advice += (
                        f"&nbsp;&nbsp;&nbsp;&nbsp;{icon} **{strat}**（{score_label}）：{reasons[strat]}<br>"
                        f"&nbsp;&nbsp;&nbsp;&nbsp;🧠 **市場心理角度**：{psych_label}<br>"
                    )
                advice += (
                    f"<br>🧭 **三核心合併解讀**：{three_core_psychology['overall']}<br>"
                    "⚠️ **解讀方式**：以上只表示目前數值與研究門檻的相符程度。各因子尚未完成獨立樣本外比較，"
                    "不代表長期報酬優勢，也不構成買賣或部位建議。"
                )

                # ======= 視覺化圖表整合 (Radar Chart) =======
                diag_c1, diag_c2 = st.columns([1.5, 1])
                with diag_c1:
                    st.markdown(
                        f"<div style='background-color:rgba(30,30,30,0.7); padding:14px 18px; border-left:6px solid {valuation_color}; border-radius:4px; margin-bottom:15px; height: 100%;'>"
                        f"<h5 style='margin:0; color:white;'>⚖️ 三核心研究條件比對</h5>"
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
                
                st.info(f"🔮 【{c_name}】{selected_tf} 價量動能估算：{chip_status_text}")

                with st.expander(f"🧠 AI 雙核心個股研究報告與 Email 傳送【{c_name}】", expanded=(f"ai_report_{selected_stock}" in st.session_state)):
                    branch_ai_snapshot_key = _broker_branch_ai_state_key(auth_user.get("id", "guest"), selected_stock)
                    current_branch_ai_context = format_broker_branch_ai_context(
                        st.session_state.get(branch_ai_snapshot_key), selected_stock
                    )
                    current_branch_snapshot = st.session_state.get(branch_ai_snapshot_key)
                    current_branch_ai_interpretation = ""
                    if current_branch_ai_context:
                        current_branch_ai_interpretation = build_broker_branch_ai_interpretation(
                            current_branch_snapshot,
                            pd.to_numeric(df_chart["Close"], errors="coerce").dropna().tail(20),
                            selected_tf,
                        )
                    if current_branch_ai_context:
                        st.info(
                            "目前登入工作階段已有這檔股票的分點摘要。重新產生 AI 報告時會將家數、淨額與集中度摘要傳送給你選擇的 AI 服務；"
                            "不會傳送完整分點明細。若目前已經有舊報告，請重新分析才會納入這份分點資料。"
                        )
                    else:
                        st.caption("若先在「籌碼流向參考」匯入這檔股票的分點資料，重新產生報告時會一併分析。")
                    ai_provider = st.radio(
                        "AI 分析來源",
                        ["Gemini", "ChatGPT（含網路搜尋）"],
                        horizontal=True,
                        index=1 if openai_api_key_input else 0,
                        key=f"ai_provider_{selected_stock}",
                        help="Gemini 使用本系統行情與可取得的新聞；ChatGPT 會額外搜尋近期市場、產業題材與同業資料並附來源。",
                    )
                    if st.button(f"🧠 啟動 AI 投資分析報告【{c_name}】", use_container_width=True, type="primary"):
                        provider_name = "ChatGPT" if ai_provider.startswith("ChatGPT") else "Gemini"
                        provider_key = openai_api_key_input if provider_name == "ChatGPT" else api_key_input
                        if not provider_key:
                            label = "ChatGPT API Key" if provider_name == "ChatGPT" else "Gemini API Key"
                            st.error(f"⚠️ 請先在左側邊欄設定 {label}，或改選另一個分析來源。")
                        else:
                            with st.spinner(f"🤖 正在取得歷史日線、執行樣本外回測並整理 {c_name} 研究報告…"):
                                try:
                                    daily_history = yf.download(
                                        selected_stock, period="5y", interval="1d", auto_adjust=False,
                                        actions=False, progress=False, threads=False,
                                    )
                                    ai_backtest = run_ai_candidate_backtests(daily_history, is_tw_stock)
                                    market_snapshot = []
                                    for label, value, index_change_pct in (
                                        ("台灣加權指數", tw_val, tw_pct),
                                        ("標普500", sp_val, sp_pct),
                                        ("道瓊", dj_val, dj_pct),
                                        ("那斯達克", nas_val, nas_pct),
                                    ):
                                        if value is not None:
                                            market_snapshot.append(
                                                f"{label} {value:,.2f}，當次畫面漲跌 {index_change_pct:+.2f}%" if index_change_pct is not None
                                                else f"{label} {value:,.2f}"
                                            )
                                    company_context = collect_stock_research_context(
                                        stock_obj_main, info_obj_main, market_snapshot
                                    )
                                    branch_snapshot = st.session_state.get(branch_ai_snapshot_key)
                                    broker_branch_ai_context = format_broker_branch_ai_context(
                                        branch_snapshot, selected_stock
                                    )
                                    broker_branch_ai_interpretation = ""
                                    if broker_branch_ai_context:
                                        ai_close_window = pd.to_numeric(df_chart["Close"], errors="coerce").dropna().tail(20)
                                        broker_branch_ai_interpretation = build_broker_branch_ai_interpretation(
                                            branch_snapshot, ai_close_window, selected_tf
                                        )
                                    ai_psychology = build_three_core_market_psychology_notes(
                                        gvi_val, momentum_score, fcf_yield_main, peg_val_main
                                    )
                                    ai_psychology_lines = [
                                        f"• {factor}：{interpretation}"
                                        for factor, interpretation in ai_psychology["notes"].items()
                                    ]
                                    ai_psychology_lines.extend([
                                        f"• 三核心合併觀察：{ai_psychology['overall']}",
                                        f"• 價格與成交量：{build_price_volume_market_psychology(change_pct, vol_ratio, selected_tf)}",
                                        "• 解讀限制：以上是市場行為線索，不是對投資人身分或真實情緒的判定，也不是買賣訊號。",
                                    ])
                                    if broker_branch_ai_interpretation:
                                        ai_psychology_lines.insert(
                                            -1,
                                            "• 分點家數、淨額、流向集中度與近 20 筆價格：\n"
                                            + broker_branch_ai_interpretation,
                                        )
                                    market_psychology_brief = "\n".join(ai_psychology_lines)
                                    quant_summary = (
                                        f"成長價值研究分數={gvi_str}；股東權益報酬率={roe_str}；股價淨值比={pb_str}；"
                                        f"目前研究比對分數最高的單項條件={best_strategy}（{best_score}/100，這是條件比對分數，不是勝率）；"
                                        f"價量動能估算={chip_status_text}；目前股價={price_val:,.2f}；最近收盤變化={change_pct:+.2f}%\n"
                                        f"三核心白話心理解讀：\n{market_psychology_brief}"
                                    )
                                    if broker_branch_ai_context:
                                        quant_summary += (
                                            f"\n\n{broker_branch_ai_context}\n\n"
                                            "以下是本次分析必須逐項解讀的分點與價格資料；請在市場心理觀察中引用具體數字，"
                                            "不得略過家數、淨額、集中度或近 20 筆價格方向：\n"
                                            f"{broker_branch_ai_interpretation}"
                                        )
                                    ai_prompt = build_ai_investment_prompt(
                                        selected_stock, c_name, provider_name, quant_summary,
                                        company_context, ai_backtest["text"],
                                    )
                                    if provider_name == "ChatGPT":
                                        success_report, ai_sources = generate_openai_research(
                                            ai_prompt, openai_api_key_input, openai_model_input
                                        )
                                    else:
                                        success_report = generate_content_with_retry(ai_prompt, api_key_input)
                                        ai_sources = []
                                    if broker_branch_ai_interpretation:
                                        success_report = (
                                            str(success_report).rstrip()
                                            + "\n\n### 分點流向與近 20 筆價格方向｜資料補充解讀\n"
                                            + broker_branch_ai_interpretation
                                        )
                                    report_key = f"ai_report_{selected_stock}"
                                    st.session_state[report_key] = success_report
                                    st.session_state[f"{report_key}_provider"] = provider_name
                                    st.session_state[f"{report_key}_backtest"] = ai_backtest
                                    st.session_state[f"{report_key}_sources"] = ai_sources
                                    st.session_state[f"{report_key}_psychology"] = market_psychology_brief
                                    st.session_state[f"{report_key}_branch_context"] = broker_branch_ai_context
                                except Exception as ai_e:
                                    st.error(f"{provider_name} 投資研究報告產生失敗：{ai_e}")
                    
                    if f"ai_report_{selected_stock}" in st.session_state:
                        report_provider = st.session_state.get(f"ai_report_{selected_stock}_provider", "AI")
                        st.success(f"📋 {report_provider}【{c_name}】市場與策略研究報告")
                        saved_report_branch_context = st.session_state.get(
                            f"ai_report_{selected_stock}_branch_context", ""
                        )
                        saved_report_text = str(st.session_state.get(f"ai_report_{selected_stock}", ""))
                        branch_report_heading = "### 分點流向與近 20 筆價格方向｜資料補充解讀"
                        report_has_branch_interpretation = branch_report_heading in saved_report_text
                        if current_branch_ai_interpretation and (
                            saved_report_branch_context != current_branch_ai_context
                            or not report_has_branch_interpretation
                        ):
                            st.warning("目前這份 AI 報告尚未納入目前工作階段的分點資料。請重新產生報告；以下先列出本次分點與價格的資料補充解讀。")
                            st.markdown("##### 分點流向與近 20 筆價格方向｜資料補充解讀")
                            st.markdown(current_branch_ai_interpretation)
                        saved_psychology_brief = st.session_state.get(f"ai_report_{selected_stock}_psychology")
                        if saved_psychology_brief:
                            st.markdown("#### 🧠 市場心理重點")
                            st.markdown(saved_psychology_brief)
                        saved_branch_context = st.session_state.get(f"ai_report_{selected_stock}_branch_context", "")
                        if saved_branch_context:
                            with st.expander("查看本次 AI 分析使用的分點摘要", expanded=False):
                                st.markdown(saved_branch_context)
                        ai_report = st.session_state[f"ai_report_{selected_stock}"]
                        st.markdown(ai_report)
                        saved_backtest = st.session_state.get(f"ai_report_{selected_stock}_backtest", {})
                        saved_backtest_table = saved_backtest.get("table") if isinstance(saved_backtest, dict) else None
                        if isinstance(saved_backtest_table, pd.DataFrame) and not saved_backtest_table.empty:
                            with st.expander("查看系統自動回測比較", expanded=False):
                                display_backtest = saved_backtest_table.copy()
                                for metric_column in (
                                    "前70%平均每筆損益(%)", "後30%平均每筆損益(%)",
                                    "後30%最大回撤(%)",
                                ):
                                    if metric_column in display_backtest:
                                        display_backtest[metric_column] = pd.to_numeric(
                                            display_backtest[metric_column], errors="coerce"
                                        ).round(2)
                                if "後30%賺賠金額比" in display_backtest:
                                    display_backtest["後30%賺賠金額比"] = pd.to_numeric(
                                        display_backtest["後30%賺賠金額比"], errors="coerce"
                                    ).round(2)
                                st.dataframe(display_backtest, use_container_width=True, hide_index=True)
                                st.caption(saved_backtest.get("assumptions", ""))
                        elif isinstance(saved_backtest, dict) and saved_backtest.get("text"):
                            st.info(saved_backtest["text"])
                        recipient_email = st.text_input(
                            "寄送分析報告到 Email",
                            key=f"ai_report_email_recipient_{selected_stock}",
                            placeholder="請輸入收件信箱",
                        )
                        if st.button(
                            "✉️ 寄出這份分析",
                            key=f"send_ai_report_email_{selected_stock}",
                            use_container_width=True,
                        ):
                            if not recipient_email.strip() or "@" not in recipient_email:
                                st.error("請輸入有效的收件 Email 地址。")
                            else:
                                try:
                                    gmail_sender, gmail_app_password = get_gmail_secret_values(st.secrets)
                                except Exception:
                                    gmail_sender, gmail_app_password = "", ""
                                if not gmail_sender or not gmail_app_password:
                                    st.error(
                                        "尚未設定 Gmail 寄件資料。請在 Streamlit Cloud 的 App Settings → Secrets "
                                        "加入 [gmail] 區段（sender_email、app_password），儲存後重新啟動。"
                                    )
                                else:
                                    email_subject = f"AI 個股分析｜{selected_stock} {c_name}"
                                    email_psychology_brief = st.session_state.get(
                                        f"ai_report_{selected_stock}_psychology",
                                        "此報告產生於市場心理摘要功能加入前；目前報告內可能已有心理面分析。",
                                    )
                                    saved_email_branch_context = st.session_state.get(
                                        f"ai_report_{selected_stock}_branch_context", ""
                                    )
                                    email_branch_context = current_branch_ai_context or saved_email_branch_context
                                    email_branch_interpretation = ""
                                    email_branch_is_current = bool(
                                        current_branch_ai_interpretation
                                        and saved_email_branch_context == current_branch_ai_context
                                        and branch_report_heading in str(ai_report)
                                    )
                                    if current_branch_ai_interpretation and not email_branch_is_current:
                                        email_branch_interpretation = (
                                            "【目前分點資料補充解讀｜AI 報告尚未重新產生】\n"
                                            + current_branch_ai_interpretation
                                            + "\n\n"
                                        )
                                    email_branch_section = (
                                        f"【本次 AI 分析使用的分點摘要】\n{email_branch_context}\n\n"
                                        if email_branch_context else ""
                                    )
                                    email_body = (
                                        f"個股：{selected_stock} {c_name}\n"
                                        f"寄送時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                                        f"{email_branch_section}"
                                        f"{email_branch_interpretation}"
                                        f"【市場心理重點】\n{email_psychology_brief}\n\n"
                                        f"{report_provider} 分析結果：\n\n{ai_report}"
                                    )
                                    try:
                                        with st.spinner("正在寄送分析報告…"):
                                            send_gmail_report(
                                                recipient_email,
                                                email_subject,
                                                email_body,
                                                gmail_sender,
                                                gmail_app_password,
                                            )
                                        st.success(f"分析報告已寄送至 {recipient_email.strip()}。")
                                    except Exception as mail_error:
                                        st.error(f"寄信失敗：{mail_error}")
                st.markdown("---")
        except Exception as e:
            st.error(f"數據載入異常：{e}")

    try:
        st.markdown(f"""
            <div style="margin: 1.2rem 0 1.2rem 0; background: linear-gradient(90deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.4) 100%);
                        border-left: 5px solid #3B82F6; border-radius: 0 14px 14px 0; padding: 0.85rem 1.4rem;
                        display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.8rem;
                        box-shadow: 0 6px 20px rgba(0,0,0,0.3); border-top: 1px solid rgba(255,255,255,0.06); border-bottom: 1px solid rgba(255,255,255,0.06);">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.4rem;">🎯</span>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: 0.02em;">
                        當前檢視策略面板：<span style="color: #60A5FA;">{selected_module}</span>
                    </div>
                </div>
                <div style="background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.35); padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; color: #93C5FD;">
                    👈 可隨時於左側邊欄切換 10 種策略模組
                </div>
            </div>
        """, unsafe_allow_html=True)

        if df_chart is None or df_chart.empty:
            if selected_module in ["📊 彩色 K 線圖與成交量", "📦 籌碼流向參考", "🤖 網格自動生成器", "⚖️ 網格／持有／定期定額比較", "🧪 技術指標自訂策略回測", "💸 配息複利策略"]:
                st.error(f"❌ 無此標的或無法取得數據：【{selected_stock}】，請檢查股票代碼是否正確。")
        else:
            
            # ==============================================================================
            # 【Tab 1: Plotly 雙子圖原生 K 線圖 + 成交量 + 完整畫線與文字工具箱】
            # ==============================================================================
            if selected_module == "📊 彩色 K 線圖與成交量":
                ma_display_html = "<div style='background-color:rgba(20,20,20,0.8); padding:4px 8px; border:1px solid #444; border-radius:6px; display:inline-block; font-size:12px; color:white; vertical-align:middle;'>"
                for ma in personal_ma_configs:
                    p, c = ma["period"], ma["color"]
                    ma_series = df_chart['Close'].rolling(window=p).mean().dropna()
                    if not ma_series.empty:
                        latest_ma_val = ma_series.to_numpy().flatten()[-1]
                        ma_display_html += f"<span style='color:{c}; font-weight:bold; margin-right:8px;'>■ {p}MA: {latest_ma_val:,.2f}</span>"
                ma_display_html += "</div>"
                st.markdown(f"### 📊 【{c_name}】專業 K 線圖與成交量 {ma_display_html}", unsafe_allow_html=True)
                st.info("🧠 市場行為觀察：" + build_price_volume_market_psychology(change_pct, vol_ratio, selected_tf))

                with st.expander("🎨 畫線顏色與粗細設定", expanded=False):
                    draw_color_col, draw_width_col = st.columns(2)
                    with draw_color_col:
                        draw_color = st.color_picker("自訂畫線／文字顏色", value="#FF3333", key="draw_line_color_picker")
                    with draw_width_col:
                        draw_line_width = st.slider("線條粗細", min_value=0.5, max_value=10.0, value=2.5, step=0.5, key=f"draw_line_width_{selected_stock}",
                                                    help="設定接下來繪製的趨勢線、矩形與圓形邊框粗細。")
                clear_shapes = st.button("🧹 清除全部畫線", key=f"clear_shapes_{auth_user['id']}_{selected_stock}_{selected_tf}", use_container_width=True)
                if clear_shapes:
                    clear_key = f"shape_clear_revision_{auth_user['id']}_{selected_stock}_{selected_tf}"
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
                    editrevision=int(st.session_state.get(f"shape_clear_revision_{auth_user['id']}_{selected_stock}_{selected_tf}", 0)),
                    height=520,
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
                
                chart_key = f"candlestick_chart_{auth_user['id']}_{selected_stock}_{selected_tf}"
                clear_revision = int(st.session_state.get(f"shape_clear_revision_{auth_user['id']}_{selected_stock}_{selected_tf}", 0))
                fig.update_layout(meta={"codex_chart_key": chart_key})
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    key=chart_key,
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
                        'edits': {'shapePosition': True},
                        'scrollZoom': True,
                        'responsive': True
                    }
                )
                render_plotly_delete_hotkey(st, chart_key, clear_revision)
                st.caption("單筆刪除：先點標線邊界使其選取，再按工具列橡皮擦；電腦也可將游標移到圖表內按 Delete。手機可左右滑動工具列找到橡皮擦，或按上方清除全部。")

            if selected_module == "💸 配息複利策略":
                render_dividend_tab(st, default_ticker=selected_stock)
            
            if selected_module == "📦 籌碼流向參考":
                st.markdown("### 📦 市場走勢與資金流向參考")
                st.caption("觀察價格趨勢、成交熱度、三大法人當日動向，以及集保持股級距的週變化。沒有資料時會明確標示，不把缺漏當成零。")
                price_history = df_chart[[column for column in ("Close", "Volume") if column in df_chart.columns]].copy()
                institutional_trade = None
                issued_shares = None
                inst_errors = []
                stock_code = selected_stock.replace('.TWO', '').replace('.TW', '')
                is_taiwan_stock = selected_stock.endswith(('.TW', '.TWO'))
                if selected_stock.endswith(('.TW', '.TWO')):
                    with st.spinner('讀取官方法人交易與已發行股數資料…'):
                        inst_trades, inst_shares, inst_errors = fetch_institutional_market_data()
                    institutional_trade = inst_trades.get(stock_code)
                    issued_shares = inst_shares.get(stock_code)
                daily_volume = None
                trade_date = _normalize_chip_trade_date(institutional_trade.get('日期')) if institutional_trade else None
                if selected_tf == "1日" and trade_date and "Volume" in price_history.columns:
                    for index_value, volume_value in pd.to_numeric(price_history["Volume"], errors="coerce").dropna().items():
                        if _normalize_chip_trade_date(index_value) == trade_date:
                            daily_volume = _finite_number(volume_value)
                            break
                if is_taiwan_stock:
                    save_chip_flow_observation(stock_code, institutional_trade, issued_shares, daily_volume)

                flow_summary = build_chip_flow_summary(price_history, institutional_trade, issued_shares, selected_tf)
                flow_cards = [
                    ("trend", "最近價格走勢"),
                    ("volume", "最近成交熱度"),
                    ("institutions", "法人當日方向"),
                ]
                if selected_tf == "1日":
                    flow_cards.append(("high_120", "近120日日線位置"))
                flow_cols = st.columns(len(flow_cards))
                available_statuses = []
                for col, (key, label) in zip(flow_cols, flow_cards):
                    item = flow_summary.get(key)
                    with col:
                        st.metric(label, item["status"] if item else "資料暫缺")
                        st.caption(item["detail"] if item else "目前資料不足或來源沒有回傳；不以零代替。")
                    if item:
                        available_statuses.append(item["status"].split("｜")[0])

                if available_statuses:
                    st.info("簡單看：" + "；".join(available_statuses) + "。這些是近期市場狀況描述，不是單獨的買進或賣出訊號。")
                if selected_stock.endswith(('.TW', '.TWO')) and inst_errors:
                    st.caption("法人來源目前未完整回傳；上方只顯示實際取得的資料。")
                st.caption("價格趨勢使用近 20 筆收盤價；成交熱度比較近 5 筆與前段平均成交量；法人方向使用官方當日三大法人資料。各項期間不同，請分開解讀。")

                if institutional_trade:
                    investor_rows = []
                    for investor_name in ("外資", "投信", "自營商"):
                        net_shares = _finite_number(institutional_trade.get(f"{investor_name}買賣超股數"))
                        net_pct = (net_shares / issued_shares * 100.0
                                   if net_shares is not None and issued_shares and issued_shares > 0 else None)
                        investor_rows.append({
                            "法人類別": investor_name,
                            "當日買賣超（股）": net_shares,
                            "占已發行股數（%）": net_pct,
                        })
                    st.markdown("#### 🏦 三大法人當日明細")
                    st.caption(f"資料日期：{institutional_trade.get('日期', '未提供')}；不同法人分開列示，缺漏資料保留空白。")
                    st.dataframe(pd.DataFrame(investor_rows), hide_index=True, use_container_width=True)

                st.markdown("#### 📚 法人累積觀察（本系統開始記錄後）")
                history = load_chip_flow_observations(stock_code) if is_taiwan_stock else pd.DataFrame()
                if not history.empty:
                    complete_history = history.dropna(subset=["foreign_net", "trust_net", "dealer_net"])
                    st.caption(
                        f"目前記錄 {len(history)} 個交易日，其中 {len(complete_history)} 筆含完整三大法人資料。"
                        "只保存本頁曾成功取得的官方資料；未造補缺漏日期，也不代表連續交易日。"
                    )
                    history_cols = st.columns(2)
                    for col, window in zip(history_cols, (5, 20)):
                        label = f"最近 {window} 筆完整觀察合計"
                        with col:
                            if len(complete_history) >= window:
                                sample = complete_history.head(window)
                                total_net = float(sample[["foreign_net", "trust_net", "dealer_net"]].to_numpy().sum())
                                share_values = pd.to_numeric(sample["issued_shares"], errors="coerce").dropna()
                                share_pct = total_net / float(share_values.iloc[0]) * 100.0 if not share_values.empty and share_values.iloc[0] > 0 else None
                                st.metric(label, f"{total_net:+,.0f} 股", delta_color="off")
                                pct_text = f"（約 {share_pct:+.4f}% 已發行股數）" if share_pct is not None else ""
                                st.caption(
                                    f"觀察區間：{sample.iloc[-1]['trade_date']} 至 {sample.iloc[0]['trade_date']}{pct_text}。"
                                    "這是法人買賣超加總，不等於法人持股增減。"
                                )
                            else:
                                st.metric(label, f"累積中 {len(complete_history)}/{window} 筆", delta_color="off")
                                st.caption("尚未累積足夠的完整官方資料，不顯示期間合計。")
                    investor_period_rows = []
                    for window in (5, 20):
                        for column, investor_name in (
                            ("foreign_net", "外資"),
                            ("trust_net", "投信"),
                            ("dealer_net", "自營商"),
                        ):
                            available = history.dropna(subset=[column]).head(window)
                            row = {"觀察期間": f"最近 {window} 筆", "法人類別": investor_name}
                            if len(available) >= window:
                                net_total = float(available[column].sum())
                                share_values = pd.to_numeric(available["issued_shares"], errors="coerce").dropna()
                                net_pct = (net_total / float(share_values.iloc[0]) * 100.0
                                           if not share_values.empty and share_values.iloc[0] > 0 else None)
                                row["累計買賣超（股）"] = f"{net_total:+,.0f}"
                                row["約占已發行股數（%）"] = f"{net_pct:+.4f}%" if net_pct is not None else "資料暫缺"
                                volume_sample = history.dropna(subset=[column, "volume_shares"]).head(window)
                                if (len(volume_sample) >= window
                                        and set(volume_sample["trade_date"]) == set(available["trade_date"])):
                                    total_volume = float(pd.to_numeric(volume_sample["volume_shares"], errors="coerce").sum())
                                    row["占同期成交量（%）"] = f"{net_total / total_volume * 100.0:+.2f}%" if total_volume > 0 else "資料暫缺"
                                else:
                                    row["占同期成交量（%）"] = "日線成交量資料不足"
                                row["狀態"] = "已累積"
                            else:
                                row["累計買賣超（股）"] = "—"
                                row["約占已發行股數（%）"] = "—"
                                row["占同期成交量（%）"] = "—"
                                row["狀態"] = f"累積中 {len(available)}/{window} 筆"
                            investor_period_rows.append(row)
                    st.dataframe(pd.DataFrame(investor_period_rows), hide_index=True, use_container_width=True)
                    st.caption("上表按法人分開計算；成交量占比只在資料庫有同日線成交量時顯示。買賣超是交易流量，不能直接解讀為持股增加或未來漲跌。")
                elif is_taiwan_stock:
                    st.info("資料累積中：成功取得每日官方資料後會依股票與交易日保存，足夠筆數前不顯示 5／20 筆合計。")
                else:
                    st.info("法人累積觀察僅適用台股。")

                st.markdown("#### 👥 集保持股級距變化（週資料）")
                st.caption("以持股級距觀察小額與大額持有比例變化。這是持有分布，不代表買賣方向，也無法直接判定持有人是散戶或主力。")
                if is_taiwan_stock:
                    tdcc_data = None
                    try:
                        with st.spinner('讀取集保中心持股級距資料…'):
                            tdcc_data = fetch_tdcc_concentration(selected_stock)
                    except Exception:
                        tdcc_data = None
                    if tdcc_data:
                        retail_delta = tdcc_data.get('retail_change_pp')
                        big_delta = tdcc_data.get('big_change_pp')
                        tdcc_cols = st.columns(2)
                        with tdcc_cols[0]:
                            st.metric(
                                "小額持股級距（≤5,000 股）",
                                f"{tdcc_data['retail_pct']:.2f}%",
                                delta=f"{retail_delta:+.2f} 個百分點" if retail_delta is not None else None,
                                delta_color="off",
                            )
                        with tdcc_cols[1]:
                            st.metric(
                                "大額持股級距（≥400,001 股）",
                                f"{tdcc_data['big_pct']:.2f}%",
                                delta=f"{big_delta:+.2f} 個百分點" if big_delta is not None else None,
                                delta_color="off",
                            )
                        comparison = (
                            f"比較日期：{tdcc_data['previous_date']} → {tdcc_data['date']}。"
                            if tdcc_data.get('previous_date') else f"資料日期：{tdcc_data['date']}；目前只有一個可比較週期。"
                        )
                        st.caption(comparison + "變化以集保庫存占比計算；百分點增加不等同該級距淨買超。")
                    else:
                        st.info("目前無法取得這檔股票的集保持股級距資料；不以零或推估值代替。")
                else:
                    st.info("集保持股級距資料目前僅適用台股。")

                st.caption(
                    f"個股分點參考目前對應 {stock_code}；要查看其他股票，請先在左側更換分析標的，"
                    "並匯入該股票的券商分點 CSV。"
                )
                with st.expander("🔎 選用：券商分點買賣明細", expanded=False):
                    st.caption("分點資料反映券商帳戶交易彙整，無法辨認最終投資人身分，也不等於主力或散戶流向。")
                    if not is_taiwan_stock:
                        st.info("券商分點資料目前僅適用台股。")
                    else:
                        def _render_branch_summary(branch_summary, source_label):
                            if not branch_summary:
                                return
                            if branch_summary.get('error'):
                                st.warning(branch_summary['error'])
                                return
                            branch_cols = st.columns(3)
                            count_suffix = "家" if branch_summary["count_unit"] == "分點" else "列"
                            with branch_cols[0]:
                                st.metric("分點淨額合計（核對）", f"{branch_summary['net_shares']:+,.0f} 股", delta_color="off")
                            with branch_cols[1]:
                                st.metric(f"淨買超{branch_summary['count_unit']}家數", f"{branch_summary['buyers']:,} {count_suffix}", delta_color="off")
                            with branch_cols[2]:
                                st.metric(f"淨賣超{branch_summary['count_unit']}家數", f"{branch_summary['sellers']:,} {count_suffix}", delta_color="off")
                            buy_share = branch_summary.get("top_net_buy_share_pct")
                            sell_share = branch_summary.get("top_net_sell_share_pct")
                            buy_share_text = f"，約占全部淨買超 {buy_share:.1f}%" if buy_share is not None else ""
                            sell_share_text = f"，約占全部淨賣超 {sell_share:.1f}%" if sell_share is not None else ""
                            st.caption(
                                f"{source_label}；淨買超前 15 個{branch_summary['count_unit']}合計 "
                                f"{branch_summary['top_net_buy']:+,.0f} 股{buy_share_text}；淨賣超前 15 個"
                                f"{branch_summary['count_unit']}合計 {branch_summary['top_net_sell']:+,.0f} 股{sell_share_text}。"
                            )
                            if buy_share is not None and sell_share is not None:
                                if buy_share > sell_share:
                                    concentration_label = "買方分點相對集中"
                                    concentration_delta = f"買方高 {buy_share - sell_share:.1f} 個百分點"
                                elif sell_share > buy_share:
                                    concentration_label = "賣方分點相對集中"
                                    concentration_delta = f"賣方高 {sell_share - buy_share:.1f} 個百分點"
                                else:
                                    concentration_label = "買賣兩側相同"
                                    concentration_delta = "前 15 個分點占比相同"
                                concentration_value = f"買 {buy_share:.1f}%｜賣 {sell_share:.1f}%"
                            elif buy_share is not None:
                                concentration_label = "賣方集中度資料不足"
                                concentration_delta = "僅有淨買超占比"
                                concentration_value = f"買 {buy_share:.1f}%｜賣—"
                            elif sell_share is not None:
                                concentration_label = "買方集中度資料不足"
                                concentration_delta = "僅有淨賣超占比"
                                concentration_value = f"買—｜賣 {sell_share:.1f}%"
                            else:
                                concentration_label = "資料不足"
                                concentration_delta = None
                                concentration_value = "—"
                            st.metric(
                                "分點淨流向集中度（前 15 個）",
                                concentration_value,
                                delta=concentration_label if concentration_delta is None else concentration_delta,
                                delta_color="off",
                                help="比較淨買超與淨賣超各自前 15 個分點，占該方向全部淨額的比例。這只表示券商交易渠道的相對集中程度，不能判定背後是主力或散戶。",
                            )
                            st.caption(
                                f"集中度比較：{concentration_label}。這表示淨買／賣超金額在少數分點的分布差異，"
                                "不是持股集中度，也無法由券商分點辨認主力或散戶身分。"
                            )
                            st.caption(
                                "讀法：買超／賣超家數只表示各分點淨額為正或負的數量，不代表散戶／大戶人數。"
                                "在資料涵蓋完整且口徑一致時，分點淨額合計接近 0 常是買賣相抵的核對結果，並非方向訊號；"
                                "前 15 比例反映分點淨額集中程度，也不能識別最終投資人。"
                            )
                            trend_item = flow_summary.get("trend")
                            trend_status = trend_item.get("status", "") if trend_item else ""
                            if "偏強" in trend_status:
                                trend_label = "偏強"
                            elif "偏弱" in trend_status:
                                trend_label = "偏弱"
                            elif "整理" in trend_status:
                                trend_label = "整理"
                            else:
                                trend_label = "資料不足"
                            net_shares = branch_summary["net_shares"]
                            branch_direction = "淨買超" if net_shares > 0 else "淨賣超" if net_shares < 0 else "買賣接近平衡"
                            st.info(
                                f"個股參考（{stock_code}）：近期價格走勢{trend_label}；"
                                f"該分點資料期間合計{branch_direction} {abs(net_shares):,.0f} 股。"
                                "價格趨勢與分點資料涵蓋期間可能不同，請分開解讀；這是市場觀察參考，不代表因果或買賣訊號。"
                            )
                            if trend_item:
                                price_change_pct = trend_item.get("period_change_pct")
                                period_bars = int(trend_item.get("period_bars", 20) or 20)
                                st.markdown(
                                    "**📈 股價方向 × 分點家數：** "
                                    + build_broker_branch_price_breadth_reading(branch_summary, price_change_pct)
                                )
                                st.caption(
                                    f"價格方向取最近 {period_bars} 筆「{selected_tf}」收盤資料；分點家數取{source_label}。"
                                    "若 CSV 沒有資料日期，兩者無法確認為同日或同期間；這是方向搭配觀察，不是統計相關係數。"
                                    "要判斷是否穩定同向，需累積多個日期且日期對齊的分點與股價資料。"
                                )
                            st.markdown("##### 🧠 市場心理觀察：怎麼理解這些分點數字？")
                            buyer_count = int(branch_summary["buyers"])
                            seller_count = int(branch_summary["sellers"])
                            if buyer_count > seller_count:
                                st.markdown(
                                    f"**參與廣度：** 淨買超分點有 {buyer_count} 個，淨賣超分點有 {seller_count} 個，"
                                    "表示這份資料中淨買方分點參與面較廣。這可以作為買方參與較廣的線索，"
                                    "但不能推論買方是散戶，或賣方是主力。"
                                )
                            elif seller_count > buyer_count:
                                st.markdown(
                                    f"**參與廣度：** 淨賣超分點有 {seller_count} 個，淨買超分點有 {buyer_count} 個，"
                                    "表示這份資料中淨賣方分點參與面較廣。這可以作為賣方參與較廣的線索，"
                                    "但不能推論賣方是散戶，或買方是主力。"
                                )
                            else:
                                st.markdown(
                                    f"**參與廣度：** 淨買超與淨賣超分點數相同（各 {buyer_count} 個）；"
                                    "家數本身沒有顯示哪一方參與面較廣。"
                                )

                            concentration_notes = []
                            if buy_share is not None:
                                concentration_notes.append(
                                    f"前 15 個淨買超分點占全部淨買超 {buy_share:.1f}%"
                                )
                            if sell_share is not None:
                                concentration_notes.append(
                                    f"前 15 個淨賣超分點占全部淨賣超 {sell_share:.1f}%"
                                )
                            if concentration_notes:
                                st.markdown(
                                    "**淨流向集中度：** " + "；".join(concentration_notes) +
                                    "。比例較高表示該方向的淨額較集中在排名靠前的分點；它不是持股集中度，也不能辨認分點背後的投資人。"
                                )
                            if trend_label != "資料不足":
                                st.markdown(
                                    f"**價格情緒線索：** 近 20 筆收盤價相對平均價呈{trend_label}；"
                                    "可與分點流向一起觀察市場是否同向，但兩者期間若不一致，就不能據此判定背離或趨勢延續。"
                                )
                            st.caption(
                                "市場心理不是這些欄位直接測得的結果。單日或單份 CSV 只能描述該資料範圍的成交分布；"
                                "請勿只憑分點家數判定吸籌、出貨或預測漲跌。"
                            )
                            breakdown = branch_summary.get("breakdown")
                            if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
                                top_rows = pd.concat([breakdown.head(10), breakdown.nsmallest(10, "買賣超")]).drop_duplicates()
                                st.dataframe(top_rows, hide_index=True, use_container_width=True)

                        st.markdown("##### 選擇分點資料來源")
                        st.caption("可輸入自己的 FinMind Token 自動帶入，或改用 CSV；FinMind 資料通常於交易日盤後更新。")
                        st.warning(
                            "授權提醒：FinMind 將商業 web／app 使用列為 Sponsor Pro 授權範圍，並禁止原始資料再散布或建立鏡像 API。"
                            "即使使用者自行輸入 Token，也不會自動取得本網站展示資料的授權；公開提供前請管理者向 FinMind 確認。"
                        )
                        st.markdown("[查看 FinMind 方案與資料授權](https://finmind.github.io/en/Pricing/)")

                        account_id = str(auth_user.get("id", "guest"))
                        user_token_key = f"finmind_user_token_{account_id}"
                        source_key = f"branch_data_source_{account_id}"
                        shared_token = get_finmind_api_token()
                        source_options = ["輸入自己的 FinMind Token（自動帶入）", "手動上傳 CSV"]
                        if shared_token:
                            source_options.insert(1, "使用網站管理者設定的共用 Token")
                        if st.session_state.get(source_key) not in source_options:
                            st.session_state[source_key] = source_options[0]
                        st.caption("自填 Token 需要每位使用者自己的 FinMind Sponsor 以上方案，且方案需包含券商分點資料。")
                        branch_source = st.radio(
                            "請選擇一種方式",
                            source_options,
                            key=source_key,
                            help="自填 Token 只保留在目前登入工作階段；CSV 不需要 API Token。",
                        )

                        if branch_source == "輸入自己的 FinMind Token（自動帶入）":
                            st.caption(
                                "請使用你自己的 FinMind Token。Token 會由本 App 伺服器代為呼叫 FinMind；"
                                "本版本不會寫入帳號資料庫，登出時會清除。請只在你信任的服務輸入 Token。"
                            )
                            finmind_token = st.text_input(
                                "你的 FinMind API Token",
                                type="password",
                                key=user_token_key,
                                help="券商分點 API 需要你的 FinMind 帳號方案包含此資料集。",
                            ).strip()
                            token_ack_key = f"finmind_token_ack_{account_id}"
                            st.button(
                                "清除目前輸入的 Token",
                                key=f"clear_finmind_token_{account_id}",
                                on_click=_clear_finmind_session_token,
                                args=(user_token_key, token_ack_key),
                            )
                            if finmind_token:
                                token_acknowledged = st.checkbox(
                                    "我確認此 Token 屬於我、我的方案包含券商分點資料，並同意本次由 App 伺服器代為查詢。",
                                    key=token_ack_key,
                                )
                            else:
                                token_acknowledged = False
                                st.info("請輸入自己的 FinMind Token；若不使用 API，請改選「手動上傳 CSV」。")
                            active_finmind_token = finmind_token
                            active_finmind_ack = token_acknowledged
                            source_label = "FinMind（個人 Token）"

                        elif branch_source == "使用網站管理者設定的共用 Token":
                            st.info("此 Token 由網站管理者設定，所有選用此方式的使用者會共用同一資料方案與查詢額度。")
                            token_ack_key = f"finmind_shared_ack_{account_id}"
                            active_finmind_ack = st.checkbox(
                                "我了解這是網站共用資料來源，並同意本次查詢。",
                                key=token_ack_key,
                            )
                            active_finmind_token = shared_token
                            source_label = "FinMind（網站共用 Token）"

                        else:
                            active_finmind_token = ""
                            active_finmind_ack = False
                            source_label = ""

                        if branch_source == "手動上傳 CSV":
                            st.info(
                                f"目前分析標的是 {stock_code}。分點 CSV 僅用來分析該檔股票；如要分析其他股票，"
                                "請先在左側更換股票代碼，再匯入該股票的 CSV。"
                            )
                            branch_file = st.file_uploader(
                                "選擇券商分點 CSV 檔",
                                type=["csv"],
                                key=f"chip_branch_upload_{account_id}_{selected_stock}",
                            )
                            if branch_file is not None:
                                branch_summary = summarize_broker_branch_csv(branch_file, stock_code)
                                saved_branch_context = save_broker_branch_ai_summary(
                                    st.session_state, account_id, stock_code, branch_summary, "使用者上傳 CSV"
                                )
                                if saved_branch_context:
                                    st.caption("已把這檔股票的分點摘要暫存於目前登入工作階段；啟動 AI 報告時會一併使用摘要，不傳送完整分點明細。")
                                _render_branch_summary(branch_summary, "依上傳檔案所含資料期間彙總")
                        elif active_finmind_token and active_finmind_ack:
                            from datetime import datetime as _datetime
                            from zoneinfo import ZoneInfo as _ZoneInfo
                            default_branch_date = pd.Timestamp(trade_date).date() if trade_date else _datetime.now(_ZoneInfo("Asia/Taipei")).date()
                            branch_trade_date = st.date_input(
                                "分點資料日期",
                                value=default_branch_date,
                                key=f"chip_branch_date_{account_id}_{stock_code}",
                            )
                            branch_date_text = branch_trade_date.strftime("%Y-%m-%d")
                            with st.spinner(f"自動讀取 {stock_code} 的券商分點資料…"):
                                branch_frame, branch_error = fetch_finmind_broker_branch(
                                    stock_code, branch_date_text, active_finmind_token, account_scope=account_id
                                )
                            if branch_error:
                                save_broker_branch_ai_summary(
                                    st.session_state, account_id, stock_code,
                                    {"error": branch_error}, f"{source_label}｜{branch_date_text}",
                                )
                                st.info(f"FinMind：{branch_error}")
                            else:
                                auto_summary = summarize_broker_branch_frame(branch_frame, stock_code)
                                saved_branch_context = save_broker_branch_ai_summary(
                                    st.session_state, account_id, stock_code, auto_summary,
                                    f"{source_label}｜{branch_date_text}",
                                )
                                if saved_branch_context:
                                    st.caption("已把此日期的分點摘要暫存於目前登入工作階段；啟動 AI 報告時會一併使用摘要，不傳送完整分點明細。")
                                _render_branch_summary(auto_summary, f"{source_label}｜{branch_date_text}")
                        elif branch_source != "手動上傳 CSV" and active_finmind_token and not active_finmind_ack:
                            st.info("請先完成上方確認，才會使用所選 Token 發出查詢。")
            
            # ==============================================================================
            # 【Tab 3: 3.12 智慧型動態網格策略與資金自動規劃器 (含三大機制解析與複利警示)】
            # ==============================================================================
            if selected_module == "🤖 網格自動生成器":
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

                grid_p = round(float(price_val), 2)
                is_tw = ".TW" in selected_stock or ".TWO" in selected_stock
                unit_label = "股" if not is_tw else "股 (台股預設)"

                grid_history = fetch_grid_daily_history(selected_stock)
                recommended_lower, recommended_upper, recommended_count, recommendation_reason = recommend_dynamic_grid(grid_p, grid_history)
                if st.session_state.get("_grid_default_stock") != selected_stock:
                    st.session_state["_grid_default_stock"] = selected_stock
                    st.session_state["grid_lower"] = recommended_lower
                    st.session_state["grid_upper"] = recommended_upper
                    st.session_state["grid_num"] = recommended_count

                st.info(f"**本檔建議預設理由：**{recommendation_reason}")
                if st.button("恢復本檔建議預設值", key="grid_restore_recommended"):
                    st.session_state["grid_lower"] = recommended_lower
                    st.session_state["grid_upper"] = recommended_upper
                    st.session_state["grid_num"] = recommended_count

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
                    input_lower = st.number_input("網格下限價格", min_value=0.01, step=0.01, format="%.2f", key="grid_lower")
                with g_col2: 
                    input_upper = st.number_input("網格上限價格", min_value=0.01, step=0.01, format="%.2f", key="grid_upper")
                with g_col3: 
                    input_num = st.number_input("規劃總格數", min_value=3, max_value=50, step=1, key="grid_num")

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

                buy_level_count = int(np.sum(levels < grid_p))
                sell_level_count = int(np.sum(levels > grid_p))
                reference_level_count = int(input_num - buy_level_count - sell_level_count)
                if input_upper <= grid_p:
                    st.warning(f"現價 {grid_p:.2f} 高於或等於網格上限 {input_upper:.2f}，目前沒有賣出價位；這通常是上下限沿用舊股票設定所致，請提高上限或恢復本檔建議預設值。")
                elif input_lower >= grid_p:
                    st.warning(f"現價 {grid_p:.2f} 低於或等於網格下限 {input_lower:.2f}，目前沒有買進價位；請降低下限或恢復本檔建議預設值。")
                else:
                    st.caption(f"依目前設定，現價下方有 {buy_level_count} 個買進價位、上方有 {sell_level_count} 個賣出價位；規劃格數包含 {reference_level_count} 個現價參考價位（若有）。")

                for idx, level in enumerate(levels):
                    action = "🟢 買進掛單" if level < grid_p else ("🔴 賣出掛單" if level > grid_p else "⚪ 現價參考價")
                    
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
                        "網格層級": f"第 {idx+1:02d} 格",
                        "目標觸發價": round(level, 2),
                        "偏離現價 (%)": f"{((level - grid_p) / grid_p * 100):+.2f}%",
                        "預計動作": action,
                        "資金/部位管理機制": trade_desc
                    })

                st.dataframe(pd.DataFrame(grid_details), use_container_width=True, height=280)
                st.caption("「網格層級」是價格分段；「目標觸發價」是策略參考價；「偏離現價」顯示距離；「預計動作」與資金管理方式是依上方假設試算，並非實際掛單或保證成交。")
                st.caption("🧠 心理角度：網格把低買高賣變成固定規則，有助減少臨場猶豫；但若價格持續單向移動，使用者可能因錨定原本區間而不願調整，或因害怕錯過而追著改參數。先訂好資金上限和檢視頻率。")

                st.warning(
                    "⚠️ **操盤手複利與風險特別提示：**\n"
                    "1. **網格獲利 ≠ 長期投資正期望值**：網格在「箱型震盪行情」中最能發揮低買高賣優勢；但在「強勢單邊大漲」時，會因持續賣出股票而落後於買進持有 (Buy & Hold)；在「持續單邊下跌」時，則會逐步買進並累積帳面虧損。\n"
                    "2. **交易成本與稅費影響**：頻繁再平衡會產生手續費與證券交易稅，長期運作下會侵蝕部分複利效益。\n"
                    "3. **驗證建言**：若目標是建立長期複利系統，請切換至 **「Tab 4 策略自訂回測與網格複利對比」**，檢視歷史 **年化報酬率 (CAGR)**、**最大回撤 (MDD)** 與 **扣除稅費後的淨報酬率**。"
                )

            # ==============================================================================
            # 【Tab 4: 3.12 「網格再平衡 vs 買進持有」歷史線路模擬與全指標對比】
            # ==============================================================================
            if selected_module == "⚖️ 網格／持有／定期定額比較":
                st.markdown(f"### ⚖️ 【{c_name}】網格、買進持有與定期定額歷史比較")
                st.caption("三種方式使用同一標的、價格期間及交易費率。定期定額預設把相同總預算平均分到每個資料月份；年化報酬按每次實際投入日期計算。")
                
                st.markdown("#### 網格策略、一次買進持有、每月定期定額")
                
                gc_col1, gc_col2, gc_col3 = st.columns(3)
                with gc_col1:
                    sim_capital = st.number_input("💵 模擬初始投入本金 (元/\\$)", min_value=1.0, value=100000.0, step=10000.0, key="sim_cap_input")
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

                comparison_prices = pd.to_numeric(df_chart["Close"], errors="coerce").dropna()
                comparison_index = pd.to_datetime(comparison_prices.index, errors="coerce")
                comparison_months = max(1, int(comparison_index[~comparison_index.isna()].to_period("M").nunique()))
                default_dca_monthly = float(sim_capital) / comparison_months
                if "sim_dca_monthly_input" not in st.session_state:
                    st.session_state["sim_dca_monthly_input"] = default_dca_monthly
                dca_cols = st.columns([2, 1])
                with dca_cols[1]:
                    if st.button("恢復相同總預算", key="sim_dca_reset"):
                        st.session_state["sim_dca_monthly_input"] = default_dca_monthly
                with dca_cols[0]:
                    dca_monthly = st.number_input(
                        "💵 定期定額每月投入金額",
                        min_value=0.0, value=float(default_dca_monthly), step=1_000.0,
                        key="sim_dca_monthly_input",
                        help="預設把上方模擬本金平均分配至回測期間的每個月份；可自行更改。"
                    )
                dca_budget_preview = float(dca_monthly) * comparison_months
                st.caption(f"期間包含 {comparison_months} 個資料月份；依目前月投入估算，定期定額累計投入約 {dca_budget_preview:,.0f} 元。預設與上方模擬本金相同。")
                if selected_tf in ("1分鐘", "5分鐘", "30分鐘", "60分鐘"):
                    st.info("目前是短線圖表，資料期間不到完整長期月扣款週期；定期定額只會在資料涵蓋月份的第一筆價格投入。若要比較長期月扣款，請切換日線、週線或月線。")
                elif selected_tf in ("1季", "半年", "1年"):
                    st.info("目前圖表是季線或更長週期，沒有每個月的價格；定期投入只能使用每個有資料月份的第一筆價格，並非逐月扣款。若要比較月扣款，請切換日線、週線或月線。")

                closes_arr = df_chart['Close'].astype(float).to_numpy()
                
                if len(closes_arr) >= 5:
                    initial_p = closes_arr[0]
                    bh_shares = (sim_capital * (1.0 - fee_rate)) / initial_p
                    bh_asset_curve = bh_shares * closes_arr
                    bh_final_val = float(bh_asset_curve[-1])
                    bh_total_ret = ((bh_final_val - sim_capital) / sim_capital) * 100.0 if sim_capital > 0 else None

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
                    grid_total_ret = ((grid_final_val - sim_capital) / sim_capital) * 100.0 if sim_capital > 0 else None
                    grid_span_years = max(
                        (pd.Timestamp(df_chart.index[-1]) - pd.Timestamp(df_chart.index[0])).total_seconds()
                        / (365.25 * 24 * 60 * 60),
                        0.0,
                    )
                    # Use actual timestamp span for every chart interval; fixed bars/year assumptions
                    # badly distorted intraday, weekly, monthly and long-interval CAGR values.
                    bh_cagr = (
                        ((bh_final_val / float(sim_capital)) ** (1.0 / grid_span_years) - 1.0) * 100.0
                        if grid_span_years >= 0.5 and bh_final_val > 0 and float(sim_capital) > 0 else None
                    )
                    grid_cagr = (
                        ((grid_final_val / float(sim_capital)) ** (1.0 / grid_span_years) - 1.0) * 100.0
                        if grid_span_years >= 0.5 and grid_final_val > 0 and float(sim_capital) > 0 else None
                    )

                    dca_result = None
                    try:
                        dca_series = pd.Series(closes_arr, index=df_chart.index, name="Close")
                        dca_result = simulate_monthly_dca(dca_series, float(dca_monthly), fee_rate)
                    except Exception as dca_error:
                        st.warning(f"定期定額暫時無法計算：{dca_error}")

                    def calc_mdd(curve):
                        arr = np.array(curve)
                        peak = np.maximum.accumulate(arr)
                        drawdown = (arr - peak) / peak
                        return float(np.min(drawdown)) * 100.0

                    bh_mdd = calc_mdd(bh_asset_curve)
                    grid_mdd = calc_mdd(grid_asset_curve)
                    grid_advice_cagr = (
                        (grid_final_val / float(sim_capital)) ** (1.0 / grid_span_years) - 1.0
                        if grid_span_years >= 0.5 and grid_final_val > 0 and float(sim_capital) > 0 else None
                    )
                    grid_advice_metrics = {
                        "cumulative_return": grid_total_ret / 100.0 if grid_total_ret is not None else None,
                        "cagr": grid_advice_cagr,
                        "mdd": grid_mdd / 100.0,
                        "strategy": grid_sim_strategy,
                        "capital": float(sim_capital),
                        "fee_rate": float(fee_rate),
                        "tax_rate": float(tax_rate),
                    }

                    st.markdown("#### 三種方式結果摘要")
                    summary_cols = st.columns(4)
                    summary_cols[0].metric("買進持有期末資產", f"{bh_final_val:,.0f} 元", delta=f"損益 {bh_final_val - sim_capital:+,.0f} 元",
                                           help="以期初本金扣除買進手續費後持有至最後一筆價格的市值；損益為期末資產減期初本金。")
                    summary_cols[1].metric("網格策略期末資產", f"{grid_final_val:,.0f} 元", delta=f"損益 {grid_final_val - sim_capital:+,.0f} 元",
                                           help="依網格觸發規則及設定交易成本模擬的期末股票與現金合計；不代表實際成交。")
                    if dca_result is not None:
                        dca_metrics = dca_result["metrics"]
                        summary_cols[2].metric("定期定額期末資產", f"{dca_metrics['ending_value']:,.0f} 元", delta=f"損益 {dca_metrics['profit_loss']:+,.0f} 元",
                                               help="每次投入買進後，所有持倉在期末的估算市值；損益已相對累計投入計算。")
                        summary_cols[3].metric("定期定額累計投入", f"{dca_metrics['total_contributed']:,.0f} 元", delta=f"{dca_metrics['contribution_count']} 次投入",
                                               help="回測期間所有月投入金額的加總；比較期末資產時，請同時考慮累計投入及投入時點。")

                    comparison_rows = [
                        {"投入方式": "一開始買進並持有", "累計投入(元)": float(sim_capital), "期末資產(元)": bh_final_val,
                         "總損益(元)": bh_final_val - float(sim_capital), "年化報酬(%)": bh_cagr,
                         "最大跌幅(%)": bh_mdd},
                        {"投入方式": f"網格｜{grid_sim_strategy.split(' ')[1]}", "累計投入(元)": float(sim_capital), "期末資產(元)": grid_final_val,
                         "總損益(元)": grid_final_val - float(sim_capital), "年化報酬(%)": grid_cagr,
                         "最大跌幅(%)": grid_mdd},
                    ]
                    if dca_result is not None:
                        dca_metrics = dca_result["metrics"]
                        comparison_rows.append({
                            "投入方式": "每月定期定額",
                            "累計投入(元)": dca_metrics["total_contributed"],
                            "期末資產(元)": dca_metrics["ending_value"],
                            "總損益(元)": dca_metrics["profit_loss"],
                            "年化報酬(%)": dca_metrics["money_weighted_return"] * 100 if dca_metrics["money_weighted_return"] is not None else None,
                            "最大跌幅(%)": dca_metrics["mdd"] * 100,
                        })
                    st.dataframe(pd.DataFrame(comparison_rows).style.format({
                        "累計投入(元)": "{:,.0f}", "期末資產(元)": "{:,.0f}", "總損益(元)": "{:+,.0f}",
                        "年化報酬(%)": "{:+.2f}%", "最大跌幅(%)": "{:.2f}%",
                    }, na_rep="—"), use_container_width=True, hide_index=True)
                    st.caption("累計投入是投入金額加總；期末資產是回測末端估值；總損益為兩者差額。CAGR 以實際資料起訖日期計算，期間不足半年時不顯示年化值。定期定額年化報酬依每次投入日期計算；最大跌幅先扣除新增投入的影響。其他兩種方式從期初投入全部預算。三者均未另計股息；買進持有與定期定額以持倉市值估值，未假設期末賣出。")
                    st.caption("🧠 心理角度：買進持有考驗承受回撤的耐心；定期定額可減少一次押時點的壓力，但不會消除下跌；網格提供規則化操作，遇到長期單邊行情卻可能太早賣出或持續加碼。結果要配合自己能承受的過程，而不只挑期末數字最高的做法。")

                    strat_display_name = grid_sim_strategy.split(" ")[1]

                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=date_strings, y=bh_asset_curve, mode='lines', name='單純買進持有 (Buy & Hold)', line=dict(color='#00CC66', width=2)))
                    fig_compare.add_trace(go.Scatter(x=date_strings, y=grid_asset_curve, mode='lines', name=strat_display_name, line=dict(color='#FF9900', width=2, dash='dash')))
                    if dca_result is not None:
                        if "分鐘" in selected_tf:
                            dca_axis_dates = dca_result["history"].index.strftime("%m-%d %H:%M").tolist()
                        else:
                            dca_axis_dates = dca_result["history"].index.strftime("%Y-%m-%d").tolist()
                        fig_compare.add_trace(go.Scatter(
                            x=dca_axis_dates,
                            y=dca_result["history"]["portfolio_value"],
                            mode="lines", name="每月定期定額",
                            line=dict(color="#4C78A8", width=2, dash="dot"),
                        ))
                    fig_compare.update_layout(
                        title=f"📈 【{c_name}】網格、買進持有與定期定額資產變化（計入買進費用）",
                        height=380,
                        xaxis=dict(type='category', showgrid=True, gridcolor="rgba(128,128,128,0.2)", nticks=12),
                        yaxis=dict(title="資產總價值 (元)", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                        margin=dict(l=10, r=20, t=40, b=10)
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

            # ==============================================================================
            # 【Tab 5: 3.12 技術指標多空策略與動態停損回測器】
            # ============================================================================== 
            if selected_module == "🧪 技術指標自訂策略回測":
                st.markdown(f"### 🧪 【{c_name}】技術指標多空策略與動態停損回測器 (3.12 版)")
                default_bt_start = pd.Timestamp(df_chart.index[0]).date()
                default_bt_end = pd.Timestamp(df_chart.index[-1]).date()
                bt_date_selection = st.date_input(
                    "本次回測資料起訖日",
                    value=(default_bt_start, default_bt_end),
                    key=f"bt_date_range_{selected_stock}_{selected_tf}",
                    help="預設使用目前圖表已有資料的完整期間；若輸入範圍超出圖表資料，系統會按目前 K 線週期重新下載。",
                )
                if isinstance(bt_date_selection, (tuple, list)):
                    if len(bt_date_selection) >= 2:
                        bt_start_date, bt_end_date = bt_date_selection[0], bt_date_selection[1]
                    elif len(bt_date_selection) == 1:
                        bt_start_date, bt_end_date = bt_date_selection[0], default_bt_end
                    else:
                        bt_start_date, bt_end_date = default_bt_start, default_bt_end
                else:
                    bt_start_date, bt_end_date = bt_date_selection, bt_date_selection
                if bt_start_date > bt_end_date:
                    st.warning("起始日期晚於結束日期，已自動對調日期。")
                    bt_start_date, bt_end_date = bt_end_date, bt_start_date
                st.caption(
                    f"目前 K 線週期：{selected_tf}；均線數字代表 K 棒資料根數，例如週線的 20 根是 20 週線，不是 20 個日曆天。"
                )
                bt_col1, bt_col2 = st.columns(2)
                
                with bt_col1:
                    st.markdown("##### 🟢 進場 (買進) 策略下拉式選單")
                    entry_ma_mode = st.selectbox("1. 均線策略 (MA)", ["停用", "突破均線", "站穩均線上"], index=1, key="bt_entry_ma_sel")
                    entry_ma_p = st.number_input("買進均線資料根數 (MA)", min_value=1, max_value=240, value=20, key="bt_entry_ma_p")

                    entry_ma_cross_pairs = []
                    with st.expander("新增多組均線黃金交叉買進條件", expanded=False):
                        st.caption("快線由下往上穿越慢線時成立。每組會依下方 ALL／ANY 規則，與其他已啟用買進條件一起判斷。最多可設定 10 組。")
                        entry_ma_cross_count = st.number_input(
                            "買進交叉組數", min_value=0, max_value=10, value=0, step=1,
                            key="bt_entry_ma_cross_count",
                        )
                        default_cross_pairs = [
                            (5, 20), (10, 20), (10, 60), (20, 60), (20, 120),
                            (60, 120), (60, 240), (5, 60), (20, 240), (120, 240),
                        ]
                        invalid_entry_cross = False
                        for pair_index in range(int(entry_ma_cross_count)):
                            fast_default, slow_default = default_cross_pairs[pair_index]
                            pair_cols = st.columns(2)
                            with pair_cols[0]:
                                fast_period = st.number_input(
                                    f"第 {pair_index + 1} 組快線資料根數",
                                    min_value=1, max_value=240, value=fast_default, step=1,
                                    key=f"bt_entry_ma_cross_fast_{pair_index + 1}",
                                )
                            with pair_cols[1]:
                                slow_period = st.number_input(
                                    f"第 {pair_index + 1} 組慢線資料根數",
                                    min_value=1, max_value=240, value=slow_default, step=1,
                                    key=f"bt_entry_ma_cross_slow_{pair_index + 1}",
                                )
                            if fast_period >= slow_period:
                                invalid_entry_cross = True
                            else:
                                entry_ma_cross_pairs.append((int(fast_period), int(slow_period)))
                        entry_ma_cross_pairs = _normalize_ma_cross_pairs(entry_ma_cross_pairs)
                        if invalid_entry_cross:
                            st.warning("快線天數必須小於慢線天數；不符合的組別會略過。")
                    
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
                    bt_rr_enabled = st.checkbox(
                        "買點成立後啟用風險報酬停利目標（預設 1:3）",
                        value=True, key="bt_rr_enabled",
                        help="停利價會依進場價與價格型停損距離計算；停損條件不會被移除。"
                    )
                    bt_rr_multiple = st.number_input(
                        "目標報酬倍數（風險 1，報酬 X）",
                        min_value=0.5, max_value=10.0, value=3.0, step=0.5,
                        key="bt_rr_multiple", disabled=not bt_rr_enabled,
                        help="預設 3 代表風險報酬比 1:3；可改成 1.5、2、4 等。"
                    )
                    if bt_rr_enabled and stop_loss_pct <= 0 and trailing_stop_pct <= 0:
                        st.warning("目前固定停損與移動停損都為 0，沒有價格型停損距離可供計算；風險報酬停利目標不會生效。")
                    else:
                        st.caption("初始風險取固定停損與移動停損中較靠近進場價的價格型停損；目標價在買點成立時鎖定，移動停損後續上移不會改寫目標價。")
                    
                    exit_ma_mode = st.selectbox("1. 均線平倉策略 (MA)", ["停用", "跌破均線"], index=1, key="bt_exit_ma_sel")
                    exit_ma_p = st.number_input("賣出均線資料根數 (MA)", min_value=1, max_value=240, value=20, key="bt_exit_ma_p")

                    exit_ma_cross_pairs = []
                    with st.expander("新增多組均線死亡交叉平倉條件", expanded=False):
                        st.caption("快線由上往下跌破慢線時成立；任一組死亡交叉成立，就會作為一項平倉訊號。最多可設定 10 組。")
                        exit_ma_cross_count = st.number_input(
                            "平倉交叉組數", min_value=0, max_value=10, value=0, step=1,
                            key="bt_exit_ma_cross_count",
                        )
                        invalid_exit_cross = False
                        for pair_index in range(int(exit_ma_cross_count)):
                            fast_default, slow_default = default_cross_pairs[pair_index]
                            pair_cols = st.columns(2)
                            with pair_cols[0]:
                                fast_period = st.number_input(
                                    f"第 {pair_index + 1} 組快線資料根數",
                                    min_value=1, max_value=240, value=fast_default, step=1,
                                    key=f"bt_exit_ma_cross_fast_{pair_index + 1}",
                                )
                            with pair_cols[1]:
                                slow_period = st.number_input(
                                    f"第 {pair_index + 1} 組慢線資料根數",
                                    min_value=1, max_value=240, value=slow_default, step=1,
                                    key=f"bt_exit_ma_cross_slow_{pair_index + 1}",
                                )
                            if fast_period >= slow_period:
                                invalid_exit_cross = True
                            else:
                                exit_ma_cross_pairs.append((int(fast_period), int(slow_period)))
                        exit_ma_cross_pairs = _normalize_ma_cross_pairs(exit_ma_cross_pairs)
                        if invalid_exit_cross:
                            st.warning("快線天數必須小於慢線天數；不符合的組別會略過。")
                    
                    exit_kd_mode = st.selectbox("2. KD 離場策略", ["停用", "死亡交叉"], index=0, key="bt_exit_kd_sel")
                    exit_macd_mode = st.selectbox("3. MACD 離場策略", ["停用", "死亡交叉"], index=0, key="bt_exit_macd_sel")
                    exit_vol_mode = st.selectbox("4. 成交量離場策略", ["停用", "極端爆量倒貨 (大於20日均量2.5倍)"], index=0, key="bt_exit_vol_sel")
                    exit_rsi_mode = st.selectbox("5. RSI 離場策略", ["停用", "超買警戒 (RSI跌破70)"], index=0, key="bt_exit_rsi_sel")
                    exit_bb_mode = st.selectbox("6. 布林通道離場策略", ["停用", "跌破布林下軌", "觸及上軌拉回"], index=0, key="bt_exit_bb_sel")

                with st.expander("💰 績效分析設定（本金與交易成本）", expanded=False):
                    cost_cols = st.columns(4)
                    with cost_cols[0]:
                        bt_initial_capital = st.number_input("回測初始資金", min_value=1000, max_value=1000000000,
                                                              value=100000, step=10000, key="bt_initial_capital")
                    with cost_cols[1]:
                        bt_commission_pct = st.number_input("手續費（每邊，%）", min_value=0.0, max_value=5.0,
                                                            value=0.1425 if is_tw_stock else 0.0, step=0.01,
                                                            format="%.4f", key="bt_commission_pct")
                    with cost_cols[2]:
                        bt_sell_tax_pct = st.number_input("賣出交易稅（%）", min_value=0.0, max_value=5.0,
                                                          value=0.3 if is_tw_stock else 0.0, step=0.01,
                                                          format="%.3f", key="bt_sell_tax_pct")
                    with cost_cols[3]:
                        bt_slippage_pct = st.number_input("滑價（每邊，%）", min_value=0.0, max_value=5.0,
                                                          value=0.05, step=0.01, format="%.2f",
                                                          key="bt_slippage_pct")
                    st.caption("成本按每次買進／賣出分別估算；台股預設值可依券商折扣、標的與交易方式調整。")

                if st.button(f"🚀 開始執行【{c_name}】全指標量化回測", type="primary", use_container_width=True, key="run_bt_btn"):
                    chart_start = pd.Timestamp(df_chart.index[0]).date()
                    chart_end = pd.Timestamp(df_chart.index[-1]).date()
                    requested_start = pd.Timestamp(bt_start_date)
                    requested_end = pd.Timestamp(bt_end_date)
                    if bt_start_date < chart_start or bt_end_date > chart_end:
                        try:
                            with st.spinner("正在下載您指定日期範圍的歷史 K 線資料…"):
                                raw_backtest = yf.download(
                                    selected_stock,
                                    start=requested_start.strftime("%Y-%m-%d"),
                                    end=(requested_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                    interval=cfg["i"], progress=False,
                                )
                            if isinstance(raw_backtest.columns, pd.MultiIndex):
                                raw_backtest.columns = raw_backtest.columns.get_level_values(0)
                            if not raw_backtest.empty and raw_backtest.index.tz is not None:
                                raw_backtest.index = raw_backtest.index.tz_localize(None)
                            if raw_backtest.empty:
                                df_backtest = df_chart.iloc[0:0].copy()
                                st.warning("資料來源沒有回傳這段期間的 K 線；請確認日期範圍或目前週期的歷史資料限制。")
                            else:
                                df_backtest, _, _ = calculate_chip_and_backtest(
                                    selected_stock, raw_backtest, selected_tf
                                )
                                df_backtest = _slice_history_for_date_range(
                                    df_backtest, bt_start_date, bt_end_date
                                )
                        except Exception as period_download_error:
                            df_backtest = df_chart.iloc[0:0].copy()
                            st.warning(f"指定期間資料下載失敗：{period_download_error}")
                    else:
                        df_backtest = _slice_history_for_date_range(
                            df_chart, bt_start_date, bt_end_date
                        )

                    if "分鐘" in selected_tf:
                        backtest_date_strings = df_backtest.index.strftime("%m-%d %H:%M").tolist()
                    else:
                        backtest_date_strings = df_backtest.index.strftime("%Y-%m-%d").tolist()

                    entry_conds = {
                        'ma_mode': entry_ma_mode,
                        'ma_p': entry_ma_p,
                        'ma_cross_pairs': entry_ma_cross_pairs,
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
                        'ma_cross_pairs': exit_ma_cross_pairs,
                        'kd_mode': exit_kd_mode,
                        'macd_mode': exit_macd_mode,
                        'vol_mode': exit_vol_mode,
                        'rsi_mode': exit_rsi_mode,
                        'bb_mode': exit_bb_mode
                    }
                    
                    df_calc = compute_backtest_indicators(
                        df_backtest, ma_entry_p=entry_ma_p, ma_exit_p=exit_ma_p,
                        entry_ma_cross_pairs=entry_ma_cross_pairs,
                        exit_ma_cross_pairs=exit_ma_cross_pairs,
                    )
                    entry_cross_counts = _count_ma_cross_events(
                        df_calc, entry_ma_cross_pairs, "golden"
                    )
                    exit_cross_counts = _count_ma_cross_events(
                        df_calc, exit_ma_cross_pairs, "death"
                    )
                    df_trades = run_strategy_backtest(
                        df_calc, entry_conds, exit_conds, backtest_date_strings,
                        risk_reward_ratio=float(bt_rr_multiple) if bt_rr_enabled else None,
                    )
                    analysis = analyze_indicator_backtest(
                        df_calc, df_trades, float(bt_initial_capital), float(bt_commission_pct),
                        float(bt_sell_tax_pct), float(bt_slippage_pct), selected_tf, is_tw_stock)

                    if df_trades.empty:
                        st.info("這段期間沒有買賣訊號；下方仍會顯示策略資金變化，以及直接買進後持有的參考結果。")
                        if entry_ma_cross_pairs:
                            cross_count_summary = "、".join(
                                f"MA{fast}/MA{slow}：{count} 次"
                                for fast, slow, count in entry_cross_counts
                            )
                            st.caption(f"所選日期內黃金交叉次數：{cross_count_summary}。交叉只在快線穿越慢線的當根 K 棒成立。")
                            if not any(count for _fast, _slow, count in entry_cross_counts):
                                st.warning("所選日期範圍內沒有出現設定的黃金交叉；可延長回測期間、縮短均線根數，或確認目前 K 線週期。")
                            else:
                                other_entry_enabled = any((
                                    entry_ma_mode != "停用", entry_kd_mode != "停用",
                                    entry_macd_mode != "停用", entry_vol_mode != "停用",
                                    entry_rsi_mode != "停用", entry_bb_mode != "停用",
                                ))
                                if entry_conds["match_mode"] == "ALL" and other_entry_enabled:
                                    st.warning("期間內有黃金交叉，但買進判定為 ALL，還必須同一根 K 棒符合其他啟用的買進條件。若只想測交叉，請把原均線策略及其他買進條件設為「停用」；若希望任一條件成立就買進，請改成 ANY。")
                        if exit_ma_cross_pairs:
                            cross_count_summary = "、".join(
                                f"MA{fast}/MA{slow}：{count} 次"
                                for fast, slow, count in exit_cross_counts
                            )
                            st.caption(f"所選日期內死亡交叉次數：{cross_count_summary}。若沒有任何買進訊號，平倉交叉不會單獨產生交易。")
                    else:
                        trade_display = df_trades.drop(columns=[
                            "_entry_index", "_exit_index", "_entry_fill_price", "_exit_fill_price"
                        ], errors="ignore").rename(columns={
                            "平倉報酬率 (%)": "未扣費用損益率 (%)",
                            "持有K棒數": "持有資料筆數",
                        })
                        if analysis is not None and not analysis["trade_analysis"].empty:
                            trade_display = trade_display.copy()
                            trade_display["扣除費用後損益 (%)"] = analysis["trade_analysis"]["成本後報酬率 (%)"].round(2).to_numpy()
                        st.markdown("#### 逐筆交易結果")
                        st.dataframe(trade_display, use_container_width=True, hide_index=True)
                        st.caption("每列是一筆買進到賣出的模擬紀錄；損益率已扣除上方設定的交易費用。期末仍持有的股票，以最後收盤價估算。")

                    if analysis is None:
                        st.warning("價格資料不足或有缺漏，暫時無法計算這段回測結果。")
                    else:
                        strategy_metrics = analysis["strategy_metrics"]
                        benchmark_metrics = analysis["benchmark_metrics"]
                        st.markdown("#### 回測結果重點")
                        st.caption("以下是依目前設定回看歷史資料的模擬結果，不代表未來表現；「—」表示資料不足或該項目不適用。")

                        def show_pct(value, signed=False):
                            if value is None or not np.isfinite(value):
                                return "—"
                            return f"{value:+.2%}" if signed else f"{value:.2%}"

                        def show_number(value, digits=2):
                            return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

                        def show_percent_points(value):
                            return "—" if value is None or not np.isfinite(value) else f"{value:+.2f}%"

                        backtest_dca_result = None
                        try:
                            dca_prices = pd.to_numeric(df_calc["Close"], errors="coerce").dropna()
                            dca_dates = pd.to_datetime(dca_prices.index, errors="coerce")
                            dca_month_count = max(1, int(dca_dates[~dca_dates.isna()].to_period("M").nunique()))
                            dca_monthly_amount = float(bt_initial_capital) / dca_month_count
                            dca_buy_cost = (float(bt_commission_pct) + float(bt_slippage_pct)) / 100.0
                            dca_exit_cost = (float(bt_commission_pct) + float(bt_sell_tax_pct) + float(bt_slippage_pct)) / 100.0
                            backtest_dca_result = simulate_monthly_dca(
                                dca_prices, dca_monthly_amount, commission_rate=dca_buy_cost,
                                exit_cost_rate=dca_exit_cost,
                            )
                        except Exception as dca_compare_error:
                            st.caption(f"定期定額參考結果暫時無法計算：{dca_compare_error}")

                        metric_row1 = st.columns(4)
                        excess_return = strategy_metrics["cumulative_return"] - benchmark_metrics["cumulative_return"]
                        metric_row1[0].metric("整段期間損益（已扣費用）", show_pct(strategy_metrics["cumulative_return"], True),
                                              delta=f"比直接持有多 {excess_return:+.2%}" if excess_return >= 0 else f"比直接持有少 {abs(excess_return):.2%}",
                                              help="從回測開始到結束，資金總共增加或減少的比例；已扣除設定的交易費用。")
                        metric_row1[1].metric("平均每年報酬", show_pct(strategy_metrics["cagr"], True),
                                              help="把整段期間的資金變化換算成平均每年增減幅度，方便比較不同長度的回測；不是每一年都會有這個報酬。")
                        metric_row1[2].metric("從高點算起最大跌幅", show_pct(strategy_metrics["mdd"]),
                                              help="回測期間曾經從資產高點跌到多低。數字越接近零，代表歷史上最深的跌幅越小。")
                        metric_row1[3].metric("夏普比率 Sharpe", show_number(strategy_metrics["sharpe"]),
                                              help="把歷史報酬和整體起伏一起衡量，再換算成年化；本頁以 0% 無風險利率估算。數值較高代表過去每承受相同程度起伏時，換到的報酬較多，不保證未來。")
                        metric_row2 = st.columns(4)
                        metric_row2[0].metric("索提諾比率 Sortino", show_number(strategy_metrics["sortino"]),
                                              help="平均報酬相對於下跌波動的比值並年化；只把低於 0 的報酬視為下行風險，並非虧損機率。")
                        metric_row2[1].metric("賺錢交易比例", show_pct(strategy_metrics["win_rate"]),
                                              help="所有交易中，扣除費用後仍賺錢的比例。比例高不一定代表總收益高，還要一起看每次賺多少、賠多少。")
                        metric_row2[2].metric("賺賠金額比", show_number(strategy_metrics["profit_factor"]),
                                              help="比較賺錢交易和賠錢交易的金額總和。若數值是 1.5，代表歷史模擬中每賠 1 元，約賺 1.5 元；已扣設定費用。沒有賠錢交易時不顯示。")
                        metric_row2[3].metric("持有股票的時間比例", show_pct(strategy_metrics["exposure"]),
                                              help="回測期間有持有股票的時間占全部時間多少；其餘時間策略資金留在現金。")

                        st.markdown("#### 回測策略與長期持有、定期定額及網格比較")
                        comparison_rows = [
                            {"做法": "依訊號買進或賣出", "整段期間損益(%)": strategy_metrics["cumulative_return"] * 100,
                             "平均每年報酬(%)": strategy_metrics["cagr"] * 100 if strategy_metrics["cagr"] is not None else None,
                             "從高點算起最大跌幅(%)": strategy_metrics["mdd"] * 100,
                             "資金起伏幅度(年化 %)": strategy_metrics["annualized_volatility"] * 100 if strategy_metrics["annualized_volatility"] is not None else None,
                             "夏普比率（0%利率）": strategy_metrics["sharpe"], "結束時資產(元)": strategy_metrics["ending_value"]},
                            {"做法": "一開始買進並持有", "整段期間損益(%)": benchmark_metrics["cumulative_return"] * 100,
                             "平均每年報酬(%)": benchmark_metrics["cagr"] * 100 if benchmark_metrics["cagr"] is not None else None,
                             "從高點算起最大跌幅(%)": benchmark_metrics["mdd"] * 100,
                             "資金起伏幅度(年化 %)": benchmark_metrics["annualized_volatility"] * 100 if benchmark_metrics["annualized_volatility"] is not None else None,
                             "夏普比率（0%利率）": benchmark_metrics["sharpe"], "結束時資產(元)": benchmark_metrics["ending_value"]},
                        ]
                        if backtest_dca_result is not None:
                            dca_metrics = backtest_dca_result["metrics"]
                            dca_return = dca_metrics["ending_value"] / dca_metrics["total_contributed"] - 1.0 if dca_metrics["total_contributed"] > 0 else None
                            comparison_rows.append({
                                "做法": "每月定期定額（同總投入）",
                                "整段期間損益(%)": dca_return * 100 if dca_return is not None else None,
                                "平均每年報酬(%)": dca_metrics["money_weighted_return"] * 100 if dca_metrics["money_weighted_return"] is not None else None,
                                "從高點算起最大跌幅(%)": dca_metrics["mdd"] * 100,
                                "資金起伏幅度(年化 %)": None,
                                 "夏普比率（0%利率）": None,
                                "結束時資產(元)": dca_metrics["ending_value"],
                            })
                        if grid_advice_metrics is not None:
                            comparison_rows.append({
                                "做法": "網格（依比較頁設定）",
                                "整段期間損益(%)": grid_advice_metrics["cumulative_return"] * 100,
                                "平均每年報酬(%)": grid_advice_metrics["cagr"] * 100 if grid_advice_metrics.get("cagr") is not None else None,
                                "從高點算起最大跌幅(%)": grid_advice_metrics["mdd"] * 100,
                                "資金起伏幅度(年化 %)": None,
                                 "夏普比率（0%利率）": None,
                                "結束時資產(元)": float(bt_initial_capital) * (1.0 + grid_advice_metrics["cumulative_return"]),
                            })
                        comparison = pd.DataFrame(comparison_rows)
                        comparison_formats = {
                            "整段期間損益(%)": "{:+.2f}%",
                            "平均每年報酬(%)": "{:+.2f}%",
                            "從高點算起最大跌幅(%)": "{:.2f}%",
                            "資金起伏幅度(年化 %)": "{:.2f}%",
                            "夏普比率（0%利率）": "{:.2f}",
                            "結束時資產(元)": "{:,.0f}",
                        }
                        comparison_formatters = {
                            column: (lambda value, spec=spec: _format_optional_cell(value, spec))
                            for column, spec in comparison_formats.items()
                        }
                        st.dataframe(comparison.style.format(comparison_formatters, na_rep="—"),
                                     use_container_width=True, hide_index=True)
                        st.caption("表中年化波動率是報酬起伏幅度，夏普比率是每單位起伏對應的年化報酬參考，CAGR 是整段複利換算成年報酬。策略與買進持有使用相同本金、期間和回測成本。定期定額把相同總預算分次投入，並將期末賣出成本計入；其年化報酬按實際投入日期計算，最大跌幅扣除新增投入影響。網格報酬與最大跌幅沿用網格比較頁設定，期末資產按回測本金等比例換算；請確認期間與成本相同再比較。股息未計入。")
                        if selected_tf in ("1分鐘", "5分鐘", "30分鐘", "60分鐘"):
                            st.info("目前資料是短線週期，涵蓋時間不足以代表長期月扣款；定期定額只在資料涵蓋月份的第一筆價格投入。")
                        elif selected_tf in ("1季", "半年", "1年"):
                            st.info("目前資料是季線或更長週期，沒有每個月的價格；定期投入只在有資料的月份執行，並非逐月扣款。")

                        if not df_trades.empty and not analysis["trade_analysis"].empty:
                            reason_frame = pd.DataFrame({
                                "離場時觸發的條件": df_trades["離場觸發原因"].to_numpy(),
                                "扣除費用後損益 (%)": analysis["trade_analysis"]["成本後報酬率 (%)"].to_numpy(),
                            })
                            reason_summary = reason_frame.groupby("離場時觸發的條件", dropna=False).agg(
                                交易筆數=("扣除費用後損益 (%)", "size"),
                                賺錢交易比例=("扣除費用後損益 (%)", lambda values: float((values > 0).mean())),
                                每筆平均損益=("扣除費用後損益 (%)", "mean"),
                            ).reset_index()
                            st.markdown("#### 不同離場條件下的交易結果")
                            st.dataframe(reason_summary.style.format({"賺錢交易比例": "{:.1%}", "每筆平均損益": "{:+.2f}%"}),
                                         use_container_width=True, hide_index=True)
                            st.caption("交易筆數是符合該離場條件的次數；賺錢交易比例是扣費後仍賺錢的占比；每筆平均損益是這些交易平均賺或賠的幅度。同一筆交易若同時觸發數個條件，會合併歸類，這只是整理結果，不代表某一條件單獨造成盈虧。")

                        st.markdown("#### 費用增加後，結果可能如何變化")
                        cost_rows = []
                        for scenario_name, multiplier in (("設定成本", 1.0), ("成本提高 50%", 1.5), ("成本加倍", 2.0)):
                            scenario_result = analyze_indicator_backtest(
                                df_calc, df_trades, float(bt_initial_capital),
                                float(bt_commission_pct) * multiplier,
                                float(bt_sell_tax_pct) * multiplier,
                                float(bt_slippage_pct) * multiplier,
                                selected_tf, is_tw_stock)
                            if scenario_result is not None:
                                scenario_metrics = scenario_result["strategy_metrics"]
                                cost_rows.append({
                                    "費用假設": scenario_name,
                                    "整段期間損益(%)": scenario_metrics["cumulative_return"] * 100,
                                    "從高點算起最大跌幅(%)": scenario_metrics["mdd"] * 100,
                                    "結束時資產(元)": scenario_metrics["ending_value"],
                                })
                        if cost_rows:
                            st.dataframe(pd.DataFrame(cost_rows).style.format({
                                "整段期間損益(%)": "{:+.2f}%", "從高點算起最大跌幅(%)": "{:.2f}%", "結束時資產(元)": "{:,.0f}",
                            }), use_container_width=True, hide_index=True)
                            st.caption("這裡固定使用同一批買賣訊號，只調高手續費、稅費和滑價假設；用來觀察費用變貴時，獲利可能縮水多少。")

                        oos_advice_metrics = None
                        if len(df_calc) >= 40:
                            split_index = int(len(df_calc) * 0.7)
                            period_rows = []
                            if split_index >= 2 and len(df_calc) - split_index >= 2:
                                if not df_trades.empty and {"_entry_index", "_exit_index"}.issubset(df_trades.columns):
                                    in_sample_trades = df_trades[
                                        (df_trades["_entry_index"] >= 0) & (df_trades["_exit_index"] < split_index)
                                    ].copy()
                                    oos_trades = df_trades[
                                        (df_trades["_entry_index"] >= split_index) &
                                        (df_trades["_exit_index"] < len(df_calc))
                                    ].copy()
                                else:
                                    in_sample_trades = pd.DataFrame()
                                    oos_trades = pd.DataFrame()
                                oos_trades["_entry_index"] = oos_trades.get("_entry_index", pd.Series(dtype=float)) - split_index
                                oos_trades["_exit_index"] = oos_trades.get("_exit_index", pd.Series(dtype=float)) - split_index
                                for period_name, period_data, period_trades in (
                                    ("較早的前 70% 資料", df_calc.iloc[:split_index], in_sample_trades),
                                    ("較晚的後 30% 資料", df_calc.iloc[split_index:], oos_trades),
                                ):
                                    period_result = analyze_indicator_backtest(
                                        period_data, period_trades, float(bt_initial_capital),
                                        float(bt_commission_pct), float(bt_sell_tax_pct), float(bt_slippage_pct),
                                        selected_tf, is_tw_stock)
                                    if period_result is not None:
                                        period_metrics = period_result["strategy_metrics"]
                                        if period_name == "較晚的後 30% 資料":
                                            oos_advice_metrics = period_metrics.copy()
                                        period_rows.append({
                                            "期間": period_name,
                                            "整段期間損益(%)": period_metrics["cumulative_return"] * 100,
                                            "平均每年報酬(%)": period_metrics["cagr"] * 100 if period_metrics["cagr"] is not None else None,
                                            "從高點算起最大跌幅(%)": period_metrics["mdd"] * 100,
                                            "交易筆數": period_metrics["trade_count"],
                                            "賺錢交易比例": period_metrics["win_rate"],
                                        })
                            if period_rows:
                                st.markdown("#### 較早資料和較晚資料的結果比較")
                                period_formats = {
                                    "整段期間損益(%)": "{:+.2f}%",
                                    "平均每年報酬(%)": "{:+.2f}%",
                                    "從高點算起最大跌幅(%)": "{:.2f}%",
                                    "賺錢交易比例": "{:.1%}",
                                }
                                period_formatters = {
                                    column: (lambda value, spec=spec: _format_optional_cell(value, spec))
                                    for column, spec in period_formats.items()
                                }
                                st.dataframe(pd.DataFrame(period_rows).style.format(period_formatters, na_rep="—"),
                                             use_container_width=True, hide_index=True)
                                st.caption(f"以 {df_calc.index[split_index]:%Y-%m-%d} 分成前後兩段，各段從相同本金重新計算；跨過分界日的交易不列入。表中的損益、最大跌幅、交易筆數和賺錢比例，分別表示整體賺賠、曾經最深跌幅、買賣次數和賺錢交易占比。後段用來看看同一套設定在較晚資料的表現，但設定沒有另外調整，因此不代表未來一定能維持相同結果。")

                        dca_advice_metrics = backtest_dca_result["metrics"] if backtest_dca_result is not None else None
                        backtest_advice = build_backtest_coach_advice(
                            strategy_metrics, benchmark_metrics, dca_advice_metrics,
                            grid_advice_metrics, oos_advice_metrics,
                        )
                        condition_label = describe_backtest_conditions(
                            entry_conds, exit_conds,
                            float(bt_rr_multiple) if bt_rr_enabled else None,
                        )
                        st.markdown("#### 🧭 操盤手回測條件提示")
                        st.info(f"**判讀：{backtest_advice['status']}**\n\n" + "\n\n".join(f"- {note}" for note in backtest_advice["notes"]))
                        st.caption("提示會一起參考交易筆數、賺錢交易比例、賺賠金額比、每筆平均損益、同期比較和較晚期間。40%／55% 勝率、20／30 筆、較晚期間 10 筆及 1.2 賺賠比只是畫面提示門檻，不是保證獲利標準。網格比較沿用另一頁的資金與成本設定；請先確認設定相同。")

                        dca_total_return = None
                        if dca_advice_metrics and dca_advice_metrics.get("total_contributed", 0) > 0:
                            dca_total_return = (dca_advice_metrics["ending_value"] / dca_advice_metrics["total_contributed"] - 1.0) * 100.0
                        grid_total_return_for_advice = grid_advice_metrics["cumulative_return"] * 100.0 if grid_advice_metrics else None
                        advice_record = {
                            "ticker": selected_stock,
                            "stock_name": c_name,
                            "condition": condition_label,
                            "timeframe": selected_tf,
                            "period": f"{df_calc.index[0]:%Y-%m-%d}～{df_calc.index[-1]:%Y-%m-%d}",
                            "trade_count": int(strategy_metrics["trade_count"]),
                            "win_rate_pct": strategy_metrics["win_rate"] * 100.0 if strategy_metrics["win_rate"] is not None else None,
                            "total_return_pct": strategy_metrics["cumulative_return"] * 100.0,
                            "profit_factor": strategy_metrics["profit_factor"],
                            "buy_hold_pct": benchmark_metrics["cumulative_return"] * 100.0,
                            "dca_pct": dca_total_return,
                            "grid_pct": grid_total_return_for_advice,
                            "oos_pct": oos_advice_metrics["cumulative_return"] * 100.0 if oos_advice_metrics else None,
                            "status": backtest_advice["status"],
                            "advice": "；".join(backtest_advice["notes"][:2]),
                        }
                        history_key = (advice_record["ticker"], advice_record["timeframe"], advice_record["period"], advice_record["condition"])
                        history = list(st.session_state.get("trader_backtest_history", []))
                        history = [item for item in history if (
                            item.get("ticker"), item.get("timeframe"), item.get("period"), item.get("condition")
                        ) != history_key]
                        history.append(advice_record)
                        st.session_state["trader_backtest_history"] = history[-50:]

                        st.markdown("#### 資產變化和從高點下跌的幅度")
                        equity_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                                   row_heights=[0.68, 0.32],
                                                   subplot_titles=("扣除費用後的資產變化", "距離過去高點的跌幅"))
                        equity_fig.add_trace(go.Scatter(x=analysis["strategy_equity"].index,
                                                       y=analysis["strategy_equity"].values,
                                                       name="依訊號買賣", line=dict(color="#2E86DE", width=2)), row=1, col=1)
                        equity_fig.add_trace(go.Scatter(x=analysis["benchmark_equity"].index,
                                                       y=analysis["benchmark_equity"].values,
                                                       name="直接買進並持有", line=dict(color="#7F8C8D", width=1.5)), row=1, col=1)
                        if backtest_dca_result is not None:
                            equity_fig.add_trace(go.Scatter(
                                x=backtest_dca_result["history"].index,
                                y=backtest_dca_result["history"]["portfolio_value"].values,
                                name="每月定期定額", line=dict(color="#4C78A8", width=1.5, dash="dot")), row=1, col=1)
                        equity_fig.add_trace(go.Scatter(x=analysis["strategy_drawdown"].index,
                                                       y=analysis["strategy_drawdown"].values * 100,
                                                       name="低於過去高點的幅度", fill="tozeroy",
                                                       line=dict(color="#E74C3C", width=1.5)), row=2, col=1)
                        equity_fig.update_yaxes(title_text="資產金額（元）", row=1, col=1)
                        equity_fig.update_yaxes(title_text="比過去高點低（%）", ticksuffix="%", row=2, col=1)
                        equity_fig.update_layout(height=560, hovermode="x unified", legend=dict(orientation="h", y=1.08),
                                                 margin=dict(l=20, r=20, t=65, b=20))
                        st.plotly_chart(equity_fig, use_container_width=True, config={"responsive": True})
                        st.caption("上圖比較策略資產和直接買進持有的資產變化；下圖顯示策略資產比之前最高點低多少，越往下代表當時跌得越深。")

                        period_returns = analysis["period_returns"].dropna()
                        if not period_returns.empty:
                            period_frame = period_returns.rename("期間報酬").to_frame()
                            period_frame["年度"] = period_frame.index.year
                            period_type = analysis["period_type"]
                            if period_type == "quarter":
                                period_frame["期間"] = period_frame.index.quarter
                                period_axis = [f"{period}季" for period in range(1, analysis["period_count"] + 1)]
                            elif period_type == "half_year":
                                period_frame["期間"] = (period_frame.index.month - 1) // 6 + 1
                                period_axis = ["上半年", "下半年"]
                            elif period_type == "year":
                                period_frame["期間"] = 1
                                period_axis = ["年度"]
                            else:
                                period_frame["期間"] = period_frame.index.month
                                period_axis = [f"{period}月" for period in range(1, analysis["period_count"] + 1)]
                            monthly_pivot = period_frame.pivot(index="年度", columns="期間", values="期間報酬").reindex(columns=range(1, analysis["period_count"] + 1))
                            st.markdown(f"#### 每期報酬顏色表（{analysis['period_label']}）")
                            heatmap = go.Figure(go.Heatmap(
                                z=monthly_pivot.to_numpy() * 100,
                                x=period_axis,
                                y=[str(year) for year in monthly_pivot.index],
                                colorscale="RdYlGn", zmid=0,
                                colorbar=dict(title="報酬 %"),
                                hovertemplate="%{y} %{x}<br>報酬 %{z:+.2f}%<extra></extra>",
                            ))
                            heatmap.update_layout(height=max(250, 48 * len(monthly_pivot) + 100),
                                                  margin=dict(l=20, r=20, t=20, b=20))
                            st.plotly_chart(heatmap, use_container_width=True, config={"responsive": True})
                            st.caption("每格代表該月、該季、該半年或該年度的資產變化；綠色表示上漲，紅色表示下跌，顏色越深代表幅度越大。沒有顏色表示該期沒有資料。")

                        st.caption(
                            f"交易筆數 {strategy_metrics['trade_count']}；每筆平均損益 {show_percent_points(strategy_metrics['expectancy_pct'])}；"
                            f"平均賺錢幅度 {show_percent_points(strategy_metrics['average_win_pct'])}；平均賠錢幅度 {show_percent_points(strategy_metrics['average_loss_pct'])}；"
                            f"平均賺賠幅度比 {show_number(strategy_metrics['payoff_ratio'])}；最多連續賠錢 {strategy_metrics['longest_loss_streak']} 筆；"
                            f"最長未回到先前高點的時間 {strategy_metrics['longest_drawdown_bars']} 根走勢資料。"
                        )
                        if strategy_metrics["cagr"] is None:
                            st.info("回測時間跨度或有效資料筆數不足，為避免年化數字失真，平均每年報酬和報酬起伏參考數值暫不顯示。")
                        st.caption(
                            "模擬假設所有本金可投入且可買零碎股；進場及指標訊號依收盤價處理。啟用風險報酬停利時，停損／停利以當根最高價、最低價判斷並以觸發價成交；若開盤跳空越過觸發價，按開盤價估算。"
                            "同一根 K 棒同時碰到停損與停利時，採較保守的停損先出；日線無法還原當日價格先後。直接買進持有使用相同價格資料，沒有另外估算股息。"
                        )

    except Exception as ex_tab:
        st.error(f"❌ 畫面渲染異常：{ex_tab}")

    if selected_module == "🚀 華爾街機構級三核心策略雷達":
        st.markdown("---")
        st.markdown("### 🚀 華爾街機構級三核心策略雷達")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            strat_gvi = st.checkbox("開啟 GVI 價值雷達", value=True, key="strat_gvi_check")
            st.caption("研究假設：以 ROE 與淨值資料計算 GVI。財報更新較慢，產業間可比性有限。")
        with col_s2:
            strat_momentum = st.checkbox("開啟動能突破雷達", value=False, key="strat_momentum_check")
            st.caption("研究假設：以可用歷史高點計算距離。歷史資料少於 50 根時不評分、不納入篩選。")
        with col_s3:
            strat_qarp = st.checkbox("開啟 QARP 現金流雷達", value=False, key="strat_qarp_check")
            st.caption("研究假設：以自由現金流收益率與 PEG 作條件；缺少任一資料時不計入。")

        with st.expander("📐 GVI、動能與 QARP 的獨立樣本外驗證", expanded=False):
            st.warning(
                "這些門檻目前是研究假設，尚未證明具有穩定優勢。本版本沒有逐因子樣本外績效報告，選股分數不代表正期望值或買賣建議。"
            )
            validation_status = pd.DataFrame([
                {"因子": "GVI", "驗證狀態": "未驗證", "資料限制": "目前只有當期 ROE／淨值，沒有歷史可得日期的財報快照。"},
                {"因子": "動能", "驗證狀態": "未驗證", "資料限制": "可取歷史價格，但尚未執行獨立 walk-forward 樣本外比較。"},
                {"因子": "QARP", "驗證狀態": "未驗證", "資料限制": "目前只有當期自由現金流／PEG，沒有歷史點時資料。"},
            ])
            st.dataframe(validation_status, use_container_width=True, hide_index=True)
            st.caption(
                "有效比較需以相同股票池與期間，比較未加條件的基準組和逐一加入單一因子的組合；先鎖定參數，再用未參與調參的後段資料檢查超額報酬、回撤、換手與交易成本。"
                "缺少歷史點時財報時不會以當期基本面回填過去，避免前視偏誤。"
            )

        st.info(
            "🧠 **三核心的市場心理讀法**\n\n"
            "**價值（GVI）：**可能吸引重視公司獲利與帳面價值的投資者，但低估值不等於安全。\n\n"
            "**趨勢（動能）：**接近近期高位可能吸引順勢買盤，也要留意追價過度與交易擁擠。\n\n"
            "**現金流與估值（QARP）：**可能吸引重視現金創造力和價格合理性的投資者，仍要查證現金流是否持續、成長預估是否可靠。\n\n"
            "條件分歧時先找出原因，不要只挑支持原有看法的指標；分數只表示本次條件比對，不是勝率或績效保證。"
        )

        st.markdown("#### ✍️ 操盤手自訂觀察代碼掃描區")
        st.caption("每個欄位輸入一檔股票代碼。台股上市請加 .TW、上櫃請加 .TWO；美股輸入代碼即可。下方掃描結果沿用台股／美股核心池的明細欄位與策略評分方式。")
        watch_defaults = ["2330.TW", "3293.TWO", "8069.TWO", "NVDA", "AAPL"] + [""] * 7
        saved_watch_codes = list(st.session_state.get("trader_watchlist_values", watch_defaults))
        sync_watch_widgets = bool(st.session_state.pop("trader_watchlist_sync_needed", False))
        for watch_index in range(12):
            watch_key = f"trader_watch_ticker_{watch_index + 1}"
            if sync_watch_widgets or watch_key not in st.session_state:
                st.session_state[watch_key] = saved_watch_codes[watch_index]
        watch_columns = st.columns(3)
        custom_scan_list = []
        duplicate_watch_codes = []
        for watch_index in range(12):
            with watch_columns[watch_index % 3]:
                watch_code = st.text_input(
                    f"觀察代碼 {watch_index + 1}",
                    value=watch_defaults[watch_index],
                    placeholder="例如 2330.TW 或 NVDA",
                    key=f"trader_watch_ticker_{watch_index + 1}",
                ).strip().upper()
            if watch_code:
                if watch_code in custom_scan_list:
                    duplicate_watch_codes.append(watch_code)
                else:
                    custom_scan_list.append(watch_code)
        st.session_state["trader_watchlist_values"] = [
            str(st.session_state.get(f"trader_watch_ticker_{index + 1}", watch_defaults[index])).strip().upper()
            for index in range(12)
        ]
        if duplicate_watch_codes:
            st.caption("重複代碼已自動略過：" + "、".join(dict.fromkeys(duplicate_watch_codes)))
        st.caption(f"目前輸入 {len(custom_scan_list)} 檔；掃描結果會列出排名、股票代碼／名稱、股價、策略評分及已啟用指標明細。")

        coach_history = st.session_state.get("trader_backtest_history", [])
        if coach_history:
            matched_history = [item for item in coach_history if item.get("ticker") in custom_scan_list]
            if not matched_history:
                matched_history = coach_history[-20:]
                st.caption("目前觀察代碼沒有對應的回測紀錄，以下先顯示最近測試過的條件；加入該股票代碼後會一併列入觀察名單。")
            if matched_history:
                st.markdown("#### 🧭 技術指標回測條件觀察紀錄")
                st.caption("僅整理您在回測頁手動執行過的條件；不會自動改動選股分數或替您下單。")
                status_rank = {"可優先列入觀察": 0, "繼續觀察": 1, "樣本不足": 2, "暫不建議直接採用": 3}
                matched_history.sort(key=lambda item: (
                    status_rank.get(item.get("status"), 4),
                    -(item.get("win_rate_pct") if item.get("win_rate_pct") is not None else -1),
                    -(item.get("total_return_pct") if item.get("total_return_pct") is not None else -1),
                ))
                coach_view = pd.DataFrame(matched_history[:20])
                coach_columns = ["ticker", "stock_name", "condition", "timeframe", "period", "trade_count", "win_rate_pct",
                                 "total_return_pct", "buy_hold_pct", "dca_pct", "grid_pct", "oos_pct",
                                 "profit_factor", "status", "advice"]
                coach_view = coach_view[[column for column in coach_columns if column in coach_view.columns]].rename(columns={
                    "ticker": "股票代碼", "stock_name": "股票名稱", "condition": "回測條件", "timeframe": "週期", "period": "資料期間",
                    "trade_count": "交易筆數", "win_rate_pct": "勝率(%)", "total_return_pct": "整段報酬(%)",
                    "buy_hold_pct": "長期持有(%)", "dca_pct": "定期定額(%)", "grid_pct": "網格(%)",
                    "oos_pct": "較晚期間(%)", "profit_factor": "賺賠金額比", "status": "操盤手提示", "advice": "建議摘要",
                })
                coach_formats = {
                    "勝率(%)": "{:.1f}%",
                    "整段報酬(%)": "{:+.2f}%",
                    "長期持有(%)": "{:+.2f}%",
                    "定期定額(%)": "{:+.2f}%",
                    "網格(%)": "{:+.2f}%",
                    "較晚期間(%)": "{:+.2f}%",
                    "賺賠金額比": "{:.2f}",
                }
                coach_formatters = {
                    column: (lambda value, spec=spec: _format_optional_cell(value, spec))
                    for column, spec in coach_formats.items()
                }
                st.dataframe(coach_view.style.format(coach_formatters, na_rep="—"),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("目前觀察代碼中尚無技術指標回測紀錄；請先在回測頁執行條件測試。")
        else:
            st.caption("技術指標回測完成後，條件勝率與報酬建議會出現在此處；紀錄保留於目前工作階段。")

        def render_stock_screen_results(result_view, title):
            st.markdown(f"#### {title}（依已啟用研究條件排名，最多 20 檔）")
            if result_view.empty:
                st.info("本次無符合標的。已完整套用目前勾選的條件，沒有放寬條件或回退顯示未通過的股票。")
                return
            st.caption("GVI、動能、自由現金流與成長估值是研究用估算；策略評分只表示本次股票池中的相對排序，不是勝率或預期報酬。價格接近高位可能吸引順勢買盤，也可能增加追價風險；價量動能不是實際籌碼或法人資料。")
            st.caption("點選明細表中的股票列，再選擇個股即時診斷或加入操盤手自訂觀察代碼。")
            table_event = st.dataframe(
                result_view,
                use_container_width=True,
                hide_index=True,
                key="stock_screen_results_table",
                on_select="rerun",
                selection_mode="single-row",
            )
            selected_rows = list(table_event.selection.rows)
            if selected_rows:
                selected_row = int(selected_rows[0])
                if 0 <= selected_row < len(result_view):
                    selected_result = result_view.iloc[selected_row]
                    st.session_state["stock_screen_selected_ticker"] = str(selected_result["股票代碼"])
                    selected_name = str(selected_result.get("股票名稱", ""))
                    selected_ticker = str(selected_result["股票代碼"])
                    _render_stock_actions(selected_ticker, selected_name, "stock_screen_selected")
            else:
                st.session_state.pop("stock_screen_selected_ticker", None)

        def load_data_from_sqlite_and_render(filter_list, title, is_custom_mode=False):
            st.session_state["stock_screen_result"] = None
            st.session_state.pop("stock_screen_selected_ticker", None)
            st.session_state.pop("stock_screen_results_table", None)
            if not any((strat_gvi, strat_momentum, strat_qarp)):
                st.info("請至少開啟一項研究條件，再執行掃描。")
                return
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

            if df_db.empty:
                st.info("目前沒有可供篩選的快取資料。")
                return
            
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
                    close_series = pd.to_numeric(df_hist['Close'], errors='coerce').dropna() if 'Close' in df_hist.columns else pd.Series(dtype=float)
                    high_252d = float(close_series.tail(min(252, len(close_series))).max()) if len(close_series) >= 50 else np.nan
                    momentum_value = (close_today / high_252d) * 100 if pd.notna(high_252d) and high_252d > 0 else np.nan
                
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
                        '動能分數': momentum_value, 
                        '自由現金流收益': fcf_yield, 
                        '本益成長比(PEG)': peg_val, 
                        '⚡ 價量動能估算': chip_status_text, 
                        'gvi_raw': gvi_live, 
                        'mo_raw': momentum_value, 
                        'fcf_raw': fcf_yield
                    })
                except Exception as e_item: 
                    st.sidebar.caption(f"標的跳過 [{t}]: {e_item}")
                
            if raw_portfolio:
                df_res = pd.DataFrame(raw_portfolio)
            
                if strat_gvi: df_res = df_res[df_res['gvi_raw'].notnull() & (df_res['gvi_raw'] >= 0.20)]
                if strat_momentum: df_res = df_res[df_res['mo_raw'].notnull() & (df_res['mo_raw'] >= 80.0)]
                if strat_qarp: df_res = df_res[df_res['fcf_raw'].notnull() & (df_res['fcf_raw'] >= 2.5) & df_res['本益成長比(PEG)'].notnull() & (df_res['本益成長比(PEG)'] <= 1.5)]

                if df_res.empty:
                    empty_view = pd.DataFrame(columns=['排名', '股票代碼', '股票名稱', '目前股價', '策略評分'])
                    st.session_state["stock_screen_result"] = {"view": empty_view, "title": title}
                    render_stock_screen_results(empty_view, title)
                    return
                
                df_res['GVI_Rank'] = df_res['gvi_raw'].rank(pct=True)
                df_res['MO_Rank'] = df_res['mo_raw'].rank(pct=True)
                df_res['FCF_Rank'] = df_res['fcf_raw'].rank(pct=True)
                df_res['綜合分數'] = 0.0
                active_strats = 0
            
                if strat_gvi: df_res['綜合分數'] += df_res['GVI_Rank']; active_strats += 1
                if strat_momentum: df_res['綜合分數'] += df_res['MO_Rank']; active_strats += 1
                if strat_qarp: df_res['綜合分數'] += df_res['FCF_Rank']; active_strats += 1
                if active_strats == 0:
                    st.info("請至少開啟一項研究條件，再執行掃描。")
                    return
            
                df_res['策略評分_num'] = ((df_res['綜合分數'] / active_strats) * 100).round(1)
                df_res = df_res.sort_values(by='綜合分數', ascending=False).head(20).reset_index(drop=True)
            
                df_res['排名'] = [f"第 {i+1} 名" for i in range(len(df_res))]
                df_res['策略評分'] = df_res['策略評分_num'].astype(str) + " 分"
                df_res['GVI價值指標'] = df_res['GVI值'].apply(lambda x: f"{x:.4f}" if (pd.notnull(x) and x is not None) else "資料不足")
                df_res['動能(創高距離)'] = df_res['動能分數'].apply(lambda value: f"{value:.1f}%" if pd.notna(value) else "資料不足")
                df_res['自由現金流收益率'] = df_res['自由現金流收益'].apply(lambda x: f"{x:.2f}%" if (pd.notnull(x) and x is not None) else "資料不足")
                df_res['本益成長比(PEG)'] = df_res['本益成長比(PEG)'].apply(lambda x: f"{x:.2f}" if (pd.notnull(x) and x is not None) else "資料不足")
            
                cols = ['排名', '股票代碼', '股票名稱', '目前股價', '策略評分']
                if strat_gvi: cols.append('GVI價值指標')
                if strat_momentum: cols.append('動能(創高距離)')
                if strat_qarp: cols.extend(['自由現金流收益率', '本益成長比(PEG)'])
                if not strat_gvi and not strat_momentum and not strat_qarp:
                    cols.extend(['GVI價值指標', '動能(創高距離)', '自由現金流收益率', '本益成長比(PEG)'])
                cols.append('⚡ 價量動能估算')
            
                df_final_view = df_res[cols]
                st.session_state["stock_screen_result"] = {
                    "view": df_final_view.copy(),
                    "title": title,
                }
                render_stock_screen_results(df_final_view, title)
            else:
                st.info("沒有取得可計算的股票資料，本次無符合標的。")

        scan_triggered_this_run = False
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🇹🇼 一鍵執行：全自動過濾全台股核心池", type="secondary", use_container_width=True):
                scan_triggered_this_run = True
                load_data_from_sqlite_and_render(AUTO_TW_UNIVERSE, "🇹🇼 台灣股市核心策略篩選結果")
        with col_btn2:
            if st.button("🇺🇸 一鍵執行：全自動過濾全美股核心池", type="secondary", use_container_width=True):
                scan_triggered_this_run = True
                load_data_from_sqlite_and_render(AUTO_US_UNIVERSE, "🇺🇸 美國股市核心策略篩選結果")
        with col_btn3:
            if st.button("🚀 執行：自訂名單多因子本地精準過濾", type="primary", use_container_width=True):
                scan_triggered_this_run = True
                load_data_from_sqlite_and_render(custom_scan_list, "🎯 操盤手自訂名單策略篩選結果", is_custom_mode=True)
        if not scan_triggered_this_run:
            saved_screen_result = st.session_state.get("stock_screen_result")
            if isinstance(saved_screen_result, dict) and isinstance(saved_screen_result.get("view"), pd.DataFrame):
                render_stock_screen_results(saved_screen_result["view"], saved_screen_result.get("title", "選股策略篩選結果"))

    if selected_module == "🏦 台股三大法人獨立選股策略":
        st.markdown("---")
        st.markdown("### 🏦 台股三大法人獨立選股策略")
        st.caption("只掃描自訂觀察名單中的台股普通股。百分比是把法人當日交易量放到公司已發行股數的規模中比較；正值代表買進多於賣出，負值代表賣出多於買進。這是當日交易流量，不是法人持股比例，也不代表後續方向。")
        inst_col1, inst_col2 = st.columns([1, 2])
        with inst_col1:
            inst_investor = st.selectbox("篩選法人", ['外資', '投信', '自營商'], key='inst_screen_investor')
            inst_min_net_pct = st.number_input("淨買賣超佔股本至少 (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.01, format='%.2f', key='inst_min_net_pct')
        with inst_col2:
            st.write("篩選條件：所選法人的當日淨買進規模需達到你設定的門檻。表格仍列出外資、投信、自營商三方的買進、賣出與淨流向規模，方便比較交易方向和強弱。")
            st.caption("🧠 **心理讀法：**單日法人買超容易引發跟隨心理，但它只代表已發生的交易。觀察資料日期、連續性、股價與成交量；不能由買超直接推論法人正在吸籌或股價必漲。")
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

    if selected_module == "📊 當日全市場三大法人佔股本比 Top 10":
        st.markdown("#### 當日全市場三大法人佔股本比 Top 10")
        st.caption("表格的百分比把法人當日買進、賣出或淨流向，換成相對公司已發行股數的規模，並非法人持股比例。正值代表買進多於賣出，負值代表賣出多於買進。榜單列出淨買超較高、淨賣超較高，以及買進或賣出較活躍的前 10 檔；上市與上櫃採各自官方最新資料日。點選股票列，可選擇即時診斷或加入自訂觀察代碼。")
        st.info("🧠 **市場心理重點：**榜單名次容易讓人把「排前面」誤認成「比較會漲」。單日交易規模不代表持倉變化或未來方向；先看資料日期，再觀察是否連續出現，以及價格有沒有同步反應。")
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
                            show_display = show.copy()
                            show_display['佔股本比(%)'] = show_display['佔股本比(%)'].apply(
                                lambda value: f"{value:+.4f}%" if pd.notna(value) else "—"
                            )
                            with ranking_cols[index % 2]:
                                st.markdown(f"**{title} Top 10**")
                                table_key = f"inst_top10_{investor}_{index}"
                                top_event = st.dataframe(
                                    show_display,
                                    use_container_width=True,
                                    hide_index=True,
                                    key=table_key,
                                    on_select="rerun",
                                    selection_mode="single-row",
                                )
                                top_rows = list(top_event.selection.rows)
                                if top_rows:
                                    chosen = show.iloc[int(top_rows[0])]
                                    ticker_suffix = ".TWO" if str(chosen["市場"]) == "上櫃" else ".TW"
                                    selected_ticker = f"{chosen['股票代碼']}{ticker_suffix}"
                                    action_key = f"inst_action_{investor}_{index}"
                                    _render_stock_actions(selected_ticker, str(chosen["股票名稱"]), action_key)
                if market_errors:
                    st.caption("部分官方來源未能連線：" + "；".join(market_errors[:2]))


    # ==============================================================================
    # 【Tab 10：投資組合理財】獨立於選股策略；只計算與提供建議，不會自動下單。
    # ==============================================================================
    if selected_module == "🧭 投資組合理財":
        st.subheader("🧭 投資組合理財")
        st.caption("本模組只管理投資組合，不管理個人收入／支出，也不會自動下單。初始配置僅為可修改的示範值，不代表最佳配置或回撤保證。")
        portfolio_user_id = int(auth_user["id"])

        def _portfolio_load_settings():
            with sqlite3.connect(DB_FILE) as _conn:
                _rows = _conn.execute(
                    "SELECT setting_key, setting_value FROM portfolio_user_settings WHERE user_id=?",
                    (portfolio_user_id,),
                ).fetchall()
            return {str(k): float(v) for k, v in _rows}

        def _portfolio_save_setting(key, value):
            with sqlite3.connect(DB_FILE) as _conn:
                _conn.execute(
                    "INSERT INTO portfolio_user_settings(user_id,setting_key,setting_value) VALUES (?,?,?) "
                    "ON CONFLICT(user_id,setting_key) DO UPDATE SET setting_value=excluded.setting_value",
                    (portfolio_user_id, str(key), float(value))
                )
                _conn.commit()

        def _portfolio_positions():
            with sqlite3.connect(DB_FILE) as _conn:
                return pd.read_sql_query(
                    "SELECT id, symbol AS 代碼, name AS 名稱, asset_type AS 資產類型, currency AS 幣別, "
                    "quantity AS 數量, price AS 最新價格, fx_to_twd AS 換算台幣匯率, "
                    "cost_basis_twd AS 台幣成本, updated_at AS 更新時間 "
                    "FROM portfolio_positions WHERE user_id=? ORDER BY id DESC",
                    _conn, params=(portfolio_user_id,)
                )

        settings = _portfolio_load_settings()
        st.markdown("#### 1. 配置與風險規則")
        with st.form("portfolio_settings_form"):
            cfg1, cfg2, cfg3 = st.columns(3)
            with cfg1:
                target_stock = st.number_input("股票目標配置 (%)", min_value=0.0, max_value=100.0, value=float(settings.get("target_stock_pct", 40)), step=1.0)
                target_bond = st.number_input("債券目標配置 (%)", min_value=0.0, max_value=100.0, value=float(settings.get("target_bond_pct", 50)), step=1.0)
                target_cash = st.number_input("現金目標配置 (%)", min_value=0.0, max_value=100.0, value=float(settings.get("target_cash_pct", 10)), step=1.0)
            with cfg2:
                rebalance_threshold = st.number_input("再平衡偏離門檻（百分點）", min_value=1.0, max_value=30.0, value=float(settings.get("rebalance_threshold_pct", 5)), step=1.0)
                minimum_cash = st.number_input("最低保留現金（新台幣）", min_value=0.0, value=float(settings.get("minimum_cash_twd", 0)), step=1000.0,
                                               help="只在計算後檢查現金市值是否低於此金額並提醒；不會自動改變持倉或再平衡金額。")
            with cfg3:
                dd_warn = st.number_input("回撤警示 (%)", min_value=1.0, max_value=30.0, value=float(settings.get("drawdown_warning_pct", 8)), step=1.0)
                dd_review = st.number_input("回撤檢查 (%)", min_value=1.0, max_value=40.0, value=float(settings.get("drawdown_review_pct", 12)), step=1.0)
                dd_limit = st.number_input("回撤風險審查門檻 (%)", min_value=1.0, max_value=60.0, value=float(settings.get("drawdown_limit_pct", 15)), step=1.0)
            save_cfg = st.form_submit_button("儲存配置規則")
        if save_cfg:
            if abs(target_stock + target_bond + target_cash - 100.0) > 0.01:
                st.error("股票、債券、現金目標比例加總必須等於 100%。")
            elif not (dd_warn < dd_review < dd_limit):
                st.error("回撤門檻需依序遞增：警示 < 檢查 < 風險審查。")
            else:
                for key, value in {
                    "target_stock_pct": target_stock, "target_bond_pct": target_bond,
                    "target_cash_pct": target_cash, "rebalance_threshold_pct": rebalance_threshold,
                    "minimum_cash_twd": minimum_cash, "drawdown_warning_pct": dd_warn,
                    "drawdown_review_pct": dd_review, "drawdown_limit_pct": dd_limit,
                }.items():
                    _portfolio_save_setting(key, value)
                st.success("配置規則已儲存。")
                settings = _portfolio_load_settings()

        st.markdown("#### 2. 新增持倉")
        with st.form("portfolio_add_position_form", clear_on_submit=True):
            p1, p2, p3 = st.columns(3)
            with p1:
                p_symbol = st.text_input("代碼／識別名稱", placeholder="例如 2330.TW、BOND-01、CASH-TWD").strip().upper()
                p_name = st.text_input("資產名稱", placeholder="例如台積電、短天期美債 ETF")
                p_type = st.selectbox("資產類型", ["股票", "債券", "現金"])
            with p2:
                p_currency = st.selectbox("幣別", ["TWD", "USD", "JPY", "HKD", "EUR"])
                p_quantity = st.number_input("持有數量／單位", min_value=0.0, value=0.0, step=1.0, format="%.6f")
                p_price = st.number_input("最新價格／每單位金額", min_value=0.0, value=0.0, step=1.0, format="%.6f")
            with p3:
                p_fx = st.number_input("1 單位外幣兌新台幣匯率（TWD 資產填 1）", min_value=0.000001, value=1.0, step=0.1, format="%.6f")
                p_cost = st.number_input("此筆持倉台幣總成本（選填，0 表示未填）", min_value=0.0, value=0.0, step=1000.0)
            add_position = st.form_submit_button("新增持倉")
        if add_position:
            if not p_symbol:
                st.error("請填寫代碼或識別名稱。")
            elif p_quantity <= 0 or p_price < 0:
                st.error("數量需大於 0，價格不可小於 0。")
            elif p_currency == "TWD" and abs(p_fx - 1.0) > 0.000001:
                st.error("台幣資產的匯率請填 1。")
            else:
                with sqlite3.connect(DB_FILE) as _conn:
                    _conn.execute(
                        "INSERT INTO portfolio_positions(user_id,symbol,name,asset_type,currency,quantity,price,fx_to_twd,cost_basis_twd,updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (portfolio_user_id, p_symbol, p_name, p_type, p_currency, float(p_quantity), float(p_price),
                         float(p_fx), float(p_cost), pd.Timestamp.now().isoformat(timespec="seconds"))
                    )
                    _conn.commit()
                st.success("持倉已新增。請注意：最新價格與匯率由你輸入，需自行更新。")

        positions = _portfolio_positions()
        st.markdown("#### 3. 持倉明細與配置計算")
        if positions.empty:
            st.info("目前尚無持倉。先新增股票、債券 ETF 或現金項目，即可計算組合配置。")
        else:
            calc = positions.copy()
            calc["台幣市值"] = pd.to_numeric(calc["數量"], errors="coerce").fillna(0) * pd.to_numeric(calc["最新價格"], errors="coerce").fillna(0) * pd.to_numeric(calc["換算台幣匯率"], errors="coerce").fillna(1)
            calc["未實現損益估算"] = np.where(calc["台幣成本"] > 0, calc["台幣市值"] - calc["台幣成本"], np.nan)
            total_value = float(calc["台幣市值"].sum())
            type_map = {"股票": "stock", "債券": "bond", "現金": "cash"}
            type_values = {kind: float(calc.loc[calc["資產類型"] == label, "台幣市值"].sum())
                           for label, kind in [("股票", "stock"), ("債券", "bond"), ("現金", "cash")]}
            target_map = {
                "stock": float(settings.get("target_stock_pct", 40)),
                "bond": float(settings.get("target_bond_pct", 50)),
                "cash": float(settings.get("target_cash_pct", 10)),
            }
            minimum_cash_floor = max(0.0, float(settings.get("minimum_cash_twd", 0)))
            if minimum_cash_floor > 0 and type_values["cash"] < minimum_cash_floor:
                st.warning(f"目前現金市值 NT${type_values['cash']:,.0f}，低於設定的最低保留現金 NT${minimum_cash_floor:,.0f}。此為提醒，不會自動執行交易。")
            metrics = st.columns(4)
            with metrics[0]: st.metric("組合總市值（TWD）", f"NT${total_value:,.0f}", help="依各筆持倉數量、最新價格與台幣換算匯率估算後加總；價格與匯率需由使用者維護。")
            with metrics[1]: st.metric("股票市值", f"NT${type_values['stock']:,.0f}", f"{type_values['stock']/total_value*100:.1f}%" if total_value > 0 else "0.0%", help="股票類持倉的台幣市值；下方百分比是股票市值占組合總市值的比例。")
            with metrics[2]: st.metric("債券市值", f"NT${type_values['bond']:,.0f}", f"{type_values['bond']/total_value*100:.1f}%" if total_value > 0 else "0.0%", help="債券類持倉的台幣市值；下方百分比是債券市值占組合總市值的比例。")
            with metrics[3]: st.metric("現金市值", f"NT${type_values['cash']:,.0f}", f"{type_values['cash']/total_value*100:.1f}%" if total_value > 0 else "0.0%", help="現金類持倉的台幣市值；下方百分比是現金占組合總市值的比例。")

            allocation_rows = []
            for kind, label in [("stock", "股票"), ("bond", "債券"), ("cash", "現金")]:
                actual_pct = type_values[kind] / total_value * 100 if total_value > 0 else 0.0
                target_pct = target_map[kind]
                diff_pct = actual_pct - target_pct
                target_value = total_value * target_pct / 100
                allocation_rows.append({
                    "資產類型": label, "目前市值(TWD)": type_values[kind],
                    "目前比例(%)": actual_pct, "目標比例(%)": target_pct,
                    "偏離(百分點)": diff_pct, "目標市值(TWD)": target_value,
                    "調整金額參考(TWD)": target_value - type_values[kind],
                })
                threshold = float(settings.get("rebalance_threshold_pct", 5))
                if abs(diff_pct) >= threshold:
                    st.warning(f"{label}目前配置偏離目標 {diff_pct:+.1f} 個百分點，達到再平衡評估門檻。此為評估提示，不是直接下單指令。")
            st.dataframe(pd.DataFrame(allocation_rows).style.format({
                "目前市值(TWD)": "NT${:,.0f}", "目前比例(%)": "{:.2f}%",
                "目標比例(%)": "{:.2f}%", "偏離(百分點)": "{:+.2f}",
                "目標市值(TWD)": "NT${:,.0f}", "調整金額參考(TWD)": "NT${:+,.0f}"
            }), use_container_width=True, hide_index=True)
            st.caption("**欄位怎麼看：**台幣市值是依持有數量、最新價格與匯率估算；未實現損益是目前估值和你填寫成本的差額；目前比例呈現各資產占組合的比重；配置偏離顯示與目標比重相差多少。調整金額為回到目標配置所需的估算差額，正數表示相對不足、負數表示相對超出。成本價容易變成心理錨點，應同時重新檢查持有理由和整體風險；調整金額不是下單指示，尚未計入交易成本、稅費、價差、流動性與稅務差異。")
            st.caption("點選下方持倉股票列，可選擇使用個股即時診斷或加入操盤手自訂觀察代碼。")
            holdings_view = calc[["代碼", "名稱", "資產類型", "幣別", "數量", "最新價格", "換算台幣匯率", "台幣市值", "台幣成本", "未實現損益估算", "更新時間"]]
            holdings_display = holdings_view.copy()
            for column in ("最新價格", "換算台幣匯率"):
                holdings_display[column] = holdings_display[column].apply(lambda value: f"{value:,.4f}" if pd.notna(value) else "—")
            for column in ("台幣市值", "台幣成本"):
                holdings_display[column] = holdings_display[column].apply(lambda value: f"NT${value:,.0f}" if pd.notna(value) else "—")
            holdings_display["未實現損益估算"] = holdings_display["未實現損益估算"].apply(
                lambda value: f"NT${value:+,.0f}" if pd.notna(value) else "—"
            )
            holdings_event = st.dataframe(holdings_display, use_container_width=True, hide_index=True,
                key=f"portfolio_holdings_table_{portfolio_user_id}", on_select="rerun", selection_mode="single-row")
            holdings_rows = list(holdings_event.selection.rows)
            if holdings_rows:
                held_asset = calc.iloc[int(holdings_rows[0])]
                if str(held_asset["資產類型"]) == "股票":
                    held_ticker = str(held_asset["代碼"]).strip().upper()
                    if held_ticker.isdigit():
                        held_ticker = f"{held_ticker}.TW"
                    _render_stock_actions(held_ticker, str(held_asset["名稱"]), "portfolio_holding_selected")

            st.markdown("#### 4. 更新或刪除持倉")
            ids = positions["id"].astype(int).tolist()
            selected_id = st.selectbox(
                "選擇要更新／刪除的持倉",
                ids,
                format_func=lambda x: f"ID {x}｜{positions.loc[positions['id'] == x, '代碼'].iloc[0]}｜{positions.loc[positions['id'] == x, '名稱'].iloc[0]}",
                key=f"portfolio_selected_position_{portfolio_user_id}",
            )
            selected_row = positions.loc[positions["id"] == selected_id].iloc[0]
            update_widget_suffix = f"{portfolio_user_id}_{int(selected_id)}"
            with st.form(f"portfolio_update_position_form_{update_widget_suffix}"):
                u1, u2, u3 = st.columns(3)
                with u1:
                    u_name = st.text_input("名稱", value=str(selected_row["名稱"]), key=f"port_update_name_{update_widget_suffix}")
                    u_type = st.selectbox("資產類型", ["股票", "債券", "現金"], index=["股票", "債券", "現金"].index(selected_row["資產類型"]), key=f"port_update_type_{update_widget_suffix}")
                    u_currency = st.selectbox("幣別", ["TWD", "USD", "JPY", "HKD", "EUR"], index=["TWD", "USD", "JPY", "HKD", "EUR"].index(selected_row["幣別"]) if selected_row["幣別"] in ["TWD", "USD", "JPY", "HKD", "EUR"] else 0, key=f"port_update_currency_{update_widget_suffix}")
                with u2:
                    u_quantity = st.number_input("數量", min_value=0.0, value=float(selected_row["數量"]), step=1.0, format="%.6f", key=f"port_update_quantity_{update_widget_suffix}")
                    u_price = st.number_input("最新價格", min_value=0.0, value=float(selected_row["最新價格"]), step=1.0, format="%.6f", key=f"port_update_price_{update_widget_suffix}")
                with u3:
                    u_fx = st.number_input("換算台幣匯率", min_value=0.000001, value=float(selected_row["換算台幣匯率"]), step=0.1, format="%.6f", key=f"port_update_fx_{update_widget_suffix}")
                    u_cost = st.number_input("台幣總成本", min_value=0.0, value=float(selected_row["台幣成本"]), step=1000.0, key=f"port_update_cost_{update_widget_suffix}")
                uc1, uc2 = st.columns(2)
                with uc1: update_position = st.form_submit_button("儲存此筆更新")
                with uc2: delete_position = st.form_submit_button("刪除此筆持倉")
            if update_position:
                if u_quantity <= 0:
                    st.error("數量需大於 0。")
                elif u_currency == "TWD" and abs(u_fx - 1.0) > 0.000001:
                    st.error("台幣資產的匯率請填 1。")
                else:
                    with sqlite3.connect(DB_FILE) as _conn:
                        _conn.execute(
                            "UPDATE portfolio_positions SET name=?,asset_type=?,currency=?,quantity=?,price=?,fx_to_twd=?,cost_basis_twd=?,updated_at=? WHERE id=? AND user_id=?",
                            (u_name, u_type, u_currency, float(u_quantity), float(u_price), float(u_fx), float(u_cost),
                             pd.Timestamp.now().isoformat(timespec="seconds"), int(selected_id), portfolio_user_id)
                        )
                        _conn.commit()
                    st.success("持倉更新完成。請重新整理頁面查看最新計算。")
            if delete_position:
                with sqlite3.connect(DB_FILE) as _conn:
                    _conn.execute("DELETE FROM portfolio_positions WHERE id=? AND user_id=?", (int(selected_id), portfolio_user_id))
                    _conn.commit()
                st.success("持倉已刪除。")

            st.markdown("#### 5. 組合淨值紀錄與回撤參考")
            st.caption("按下紀錄才會新增一筆淨值快照。快照尚未校正入金／出金，因此回撤僅是粗略參考，不能當成正式績效或最大回撤統計。")
            if st.button("📌 記錄目前組合市值", key="portfolio_record_snapshot"):
                with sqlite3.connect(DB_FILE) as _conn:
                    _conn.execute(
                        "INSERT INTO portfolio_snapshots(user_id,snapshot_date,total_value_twd,stock_value_twd,bond_value_twd,cash_value_twd,note) VALUES(?,?,?,?,?,?,?)",
                        (portfolio_user_id, pd.Timestamp.now().isoformat(timespec="seconds"), total_value, type_values["stock"], type_values["bond"], type_values["cash"], "手動快照；未調整外部現金流")
                    )
                    _conn.commit()
                st.success("已記錄目前市值快照。")
            with sqlite3.connect(DB_FILE) as _conn:
                snapshots = pd.read_sql_query("SELECT snapshot_date AS 記錄時間,total_value_twd AS 組合市值,stock_value_twd AS 股票市值,bond_value_twd AS 債券市值,cash_value_twd AS 現金市值 FROM portfolio_snapshots WHERE user_id=? ORDER BY id", _conn, params=(portfolio_user_id,))
            if not snapshots.empty:
                peak = float(snapshots["組合市值"].cummax().max())
                latest_snapshot = float(snapshots["組合市值"].iloc[-1])
                rough_dd = (latest_snapshot / peak - 1) * 100 if peak > 0 else 0.0
                st.metric("快照最高值以來的估算回撤", f"{rough_dd:.2f}%", help="最近一筆手動快照相對歷史快照最高值的跌幅；沒有調整入金或出金，僅供粗略觀察。")
                if rough_dd <= -float(settings.get("drawdown_limit_pct", 15)):
                    st.error("估算回撤已達風險審查門檻：請檢查資產配置、價格／匯率資料與風險來源。")
                elif rough_dd <= -float(settings.get("drawdown_review_pct", 12)):
                    st.warning("估算回撤已達檢查門檻：暫緩增加主動風險，重新評估配置。")
                elif rough_dd <= -float(settings.get("drawdown_warning_pct", 8)):
                    st.warning("估算回撤已達警示門檻：檢查各資產損失來源與配置偏離。")
                st.dataframe(snapshots, use_container_width=True, hide_index=True)
                if st.button("🗑️ 清除所有淨值快照", key="portfolio_clear_snapshots"):
                    with sqlite3.connect(DB_FILE) as _conn:
                        _conn.execute("DELETE FROM portfolio_snapshots WHERE user_id=?", (portfolio_user_id,))
                        _conn.commit()
                    st.success("已清除快照紀錄。")
            else:
                st.info("尚無快照紀錄。要觀察回撤變化，請在不同日期更新價格後手動記錄。")

        st.markdown("---")
        st.caption("重要限制：目前價格與匯率由使用者維護；債券存續期間、信用評等、配息再投入、交易成本及外部現金流調整尚未納入。目標回撤 15% 是風險管理目標，不是保證。")
