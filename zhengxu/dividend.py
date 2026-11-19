"""Dividend quality, official dividend data and reinvestment research."""

from contextvars import ContextVar
from typing import Any
from typing import Mapping
from typing import Optional
import io
import numpy as np
import pandas as pd
import re
import time


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

_TWSE_DIVIDEND_CACHE = ContextVar("twse_dividend_cache", default=(0.0, ()))
_TWSE_TR_CACHE = ContextVar("twse_total_return_cache", default=(0.0, pd.DataFrame()))


def reset_dividend_source_cache():
    """Keep the source caches scoped to a single Streamlit execution."""
    _TWSE_DIVIDEND_CACHE.set((0.0, []))
    _TWSE_TR_CACHE.set((0.0, pd.DataFrame()))


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
    if not str(symbol).endswith(".TW") or str(symbol).endswith(".TWO"):
        return {}, None
    now = time.time()
    cached_at, records = _TWSE_DIVIDEND_CACHE.get()
    if now - cached_at > 3600 or not records:
        try:
            import requests
            response = requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap45_L",
                                    headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, list):
                records = payload
                _TWSE_DIVIDEND_CACHE.set((now, records))
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
    cached_at, cached_frame = _TWSE_TR_CACHE.get()
    if cached_frame.empty or time.time() - cached_at > 3600:
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.twse.com.tw/"}
            response = requests.get("https://www.twse.com.tw/indicesReport/MFI94U?response=open_data",
                                    headers=headers, timeout=25)
            response.raise_for_status()
            parsed = _parse_twse_tr_csv(response.content)
            _TWSE_TR_CACHE.set((time.time(), parsed))
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


def _free_cash_flow_from_statements(stock_obj):
    """Return a defensible FCF estimate, never annualizing incomplete quarters or missing capex."""
    operating_cashflow_labels = (
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    )
    capex_labels = ("Capital Expenditure", "Capital Expenditures")

    def read_row(frame, labels):
        if frame is None or frame.empty:
            return None
        for label in labels:
            if label in frame.index:
                values = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
                if not values.empty:
                    parsed_dates = pd.to_datetime(values.index, errors="coerce")
                    if len(parsed_dates) and not pd.isna(parsed_dates).any():
                        values.index = parsed_dates
                        values = values.sort_index(ascending=False)
                    return values
        return None

    try:
        quarterly = stock_obj.quarterly_cashflow
    except Exception:
        quarterly = pd.DataFrame()
    ocf_quarterly = read_row(quarterly, operating_cashflow_labels)
    capex_quarterly = read_row(quarterly, capex_labels)
    if ocf_quarterly is not None and capex_quarterly is not None:
        if len(ocf_quarterly) >= 4 and len(capex_quarterly) >= 4:
            ttm_ocf = float(ocf_quarterly.iloc[:4].sum())
            ttm_capex = abs(float(capex_quarterly.iloc[:4].sum()))
            return ttm_ocf - ttm_capex, "近四季財報估算"

    # If quarterly coverage is incomplete, use only the latest full-year
    # statement. Never sum four annual columns or multiply one quarter by four.
    try:
        annual = stock_obj.cashflow
    except Exception:
        annual = pd.DataFrame()
    ocf_annual = read_row(annual, operating_cashflow_labels)
    capex_annual = read_row(annual, capex_labels)
    if ocf_annual is not None and capex_annual is not None:
        return float(ocf_annual.iloc[0]) - abs(float(capex_annual.iloc[0])), "最新年度財報估算"
    return None, "財報現金流欄位不足，未估算"


def _peg_from_company_data(info):
    """Use reported PEG or estimate from annual earnings growth, never one-quarter growth."""
    info = info or {}
    reported_peg = _finite_number(info.get("pegRatio"))
    if reported_peg is not None and reported_peg > 0:
        return reported_peg, "yfinance PEG 欄位"

    pe = _finite_number(info.get("trailingPE"))
    if pe is None or pe <= 0:
        pe = _finite_number(info.get("forwardPE"))
    annual_growth = _finite_number(info.get("earningsGrowth"))
    if pe is None or pe <= 0 or annual_growth is None or annual_growth <= 0:
        return None, "PEG 或年度盈餘成長資料不足"

    # yfinance's earningsGrowth value is a decimal ratio (0.20 = 20%).
    growth_percent = annual_growth * 100.0
    if growth_percent <= 0:
        return None, "年度盈餘成長資料不足"
    return pe / growth_percent, "以本益比與年度盈餘成長估算"


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
    st.line_chart(history[["portfolio_value"]], use_container_width=True, height=360)
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
