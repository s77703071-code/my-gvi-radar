"""Read-only official regular-session closes for the Taiwan pilot symbols."""

from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

import requests
import streamlit as st

from .storage import BASE_DIR

TAIPEI = ZoneInfo("Asia/Taipei")
OFFICIAL_CLOSE_CACHE = Path(BASE_DIR) / "official_close_cache.db"
TWSE_CLOSE = "https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
TPEX_CLOSE = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
PILOT_SOURCES = {"2330.TW": "TWSE 證交所", "3293.TWO": "TPEx 櫃買中心"}


@dataclass(frozen=True)
class OfficialClose:
    trading_day: date
    price: Decimal
    retrieved_at: datetime


def _roc_date(value):
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]{3}(?:/[0-9]{2}/[0-9]{2}|[0-9]{4})", text):
        raise ValueError("盤後資料日期格式不符")
    text = text.replace("/", "")
    return date(int(text[:3]) + 1911, int(text[3:5]), int(text[5:]))


def _twse_rows(symbol, reference_day):
    response = requests.get(
        TWSE_CLOSE, params={"date": reference_day.strftime("%Y%m%d"),
                            "stockNo": symbol, "response": "json"}, timeout=8,
    )
    response.raise_for_status()
    report = response.json()
    if not isinstance(report, dict):
        raise ValueError("證交所盤後格式不符")
    if report.get("stat") in ("很抱歉，沒有符合條件的資料!", "很抱歉，沒有符合條件的資料！"):
        return []
    fields, rows = report.get("fields"), report.get("data")
    if (report.get("stat") != "OK" or not isinstance(fields, list)
            or fields.count("日期") != 1 or fields.count("收盤價") != 1
            or not isinstance(rows, list)
            or not re.search(rf"(?<![0-9]){symbol}(?![0-9])", str(report.get("title", "")))):
        raise ValueError("證交所盤後格式或股票代號不符")
    day_column, close_column = fields.index("日期"), fields.index("收盤價")
    result = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(fields):
            raise ValueError("證交所盤後欄位數不符")
        trading_day = _roc_date(row[day_column])
        if (trading_day.year, trading_day.month) != (reference_day.year, reference_day.month):
            raise ValueError("證交所盤後資料月份不符")
        result.append((trading_day, row[close_column]))
    return result


def fetch_official_close(ticker, *, now=None):
    if ticker not in PILOT_SOURCES:
        raise ValueError("股票不在本次盤後試行範圍")
    observed = now.astimezone(TAIPEI) if now is not None else datetime.now(TAIPEI)
    last_eligible_day = observed.date()
    if observed.time() < time(13, 33):
        last_eligible_day -= timedelta(days=1)

    code, market = ticker.split(".")
    if market == "TW":
        rows = _twse_rows(code, last_eligible_day)
        if not rows:
            previous_month = last_eligible_day.replace(day=1) - timedelta(days=1)
            rows = _twse_rows(code, previous_month)
    else:
        response = requests.get(TPEX_CLOSE, timeout=8)
        response.raise_for_status()
        table = response.json()
        if not isinstance(table, list):
            raise ValueError("櫃買中心盤後格式不符")
        rows = [
            (_roc_date(row.get("Date")), row.get("Close"))
            for row in table
            if isinstance(row, dict) and str(row.get("SecuritiesCompanyCode", "")).strip() == code
        ]

    if any(day > observed.date() for day, _ in rows):
        raise ValueError("盤後資料出現未來日期")
    eligible = [(day, value) for day, value in rows if day <= last_eligible_day]
    if not eligible:
        raise ValueError("暫無已確認的盤後資料")
    trading_day = max(day for day, _ in eligible)
    values = [value for day, value in eligible if day == trading_day]
    if len(values) != 1:
        raise ValueError("盤後資料同日重複")
    price_text = str(values[0]).strip()
    if not re.fullmatch(r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", price_text):
        raise ValueError("最新資料無有效收盤價")
    price = Decimal(price_text.replace(",", ""))
    if price <= 0:
        raise ValueError("最新資料無有效收盤價")
    return OfficialClose(trading_day, price, datetime.now(TAIPEI) if now is None else observed)


def _decode_cached_close(row):
    if row is None:
        return None
    trading_day, price, retrieved_at = row
    close = OfficialClose(date.fromisoformat(trading_day), Decimal(price),
                          datetime.fromisoformat(retrieved_at))
    if (not close.price.is_finite() or close.price <= 0 or close.retrieved_at.tzinfo is None
            or close.trading_day > close.retrieved_at.astimezone(TAIPEI).date()
            or close.trading_day > datetime.now(TAIPEI).date()):
        raise ValueError("Invalid saved official close")
    return close


def _read_cached_close(ticker):
    if not OFFICIAL_CLOSE_CACHE.exists():
        return None
    database_uri = OFFICIAL_CLOSE_CACHE.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True, timeout=2)) as db:
        row = db.execute(
            "SELECT trading_day, price, retrieved_at FROM official_closes WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    return _decode_cached_close(row)


def _save_cached_close(ticker, close):
    with closing(sqlite3.connect(OFFICIAL_CLOSE_CACHE, timeout=2)) as db, db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS official_closes (
                ticker TEXT PRIMARY KEY, trading_day TEXT NOT NULL,
                price TEXT NOT NULL, retrieved_at TEXT NOT NULL
            )
        """)
        db.execute("""
            INSERT INTO official_closes (ticker, trading_day, price, retrieved_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                trading_day = excluded.trading_day, price = excluded.price,
                retrieved_at = excluded.retrieved_at
            WHERE (excluded.trading_day, excluded.retrieved_at) >=
                  (official_closes.trading_day, official_closes.retrieved_at)
        """, (ticker, close.trading_day.isoformat(), str(close.price),
              close.retrieved_at.astimezone(TAIPEI).isoformat(timespec="microseconds")))
        row = db.execute(
            "SELECT trading_day, price, retrieved_at FROM official_closes WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        return _decode_cached_close(row)


def _refresh_official_close(ticker, previous):
    cache_warning = None
    try:
        saved = _read_cached_close(ticker)
        previous = max(filter(None, (previous, saved)),
                       key=lambda close: (close.trading_day, close.retrieved_at), default=None)
    except (OSError, sqlite3.Error, ValueError, InvalidOperation):
        cache_warning = "本機收盤快取暫時無法讀取。"

    try:
        close = fetch_official_close(ticker)
    except requests.RequestException:
        return previous, "官方來源暫時無法取得", cache_warning
    except ValueError as error:
        return previous, str(error), cache_warning
    if previous is not None and close.trading_day < previous.trading_day:
        return previous, "來源回傳較早的交易日", cache_warning

    try:
        saved = _save_cached_close(ticker, close)
    except (OSError, sqlite3.Error, ValueError, InvalidOperation):
        return close, None, "收盤價暫時無法寫入本機快取，重啟後可能需重新取得。"
    if saved.trading_day > close.trading_day:
        return saved, "來源回傳較早的交易日", None
    return saved, None, None


def render_official_close(ticker):
    if ticker not in PILOT_SOURCES:
        return
    key = f"official_close_pilot_{ticker}"
    st.markdown("#### 台股正式收盤價｜試行")
    refresh = st.button("手動刷新正式收盤價", key=f"{key}_refresh")
    if key not in st.session_state or refresh:
        previous = st.session_state[key][0] if key in st.session_state else None
        st.session_state[key] = _refresh_official_close(ticker, previous)

    close, error, cache_warning = st.session_state[key]
    if cache_warning:
        st.warning(cache_warning)
    source = PILOT_SOURCES[ticker]
    if close is None:
        st.info(f"正式收盤價資料暫缺 · {source} · {error or '等待發布'}")
        return
    st.metric(f"{ticker} 正式收盤價", f"NT$ {close.price:,.2f}")
    st.caption(f"{source} · 交易日 {close.trading_day:%Y-%m-%d} · "
               f"取得時間 {close.retrieved_at:%Y-%m-%d %H:%M:%S}（台北）")
    if error:
        st.warning(f"{error}。顯示最後有效收盤價。")
    elif close.trading_day < datetime.now(TAIPEI).date():
        st.caption("顯示已取得的最近盤後資料，請以交易日辨識。")
