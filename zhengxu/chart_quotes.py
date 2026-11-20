"""Validate chart quotes and preserve the last usable response per view."""

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def _current_chart_frame(ticker, config):
    frame = yf.download(ticker, period=config["p"], interval=config["i"], progress=False)
    if not isinstance(frame, pd.DataFrame) or not isinstance(frame.index, pd.DatetimeIndex):
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.droplevel(list(range(1, frame.columns.nlevels)), axis="columns")
    prices = ["Open", "High", "Low", "Close"]
    if not frame.columns.is_unique or not set(prices + ["Volume"]).issubset(frame.columns):
        return None
    frame = frame.copy()
    for name in prices + ["Volume"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    quote = frame[prices]
    valid = frame.index.notna() & np.isfinite(quote).all(axis=1) & (quote > 0).all(axis=1)
    valid &= frame["High"].ge(quote.max(axis=1)) & frame["Low"].le(quote.min(axis=1))
    frame = frame.loc[valid].sort_index(kind="stable")
    return frame.loc[~frame.index.duplicated(keep="last")] if not frame.empty else None


def load_chart_quote(ticker, config):
    key = (ticker, config["p"], config["i"])
    cache = st.session_state.setdefault("public_chart_quote_cache", {})
    previous = cache.get(key)
    try:
        frame = _current_chart_frame(ticker, config)
    except Exception:
        frame = None
    if frame is None or (previous is not None and frame.index[-1].value < previous.index[-1].value):
        return previous.copy() if previous is not None else None, True
    cache.pop(key, None)
    cache[key] = frame
    if len(cache) > 12:
        cache.pop(next(iter(cache)))
    return frame.copy(), False


def chart_time_label(timestamp, interval, ticker):
    if not interval.endswith("m"):
        return f"K 棒日期 {timestamp:%Y-%m-%d}"
    if timestamp.tzinfo is not None and ticker.endswith((".TW", ".TWO")):
        timestamp = timestamp.tz_convert(ZoneInfo("Asia/Taipei"))
    clock = f" {timestamp:%Z}" if timestamp.tzinfo is not None else ""
    return f"K 棒時間 {timestamp:%Y-%m-%d %H:%M}{clock}"
