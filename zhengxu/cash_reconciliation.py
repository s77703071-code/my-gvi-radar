"""Summarize recorded cash movements without changing portfolio balances."""

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, localcontext
from zoneinfo import ZoneInfo

import streamlit as st

from .foundation import failure_message


DEMO_TRADES = (
    {"operation_id": "demo-buy", "executed_on": "2026-01-05", "side": "buy", "symbol": "AAPL",
     "currency": "USD", "quantity": "2", "unit_price": "150", "fee": "1", "tax": "0"},
    {"operation_id": "demo-sell", "executed_on": "2026-01-12", "side": "sell", "symbol": "AAPL",
     "currency": "USD", "quantity": "1", "unit_price": "170", "fee": "1", "tax": "0"},
    {"operation_id": "demo-tw-buy", "executed_on": "2026-01-08", "side": "buy", "symbol": "2330.TW",
     "currency": "TWD", "quantity": "1", "unit_price": "1000", "fee": None, "tax": "0"},
)
DEMO_INCOME = (
    {"operation_id": "demo-dividend", "booked_on": "2026-01-15", "income_type": "dividend",
     "currency": "USD", "net_amount": "4", "reference": "AAPL"},
    {"operation_id": "demo-interest", "booked_on": "2026-01-20", "income_type": "interest",
     "currency": "TWD", "net_amount": "10", "reference": ""},
)

SUMMARY_FIELDS = ("買入成交額", "賣出成交額", "股息實收", "利息實收")


def _check_dates(rows, field):
    seen = set()
    for row in rows:
        identity = row["operation_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Invalid cash journal entry")
        seen.add(identity)
        yield row, date.fromisoformat(row[field])


def summarize_cash_activity(trades, income, *, start=None, end=None):
    totals = defaultdict(lambda: dict.fromkeys(SUMMARY_FIELDS, Decimal(0)))
    details = []
    incomplete = defaultdict(int)
    net_by_currency = defaultdict(Decimal)
    with localcontext() as context:
        # The stored decimal strings can hold 24 integer and 12 fractional digits.
        # Multiplying two such values can require 72 digits before aggregation.
        context.prec = 96
        for rows, day_field in ((trades, "executed_on"), (income, "booked_on")):
            for row, day in _check_dates(rows, day_field):
                if (start is not None and day < start) or (end is not None and day > end):
                    continue
                currency = row["currency"]
                total = totals[currency]
                if day_field == "executed_on":
                    gross = Decimal(row["quantity"]) * Decimal(row["unit_price"])
                    bucket = {"buy": "買入成交額", "sell": "賣出成交額"}[row["side"]]
                    ref = row["symbol"]
                    fee = Decimal(row["fee"]) if row["fee"] is not None else None
                    tax = Decimal(row["tax"]) if row["tax"] is not None else None
                    if fee is None or tax is None:
                        incomplete[currency] += 1
                        net = None
                    else:
                        net = (-gross if row["side"] == "buy" else gross) - fee - tax
                else:
                    gross = net = Decimal(row["net_amount"])
                    bucket = {"dividend": "股息實收", "interest": "利息實收"}[row["income_type"]]
                    ref, fee, tax = row["reference"], None, None
                total[bucket] += gross
                if net is not None:
                    net_by_currency[currency] += net
                details.append({
                    "日期": day.isoformat(), "項目": bucket, "標的或帳戶": ref or "",
                    "幣別": currency, "金額": gross, "手續費": fee, "交易稅": tax,
                    "紀錄淨流量": net,
                })

        summary = []
        for currency, total in sorted(totals.items()):
            net = None if incomplete[currency] else net_by_currency[currency]
            summary.append({"幣別": currency, **total, "費用待補筆數": incomplete[currency],
                            "紀錄淨流量": net})
    return summary, sorted(details, key=lambda row: row["日期"], reverse=True)


def _display_amount(value):
    if value is None:
        return "—"
    if isinstance(value, Decimal):
        text = format(value, ",f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    return value


def _display_rows(rows):
    return [{key: _display_amount(value) for key, value in row.items()} for row in rows]


def render_cash_activity(load_entries, *, scope):
    st.markdown("#### 成交與收入對帳（唯讀）")
    demo = st.toggle("使用對帳示範資料", key=f"{scope}_reconciliation_demo")
    period_key = f"{scope}_reconciliation_period"
    period = st.date_input("對帳期間", value=(), key=period_key,
                           min_value=date(1900, 1, 1), max_value=datetime.now(ZoneInfo("Asia/Taipei")).date(),
                           help="選擇起訖日，含當日。留空顯示所有已登錄紀錄。")
    if period:
        st.button("清除期間", key=f"{scope}_reconciliation_clear",
                  on_click=st.session_state.__setitem__, args=(period_key, ()))
    if len(period) == 1:
        st.info("請再選擇結束日期。")
        return
    start, end = period if period else (None, None)
    if demo:
        st.info("目前顯示固定示範資料。操作不會寫入帳本。")

    try:
        trades, income = (DEMO_TRADES, DEMO_INCOME) if demo else load_entries()
        summary, details = summarize_cash_activity(trades, income, start=start, end=end)
    except (KeyError, TypeError, ValueError, ArithmeticError, RuntimeError) as error:
        st.error(failure_message(error, "暫時無法核對完整的成交與收入紀錄。請稍後再試。"))
        return

    st.caption("各幣別分開加總。紀錄淨流量為賣出與實收收入，扣除買入、手續費及交易稅。"
               "成交與收入對帳只涵蓋已登錄紀錄，帳本現金餘額仍以持倉頁為準。")
    if not summary:
        st.info("這個期間沒有已登錄的成交或收入。")
        return
    st.dataframe(_display_rows(summary), hide_index=True, width="stretch")
    if any(row["費用待補筆數"] for row in summary):
        st.warning("有成交未填手續費或交易稅。該幣別先顯示成交額與收入，待費用補齊才顯示紀錄淨流量。")
    with st.expander("查看逐筆紀錄", expanded=False):
        st.dataframe(_display_rows(details), hide_index=True, width="stretch")
