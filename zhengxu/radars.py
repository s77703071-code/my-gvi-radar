"""Existing GVI, momentum and QARP calculations and screening rules."""

import pandas as pd


def calculate_gvi(price, book_value, roe):
    return (book_value / price) * ((1 + roe) ** 5)


def calculate_momentum(price, closes):
    if len(closes) < 50:
        return None
    high = float(closes.tail(252).max())
    return (price / high) * 100 if pd.notna(high) and high > 0 else None


def calculate_fcf_yield(free_cash_flow, market_cap):
    if free_cash_flow is None or market_cap is None or not market_cap > 0:
        return None
    return free_cash_flow / market_cap * 100.0


def score_gvi(value):
    if value is None:
        return None, "基本面資料不足，未評分。"
    if value >= 0.20:
        return (90 if value >= 0.35 else 75), (
            f"成長價值分數為 {value:.4f}，達到目前設定的價值研究門檻；尚未證明能帶來較佳績效。"
        )
    return 40, f"成長價值分數為 {value:.4f}，未達目前設定的價值研究門檻。"


def score_momentum(value, window):
    if value is None:
        return None, "有效歷史收盤資料少於 50 根，動能值缺失且不納入評分。"
    if value >= 80:
        return (95 if value >= 95 else 85), (
            f"目前價格約在{window}可用最高收盤價的 {value:.1f}%，達到趨勢研究門檻。"
        )
    return 50, f"目前價格約在{window}可用最高收盤價的 {value:.1f}%，未達趨勢研究門檻。"


def score_qarp(fcf_yield, peg):
    if fcf_yield is None or peg is None:
        return None, "自由現金流收益率或 PEG 資料不足，未評分。"
    detail = f"現金流收益參考為 {fcf_yield:.2f}%，成長估值參考為 {peg:.2f}"
    if fcf_yield >= 2.5 and peg <= 1.5:
        return 90, f"{detail}，兩項均達目前研究門檻。"
    return 45, f"{detail}；至少一項未達目前研究門檻。"


def filter_radar_matches(rows, gvi, momentum, qarp):
    if gvi:
        rows = rows[rows["gvi_raw"].notnull() & (rows["gvi_raw"] >= 0.20)]
    if momentum:
        rows = rows[rows["mo_raw"].notnull() & (rows["mo_raw"] >= 80.0)]
    if qarp:
        rows = rows[
            rows["fcf_raw"].notnull() & (rows["fcf_raw"] >= 2.5)
            & rows["本益成長比(PEG)"].notnull() & (rows["本益成長比(PEG)"] <= 1.5)
        ]
    return rows


def rank_radar_matches(rows, gvi, momentum, qarp):
    ranked = rows.assign(
        GVI_Rank=rows["gvi_raw"].rank(pct=True),
        MO_Rank=rows["mo_raw"].rank(pct=True),
        FCF_Rank=rows["fcf_raw"].rank(pct=True),
    )
    ranked["綜合分數"] = 0.0
    active = 0
    for selected, column in ((gvi, "GVI_Rank"), (momentum, "MO_Rank"), (qarp, "FCF_Rank")):
        if selected:
            ranked["綜合分數"] += ranked[column]
            active += 1
    if active:
        ranked["策略評分_num"] = ((ranked["綜合分數"] / active) * 100).round(1)
    return ranked
