"""Read-only position research for the existing personal portfolio sources."""

from dataclasses import dataclass
from decimal import Decimal
import re

import streamlit as st

from .asset_exposure import build_asset_exposure
from .asset_state import build_asset_state, ledger_number


DEMO_POSITIONS = (
    {"symbol": "AAPL", "name": "示範美股", "asset_type": "股票", "currency": "USD",
     "quantity": "2", "price": "150", "fx_to_twd": "32", "cost_basis_twd": "8000"},
    {"symbol": "CASH", "name": "示範台幣現金", "asset_type": "現金", "currency": "TWD",
     "quantity": "400", "price": "1", "fx_to_twd": "1"},
)


@dataclass(frozen=True)
class PositionReconciliation:
    matched_cost: Decimal | None
    unrealized: Decimal | None
    noncash_matched: int
    noncash_count: int
    costs: tuple[Decimal | None, ...]
    gains: tuple[Decimal | None, ...]


def reconcile_positions(positions, values):
    matched_costs = []
    costs = []
    gains = []
    noncash_count = 0
    for row, value in zip(positions, values, strict=True):
        if row.get("asset_type") == "現金":
            costs.append(None)
            gains.append(None)
            continue

        noncash_count += 1
        cost = ledger_number(row.get("cost_basis_twd"))
        costs.append(cost if cost is not None and cost > 0 else None)
        if value is None or value <= 0 or costs[-1] is None:
            gains.append(None)
        else:
            matched_costs.append(costs[-1])
            gains.append(value - costs[-1])

    return PositionReconciliation(
        matched_cost=sum(matched_costs, Decimal(0)) if matched_costs else None,
        unrealized=sum((gain for gain in gains if gain is not None), Decimal(0)) if matched_costs else None,
        noncash_matched=len(matched_costs),
        noncash_count=noncash_count,
        costs=tuple(costs),
        gains=tuple(gains),
    )


def _chart_symbol(row):
    if row.get("asset_type") != "股票":
        return None
    symbol = str(row.get("symbol", "")).strip().upper()
    currency = row.get("currency")
    if currency == "TWD" and re.fullmatch(r"[0-9]{4,6}[A-Z]?\.(?:TW|TWO)", symbol):
        return symbol
    if currency == "USD" and re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", symbol):
        return symbol
    return None


def format_twd(value):
    return f"NT$ {value:,.2f}" if value is not None else "資料不足"


def valuation_total_label(total):
    return "可估值部分小計（TWD）" if total.incomplete else "持倉估算總市值（TWD）"


def render_cash_valuation(column, cash, *, share=None):
    label = "可估值現金小計（TWD）" if cash.incomplete else "帳本現金估值（TWD）"
    column.metric(
        label, format_twd(cash.subtotal) if cash.count else "未記錄現金",
        delta=f"現金占比 {share:.2%}" if share is not None else None, delta_color="off",
        help="依標記為現金的持倉估值。成交與股息明細不會自動加減這裡的金額。",
    )


def _render_reconciliation(reconciliation, cash):
    cash_col, cost_col, gain_col = st.columns(3)
    render_cash_valuation(cash_col, cash)
    incomplete = reconciliation.noncash_matched < reconciliation.noncash_count
    cost_label = "可核對成本小計（TWD）" if incomplete else "已核對成本（TWD）"
    gain_label = "可核對未實現損益小計（TWD）" if incomplete else "未實現損益估算（TWD）"
    applicability = "無非現金持倉"
    cost_col.metric(cost_label, format_twd(reconciliation.matched_cost) if reconciliation.noncash_count
                    else applicability)
    gain_col.metric(gain_label, _signed_amount(reconciliation.unrealized) if reconciliation.noncash_count
                    else applicability)
    st.caption(
        f"現金可估值 {cash.priced}／{cash.count} 筆。"
        f"非現金可核對 {reconciliation.noncash_matched}／{reconciliation.noncash_count} 筆。"
        "未實現損益以同一筆非現金持倉的估值減去帳本台幣成本。成本為零或缺資料的持倉不列入估算。"
    )


def _signed_amount(value):
    return f"NT$ {value:+,.2f}" if value is not None else "資料不足"


def _exposure_share(value):
    return f"{value:.2%}" if value is not None else "—"


def _exposure_table(column, groups, exposure, *, label, incomplete, allow_weights=True):
    amount_label = "可估值小計（TWD）" if incomplete else "估算市值（TWD）"
    column.dataframe([{
        label: item.label,
        amount_label: format_twd(item.valuation.subtotal),
        "可估值筆數": f"{item.valuation.priced}／{item.valuation.count}",
        "占總市值比重": _exposure_share(exposure.share(item.valuation)) if allow_weights else "—",
    } for item in groups], hide_index=True, width="stretch")


def _render_asset_exposure(rows, assets):
    exposure = build_asset_exposure(rows, assets)
    classes, positions = st.columns(2)
    classes.markdown("##### 資產類別分布")
    _exposure_table(classes, exposure.classes, exposure, label="資產類別", incomplete=assets.total.incomplete)
    positions.markdown("##### 非現金標的集中度")
    if exposure.symbols:
        largest = exposure.symbols[0]
        share = exposure.share(largest.valuation) if not exposure.unidentified else None
        positions.metric("最大非現金標的占比", _exposure_share(share),
                         help="同代碼、同原幣的非現金持倉合併，除以含現金的組合總市值。")
        if share is not None:
            positions.caption(largest.label)
        _exposure_table(positions, exposure.symbols, exposure, label="標的及原幣",
                        incomplete=assets.total.incomplete, allow_weights=not exposure.unidentified)
    elif not exposure.unidentified:
        positions.info("目前沒有非現金持倉。")
    if exposure.unidentified:
        positions.warning(f"有 {exposure.unidentified} 筆非現金持倉缺少代碼或原幣，暫不顯示標的占比。")
    if assets.total.incomplete:
        st.info("部分持倉缺少估值資料，目前顯示可估值小計，待資料補齊後顯示完整占比。")
    elif exposure.denominator is None:
        st.info("總市值為零，無法計算占比。")


def render_portfolio_research(positions, *, scope, source, open_chart):
    key = f"portfolio_research_{scope}"
    with st.expander("持倉研究（唯讀）", expanded=True):
        use_demo = st.toggle("使用示範資料", key=f"{key}_demo")
        rows = DEMO_POSITIONS if use_demo else positions
        if use_demo:
            st.info("目前使用固定示範資料。示範金額與行情無關，也不會寫入帳本。")
        else:
            st.caption(f"資料來源為{source}。估值使用帳本已儲存的數量、價格及匯率。")
        if not rows:
            st.info("目前沒有持倉。可開啟上方示範資料試用。")
            return

        assets = build_asset_state(rows)
        valuation_tab, exposure_tab = st.tabs(("估值與成本", "分布與集中度"))
        with valuation_tab:
            _render_position_valuation(rows, assets, key=key, use_demo=use_demo, open_chart=open_chart)
        with exposure_tab:
            _render_asset_exposure(rows, assets)


def _render_position_valuation(rows, assets, *, key, use_demo, open_chart):
    total = assets.total
    columns = st.columns(2)
    columns[0].metric("可估值持倉", f"{total.priced}／{total.count} 筆")
    columns[1].metric(valuation_total_label(total), format_twd(total.subtotal))
    if total.incomplete:
        st.warning("部分持倉缺少有效數量、價格、幣別或匯率。占比需等全部持倉可估值後才顯示。")

    reconciliation = reconcile_positions(rows, assets.positions)
    _render_reconciliation(reconciliation, assets.cash)
    table = []
    for row, value, cost, gain in zip(rows, assets.positions, reconciliation.costs, reconciliation.gains, strict=True):
        weight = (f"{value / total.subtotal:.2%}" if not total.incomplete and total.subtotal else "—")
        table.append({
            "代碼": row["symbol"], "名稱": row["name"], "資產類型": row["asset_type"],
            "幣別": row["currency"], "估算市值": format_twd(value), "占組合比重": weight,
            "帳本台幣成本": "—" if row["asset_type"] == "現金" else format_twd(cost),
            "未實現損益估算": "—" if row["asset_type"] == "現金" else _signed_amount(gain),
        })
    st.dataframe(table, width="stretch", hide_index=True)

    choices = [index for index, row in enumerate(rows) if row["asset_type"] == "股票"]
    if not choices:
        return
    selected = st.selectbox("選擇看圖標的", choices, index=None,
                            format_func=lambda index: f"{rows[index]['symbol']}｜{rows[index]['name']}",
                            key=f"{key}_{'demo' if use_demo else 'ledger'}_symbol")
    if selected is None:
        return
    symbol = _chart_symbol(rows[selected])
    if symbol:
        st.button("開啟個股 K 線", on_click=open_chart, args=(symbol,), key=f"{key}_open_chart")
    else:
        st.caption("這筆持倉代碼未標明可辨識的行情市場。請從個股 K 線入口輸入完整代碼。")
