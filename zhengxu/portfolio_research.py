"""Read-only position research for the existing personal portfolio sources."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re

import streamlit as st


DEMO_POSITIONS = (
    {"symbol": "AAPL", "name": "示範美股", "asset_type": "股票", "currency": "USD",
     "quantity": "2", "price": "150", "fx_to_twd": "32", "cost_basis_twd": "8000"},
    {"symbol": "CASH", "name": "示範台幣現金", "asset_type": "現金", "currency": "TWD",
     "quantity": "400", "price": "1", "fx_to_twd": "1"},
)


def _ledger_number(raw):
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _position_value_twd(row):
    quantity, price, fx = (_ledger_number(row.get(field)) for field in
                           ("quantity", "price", "fx_to_twd"))
    if (quantity is None or quantity < 0 or price is None or price <= 0
            or fx is None or fx <= 0 or not row.get("currency")):
        return None
    return quantity * price * fx


def summarize_positions(positions):
    values = [_position_value_twd(row) for row in positions]
    verified = [value for value in values if value is not None]
    subtotal = sum(verified, Decimal(0)) if verified else None
    missing = len(values) - len(verified)
    return values, subtotal, missing


@dataclass(frozen=True)
class PositionReconciliation:
    cash_value: Decimal | None
    cash_priced: int
    cash_count: int
    matched_cost: Decimal | None
    unrealized: Decimal | None
    noncash_matched: int
    noncash_count: int
    costs: tuple[Decimal | None, ...]
    gains: tuple[Decimal | None, ...]


def reconcile_positions(positions, values):
    cash_values = []
    matched_costs = []
    costs = []
    gains = []
    cash_count = 0
    for row, value in zip(positions, values, strict=True):
        if row.get("asset_type") == "現金":
            cash_count += 1
            if value is not None:
                cash_values.append(value)
            costs.append(None)
            gains.append(None)
            continue

        cost = _ledger_number(row.get("cost_basis_twd"))
        costs.append(cost if cost is not None and cost > 0 else None)
        if value is None or value <= 0 or costs[-1] is None:
            gains.append(None)
        else:
            matched_costs.append(costs[-1])
            gains.append(value - costs[-1])

    return PositionReconciliation(
        cash_value=sum(cash_values, Decimal(0)) if cash_values else None,
        cash_priced=len(cash_values),
        cash_count=cash_count,
        matched_cost=sum(matched_costs, Decimal(0)) if matched_costs else None,
        unrealized=sum((gain for gain in gains if gain is not None), Decimal(0)) if matched_costs else None,
        noncash_matched=len(matched_costs),
        noncash_count=len(costs) - cash_count,
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


def _amount(value):
    return f"NT$ {value:,.2f}" if value is not None else "資料不足"


def _render_reconciliation(reconciliation):
    cash_col, cost_col, gain_col = st.columns(3)
    cash_label = ("可估值現金小計（TWD）"
                  if reconciliation.cash_priced < reconciliation.cash_count else "帳本現金估值（TWD）")
    cash_col.metric(
        cash_label, _amount(reconciliation.cash_value) if reconciliation.cash_count else "未記錄現金",
        help="依標記為現金的持倉估值。成交與股息明細不會自動加減這裡的金額。",
    )
    incomplete = reconciliation.noncash_matched < reconciliation.noncash_count
    cost_label = "可核對成本小計（TWD）" if incomplete else "已核對成本（TWD）"
    gain_label = "可核對未實現損益小計（TWD）" if incomplete else "未實現損益估算（TWD）"
    applicability = "無非現金持倉"
    cost_col.metric(cost_label, _amount(reconciliation.matched_cost) if reconciliation.noncash_count
                    else applicability)
    gain_col.metric(gain_label, _signed_amount(reconciliation.unrealized) if reconciliation.noncash_count
                    else applicability)
    st.caption(
        f"現金可估值 {reconciliation.cash_priced}／{reconciliation.cash_count} 筆。"
        f"非現金可核對 {reconciliation.noncash_matched}／{reconciliation.noncash_count} 筆。"
        "未實現損益以同一筆非現金持倉的估值減去帳本台幣成本。成本為零或缺資料的持倉不列入估算。"
    )


def _signed_amount(value):
    return f"NT$ {value:+,.2f}" if value is not None else "資料不足"


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

        values, subtotal, missing = summarize_positions(rows)
        columns = st.columns(2)
        columns[0].metric("可估值持倉", f"{len(rows) - missing}／{len(rows)} 筆")
        amount_label = "可估值部分小計（TWD）" if missing else "持倉估算總市值（TWD）"
        columns[1].metric(amount_label, _amount(subtotal))
        if missing:
            st.warning("部分持倉缺少有效數量、價格、幣別或匯率。占比需等全部持倉可估值後才顯示。")

        reconciliation = reconcile_positions(rows, values)
        _render_reconciliation(reconciliation)
        table = []
        for row, value, cost, gain in zip(rows, values, reconciliation.costs, reconciliation.gains, strict=True):
            weight = (f"{value / subtotal:.2%}" if not missing and subtotal else "—")
            table.append({
                "代碼": row["symbol"], "名稱": row["name"], "資產類型": row["asset_type"],
                "幣別": row["currency"], "估算市值": _amount(value), "占組合比重": weight,
                "帳本台幣成本": "—" if row["asset_type"] == "現金" else _amount(cost),
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

