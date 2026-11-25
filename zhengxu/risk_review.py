"""Views for the portfolio's allocation, cash reserve and snapshot risk."""

import streamlit as st

from .foundation import AccountScope, record_module_action
from .portfolio_risk import allocation_check, cash_budget, drawdown_status, snapshot_risk


DEMO_SETTINGS = {
    "target_stock_pct": 70, "target_bond_pct": 0, "target_cash_pct": 30,
    "rebalance_threshold_pct": 5, "minimum_cash_twd": 250,
    "drawdown_warning_pct": 8, "drawdown_review_pct": 12, "drawdown_limit_pct": 15,
}
DEMO_SNAPSHOTS = (
    {"snapshot_date": "2026-01-02T15:00:00+08:00", "total_value_twd": 12000},
    {"snapshot_date": "2026-01-05T15:00:00+08:00", "total_value_twd": 10000},
    {"snapshot_date": "2026-01-08T15:00:00+08:00", "total_value_twd": 10800},
)


def _twd(amount):
    return f"NT$ {amount:,.2f}" if amount is not None else "資料不足"


def render_risk_budget(assets, exposure, settings, snapshots, *, key, demo):
    budget_tab, history_tab = st.tabs(("資金預算與配置", "快照與回撤"))
    with budget_tab:
        budget = cash_budget(assets, settings)
        floor, remaining, shortfall = st.columns(3)
        floor.metric("最低保留現金", _twd(budget.floor))
        remaining.metric("保留現金後的可用預算", _twd(budget.available))
        shortfall.metric("距現金門檻缺口", _twd(budget.shortfall))
        st.caption("預算採帳本所列的現金持倉與已儲存的保留額。")
        if budget.available is None:
            st.info("請先確認配置中的最低保留現金，並核對現金持倉的數量、價格、幣別與匯率。")
        else:
            draft = st.number_input("試算投入金額（TWD）", min_value=0.0, value=None,
                                    step=100.0, format="%.2f", key=f"{key}_{'demo' if demo else 'ledger'}_risk_budget")
            if st.button("試算現金預算", key=f"{key}_risk_check", disabled=draft is None):
                decision = budget.covers(draft)
                if decision is not None:
                    notice = ("在現金預算內" if decision else "超出預算或低於保留額")
                    st.write(notice + "。")
                    account = st.session_state.get("auth_user")
                    if account:
                        scope = AccountScope.from_session(account)
                        label = "示範現金預算試算" if demo else "現金預算試算"
                        record_module_action(st.session_state, scope, label, notice)


        st.markdown("##### 配置偏離")
        rows = allocation_check(exposure, settings)
        if not rows:
            st.info("配置比對需要完整估值、有效的配置目標與偏離門檻。")
        else:
            st.dataframe([{
                "資產類別": name, "目前比重": f"{actual:.2f}%", "目標比重": f"{target:.2f}%",
                "偏離（百分點）": f"{gap:+.2f}", "狀態": "達評估門檻" if reached else "低於門檻",
            } for name, actual, target, gap, reached in rows], hide_index=True, width="stretch")
            if any(group.label not in {"股票", "債券", "現金"} for group in exposure.classes):
                st.caption("其他或未分類資產仍計入組合總市值，其比重可在「分布與集中度」查閱。")

    with history_tab:
        st.caption("使用帳本已記錄的資產快照。期間尚未校正入出金，也未重建兩次快照之間的變動。")
        try:
            risk = snapshot_risk(snapshots)
        except (KeyError, TypeError, ValueError):
            st.warning("部分快照無法核對，暫不估算回撤。可切到「歷史快照」查看紀錄。")
            return
        if risk is None:
            st.info("估算回撤至少需要兩筆有效快照，且曾有正的資產市值。")
            return
        latest, deepest = st.columns(2)
        latest.metric("最近快照回撤", f"{risk.latest_drawdown:.2f}%")
        deepest.metric("快照觀察到的最大回撤", f"{risk.deepest_drawdown:.2f}%")
        status = drawdown_status(risk.latest_drawdown, settings)
        st.caption(f"共 {risk.count} 筆快照。最近一筆 {risk.latest_at:%Y-%m-%d %H:%M}（台北），"
                   f"帳本市值 {_twd(risk.latest_value)}。")
        if status:
            st.write(status)
        else:
            st.info("尚無可用的已存回撤門檻，僅顯示快照估算。")
