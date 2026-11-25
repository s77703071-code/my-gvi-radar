"""Legacy portfolio settings, holdings and allocation workflows."""

from contextlib import closing

import numpy as np
import pandas as pd
import sqlite3
import streamlit as st

from .foundation import feature_enabled
from .portfolio import (
    _is_portfolio_pilot,
    _render_portfolio_export_link,
    _render_portfolio_pilot,
)
from .portfolio_research import render_portfolio_research
from .storage import (
    DB_FILE,
    _export_portfolio,
)


def render_legacy_portfolio(_render_stock_actions, auth_user, *, open_chart):
    st.subheader("🧭 投資組合理財")
    st.caption("本模組只管理投資組合，不管理個人收入／支出，也不會自動下單。初始配置僅為可修改的示範值，不代表最佳配置或回撤保證。")
    if _is_portfolio_pilot(auth_user):
        portfolio_enabled = any(feature_enabled(flag) for flag in (
            "GVI_PORTFOLIO_PILOT_POSITION_ADD", "GVI_PORTFOLIO_PILOT_POSITION_EDIT",
            "GVI_PORTFOLIO_PILOT_POSITION_DELETE", "GVI_PORTFOLIO_PILOT_SNAPSHOT",
            "GVI_PORTFOLIO_PILOT_INCOME", "GVI_PORTFOLIO_PILOT_TRADE"))
        cloud_label = ("雲端投資組合試行" if portfolio_enabled
                       else "雲端配置試行" if feature_enabled("GVI_PORTFOLIO_PILOT_SETTINGS_WRITE")
                       else "雲端唯讀試行")
        source = st.radio(
            "投資組合來源", ("原有投資組合", cloud_label), horizontal=True,
            index=int(st.session_state.get("portfolio_workbench_cloud", False)),
        )
        if source == cloud_label:
            _render_portfolio_pilot(auth_user, open_chart=open_chart)
            return

    portfolio_user_id = int(auth_user["id"])
    holdings_view_tab, edit_tab, snapshots_tab, config_tab = st.tabs(("持倉", "新增與調整", "歷史快照", "配置"))

    with holdings_view_tab:
        if st.button("準備匯出我的投資組合", key="portfolio_prepare_export"):
            try:
                filename, content = _export_portfolio(portfolio_user_id)
            except (sqlite3.Error, ValueError):
                st.error("目前無法匯出完整資料。請稍後再試，若仍失敗請聯絡管理員。")
            else:
                _render_portfolio_export_link(filename, content)

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

    def _portfolio_snapshots():
        try:
            with closing(sqlite3.connect(DB_FILE)) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(row) for row in conn.execute(
                    "SELECT snapshot_date, total_value_twd FROM portfolio_snapshots WHERE user_id=? ORDER BY id",
                    (portfolio_user_id,),
                ).fetchall()]
        except (sqlite3.Error, TypeError, ValueError):
            return None

    settings = _portfolio_load_settings()
    with config_tab:
        st.markdown("#### 配置與風險規則")
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

    with edit_tab:
        st.markdown("#### 新增持倉")
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
    research_rows = positions.rename(columns={
        "代碼": "symbol", "名稱": "name", "資產類型": "asset_type", "幣別": "currency",
        "數量": "quantity", "最新價格": "price", "換算台幣匯率": "fx_to_twd",
        "台幣成本": "cost_basis_twd",
    }).to_dict("records")
    with holdings_view_tab:
        render_portfolio_research(research_rows, scope=f"local_{portfolio_user_id}",
                                  source="原有投資組合", open_chart=open_chart,
                                  settings=settings, snapshots=_portfolio_snapshots())
        st.markdown("#### 持倉明細與配置計算")
    if positions.empty:
        with holdings_view_tab:
            st.info("目前尚無持倉。請切到「新增與調整」新增股票、債券 ETF 或現金。")
        with snapshots_tab:
            st.info("目前沒有持倉可記錄新快照。")
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
        with holdings_view_tab:
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

        with edit_tab:
            st.markdown("#### 更新或刪除持倉")
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

        with snapshots_tab:
            st.markdown("#### 組合淨值紀錄與回撤參考")
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

    with config_tab:
        st.markdown("---")
        st.caption("重要限制：目前價格與匯率由使用者維護；債券存續期間、信用評等、配息再投入、交易成本及外部現金流調整尚未納入。目標回撤 15% 是風險管理目標，不是保證。")
