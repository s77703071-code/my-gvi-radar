"""Investment workbench, navigation callbacks and session cleanup."""

from base64 import b64encode
from contextlib import closing
from pathlib import Path
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import requests
import sqlite3
import streamlit as st

from .asset_state import build_asset_state
from .chart_view import render_chart_workspace
from .foundation import ACTIVITY_KEY, AccountScope, failure_message, module_events, feature_enabled
from .market_data import (
    WORKBENCH_INDEXES,
    _market_index_history,
    _render_market_overview,
    get_trader_watchlist_symbols,
)
from .portfolio import (
    PILOT_INCOME_CORRECTION_STATE,
    PILOT_INCOME_STATE,
    PILOT_POSITION_DELETE_STATE,
    PILOT_POSITION_EDIT_STATE,
    PILOT_POSITION_STATE,
    PILOT_SETTINGS_STATE,
    PILOT_SNAPSHOT_STATE,
    PILOT_TRADE_CORRECTION_STATE,
    PILOT_TRADE_STATE,
    _is_portfolio_pilot,
    _pilot_trade_display,
    _read_pilot_trade_entries,
    _read_portfolio_pilot,
)
from .portfolio_research import format_twd, render_cash_valuation, valuation_total_label
from .research_workflow_view import render_research_workflow
from .storage import (
    DB_FILE,
    _read_legacy_positions,
)


def _logout_account():
    st.session_state.pop(ACTIVITY_KEY, None)
    st.session_state.pop(PILOT_SETTINGS_STATE, None)
    st.session_state.pop(PILOT_POSITION_STATE, None)
    st.session_state.pop(PILOT_POSITION_EDIT_STATE, None)
    st.session_state.pop(PILOT_POSITION_DELETE_STATE, None)
    st.session_state.pop(PILOT_SNAPSHOT_STATE, None)
    st.session_state.pop(PILOT_INCOME_STATE, None)
    st.session_state.pop(PILOT_TRADE_STATE, None)
    st.session_state.pop(PILOT_TRADE_CORRECTION_STATE, None)
    st.session_state.pop("trade_correction_selection", None)
    st.session_state.pop(PILOT_INCOME_CORRECTION_STATE, None)
    st.session_state.pop("income_correction_selection", None)
    for key in list(st.session_state.keys()):
        if key in (
            "auth_user", "pending_recovery_email", "gemini_api_key_input",
            "openai_api_key_input", "openai_model_input", "sidebar_strategy_selector", "selected_stock_input",
            "trader_backtest_history", "explanation_style_selector",
        ) or key.startswith((
            "port_", "portfolio_", "dual_grid_", "module_access_", "login_", "register_", "recovery_",
            "password_reset_", "reset_", "finmind_user_token_",
            "finmind_token_ack_", "finmind_shared_ack_", "branch_data_source_", "broker_branch_ai_",
            "chip_branch_csv_last_",
            "account_ai_report:", "ai_report_", "ai_provider_", "send_ai_report_email_",
            "stock_screen_",
            "workbench_", "trader_watchlist_", "trader_watch_ticker_", "observer_action_notice_",
        )):
            st.session_state.pop(key, None)


def _clear_finmind_session_token(token_key, acknowledgement_key):
    """Remove a user-entered provider token and its acknowledgement from this session."""
    st.session_state.pop(token_key, None)
    st.session_state.pop(acknowledgement_key, None)


def _close_workbench_on_module_change():
    st.session_state.pop("workbench_open", None)
    st.session_state.pop("workbench_watch_symbol", None)
    st.session_state.pop("portfolio_workbench_cloud", None)


def _open_workbench_module(module, cloud=False):
    _close_workbench_on_module_change()
    st.session_state["sidebar_strategy_selector"] = module
    if cloud:
        st.session_state["portfolio_workbench_cloud"] = True


def _jump_to_stock_diagnosis(ticker):
    _open_workbench_module("📊 彩色 K 線圖與成交量")
    st.session_state["selected_stock_input"] = str(ticker).strip().upper()


def _open_workbench_watch_symbol():
    ticker = st.session_state.get("workbench_watch_symbol")
    if ticker:
        _jump_to_stock_diagnosis(ticker)


def _render_workbench_watchlist():
    st.subheader("自選股")
    st.caption("與三雷達的觀察代碼共用清單，登出後會回到預設清單。")
    symbols = get_trader_watchlist_symbols()
    if symbols:
        st.selectbox(
            "選擇代碼以開啟個股診斷", symbols, index=None, placeholder="選擇自選股",
            key="workbench_watch_symbol", on_change=_open_workbench_watch_symbol,
        )
    else:
        st.info("目前沒有自選股，可到三雷達新增觀察代碼。")
    st.button("到三雷達編輯自選股", key="workbench_edit_watchlist",
              on_click=_open_workbench_module, args=("🚀 華爾街機構級三核心策略雷達",))


def _submit_workbench_search():
    ticker = st.session_state.get("workbench_search_input", "").strip().upper()
    if re.fullmatch(r"\^?[A-Z0-9][A-Z0-9.=^\-]{0,23}", ticker):
        _jump_to_stock_diagnosis(ticker)
        st.session_state.pop("workbench_search_error", None)
    else:
        st.session_state["workbench_search_error"] = True


def _render_workbench_header():
    with st.container(key="workbench_topbar"):
        search, quotes = st.columns([1, 3], vertical_alignment="center")
        with search:
            with st.form("workbench_search", border=False):
                field, action = st.columns([2.7, 1.1], gap="small")
                with field:
                    st.text_input("搜尋股票代碼", placeholder="股票代碼",
                                  key="workbench_search_input", label_visibility="collapsed")
                with action:
                    st.form_submit_button("搜尋", on_click=_submit_workbench_search, width="stretch")
            if st.session_state.get("workbench_search_error"):
                st.warning("請輸入股票代碼，例如 2330.TW 或 NVDA。")
        with quotes:
            _render_market_overview(compact=True)


def _render_workbench_market_trend(account):
    st.subheader("大盤走勢分析")
    symbols = {name: ticker for name, ticker, _ in WORKBENCH_INDEXES}
    label = st.radio("選擇市場指數", list(symbols), horizontal=True, key="workbench_index",
                     label_visibility="collapsed")
    symbol = symbols[label]
    history = _market_index_history(symbol, "1mo")
    if history is None:
        st.info("大盤走勢資料暫缺。")
        return
    closes = pd.to_numeric(history["Close"], errors="coerce").sort_index()
    closes = closes[closes.index.notna() & np.isfinite(closes) & (closes > 0)]
    if closes.empty:
        st.info("大盤走勢資料暫缺。")
        return

    ohlc = ("Open", "High", "Low", "Close")
    volume = pd.to_numeric(history["Volume"], errors="coerce") if "Volume" in history else None
    if volume is not None:
        volume = volume[volume.index.notna() & np.isfinite(volume) & (volume >= 0)].sort_index()
    has_volume = volume is not None and not volume.empty and (volume > 0).any()
    plot = (make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.04, row_heights=[.78, .22])
            if has_volume else go.Figure())
    if set(ohlc).issubset(history.columns):
        candles = history.loc[:, list(ohlc)].apply(pd.to_numeric, errors="coerce").sort_index()
        candles = candles[candles.index.notna() & (np.isfinite(candles) & (candles > 0)).all(axis=1)]
        candles = candles[(candles["High"] >= candles[["Open", "Close", "Low"]].max(axis=1))
                          & (candles["Low"] <= candles[["Open", "Close"]].min(axis=1))]
        if not candles.empty:
            plot.add_trace(go.Candlestick(x=candles.index, open=candles["Open"], high=candles["High"],
                                          low=candles["Low"], close=candles["Close"], name=label,
                                          increasing_line_color="#d14a42", decreasing_line_color="#177957"))
    if not plot.data:
        plot.add_trace(go.Scatter(x=closes.index, y=closes, name=label, mode="lines+markers",
                                  line={"color": "#177957", "width": 2}))
    for period, color in ((5, "#d99431"), (10, "#5879bf")):
        if len(closes) >= period:
            plot.add_trace(go.Scatter(x=closes.index, y=closes.rolling(period).mean(), mode="lines",
                                      name=f"MA{period}", line={"color": color, "width": 1.2}))
    if has_volume:
        plot.add_trace(go.Bar(x=volume.index, y=volume, name="成交量", marker_color="#82b8a2",
                              showlegend=False), row=2, col=1)
    span = [(closes.index[0].date() - pd.Timedelta(days=1)).isoformat(),
            (closes.index[-1].date() + pd.Timedelta(days=1)).isoformat()]
    plot.update_xaxes(type="date", range=span, rangeslider_visible=False,
                      hoverformat="%Y-%m-%d", spikemode="across", spikesnap="cursor")
    plot.update_yaxes(side="right", spikemode="across", spikesnap="cursor")
    plot.update_layout(height=465, margin=dict(l=8, r=12, t=35, b=36), template="plotly_white",
                       showlegend=len(closes) >= 5, legend=dict(orientation="h", y=1.13),
                       dragmode="pan", newshape=dict(line=dict(color="#d04d38", width=2)))
    freshness = "未更新，沿用最後有效資料" if history.attrs.get("quote_stale") else "資料可能延遲"
    st.caption(f"Yahoo Finance 日線 · 最新可用交易日 {closes.index[-1].date():%Y-%m-%d} · {freshness}")
    render_chart_workspace(plot, account["id"], f"index:{symbol}", f"{label} K 線與技術指標")


def _render_workbench_portfolio_summary(account, cloud):
    st.subheader("個人資產摘要")
    st.caption("資料來源為雲端投資組合。" if cloud else "資料來源為原有投資組合。")
    try:
        if cloud:
            positions = _read_portfolio_pilot(account)["positions"]
        else:
            uri = Path(DB_FILE).resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                positions = _read_legacy_positions(conn, int(account["id"]))
    except (KeyError, ValueError, OverflowError, RuntimeError, sqlite3.Error, requests.RequestException) as error:
        st.warning(failure_message(error, "暫時無法讀取投資組合。"))
        return
    if not positions:
        st.info("目前尚無持倉。")
        return
    assets = build_asset_state(positions)
    total = assets.total
    total_card, coverage_card, stock_card, cash_card = st.columns(4)
    total_card.metric(valuation_total_label(total), format_twd(total.subtotal))
    coverage_card.metric("可估值持倉", f"{total.priced}／{total.count} 筆")
    stock_share = "—"
    cash_share = None
    if not total.incomplete and total.subtotal:
        stock_share = (f"{assets.stock.subtotal / total.subtotal:.2%}"
                       if assets.stock.subtotal is not None else "0.00%")
        cash_share = (assets.cash.subtotal or 0) / total.subtotal
    stock_card.metric("股票市值占比", stock_share)
    render_cash_valuation(cash_card, assets.cash, share=cash_share)
    if total.incomplete:
        st.warning("部分持倉無法估值。暫以可估值部分顯示小計，完整占比待資料補齊。")
    st.caption("依帳本保存的持倉數量、價格與匯率估算。")


def _render_workbench_recent_trades(account):
    st.subheader("最近買賣成交")
    try:
        rows = _read_pilot_trade_entries(account)
    except (RuntimeError, ValueError, TypeError, requests.RequestException) as error:
        st.warning(failure_message(error, "暫時無法讀取最近成交。"))
        return
    if not rows:
        st.info("尚無買賣成交紀錄。")
        return

    columns = ("成交日", "方向", "標的", "原幣", "成交數量", "成交單價")
    display_rows = [_pilot_trade_display(row) for row in rows[:5]]
    st.dataframe([{column: row[column] for column in columns} for row in display_rows],
                 hide_index=True, width="stretch")
    st.caption("依成交日由新到舊，最多顯示 5 筆。完整歷史可在雲端投資組合查閱。")


def _render_workbench_shortcuts(cloud):
    st.subheader("研究與策略")
    st.button("個股診斷與 AI 報告", key="workbench_ai", on_click=_open_workbench_module,
              args=("📊 彩色 K 線圖與成交量",), width="stretch")
    st.button("開啟三雷達", key="workbench_radar", on_click=_open_workbench_module,
              args=("🚀 華爾街機構級三核心策略雷達",), width="stretch")
    st.button("開啟策略比較", key="workbench_compare", on_click=_open_workbench_module,
              args=("⚖️ 網格／持有／定期定額比較",), width="stretch")
    st.button("開啟雙策略網格", key="workbench_grid", on_click=_open_workbench_module,
              args=("🧠 智慧型雙策略網格系統",), width="stretch")
    st.button("開啟雲端投資組合" if cloud else "開啟投資組合",
              key="workbench_portfolio", on_click=_open_workbench_module,
              args=("🧭 投資組合理財", cloud), width="stretch")


def _render_workbench_research():
    st.subheader("最近回測")
    history = st.session_state.get("trader_backtest_history", [])
    if history:
        st.dataframe([
            {"標的": row["ticker"], "條件": row["condition"], "週期": row["timeframe"],
             "資料期間": row["period"], "評估": row["status"]}
            for row in reversed(history[-3:])
        ], hide_index=True, width="stretch")
        st.caption("顯示本次登入手動執行的最近 3 項回測條件。")
    else:
        st.info("本次登入尚無回測紀錄。")
    st.button("開啟技術指標回測", key="workbench_backtest",
              on_click=_open_workbench_module, args=("🧪 技術指標自訂策略回測",))


def _render_module_activity(account):
    events = module_events(st.session_state, AccountScope.from_session(account))
    st.caption("本次登入的模組切換紀錄，登出後清除。")
    if not events:
        st.info("本次登入還沒有模組操作紀錄。")
        return
    st.dataframe([{
        "時間（台北）": pd.Timestamp(item.occurred_at).tz_convert("Asia/Taipei").strftime("%H:%M:%S"),
        "原畫面": item.source or "—",
        "開啟畫面": item.target,
        "狀態": item.status,
    } for item in events], hide_index=True, width="stretch")


def _read_research_workflow_ledger(account, cloud):
    if cloud:
        data = _read_portfolio_pilot(account)
        return data["positions"], data["settings"], "雲端投資組合"
    try:
        uri = Path(DB_FILE).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            owner = AccountScope.from_session(account).local_id
            positions = _read_legacy_positions(conn, owner)
            settings = dict(conn.execute(
                "SELECT setting_key, setting_value FROM portfolio_user_settings WHERE user_id=?",
                (owner,),
            ).fetchall())
    except sqlite3.Error as error:
        raise RuntimeError("Ledger temporarily unavailable") from error
    return positions, settings, "原有投資組合"


def _render_investment_workbench(account, explanation_style):
    backdrop = Path(__file__).resolve().parents[1] / "assets" / "workbench-dawn.webp"
    if backdrop.is_file():
        hero_image = b64encode(backdrop.read_bytes()).decode("ascii")
        st.markdown(
            '<style class="zhengxu-workbench-theme">.st-key-workbench_hero {'
            'background-image: linear-gradient(90deg,#0a2d22e8,#0a2d2245),'
            f'url("data:image/webp;base64,{hero_image}") !important;'
            '}</style>',
            unsafe_allow_html=True,
        )
    st.markdown("""
        <style class="zhengxu-workbench-theme">
        .st-key-zhengxu_workbench [data-testid="stDataFrame"] [data-testid="stElementToolbar"] {
            display: none !important;
        }
        [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
        > [data-testid="stElementContainer"]:has(.zhengxu-workbench-theme) {
            display: none;
        }
        .st-key-zhengxu_workbench { color: #173b2c; gap: .7rem !important; }
        .st-key-zhengxu_workbench :is(h1, h2, h3, h4) { color: #173b2c !important; }
        .st-key-zhengxu_workbench h3 { font-size: 1.15rem !important; padding: 0 0 .2rem; }
        .st-key-workbench_hero {
            min-height: 114px; justify-content: center; border-radius: 13px;
            background-color: #123e2c; background-size: cover; background-position: center 47%;
            padding: .65rem 1.6rem; gap: .4rem !important;
        }
        .st-key-zhengxu_workbench .st-key-workbench_hero :is(h1, p, span) { color: #fff !important; }
        .st-key-workbench_hero h1 { font-size: 2rem !important; padding: 0 !important; line-height: 1.15; }
        .st-key-zhengxu_workbench [data-testid="stCaptionContainer"] :is(p, span) {
            color: #486657 !important;
        }
        .st-key-zhengxu_workbench .st-key-workbench_hero [data-testid="stCaptionContainer"] :is(p, span) {
            color: #f3f8ee !important;
        }
        .st-key-zhengxu_workbench [class*="st-key-workbench_card_"],
        .st-key-zhengxu_workbench [data-testid="stMetric"] {
            background: #fff !important; border: 1px solid #cadcd1 !important;
            box-shadow: none !important; border-radius: 12px !important;
        }
        .st-key-zhengxu_workbench :is([data-testid="stMetricLabel"], [data-testid="stMetricValue"],
                                      [data-testid="stWidgetLabel"]) {
            color: #173b2c !important;
        }
        .st-key-zhengxu_workbench [data-testid="stButton"] button {
            background: #236647 !important; color: #fff !important; box-shadow: none !important;
        }
        .st-key-zhengxu_workbench [data-testid="stButton"] button:focus-visible {
            outline: 3px solid #3f8c64; outline-offset: 3px;
        }
        .st-key-workbench_card_assets { padding: .55rem .85rem !important; gap: .3rem !important; }
        .st-key-workbench_card_assets [data-testid="stMetric"] { padding: .4rem .65rem !important; }
        .st-key-workbench_card_assets [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
        .st-key-workbench_card_market [data-testid="stRadio"] label p { color: #294837 !important; }
        .st-key-workbench_topbar [data-testid="stVerticalBlock"] { gap: .25rem !important; }
        .st-key-workbench_topbar [data-testid="stMetric"] {
            background: transparent !important; border: 0 !important;
            padding: 0 .3rem !important; min-width: 0;
        }
        .st-key-workbench_topbar [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
        .st-key-workbench_topbar [data-testid="stMetricDelta"] { font-size: .72rem !important; }
        .st-key-workbench_topbar [data-testid="stCaptionContainer"] p { font-size: .69rem; }
        .st-key-workbench_topbar [data-testid="stForm"] { padding: 0 !important; }
        .st-key-workbench_topbar [data-testid="stFormSubmitButton"] button {
            background: #236647 !important; box-shadow: none !important;
            padding: .22rem .42rem !important; font-size: .8rem !important;
        }
        .st-key-workbench_topbar [data-testid="stTextInput"] input {
            background: #fff !important; color: #173b2c !important; border-color: #cadcd1 !important;
        }
        .st-key-workbench_topbar [data-testid="stTextInput"] input::placeholder { color: #516c60 !important; }
        @media (max-width: 768px) {
            :is(.st-key-workbench_personal,.st-key-workbench_middle) [data-testid="stHorizontalBlock"] >
            :is([data-testid="column"], [data-testid="stColumn"]) {
                flex-basis: 100% !important; width: 100% !important; min-width: 100% !important;
            }
            .st-key-workbench_topbar > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex-basis: 100% !important; width: 100% !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    cloud = _is_portfolio_pilot(account)
    with st.container(key="zhengxu_workbench"):
        _render_workbench_header()
        with st.container(key="workbench_hero"):
            st.title("投資工作台")
            st.caption("正序 · 投資，從理解開始")
        show_trades = cloud and feature_enabled("GVI_PORTFOLIO_PILOT_TRADE")
        labels = ["總覽", "自選股與回測"]
        if show_trades:
            labels.append("最近成交")
        show_workflow = feature_enabled("GVI_RESEARCH_WORKFLOW_PREVIEW")
        if show_workflow:
            labels.append("研究工作流")
        labels.append("本次操作紀錄")
        pages = st.tabs(labels)
        with pages[0]:
            with st.container(border=True, key="workbench_card_assets"):
                _render_workbench_portfolio_summary(account, cloud)
                if explanation_style.startswith("生動童趣"):
                    st.caption("像用帳本上的標價盤點家當。市值會隨價格與匯率變動。")
            with st.container(key="workbench_middle"):
                market, research_tools = st.columns([2.15, 1])
                with market.container(border=True, key="workbench_card_market", height="stretch"):
                    _render_workbench_market_trend(account)
                with research_tools.container(border=True, key="workbench_card_tools", height="stretch"):
                    _render_workbench_shortcuts(cloud)
        with pages[1]:
            with st.container(key="workbench_personal"):
                watchlist, research = st.columns(2)
                with watchlist.container(border=True, key="workbench_card_watchlist"):
                    _render_workbench_watchlist()
                with research.container(border=True, key="workbench_card_research"):
                    _render_workbench_research()
        if show_trades:
            with pages[2], st.container(border=True, key="workbench_card_trades"):
                _render_workbench_recent_trades(account)
        if show_workflow:
            with pages[-2]:
                render_research_workflow(account, lambda: _read_research_workflow_ledger(account, cloud))
        with pages[-1]:
            _render_module_activity(account)
