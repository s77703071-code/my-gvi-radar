"""Shared analysis controls and dispatch to the active research view."""

import html
import streamlit as st

from .backtest_view import (
    render_backtest,
)
from .chart_tools import (
    _render_metric_glossary,
)
from .chart_view import (
    render_chart,
)
from .chip_view import (
    render_chip_flow,
)
from .comparison_view import (
    render_comparison,
)
from .dividend import (
    render_dividend_tab,
)
from .dual_grid_view import (
    render_dual_grid,
)
from .explanations import (
    _render_market_psychology_guide,
)
from .grid_view import (
    render_grid,
)
from .institutional_view import (
    render_institutional,
)
from .legacy_portfolio_view import (
    render_legacy_portfolio,
)
from .market_data import (
    TRADER_WATCHLIST_DEFAULTS,
    _render_market_overview,
)
from .radar_view import (
    render_radar,
)
from .rankings_view import (
    render_rankings,
)
from .research_ai import (
    get_openai_secret_values,
)
from .stock_diagnosis import (
    render_stock_diagnosis,
)
from .workbench import (
    _jump_to_stock_diagnosis,
)


def render_analysis(auth_user, selected_stock, selected_module, explanation_style):
    def _queue_observer_ticker(ticker, notice_key):
        ticker = str(ticker).strip().upper()
        existing = [str(code).strip().upper()
                    for code in st.session_state.get("trader_watchlist_values", TRADER_WATCHLIST_DEFAULTS)]
        if ticker in existing:
            st.session_state[notice_key] = f"{ticker} 已在操盤手自訂觀察名單中。"
            return
        try:
            empty_index = existing.index("")
        except ValueError:
            st.session_state[notice_key] = "自訂觀察代碼欄位已滿，請先清出一格再加入。"
            return
        existing[empty_index] = ticker
        st.session_state["trader_watchlist_values"] = existing
        st.session_state["trader_watchlist_sync_needed"] = True
        st.session_state[notice_key] = f"已將 {ticker} 加入操盤手自訂觀察代碼。"

    def _render_stock_actions(ticker, stock_name, key_prefix):
        st.caption(f"已選取：{stock_name}（{ticker}）。選擇要開啟的功能：")
        action_cols = st.columns(2)
        with action_cols[0]:
            st.button(
                "🔍 使用個股即時診斷",
                key=f"{key_prefix}_diagnose",
                on_click=_jump_to_stock_diagnosis,
                args=(ticker,),
                use_container_width=True,
            )
        with action_cols[1]:
            notice_key = f"observer_action_notice_{key_prefix}"
            st.button(
                "➕ 加入自訂觀察代碼",
                key=f"{key_prefix}_watch",
                on_click=_queue_observer_ticker,
                args=(ticker, notice_key),
                use_container_width=True,
            )
        notice = st.session_state.pop(f"observer_action_notice_{key_prefix}", None)
        if notice:
            st.info(notice)

    st.markdown(
        '<section class="zhengxu-module-hero">'
        f'<h1>{html.escape(selected_module)}</h1></section>',
        unsafe_allow_html=True,
    )

    chart_page = selected_module == "📊 彩色 K 線圖與成交量"
    radar_page = selected_module == "🚀 華爾街機構級三核心策略雷達"
    show_market_overview = chart_page or radar_page or selected_module == "📦 籌碼流向參考"
    market_indexes = _render_market_overview() if show_market_overview else None

    if chart_page or radar_page or selected_module in {
        "🧪 技術指標自訂策略回測",
        "💸 配息複利策略",
    }:
        _render_metric_glossary(st)

    st.markdown("---")

    st.sidebar.markdown("### 🧠 系統 API 金鑰設定")
    api_key_input = ""
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip() != "":
            api_key_input = st.secrets["GEMINI_API_KEY"].strip()
            st.sidebar.success("🔒 Gemini API 金鑰已從雲端安全盾自動加載")
    except Exception:
        pass

    if not api_key_input:
        api_key_input = st.sidebar.text_input(
            "請輸入您的 Gemini API Key：", type="password", key="gemini_api_key_input"
        )

    openai_api_key_input, secret_openai_model = get_openai_secret_values(st.secrets)
    if openai_api_key_input:
        st.sidebar.success("🔒 ChatGPT API 金鑰已從雲端安全設定載入")
    else:
        openai_api_key_input = st.sidebar.text_input(
            "請輸入您的 ChatGPT API Key：", type="password", key="openai_api_key_input"
        )
    openai_model_input = st.sidebar.text_input(
        "ChatGPT 模型名稱", value=secret_openai_model, key="openai_model_input",
        help="預設為 gpt-5.5；若您的 API 帳號沒有此模型權限，可改成帳號可用的模型。",
    ).strip() or "gpt-5.5"
    st.sidebar.caption("ChatGPT API 使用量依 OpenAI API 帳戶計費；ChatGPT 訂閱不包含 API 額度。")

    chart_controls = st.container() if chart_page else st.sidebar
    chart_controls.markdown("### 🎛️ 看盤資料設定")
    selected_tf = chart_controls.selectbox("⏱️ 看盤 K 線時間軸級別", options=["1分鐘", "5分鐘", "30分鐘", "60分鐘", "1日", "1週", "1個月", "1季", "半年", "1年"], index=4, key="research_chart_timeframe")
    chart_controls.caption(f"目前圖表週期：{selected_tf}")
    tf_mapping = {
        "1分鐘": {"p": "7d", "i": "1m"}, "5分鐘": {"p": "7d", "i": "5m"}, "30分鐘": {"p": "30d", "i": "30m"}, "60分鐘": {"p": "60d", "i": "60m"},
        "1日": {"p": "2y", "i": "1d"}, "1週": {"p": "5y", "i": "1wk"}, "1個月": {"p": "max", "i": "1mo"},
        "1季": {"p": "max", "i": "3mo"}, "半年": {"p": "max", "i": "3mo"}, "1年": {"p": "max", "i": "3mo"}
    }
    cfg = tf_mapping[selected_tf]

    with chart_controls.expander("📈 K 線常用技術指標", expanded=False):
        selected_chart_indicators = st.multiselect(
            "顯示在 K 線圖下方",
            options=[
                "KD", "MACD", "RSI", "BIAS（乖離率）", "ATR（波動幅度）",
                "OBV（能量潮）", "CCI（順勢指標）", "ADX（趨勢強度）",
            ],
            default=[],
            key=f"chart_technical_indicators_{auth_user['id']}",
            help="可複選；KD、MACD、RSI、BIAS、ATR、OBV、CCI、ADX 會各自顯示在 K 線下方，不會改變選股條件。",
        )
        selected_chart_overlays = st.multiselect(
            "疊加在 K 線／成交量圖",
            options=["布林通道（20，±2 標準差）", "EMA 12", "EMA 26", "成交量均線（20）"],
            default=[],
            key=f"chart_technical_overlays_{auth_user['id']}",
            help="布林通道與 EMA 疊在價格 K 線上；成交量均線疊在成交量柱上。",
        )
        st.caption(
            "KD 看短線區間；MACD 看趨勢動能；RSI／CCI 看漲跌力道；BIAS 看價格偏離均線的幅度；"
            "ATR 看波動大小；OBV 看量價累積；ADX 看趨勢強弱。週期依目前 K 線級別計算，指標不是單獨買賣訊號。"
        )

    with chart_controls.expander("🛠️ 均線指標快速視窗設定", expanded=False):
        preset_ma = st.selectbox("📊 選擇均線快捷套組", options=["標準視窗 (5日 / 10日 / 20日)", "極短線 (3日 / 5日 / 10日)", "波段趨勢 (20日 / 60日 / 120日)", "自訂均線配置"], index=0)
        default_days = [5, 10, 20, 60, 240] if preset_ma == "標準視窗 (5日 / 10日 / 20日)" else ([3, 5, 10, 20, 60] if preset_ma == "極短線 (3日 / 5日 / 10日)" else [20, 60, 120, 240, 500])
        default_actives = [True, True, True, False, False]
        personal_ma_configs = []
        default_colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3"]

        for i in range(1, 6):
            col_show, col_p, col_c = st.columns([0.8, 1.4, 0.8])
            with col_show: is_active = st.checkbox(f"MA{i}", value=default_actives[i-1], key=f"ma_active_{i}")
            with col_p: ma_p = st.number_input(f"天數{i}", min_value=1, max_value=500, value=int(default_days[i-1]), label_visibility="collapsed", key=f"personal_ma_p_{i}")
            with col_c: ma_c = str(st.color_picker(f"C{i}", value=default_colors[i-1], label_visibility="collapsed", key=f"personal_ma_c_{i}"))
            if is_active: personal_ma_configs.append({"period": int(ma_p), "color": ma_c})

    chart_area = st.container() if chart_page else None
    df_chart, c_name, change_pct, date_strings, vol_ratio, price_val, grid_advice_metrics, is_tw_stock = render_stock_diagnosis(
        api_key_input, auth_user, cfg, explanation_style, market_indexes, openai_api_key_input,
        openai_model_input, selected_stock, selected_tf, show_tactical_panel=chart_page,
        show_official_close=chart_page, show_radar_comparison=chart_page or radar_page,
        show_stock_research=chart_page,
    )

    try:
        _render_market_psychology_guide(st, selected_module, explanation_style)
        if df_chart is None or df_chart.empty:
            if selected_module in ["📊 彩色 K 線圖與成交量", "📦 籌碼流向參考", "🤖 網格自動生成器", "⚖️ 網格／持有／定期定額比較", "🧪 技術指標自訂策略回測", "💸 配息複利策略", "🧠 智慧型雙策略網格系統"]:
                st.error(f"❌ 無此標的或無法取得數據：【{selected_stock}】，請檢查股票代碼是否正確。")
        else:
            if chart_area is not None:
                with chart_area:
                    render_chart(auth_user, c_name, change_pct, date_strings, df_chart, personal_ma_configs, selected_chart_indicators, selected_chart_overlays, selected_stock, selected_tf, vol_ratio)
            if selected_module == '💸 配息複利策略':
                render_dividend_tab(st, default_ticker=selected_stock)
            if selected_module == '📦 籌碼流向參考':
                render_chip_flow(auth_user, df_chart, selected_stock, selected_tf)
            if selected_module == '🧠 智慧型雙策略網格系統':
                render_dual_grid(auth_user, selected_stock)
            if selected_module == '🤖 網格自動生成器':
                render_grid(price_val, selected_stock)
            if selected_module == '⚖️ 網格／持有／定期定額比較':
                render_comparison(c_name, date_strings, df_chart, selected_tf, is_tw_stock=is_tw_stock)
            if selected_module == '🧪 技術指標自訂策略回測':
                render_backtest(c_name, cfg, df_chart, explanation_style, grid_advice_metrics, is_tw_stock, selected_stock, selected_tf)
    except Exception as ex_tab:
        st.error(f"❌ 畫面渲染異常：{ex_tab}")

    if selected_module == '🚀 華爾街機構級三核心策略雷達':
        render_radar(_render_stock_actions, cfg, explanation_style, selected_tf)
    if selected_module == '🏦 台股三大法人獨立選股策略':
        render_institutional()
    if selected_module == '📊 當日全市場三大法人佔股本比 Top 10':
        render_rankings(_render_stock_actions)
    if selected_module == '🧭 投資組合理財':
        render_legacy_portfolio(_render_stock_actions, auth_user)
