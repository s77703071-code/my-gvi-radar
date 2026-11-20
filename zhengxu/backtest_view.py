"""Technical indicator backtest controls and result display."""

from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from .backtest import (
    _count_ma_cross_events,
    _normalize_ma_cross_pairs,
    _slice_history_for_date_range,
    analyze_indicator_backtest,
    build_backtest_coach_advice,
    compare_ma_execution_timing,
    compute_kline_indicator_frame,
    describe_backtest_conditions,
    run_strategy_backtest,
)
from .chart_tools import (
    _format_optional_cell,
)
from .grid import (
    simulate_monthly_dca,
)
from .market_data import (
    calculate_chip_and_backtest,
)


def render_backtest(c_name, cfg, df_chart, explanation_style, grid_advice_metrics, is_tw_stock, selected_stock, selected_tf):
    st.markdown(f"### 🧪 【{c_name}】技術指標多空策略與動態停損回測器 (3.12 版)")
    default_bt_start = pd.Timestamp(df_chart.index[0]).date()
    default_bt_end = pd.Timestamp(df_chart.index[-1]).date()
    bt_date_selection = st.date_input(
        "本次回測資料起訖日",
        value=(default_bt_start, default_bt_end),
        key=f"bt_date_range_{selected_stock}_{selected_tf}",
        help="預設使用目前圖表已有資料的完整期間；若輸入範圍超出圖表資料，系統會按目前 K 線週期重新下載。",
    )
    if isinstance(bt_date_selection, (tuple, list)):
        if len(bt_date_selection) >= 2:
            bt_start_date, bt_end_date = bt_date_selection[0], bt_date_selection[1]
        elif len(bt_date_selection) == 1:
            bt_start_date, bt_end_date = bt_date_selection[0], default_bt_end
        else:
            bt_start_date, bt_end_date = default_bt_start, default_bt_end
    else:
        bt_start_date, bt_end_date = bt_date_selection, bt_date_selection
    if bt_start_date > bt_end_date:
        st.warning("起始日期晚於結束日期，已自動對調日期。")
        bt_start_date, bt_end_date = bt_end_date, bt_start_date
    st.caption(
        f"目前 K 線週期：{selected_tf}；均線數字代表 K 棒資料根數，例如週線的 20 根是 20 週線，不是 20 個日曆天。"
    )
    bt_col1, bt_col2 = st.columns(2)

    with bt_col1:
        st.markdown("##### 🟢 進場條件")
        entry_indicator_options = [
            "均線 MA", "五線排列", "KD", "MACD", "成交量", "RSI", "布林通道",
            "BIAS 乖離率", "ATR 波動幅度", "OBV 能量潮", "CCI 順勢指標",
            "DMI／ADX 趨勢強度", "EMA12／EMA26",
        ]
        entry_selected_indicators = st.multiselect(
            "選擇要使用的進場指標",
            options=entry_indicator_options,
            default=["均線 MA"],
            key="bt_entry_indicators",
            help="只顯示已選指標的設定；未選指標不會加入買進條件。",
        )
        st.caption("多個進場條件可選擇同時成立（ALL）或任一成立（ANY）。")
        entry_ma_mode, entry_ma_p = "停用", 20
        entry_five_ma_mode = "停用"
        entry_kd_mode = entry_macd_mode = entry_vol_mode = "停用"
        entry_rsi_mode = entry_bb_mode = entry_bias_mode = "停用"
        entry_atr_mode = entry_obv_mode = entry_cci_mode = "停用"
        entry_adx_mode = entry_ema_mode = "停用"
        entry_atr_pct_limit, entry_adx_threshold = 4.0, 20.0
        entry_ma_cross_pairs = []
        exit_ma_cross_pairs = []
        default_cross_pairs = [
            (5, 20), (10, 20), (10, 60), (20, 60), (20, 120),
            (60, 120), (60, 240), (5, 60), (20, 240), (120, 240),
        ]
        if "均線 MA" in entry_selected_indicators:
            entry_ma_mode = st.selectbox(
                "均線買進方式", ["停用", "突破均線", "站穩均線上"], index=1,
                key="bt_entry_ma_sel_v2",
            )
            entry_ma_p = st.number_input(
                "買進均線資料根數", min_value=1, max_value=240, value=20,
                key="bt_entry_ma_p",
            )
            with st.expander("新增多組均線黃金交叉買進條件", expanded=False):
                st.caption("快線由下往上穿越慢線時成立；最多可設定 10 組。")
                entry_ma_cross_count = st.number_input(
                    "買進交叉組數", min_value=0, max_value=10, value=0, step=1,
                    key="bt_entry_ma_cross_count",
                )
                invalid_entry_cross = False
                for pair_index in range(int(entry_ma_cross_count)):
                    fast_default, slow_default = default_cross_pairs[pair_index]
                    pair_cols = st.columns(2)
                    with pair_cols[0]:
                        fast_period = st.number_input(
                            f"第 {pair_index + 1} 組快線資料根數", min_value=1,
                            max_value=240, value=fast_default, step=1,
                            key=f"bt_entry_ma_cross_fast_{pair_index + 1}",
                        )
                    with pair_cols[1]:
                        slow_period = st.number_input(
                            f"第 {pair_index + 1} 組慢線資料根數", min_value=1,
                            max_value=240, value=slow_default, step=1,
                            key=f"bt_entry_ma_cross_slow_{pair_index + 1}",
                        )
                    if fast_period >= slow_period:
                        invalid_entry_cross = True
                    else:
                        entry_ma_cross_pairs.append((int(fast_period), int(slow_period)))
                entry_ma_cross_pairs = _normalize_ma_cross_pairs(entry_ma_cross_pairs)
                if invalid_entry_cross:
                    st.warning("快線資料根數必須小於慢線；不符合的組別會略過。")
        if "五線排列" in entry_selected_indicators:
            entry_five_ma_mode = "五線開花"
            st.caption("五線開花：MA5＞MA10＞MA20＞MA60＞MA120，且均線同步上升時進場。")
        if "KD" in entry_selected_indicators:
            entry_kd_mode = "黃金交叉"
            st.caption("KD：K 線由下往上穿越 D 線時進場。")
        if "MACD" in entry_selected_indicators:
            entry_macd_mode = "黃金交叉"
            st.caption("MACD：DIF 由下往上穿越訊號線時進場。")
        if "成交量" in entry_selected_indicators:
            entry_vol_mode = st.selectbox(
                "成交量條件", ["停用", "爆量 (大於20日均量1.5倍)", "爆量 (大於20日均量2.0倍)"],
                index=1, key="bt_entry_vol_sel_v2",
            )
        if "RSI" in entry_selected_indicators:
            entry_rsi_mode = st.selectbox(
                "RSI 條件", ["停用", "超賣回升 (RSI向上突破30)", "強勢突破 (RSI向上突破50)"],
                index=1, key="bt_entry_rsi_sel_v2",
            )
        if "布林通道" in entry_selected_indicators:
            entry_bb_mode = st.selectbox(
                "布林通道條件", ["停用", "突破布林上軌", "觸及布林下軌反彈"],
                index=1, key="bt_entry_bb_sel_v2",
            )
        if "BIAS 乖離率" in entry_selected_indicators:
            entry_bias_mode = st.selectbox(
                "BIAS 條件", ["停用", "向上突破 0%", "自 −5% 向上回升"],
                index=1, key="bt_entry_bias_sel_v2",
            )
        if "ATR 波動幅度" in entry_selected_indicators:
            entry_atr_mode = "低波動篩選"
            entry_atr_pct_limit = st.number_input(
                "ATR／收盤價上限（%）", min_value=0.1, max_value=30.0,
                value=4.0, step=0.1, key="bt_entry_atr_limit",
                help="只在 ATR 相對收盤價不高於此數值時允許進場。",
            )
        if "OBV 能量潮" in entry_selected_indicators:
            entry_obv_mode = "OBV 高於前一根"
            st.caption("OBV：能量潮高於前一根 K 棒時進場。")
        if "CCI 順勢指標" in entry_selected_indicators:
            entry_cci_mode = st.selectbox(
                "CCI 條件", ["停用", "上穿 −100", "上穿 0"], index=1,
                key="bt_entry_cci_sel_v2",
            )
        if "DMI／ADX 趨勢強度" in entry_selected_indicators:
            entry_adx_mode = "趨勢增強且 +DI 領先"
            entry_adx_threshold = st.number_input(
                "ADX 趨勢強度門檻", min_value=5.0, max_value=60.0,
                value=20.0, step=1.0, key="bt_entry_adx_threshold",
            )
        if "EMA12／EMA26" in entry_selected_indicators:
            entry_ema_mode = "EMA12 黃金交叉 EMA26"
            st.caption("EMA：短期 EMA12 向上穿越 EMA26 時進場。")
        entry_match_mode = st.radio(
            "買進訊號判定邏輯",
            options=["同時滿足所有選定條件 (ALL)", "任一滿足即買進 (ANY)"],
            index=0, key="bt_match",
        )

    with bt_col2:
        st.markdown("##### 🔴 離場與風險控制")
        stop_loss_pct = st.number_input("固定停損幅度 (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="bt_sl")
        trailing_stop_pct = st.number_input("最高價移動回撤幅度 (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="bt_trail")
        bt_rr_enabled = st.checkbox(
            "買點成立後啟用風險報酬停利目標（預設 1:3）",
            value=True, key="bt_rr_enabled",
            help="停利價會依進場價與價格型停損距離計算；停損條件不會被移除。"
        )
        bt_rr_multiple = st.number_input(
            "目標報酬倍數（風險 1，報酬 X）",
            min_value=0.5, max_value=10.0, value=3.0, step=0.5,
            key="bt_rr_multiple", disabled=not bt_rr_enabled,
            help="預設 3 代表風險報酬比 1:3；可改成 1.5、2、4 等。"
        )
        exit_indicator_options = entry_indicator_options
        exit_selected_indicators = st.multiselect(
            "選擇要使用的離場指標",
            options=exit_indicator_options,
            default=["均線 MA"],
            key="bt_exit_indicators",
            help="只顯示已選指標的設定；未選指標不會加入平倉條件。固定與移動停損獨立運作。",
        )
        exit_ma_mode, exit_ma_p = "停用", 20
        exit_five_ma_mode = "停用"
        exit_kd_mode = exit_macd_mode = exit_vol_mode = "停用"
        exit_rsi_mode = exit_bb_mode = exit_bias_mode = "停用"
        exit_obv_mode = exit_cci_mode = exit_adx_mode = exit_ema_mode = "停用"
        exit_adx_threshold = 20.0
        exit_atr_stop_multiple = 0.0
        exit_ma_cross_pairs = []
        if "均線 MA" in exit_selected_indicators:
            exit_ma_mode = st.selectbox(
                "均線平倉方式", ["停用", "跌破均線"], index=1, key="bt_exit_ma_sel_v2",
            )
            exit_ma_p = st.number_input(
                "賣出均線資料根數", min_value=1, max_value=240, value=20,
                key="bt_exit_ma_p",
            )
            with st.expander("新增多組均線死亡交叉平倉條件", expanded=False):
                st.caption("快線由上往下跌破慢線時成立；最多可設定 10 組。")
                exit_ma_cross_count = st.number_input(
                    "平倉交叉組數", min_value=0, max_value=10, value=0, step=1,
                    key="bt_exit_ma_cross_count",
                )
                invalid_exit_cross = False
                for pair_index in range(int(exit_ma_cross_count)):
                    fast_default, slow_default = default_cross_pairs[pair_index]
                    pair_cols = st.columns(2)
                    with pair_cols[0]:
                        fast_period = st.number_input(
                            f"第 {pair_index + 1} 組快線資料根數", min_value=1,
                            max_value=240, value=fast_default, step=1,
                            key=f"bt_exit_ma_cross_fast_{pair_index + 1}",
                        )
                    with pair_cols[1]:
                        slow_period = st.number_input(
                            f"第 {pair_index + 1} 組慢線資料根數", min_value=1,
                            max_value=240, value=slow_default, step=1,
                            key=f"bt_exit_ma_cross_slow_{pair_index + 1}",
                        )
                    if fast_period >= slow_period:
                        invalid_exit_cross = True
                    else:
                        exit_ma_cross_pairs.append((int(fast_period), int(slow_period)))
                exit_ma_cross_pairs = _normalize_ma_cross_pairs(exit_ma_cross_pairs)
                if invalid_exit_cross:
                    st.warning("快線資料根數必須小於慢線；不符合的組別會略過。")
        if "五線排列" in exit_selected_indicators:
            exit_five_ma_mode = "排列失序"
            st.caption("任一均線排列不再維持 MA5＞MA10＞MA20＞MA60＞MA120 時離場。")
        if "KD" in exit_selected_indicators:
            exit_kd_mode = "死亡交叉"
            st.caption("KD：K 線由上往下跌破 D 線時離場。")
        if "MACD" in exit_selected_indicators:
            exit_macd_mode = "死亡交叉"
            st.caption("MACD：DIF 由上往下跌破訊號線時離場。")
        if "成交量" in exit_selected_indicators:
            exit_vol_mode = st.selectbox(
                "成交量離場條件", ["停用", "極端爆量倒貨 (大於20日均量2.5倍)"],
                index=1, key="bt_exit_vol_sel_v2",
            )
        if "RSI" in exit_selected_indicators:
            exit_rsi_mode = st.selectbox(
                "RSI 離場條件", ["停用", "超買警戒 (RSI跌破70)"],
                index=1, key="bt_exit_rsi_sel_v2",
            )
        if "布林通道" in exit_selected_indicators:
            exit_bb_mode = st.selectbox(
                "布林通道離場條件", ["停用", "跌破布林下軌", "觸及上軌拉回", "站上布林中軌"],
                index=1, key="bt_exit_bb_sel_v2",
            )
        if "BIAS 乖離率" in exit_selected_indicators:
            exit_bias_mode = st.selectbox(
                "BIAS 離場條件", ["停用", "跌破 0%", "高檔回落並跌破 +5%"],
                index=1, key="bt_exit_bias_sel_v2",
            )
        if "ATR 波動幅度" in exit_selected_indicators:
            exit_atr_stop_multiple = st.number_input(
                "ATR 停損倍數", min_value=0.5, max_value=10.0,
                value=2.0, step=0.5, key="bt_exit_atr_multiple",
                help="以進場時 ATR 固定停損價：進場價減去 ATR × 倍數。",
            )
        if "OBV 能量潮" in exit_selected_indicators:
            exit_obv_mode = "OBV 低於前一根"
            st.caption("OBV：能量潮低於前一根 K 棒時離場。")
        if "CCI 順勢指標" in exit_selected_indicators:
            exit_cci_mode = st.selectbox(
                "CCI 離場條件", ["停用", "跌破 +100", "跌破 0"],
                index=1, key="bt_exit_cci_sel_v2",
            )
        if "DMI／ADX 趨勢強度" in exit_selected_indicators:
            exit_adx_mode = st.selectbox(
                "DMI／ADX 離場條件", ["停用", "+DI 跌破 −DI", "ADX 低於門檻"],
                index=1, key="bt_exit_adx_sel_v2",
            )
            if exit_adx_mode == "ADX 低於門檻":
                exit_adx_threshold = st.number_input(
                    "ADX 離場強度門檻", min_value=5.0, max_value=60.0,
                    value=20.0, step=1.0, key="bt_exit_adx_threshold",
                )
        if "EMA12／EMA26" in exit_selected_indicators:
            exit_ema_mode = "EMA12 死亡交叉 EMA26"
            st.caption("EMA：EMA12 向下跌破 EMA26 時離場。")
        if bt_rr_enabled and stop_loss_pct <= 0 and trailing_stop_pct <= 0 and exit_atr_stop_multiple <= 0:
            st.warning("目前沒有價格型停損距離，風險報酬停利目標不會生效。")
        else:
            st.caption("價格停損與指標離場可同時運作；指標未勾選時不會加入策略。")
    with st.expander("💰 績效分析設定（本金與交易成本）", expanded=False):
        cost_cols = st.columns(4)
        with cost_cols[0]:
            bt_initial_capital = st.number_input("回測初始資金", min_value=1000, max_value=1000000000,
                                                  value=100000, step=10000, key="bt_initial_capital")
        with cost_cols[1]:
            bt_commission_pct = st.number_input("手續費（每邊，%）", min_value=0.0, max_value=5.0,
                                                value=0.1425 if is_tw_stock else 0.0, step=0.01,
                                                format="%.4f", key="bt_commission_pct")
        with cost_cols[2]:
            bt_sell_tax_pct = st.number_input("賣出交易稅（%）", min_value=0.0, max_value=5.0,
                                              value=0.3 if is_tw_stock else 0.0, step=0.01,
                                              format="%.3f", key="bt_sell_tax_pct")
        with cost_cols[3]:
            bt_slippage_pct = st.number_input("滑價（每邊，%）", min_value=0.0, max_value=5.0,
                                              value=0.05, step=0.01, format="%.2f",
                                              key="bt_slippage_pct")
        st.caption("成本按每次買進／賣出分別估算；台股預設值可依券商折扣、標的與交易方式調整。")

    compare_execution = False
    if selected_tf == "1日":
        compare_execution = st.checkbox("附加比較 MA5／MA20 的同日收盤與次日開盤成交", value=False)
        if compare_execution:
            st.caption("使用相同日線、本金及費用。固定以 MA5／MA20 交叉進出，不套用自訂停損或停利。")

    if st.button(f"🚀 開始執行【{c_name}】全指標量化回測", type="primary", use_container_width=True, key="run_bt_btn"):
        chart_start = pd.Timestamp(df_chart.index[0]).date()
        chart_end = pd.Timestamp(df_chart.index[-1]).date()
        requested_start = pd.Timestamp(bt_start_date)
        requested_end = pd.Timestamp(bt_end_date)
        if bt_start_date < chart_start or bt_end_date > chart_end:
            try:
                with st.spinner("正在下載您指定日期範圍的歷史 K 線資料…"):
                    raw_backtest = yf.download(
                        selected_stock,
                        start=requested_start.strftime("%Y-%m-%d"),
                        end=(requested_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        interval=cfg["i"], progress=False,
                    )
                if isinstance(raw_backtest.columns, pd.MultiIndex):
                    raw_backtest.columns = raw_backtest.columns.get_level_values(0)
                if not raw_backtest.empty and raw_backtest.index.tz is not None:
                    raw_backtest.index = raw_backtest.index.tz_localize(None)
                if raw_backtest.empty:
                    df_backtest = df_chart.iloc[0:0].copy()
                    st.warning("資料來源沒有回傳這段期間的 K 線；請確認日期範圍或目前週期的歷史資料限制。")
                else:
                    df_backtest, _, _ = calculate_chip_and_backtest(
                        selected_stock, raw_backtest, selected_tf
                    )
                    df_backtest = _slice_history_for_date_range(
                        df_backtest, bt_start_date, bt_end_date
                    )
            except Exception as period_download_error:
                df_backtest = df_chart.iloc[0:0].copy()
                st.warning(f"指定期間資料下載失敗：{period_download_error}")
        else:
            df_backtest = _slice_history_for_date_range(
                df_chart, bt_start_date, bt_end_date
            )

        if "分鐘" in selected_tf:
            backtest_date_strings = df_backtest.index.strftime("%m-%d %H:%M").tolist()
        else:
            backtest_date_strings = df_backtest.index.strftime("%Y-%m-%d").tolist()

        entry_conds = {
            'ma_mode': entry_ma_mode,
            'ma_p': entry_ma_p,
            'ma_cross_pairs': entry_ma_cross_pairs,
            'five_ma_mode': entry_five_ma_mode,
            'kd_mode': entry_kd_mode,
            'macd_mode': entry_macd_mode,
            'vol_mode': entry_vol_mode,
            'rsi_mode': entry_rsi_mode,
            'bb_mode': entry_bb_mode,
            'bias_mode': entry_bias_mode,
            'atr_mode': entry_atr_mode,
            'atr_pct_limit': float(entry_atr_pct_limit),
            'obv_mode': entry_obv_mode,
            'cci_mode': entry_cci_mode,
            'adx_mode': entry_adx_mode,
            'adx_threshold': float(entry_adx_threshold),
            'ema_mode': entry_ema_mode,
            'match_mode': 'ALL' if "同時" in entry_match_mode else 'ANY'
        }

        exit_conds = {
            'stop_loss_pct': stop_loss_pct,
            'trailing_stop_pct': trailing_stop_pct,
            'ma_mode': exit_ma_mode,
            'ma_p': exit_ma_p,
            'ma_cross_pairs': exit_ma_cross_pairs,
            'five_ma_mode': exit_five_ma_mode,
            'kd_mode': exit_kd_mode,
            'macd_mode': exit_macd_mode,
            'vol_mode': exit_vol_mode,
            'rsi_mode': exit_rsi_mode,
            'bb_mode': exit_bb_mode,
            'bias_mode': exit_bias_mode,
            'atr_stop_multiple': float(exit_atr_stop_multiple),
            'obv_mode': exit_obv_mode,
            'cci_mode': exit_cci_mode,
            'adx_mode': exit_adx_mode,
            'adx_threshold': float(exit_adx_threshold),
            'ema_mode': exit_ema_mode,
        }

        df_calc = compute_kline_indicator_frame(
            df_backtest, ma_entry_p=entry_ma_p, ma_exit_p=exit_ma_p,
            entry_ma_cross_pairs=entry_ma_cross_pairs,
            exit_ma_cross_pairs=exit_ma_cross_pairs,
        )
        entry_cross_counts = _count_ma_cross_events(
            df_calc, entry_ma_cross_pairs, "golden"
        )
        exit_cross_counts = _count_ma_cross_events(
            df_calc, exit_ma_cross_pairs, "death"
        )
        df_trades = run_strategy_backtest(
            df_calc, entry_conds, exit_conds, backtest_date_strings,
            risk_reward_ratio=float(bt_rr_multiple) if bt_rr_enabled else None,
        )
        analysis = analyze_indicator_backtest(
            df_calc, df_trades, float(bt_initial_capital), float(bt_commission_pct),
            float(bt_sell_tax_pct), float(bt_slippage_pct), selected_tf, is_tw_stock)

        if compare_execution:
            st.markdown("#### MA5／MA20 成交時點比較")
            try:
                comparison = compare_ma_execution_timing(
                    df_backtest, (bt_initial_capital, bt_commission_pct, bt_sell_tax_pct, bt_slippage_pct), is_tw_stock,
                )
            except ValueError as error:
                st.info(str(error))
            else:
                rows = []
                for label, trades, metrics in comparison:
                    open_position = (
                        not trades.empty and str(trades["離場觸發原因"].iloc[-1]).startswith("期末未平倉")
                    )
                    rows.append({
                        "成交時點": label,
                        "已平倉筆數": len(trades) - int(open_position),
                        "成本後累積報酬": f"{metrics['cumulative_return']:+.2%}",
                        "最大回撤": f"{metrics['mdd']:.2%}",
                        "期末資產": f"{metrics['ending_value']:,.2f}",
                        "最後一日待成交": trades.attrs.get("unfilled_signal") or "—",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                st.caption("期末仍持有的部位按末日收盤估值，並估算賣出成本。末日訊號缺少下一根開盤時不成交。")

        if df_trades.empty:
            st.info("這段期間沒有買賣訊號；下方仍會顯示策略資金變化，以及直接買進後持有的參考結果。")
            if entry_ma_cross_pairs:
                cross_count_summary = "、".join(
                    f"MA{fast}/MA{slow}：{count} 次"
                    for fast, slow, count in entry_cross_counts
                )
                st.caption(f"所選日期內黃金交叉次數：{cross_count_summary}。交叉只在快線穿越慢線的當根 K 棒成立。")
                if not any(count for _fast, _slow, count in entry_cross_counts):
                    st.warning("所選日期範圍內沒有出現設定的黃金交叉；可延長回測期間、縮短均線根數，或確認目前 K 線週期。")
                else:
                    other_entry_enabled = any((
                        entry_ma_mode != "停用", entry_kd_mode != "停用",
                        entry_macd_mode != "停用", entry_vol_mode != "停用",
                        entry_rsi_mode != "停用", entry_bb_mode != "停用",
                        entry_bias_mode != "停用", entry_atr_mode != "停用",
                        entry_obv_mode != "停用", entry_cci_mode != "停用",
                        entry_adx_mode != "停用", entry_ema_mode != "停用",
                        entry_five_ma_mode != "停用",
                    )) or bool(entry_ma_cross_pairs)
                    if entry_conds["match_mode"] == "ALL" and other_entry_enabled:
                        st.warning("期間內有黃金交叉，但買進判定為 ALL，還必須同一根 K 棒符合其他啟用的買進條件。若只想測交叉，請把原均線策略及其他買進條件設為「停用」；若希望任一條件成立就買進，請改成 ANY。")
            if exit_ma_cross_pairs:
                cross_count_summary = "、".join(
                    f"MA{fast}/MA{slow}：{count} 次"
                    for fast, slow, count in exit_cross_counts
                )
                st.caption(f"所選日期內死亡交叉次數：{cross_count_summary}。若沒有任何買進訊號，平倉交叉不會單獨產生交易。")
        else:
            trade_display = df_trades.drop(columns=[
                "_entry_index", "_exit_index", "_entry_fill_price", "_exit_fill_price"
            ], errors="ignore").rename(columns={
                "平倉報酬率 (%)": "未扣費用損益率 (%)",
                "持有K棒數": "持有資料筆數",
            })
            if analysis is not None and not analysis["trade_analysis"].empty:
                trade_display = trade_display.copy()
                trade_display["扣除費用後損益 (%)"] = analysis["trade_analysis"]["成本後報酬率 (%)"].round(2).to_numpy()
            st.markdown("#### 逐筆交易結果")
            st.dataframe(trade_display, use_container_width=True, hide_index=True)
            st.caption("每列是一筆買進到賣出的模擬紀錄；損益率已扣除上方設定的交易費用。期末仍持有的股票，以最後收盤價估算。")

        if analysis is None:
            st.warning("價格資料不足或有缺漏，暫時無法計算這段回測結果。")
        else:
            strategy_metrics = analysis["strategy_metrics"]
            benchmark_metrics = analysis["benchmark_metrics"]
            st.markdown("#### 回測結果重點")
            st.caption("以下是依目前設定回看歷史資料的模擬結果，不代表未來表現；「—」表示資料不足或該項目不適用。")

            def show_pct(value, signed=False):
                if value is None or not np.isfinite(value):
                    return "—"
                return f"{value:+.2%}" if signed else f"{value:.2%}"

            def show_number(value, digits=2):
                return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

            def show_percent_points(value):
                return "—" if value is None or not np.isfinite(value) else f"{value:+.2f}%"

            backtest_dca_result = None
            try:
                dca_prices = pd.to_numeric(df_calc["Close"], errors="coerce").dropna()
                dca_dates = pd.to_datetime(dca_prices.index, errors="coerce")
                dca_month_count = max(1, int(dca_dates[~dca_dates.isna()].to_period("M").nunique()))
                dca_monthly_amount = float(bt_initial_capital) / dca_month_count
                dca_buy_cost = (float(bt_commission_pct) + float(bt_slippage_pct)) / 100.0
                dca_exit_cost = (float(bt_commission_pct) + float(bt_sell_tax_pct) + float(bt_slippage_pct)) / 100.0
                backtest_dca_result = simulate_monthly_dca(
                    dca_prices, dca_monthly_amount, commission_rate=dca_buy_cost,
                    exit_cost_rate=dca_exit_cost,
                )
            except Exception as dca_compare_error:
                st.caption(f"定期定額參考結果暫時無法計算：{dca_compare_error}")

            metric_row1 = st.columns(4)
            excess_return = strategy_metrics["cumulative_return"] - benchmark_metrics["cumulative_return"]
            metric_row1[0].metric("整段期間損益（已扣費用）", show_pct(strategy_metrics["cumulative_return"], True),
                                  delta=f"比直接持有多 {excess_return:+.2%}" if excess_return >= 0 else f"比直接持有少 {abs(excess_return):.2%}",
                                  help="從回測開始到結束，資金總共增加或減少的比例；已扣除設定的交易費用。")
            metric_row1[1].metric("平均每年報酬", show_pct(strategy_metrics["cagr"], True),
                                  help="把整段期間的資金變化換算成平均每年增減幅度，方便比較不同長度的回測；不是每一年都會有這個報酬。")
            metric_row1[2].metric("從高點算起最大跌幅", show_pct(strategy_metrics["mdd"]),
                                  help="回測期間曾經從資產高點跌到多低。數字越接近零，代表歷史上最深的跌幅越小。")
            metric_row1[3].metric("夏普比率 Sharpe", show_number(strategy_metrics["sharpe"]),
                                  help="把歷史報酬和整體起伏一起衡量，再換算成年化；本頁以 0% 無風險利率估算。數值較高代表過去每承受相同程度起伏時，換到的報酬較多，不保證未來。")
            metric_row2 = st.columns(4)
            metric_row2[0].metric("索提諾比率 Sortino", show_number(strategy_metrics["sortino"]),
                                  help="平均報酬相對於下跌波動的比值並年化；只把低於 0 的報酬視為下行風險，並非虧損機率。")
            metric_row2[1].metric("賺錢交易比例", show_pct(strategy_metrics["win_rate"]),
                                  help="所有交易中，扣除費用後仍賺錢的比例。比例高不一定代表總收益高，還要一起看每次賺多少、賠多少。")
            metric_row2[2].metric("賺賠金額比", show_number(strategy_metrics["profit_factor"]),
                                  help="比較賺錢交易和賠錢交易的金額總和。若數值是 1.5，代表歷史模擬中每賠 1 元，約賺 1.5 元；已扣設定費用。沒有賠錢交易時不顯示。")
            metric_row2[3].metric("持有股票的時間比例", show_pct(strategy_metrics["exposure"]),
                                  help="回測期間有持有股票的時間占全部時間多少；其餘時間策略資金留在現金。")

            st.markdown("#### 回測策略與長期持有、定期定額及網格比較")
            comparison_rows = [
                {"做法": "依訊號買進或賣出", "整段期間損益(%)": strategy_metrics["cumulative_return"] * 100,
                 "平均每年報酬(%)": strategy_metrics["cagr"] * 100 if strategy_metrics["cagr"] is not None else None,
                 "從高點算起最大跌幅(%)": strategy_metrics["mdd"] * 100,
                 "資金起伏幅度(年化 %)": strategy_metrics["annualized_volatility"] * 100 if strategy_metrics["annualized_volatility"] is not None else None,
                 "夏普比率（0%利率）": strategy_metrics["sharpe"], "結束時資產(元)": strategy_metrics["ending_value"]},
                {"做法": "一開始買進並持有", "整段期間損益(%)": benchmark_metrics["cumulative_return"] * 100,
                 "平均每年報酬(%)": benchmark_metrics["cagr"] * 100 if benchmark_metrics["cagr"] is not None else None,
                 "從高點算起最大跌幅(%)": benchmark_metrics["mdd"] * 100,
                 "資金起伏幅度(年化 %)": benchmark_metrics["annualized_volatility"] * 100 if benchmark_metrics["annualized_volatility"] is not None else None,
                 "夏普比率（0%利率）": benchmark_metrics["sharpe"], "結束時資產(元)": benchmark_metrics["ending_value"]},
            ]
            if backtest_dca_result is not None:
                dca_metrics = backtest_dca_result["metrics"]
                dca_return = dca_metrics["ending_value"] / dca_metrics["total_contributed"] - 1.0 if dca_metrics["total_contributed"] > 0 else None
                comparison_rows.append({
                    "做法": "每月定期定額（同總投入）",
                    "整段期間損益(%)": dca_return * 100 if dca_return is not None else None,
                    "平均每年報酬(%)": dca_metrics["money_weighted_return"] * 100 if dca_metrics["money_weighted_return"] is not None else None,
                    "從高點算起最大跌幅(%)": dca_metrics["mdd"] * 100,
                    "資金起伏幅度(年化 %)": None,
                     "夏普比率（0%利率）": None,
                    "結束時資產(元)": dca_metrics["ending_value"],
                })
            if grid_advice_metrics is not None:
                comparison_rows.append({
                    "做法": "網格（依比較頁設定）",
                    "整段期間損益(%)": grid_advice_metrics["cumulative_return"] * 100,
                    "平均每年報酬(%)": grid_advice_metrics["cagr"] * 100 if grid_advice_metrics.get("cagr") is not None else None,
                    "從高點算起最大跌幅(%)": grid_advice_metrics["mdd"] * 100,
                    "資金起伏幅度(年化 %)": None,
                     "夏普比率（0%利率）": None,
                    "結束時資產(元)": float(bt_initial_capital) * (1.0 + grid_advice_metrics["cumulative_return"]),
                })
            comparison = pd.DataFrame(comparison_rows)
            comparison_formats = {
                "整段期間損益(%)": "{:+.2f}%",
                "平均每年報酬(%)": "{:+.2f}%",
                "從高點算起最大跌幅(%)": "{:.2f}%",
                "資金起伏幅度(年化 %)": "{:.2f}%",
                "夏普比率（0%利率）": "{:.2f}",
                "結束時資產(元)": "{:,.0f}",
            }
            comparison_formatters = {
                column: (lambda value, spec=spec: _format_optional_cell(value, spec))
                for column, spec in comparison_formats.items()
            }
            st.dataframe(comparison.style.format(comparison_formatters, na_rep="—"),
                         use_container_width=True, hide_index=True)
            st.caption("表中年化波動率是報酬起伏幅度，夏普比率是每單位起伏對應的年化報酬參考，CAGR 是整段複利換算成年報酬。策略與買進持有使用相同本金、期間和回測成本。定期定額把相同總預算分次投入，並將期末賣出成本計入；其年化報酬按實際投入日期計算，最大跌幅扣除新增投入影響。網格報酬與最大跌幅沿用網格比較頁設定，期末資產按回測本金等比例換算；請確認期間與成本相同再比較。股息未計入。")
            if selected_tf in ("1分鐘", "5分鐘", "30分鐘", "60分鐘"):
                st.info("目前資料是短線週期，涵蓋時間不足以代表長期月扣款；定期定額只在資料涵蓋月份的第一筆價格投入。")
            elif selected_tf in ("1季", "半年", "1年"):
                st.info("目前資料是季線或更長週期，沒有每個月的價格；定期投入只在有資料的月份執行，並非逐月扣款。")

            if not df_trades.empty and not analysis["trade_analysis"].empty:
                reason_frame = pd.DataFrame({
                    "離場時觸發的條件": df_trades["離場觸發原因"].to_numpy(),
                    "扣除費用後損益 (%)": analysis["trade_analysis"]["成本後報酬率 (%)"].to_numpy(),
                })
                reason_summary = reason_frame.groupby("離場時觸發的條件", dropna=False).agg(
                    交易筆數=("扣除費用後損益 (%)", "size"),
                    賺錢交易比例=("扣除費用後損益 (%)", lambda values: float((values > 0).mean())),
                    每筆平均損益=("扣除費用後損益 (%)", "mean"),
                ).reset_index()
                st.markdown("#### 不同離場條件下的交易結果")
                st.dataframe(reason_summary.style.format({"賺錢交易比例": "{:.1%}", "每筆平均損益": "{:+.2f}%"}),
                             use_container_width=True, hide_index=True)
                st.caption("交易筆數是符合該離場條件的次數；賺錢交易比例是扣費後仍賺錢的占比；每筆平均損益是這些交易平均賺或賠的幅度。同一筆交易若同時觸發數個條件，會合併歸類，這只是整理結果，不代表某一條件單獨造成盈虧。")

            st.markdown("#### 費用增加後，結果可能如何變化")
            cost_rows = []
            for scenario_name, multiplier in (("設定成本", 1.0), ("成本提高 50%", 1.5), ("成本加倍", 2.0)):
                scenario_result = analyze_indicator_backtest(
                    df_calc, df_trades, float(bt_initial_capital),
                    float(bt_commission_pct) * multiplier,
                    float(bt_sell_tax_pct) * multiplier,
                    float(bt_slippage_pct) * multiplier,
                    selected_tf, is_tw_stock)
                if scenario_result is not None:
                    scenario_metrics = scenario_result["strategy_metrics"]
                    cost_rows.append({
                        "費用假設": scenario_name,
                        "整段期間損益(%)": scenario_metrics["cumulative_return"] * 100,
                        "從高點算起最大跌幅(%)": scenario_metrics["mdd"] * 100,
                        "結束時資產(元)": scenario_metrics["ending_value"],
                    })
            if cost_rows:
                st.dataframe(pd.DataFrame(cost_rows).style.format({
                    "整段期間損益(%)": "{:+.2f}%", "從高點算起最大跌幅(%)": "{:.2f}%", "結束時資產(元)": "{:,.0f}",
                }), use_container_width=True, hide_index=True)
                st.caption("這裡固定使用同一批買賣訊號，只調高手續費、稅費和滑價假設；用來觀察費用變貴時，獲利可能縮水多少。")

            oos_advice_metrics = None
            if len(df_calc) >= 40:
                split_index = int(len(df_calc) * 0.7)
                period_rows = []
                if split_index >= 2 and len(df_calc) - split_index >= 2:
                    if not df_trades.empty and {"_entry_index", "_exit_index"}.issubset(df_trades.columns):
                        in_sample_trades = df_trades[
                            (df_trades["_entry_index"] >= 0) & (df_trades["_exit_index"] < split_index)
                        ].copy()
                        oos_trades = df_trades[
                            (df_trades["_entry_index"] >= split_index) &
                            (df_trades["_exit_index"] < len(df_calc))
                        ].copy()
                    else:
                        in_sample_trades = pd.DataFrame()
                        oos_trades = pd.DataFrame()
                    oos_trades["_entry_index"] = oos_trades.get("_entry_index", pd.Series(dtype=float)) - split_index
                    oos_trades["_exit_index"] = oos_trades.get("_exit_index", pd.Series(dtype=float)) - split_index
                    for period_name, period_data, period_trades in (
                        ("較早的前 70% 資料", df_calc.iloc[:split_index], in_sample_trades),
                        ("較晚的後 30% 資料", df_calc.iloc[split_index:], oos_trades),
                    ):
                        period_result = analyze_indicator_backtest(
                            period_data, period_trades, float(bt_initial_capital),
                            float(bt_commission_pct), float(bt_sell_tax_pct), float(bt_slippage_pct),
                            selected_tf, is_tw_stock)
                        if period_result is not None:
                            period_metrics = period_result["strategy_metrics"]
                            if period_name == "較晚的後 30% 資料":
                                oos_advice_metrics = period_metrics.copy()
                            period_rows.append({
                                "期間": period_name,
                                "整段期間損益(%)": period_metrics["cumulative_return"] * 100,
                                "平均每年報酬(%)": period_metrics["cagr"] * 100 if period_metrics["cagr"] is not None else None,
                                "從高點算起最大跌幅(%)": period_metrics["mdd"] * 100,
                                "交易筆數": period_metrics["trade_count"],
                                "賺錢交易比例": period_metrics["win_rate"],
                            })
                if period_rows:
                    st.markdown("#### 較早資料和較晚資料的結果比較")
                    period_formats = {
                        "整段期間損益(%)": "{:+.2f}%",
                        "平均每年報酬(%)": "{:+.2f}%",
                        "從高點算起最大跌幅(%)": "{:.2f}%",
                        "賺錢交易比例": "{:.1%}",
                    }
                    period_formatters = {
                        column: (lambda value, spec=spec: _format_optional_cell(value, spec))
                        for column, spec in period_formats.items()
                    }
                    st.dataframe(pd.DataFrame(period_rows).style.format(period_formatters, na_rep="—"),
                                 use_container_width=True, hide_index=True)
                    st.caption(f"以 {df_calc.index[split_index]:%Y-%m-%d} 分成前後兩段，各段從相同本金重新計算；跨過分界日的交易不列入。表中的損益、最大跌幅、交易筆數和賺錢比例，分別表示整體賺賠、曾經最深跌幅、買賣次數和賺錢交易占比。後段用來看看同一套設定在較晚資料的表現，但設定沒有另外調整，因此不代表未來一定能維持相同結果。")

            dca_advice_metrics = backtest_dca_result["metrics"] if backtest_dca_result is not None else None
            backtest_advice = build_backtest_coach_advice(
                strategy_metrics, benchmark_metrics, dca_advice_metrics,
                grid_advice_metrics, oos_advice_metrics,
            )
            condition_label = describe_backtest_conditions(
                entry_conds, exit_conds,
                float(bt_rr_multiple) if bt_rr_enabled else None,
            )
            st.markdown("#### 🧭 回測後方向參考")
            st.info(f"**判讀：{backtest_advice['status']}**\n\n" + "\n\n".join(f"- {note}" for note in backtest_advice["notes"]))
            if explanation_style.startswith("生動童趣"):
                child_backtest_explanations = {
                    "可優先列入觀察": "這套方法在舊資料練習中有些表現不錯，可以繼續觀察；但舊比賽贏過，不保證下一場也贏。",
                    "繼續觀察": "目前有一些線索，但還有問號；先多看幾段資料，不急著照做。",
                    "樣本不足": "練習的買賣次數太少，還看不出這套方法穩不穩。",
                    "暫不建議直接採用": "舊資料裡這套方法有明顯弱點，先不要只靠它決定進出場。",
                }
                st.success(
                    "🧩 用生活比喻說："
                    + child_backtest_explanations.get(
                        backtest_advice["status"],
                        "這是舊資料的練習結果，不能保證下一次會一樣。",
                    )
                )
            st.caption("提示會一起參考交易筆數、賺錢交易比例、賺賠金額比、每筆平均損益、同期比較和較晚期間。40%／55% 勝率、20／30 筆、較晚期間 10 筆及 1.2 賺賠比只是畫面提示門檻，不是保證獲利標準。網格比較沿用另一頁的資金與成本設定；請先確認設定相同。")

            dca_total_return = None
            if dca_advice_metrics and dca_advice_metrics.get("total_contributed", 0) > 0:
                dca_total_return = (dca_advice_metrics["ending_value"] / dca_advice_metrics["total_contributed"] - 1.0) * 100.0
            grid_total_return_for_advice = grid_advice_metrics["cumulative_return"] * 100.0 if grid_advice_metrics else None
            advice_record = {
                "ticker": selected_stock,
                "stock_name": c_name,
                "condition": condition_label,
                "timeframe": selected_tf,
                "period": f"{df_calc.index[0]:%Y-%m-%d}～{df_calc.index[-1]:%Y-%m-%d}",
                "trade_count": int(strategy_metrics["trade_count"]),
                "win_rate_pct": strategy_metrics["win_rate"] * 100.0 if strategy_metrics["win_rate"] is not None else None,
                "total_return_pct": strategy_metrics["cumulative_return"] * 100.0,
                "profit_factor": strategy_metrics["profit_factor"],
                "buy_hold_pct": benchmark_metrics["cumulative_return"] * 100.0,
                "dca_pct": dca_total_return,
                "grid_pct": grid_total_return_for_advice,
                "oos_pct": oos_advice_metrics["cumulative_return"] * 100.0 if oos_advice_metrics else None,
                "status": backtest_advice["status"],
                "advice": "；".join(backtest_advice["notes"][:2]),
            }
            history_key = (advice_record["ticker"], advice_record["timeframe"], advice_record["period"], advice_record["condition"])
            history = list(st.session_state.get("trader_backtest_history", []))
            history = [item for item in history if (
                item.get("ticker"), item.get("timeframe"), item.get("period"), item.get("condition")
            ) != history_key]
            history.append(advice_record)
            st.session_state["trader_backtest_history"] = history[-50:]

            st.markdown("#### 資產變化和從高點下跌的幅度")
            equity_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                       row_heights=[0.68, 0.32],
                                       subplot_titles=("扣除費用後的資產變化", "距離過去高點的跌幅"))
            equity_fig.add_trace(go.Scatter(x=analysis["strategy_equity"].index,
                                           y=analysis["strategy_equity"].values,
                                           name="依訊號買賣", line=dict(color="#2E86DE", width=2)), row=1, col=1)
            equity_fig.add_trace(go.Scatter(x=analysis["benchmark_equity"].index,
                                           y=analysis["benchmark_equity"].values,
                                           name="直接買進並持有", line=dict(color="#7F8C8D", width=1.5)), row=1, col=1)
            if backtest_dca_result is not None:
                equity_fig.add_trace(go.Scatter(
                    x=backtest_dca_result["history"].index,
                    y=backtest_dca_result["history"]["portfolio_value"].values,
                    name="每月定期定額", line=dict(color="#4C78A8", width=1.5, dash="dot")), row=1, col=1)
            equity_fig.add_trace(go.Scatter(x=analysis["strategy_drawdown"].index,
                                           y=analysis["strategy_drawdown"].values * 100,
                                           name="低於過去高點的幅度", fill="tozeroy",
                                           line=dict(color="#E74C3C", width=1.5)), row=2, col=1)
            equity_fig.update_yaxes(title_text="資產金額（元）", row=1, col=1)
            equity_fig.update_yaxes(title_text="比過去高點低（%）", ticksuffix="%", row=2, col=1)
            equity_fig.update_layout(height=560, hovermode="x unified", legend=dict(orientation="h", y=1.08),
                                     margin=dict(l=20, r=20, t=65, b=20))
            st.plotly_chart(equity_fig, use_container_width=True, config={"responsive": True})
            st.caption("上圖比較策略資產和直接買進持有的資產變化；下圖顯示策略資產比之前最高點低多少，越往下代表當時跌得越深。")

            period_returns = analysis["period_returns"].dropna()
            if not period_returns.empty:
                period_frame = period_returns.rename("期間報酬").to_frame()
                period_frame["年度"] = period_frame.index.year
                period_type = analysis["period_type"]
                if period_type == "quarter":
                    period_frame["期間"] = period_frame.index.quarter
                    period_axis = [f"{period}季" for period in range(1, analysis["period_count"] + 1)]
                elif period_type == "half_year":
                    period_frame["期間"] = (period_frame.index.month - 1) // 6 + 1
                    period_axis = ["上半年", "下半年"]
                elif period_type == "year":
                    period_frame["期間"] = 1
                    period_axis = ["年度"]
                else:
                    period_frame["期間"] = period_frame.index.month
                    period_axis = [f"{period}月" for period in range(1, analysis["period_count"] + 1)]
                monthly_pivot = period_frame.pivot(index="年度", columns="期間", values="期間報酬").reindex(columns=range(1, analysis["period_count"] + 1))
                st.markdown(f"#### 每期報酬顏色表（{analysis['period_label']}）")
                heatmap = go.Figure(go.Heatmap(
                    z=monthly_pivot.to_numpy() * 100,
                    x=period_axis,
                    y=[str(year) for year in monthly_pivot.index],
                    colorscale="RdYlGn", zmid=0,
                    colorbar=dict(title="報酬 %"),
                    hovertemplate="%{y} %{x}<br>報酬 %{z:+.2f}%<extra></extra>",
                ))
                heatmap.update_layout(height=max(250, 48 * len(monthly_pivot) + 100),
                                      margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(heatmap, use_container_width=True, config={"responsive": True})
                st.caption("每格代表該月、該季、該半年或該年度的資產變化；綠色表示上漲，紅色表示下跌，顏色越深代表幅度越大。沒有顏色表示該期沒有資料。")

            st.caption(
                f"交易筆數 {strategy_metrics['trade_count']}；每筆平均損益 {show_percent_points(strategy_metrics['expectancy_pct'])}；"
                f"平均賺錢幅度 {show_percent_points(strategy_metrics['average_win_pct'])}；平均賠錢幅度 {show_percent_points(strategy_metrics['average_loss_pct'])}；"
                f"平均賺賠幅度比 {show_number(strategy_metrics['payoff_ratio'])}；最多連續賠錢 {strategy_metrics['longest_loss_streak']} 筆；"
                f"最長未回到先前高點的時間 {strategy_metrics['longest_drawdown_bars']} 根走勢資料。"
            )
            if strategy_metrics["cagr"] is None:
                st.info("回測時間跨度或有效資料筆數不足，為避免年化數字失真，平均每年報酬和報酬起伏參考數值暫不顯示。")
            st.caption(
                "模擬假設所有本金可投入且可買零碎股；指標訊號以收盤資料確認並假設同一根收盤價成交，屬理想化假設，實際下單價可能不同。啟用風險報酬停利時，停損／停利以當根最高價、最低價判斷並以觸發價成交；若開盤跳空越過觸發價，按開盤價估算。"
                "同一根 K 棒同時碰到停損與停利時，採較保守的停損先出；日線無法還原當日價格先後。直接買進持有使用相同價格資料，沒有另外估算股息。"
            )
