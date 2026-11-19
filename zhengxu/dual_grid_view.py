"""Dual-grid screening and backtest controls."""

import pandas as pd
import re
import streamlit as st

from .administration import (
    _require_dual_grid_access,
)
from .grid import (
    _dual_grid_analyze,
    _dual_grid_backtest,
)
from .market_data import (
    _dual_grid_fetch_history,
)


def render_dual_grid(auth_user, selected_stock):
    st.markdown("### 🧠 智慧型動態網格：雙策略池與共用資金風控")
    st.caption("試用版：技術面候選排序＋資金配置草案＋單一標的示意回測。這不是自動下單，也不保證獲利或最大回撤。")
    st.warning("成長型池目前以價格趨勢／動能作為代理指標，並非完整財報品質分析；資料來源為 Yahoo Finance，台股代碼通常使用 .TW 或 .TWO。")

    with st.expander("使用方式與策略規則", expanded=False):
        st.markdown("""
        - **震盪型網格**：偏好 ADX 較低、近期波動適中、價格未明顯單邊趨勢的標的。
        - **成長型網格**：偏好長期均線趨勢向上、近 60／120 日動能為正的標的；核心持有與網格交易資金分開看待。
        - **避免重複配置**：每檔股票只分配到一個主要策略池；分數不足時標記為「暫不納入」。
        - **風控**：各池預算、單檔上限、保留現金與每檔預算均由使用者設定；這些是配置上限，不是停損保證。
        - **回測限制**：示意網格以日收盤價判斷跨格，無法重建盤中成交順序；結果可能受跳空、流動性、滑價及樣本期間影響。
        """)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        tickers_raw = st.text_area(
            "候選股票代碼（逗號或換行分隔）",
            value="2330.TW, 0050.TW, 2317.TW, AAPL, MSFT, NVDA",
            height=95,
            key="dual_grid_ticker_universe",
            help="請先輸入你要篩選的股票池；這個試用版不會自動掃描全市場。",
        )
        scan_period = st.selectbox("篩選資料期間", ["1y", "2y", "5y"], index=1, key="dual_grid_scan_period")
        scan_btn = st.button("🔎 分析候選股票", type="primary", key="dual_grid_scan_button")
    with col_b:
        total_capital = st.number_input("系統總資金（元）", min_value=10000.0, value=1000000.0, step=50000.0, key="dual_grid_total_capital")
        reserve_pct = st.slider("保留現金比例（%）", min_value=10, max_value=80, value=50, step=5, key="dual_grid_reserve_pct")
        range_pool_pct = st.slider("可投資資金中分配給震盪池（%）", min_value=0, max_value=100, value=40, step=5, key="dual_grid_range_pool_pct")
        per_stock_pct = st.slider("單檔最高占總資金（%）", min_value=5, max_value=40, value=15, step=5, key="dual_grid_per_stock_pct")

    if scan_btn:
        _require_dual_grid_access(auth_user)
        tickers = []
        for item in re.split(r"[,，;；\s]+", tickers_raw.upper()):
            item = item.strip()
            if item and item not in tickers:
                tickers.append(item)
        if not tickers:
            st.error("請至少輸入一個有效的股票代碼。")
        elif len(tickers) > 40:
            st.error("一次最多分析 40 檔，避免資料查詢過久。")
        else:
            analyses = []
            progress = st.progress(0)
            for idx, ticker in enumerate(tickers):
                with st.spinner(f"分析 {ticker}（{idx + 1}/{len(tickers)}）"):
                    hist = _dual_grid_fetch_history(ticker, scan_period)
                    item = _dual_grid_analyze(ticker, hist)
                    if item:
                        analyses.append(item)
                progress.progress((idx + 1) / len(tickers))
            st.session_state["dual_grid_analysis_results"] = analyses
            st.session_state["dual_grid_analysis_tickers"] = tickers
            if analyses:
                st.success(f"完成 {len(analyses)} 檔候選股票分析。")
            else:
                st.warning("沒有足夠的日線資料。請檢查代碼、網路連線或更換期間。")

    analyses = st.session_state.get("dual_grid_analysis_results", [])
    if analyses:
        result_df = pd.DataFrame(analyses)
        result_df = result_df.sort_values(["主要策略池", "震盪分數", "成長分數"], ascending=[True, False, False])
        st.markdown("#### 候選股與策略池分配")
        st.dataframe(result_df.round(2), use_container_width=True, hide_index=True)
        pool_counts = result_df["主要策略池"].value_counts()
        m1, m2, m3 = st.columns(3)
        m1.metric("震盪型候選", int(pool_counts.get("震盪型網格", 0)))
        m2.metric("成長型候選", int(pool_counts.get("成長型網格", 0)))
        m3.metric("暫不納入", int(pool_counts.get("暫不納入", 0)))

        investable = total_capital * (1 - reserve_pct / 100)
        range_budget = investable * range_pool_pct / 100
        growth_budget = investable - range_budget
        st.markdown("#### 共用資金配置草案")
        allocation = pd.DataFrame([
            {"資金用途": "保留現金", "預算": total_capital * reserve_pct / 100, "占總資金%": reserve_pct},
            {"資金用途": "震盪型網格池", "預算": range_budget, "占總資金%": range_budget / total_capital * 100},
            {"資金用途": "成長型網格池（含核心持有預算）", "預算": growth_budget, "占總資金%": growth_budget / total_capital * 100},
        ])
        st.dataframe(allocation.round(2), use_container_width=True, hide_index=True)
        st.caption(f"單檔最高配置上限約 {total_capital * per_stock_pct / 100:,.0f} 元。此上限套用在總資金上；同一股票只允許分配到一個主要策略池。實際下單前仍需考量整股／零股規則、手續費最低收費、稅費與匯率。")

        if not (result_df["主要策略池"] != "暫不納入").any():
            st.info("目前沒有分數達標的策略候選；依規則可以選擇暫不交易。")

    st.markdown("#### 單一標的示意回測：網格 vs 買進持有 vs 定期定額")
    st.caption("回測可獨立執行。候選清單僅供選擇股票，近期分數不參與歷史成交判定。")
    backtest_tickers = list(dict.fromkeys([selected_stock] + [item["股票代碼"] for item in analyses]))
    bt_ticker = st.selectbox("回測標的", backtest_tickers, key="dual_grid_bt_ticker")
    bt_last_day = pd.Timestamp.now(tz="Asia/Taipei").date() - pd.Timedelta(days=1)
    bt_date_range = st.date_input(
        "日線資料區間（含校準期）",
        value=(bt_last_day - pd.Timedelta(days=730), bt_last_day),
        max_value=bt_last_day,
        key="dual_grid_bt_dates",
        help="前 60 根有效日線只用於校準。延長資料時，保留相同起始日。",
    )
    bt_capital = st.number_input("回測本金（元）", min_value=10000.0, value=200000.0, step=10000.0, key="dual_grid_bt_capital")
    dca_pct = st.number_input(
        "定期定額每月投入（占初始本金 %）", min_value=0.1, max_value=100.0,
        value=10.0, step=1.0, key="dual_grid_bt_dca_pct",
        help="從評估期首根日線開始，每月第一根有效日線投入固定金額，餘額不足時只投入剩餘金額。",
    )
    fee_pct = st.number_input("單邊手續費率（%）", min_value=0.0, max_value=1.0, value=0.1425, step=0.01, format="%.4f", key="dual_grid_bt_fee_pct")
    tax_pct = st.number_input("賣出交易稅率（%）", min_value=0.0, max_value=2.0, value=0.3, step=0.05, format="%.3f", key="dual_grid_bt_tax_pct", help="台股普通股票賣出稅率常見值；ETF、不同商品及市場可能不同，請自行調整。")
    slippage_pct = st.number_input("單邊滑價假設（%）", min_value=0.0, max_value=2.0, value=0.05, step=0.01, format="%.3f", key="dual_grid_bt_slippage_pct")
    levels_count = st.slider("網格價位數", min_value=3, max_value=15, value=7, step=2, key="dual_grid_bt_levels")
    bt_inputs = (bt_ticker, bt_date_range, bt_capital, dca_pct, fee_pct, tax_pct, slippage_pct, levels_count)
    if st.button("▶️ 執行示意回測", key="dual_grid_run_backtest"):
        _require_dual_grid_access(auth_user)
        st.session_state.pop("dual_grid_backtest_result", None)
        if not bt_date_range or len(bt_date_range) != 2 or bt_date_range[0] >= bt_date_range[1]:
            st.warning("請選擇起日早於迄日的完整資料區間。")
        else:
            start_date, end_date = bt_date_range
            bt_history = _dual_grid_fetch_history(
                bt_ticker, start=start_date, end=end_date + pd.Timedelta(days=1),
            )
            if bt_history.empty:
                st.error("無法取得回測歷史資料。")
            else:
                bt_result = _dual_grid_backtest(
                    bt_history["Close"], bt_capital,
                    fee_pct / 100, tax_pct / 100, slippage_pct / 100, levels_count,
                    dca_monthly_fraction=dca_pct / 100,
                )
                if bt_result is None:
                    st.warning("至少需要 61 根有效日線，前 60 根校準，評估期從第 61 根開始。")
                else:
                    st.session_state["dual_grid_backtest_result"] = (bt_inputs, bt_result)
    saved_backtest = st.session_state.get("dual_grid_backtest_result")
    if saved_backtest and saved_backtest[0] == bt_inputs:
        metrics, chart, params = saved_backtest[1]
        st.dataframe(metrics.round(2), use_container_width=True, hide_index=True)
        st.line_chart(chart.set_index("日期"), use_container_width=True)
        st.caption(
            f"校準期：{params['calibration_start']} 至 {params['calibration_end']}。"
            f"共同評估期：{params['evaluation_start']} 至 {params['evaluation_end']}。"
            f"定期定額每月預定投入 {params['dca_monthly_amount']:,.2f} 元。"
        )
        st.caption(f"網格回測固定參考區間：{params['lower']:.2f}–{params['upper']:.2f}；估算成交 {params['trades']} 次。期間約 {params['years']:.2f} 年。")
        st.caption("三種曲線均包含未投入現金，持股按扣除假設賣出成本後的價值估算。年化報酬以初始本金為基礎。")
        st.warning("這是試用級示意模型：使用收盤價跨越價位模擬，不含盤中撮合、最低手續費、股數整數限制、股息現金流、借券、停牌、跳空的真實成交順序或匯率。歷史調整價格也可能由資料供應商修訂。")

    st.markdown("#### 風控檢查清單")
    st.markdown("- [ ] 確認單檔配置不超過設定上限，且同一股票沒有跨策略重複占用資金。\n- [ ] 檢查總投入、各池預算與保留現金是否相符。\n- [ ] 震盪型標的若出現明顯趨勢／區間失效，重新評估或暫停網格。\n- [ ] 成長型標的若基本面惡化，不能只因價格下跌就持續加碼。\n- [ ] 最大回撤 15% 僅為風險目標，不能保證達成；須用多市場、多期間和樣本外資料驗證。")
