"""Grid, buy-and-hold and periodic investment comparisons."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .grid import (
    simulate_monthly_dca,
)


def render_comparison(c_name, date_strings, df_chart, selected_tf, *, is_tw_stock=True):
    st.markdown(f"### ⚖️ 【{c_name}】網格、買進持有與定期定額歷史比較")
    st.caption("三種方式使用同一標的、價格期間及交易費率。定期定額預設把相同總預算平均分到每個資料月份；年化報酬按每次實際投入日期計算。")

    settings_tab, report_tab = st.tabs(("參數與期間", "比較結果"))
    with settings_tab:
        st.markdown("#### 網格策略、一次買進持有、每月定期定額")

        gc_col1, gc_col2, gc_col3, gc_col4 = st.columns(4)
        with gc_col1:
            sim_capital = st.number_input("💵 模擬初始投入本金 (元/\\$)", min_value=1.0, value=100000.0, step=10000.0, key="sim_cap_input")
        with gc_col2:
            fee_rate = st.number_input("💸 單邊交易手續費率 (%)", min_value=0.0, max_value=5.0, value=0.1425 if is_tw_stock else 0.0,
                                       step=0.01, format="%.4f", key="sim_fee_input" if is_tw_stock else "sim_us_fee_input") / 100.0
        with gc_col3:
            tax_rate = st.number_input("🏛️ 賣出證券交易稅率 (%)", min_value=0.0, max_value=5.0, value=0.3 if is_tw_stock else 0.0,
                                       step=0.01, format="%.4f", key="sim_tax_input" if is_tw_stock else "sim_us_tax_input") / 100.0
        with gc_col4:
            slippage_rate = st.number_input("滑價（每邊，%）", min_value=0.0, max_value=5.0, value=0.05,
                                            step=0.01, key="sim_slippage_input") / 100.0
        buy_cost_rate = fee_rate + slippage_rate
        sell_cost_rate = buy_cost_rate + tax_rate

        g_opt1, g_opt2 = st.columns(2)
        with g_opt1:
            grid_sim_strategy = st.selectbox(
                "🎯 選擇網格模擬核心策略機制",
                options=["1. 固定比例再平衡 (1:1)", "2. 固定股數交易 (每次買賣初始部位 10%)", "3. 庫存百分比管理 (每次買賣現有庫存 10%)"],
                key="sim_grid_strat_select"
            )
        with g_opt2:
            trigger_pct = st.number_input("📉 網格觸發間距 (%)", min_value=1.0, max_value=50.0, value=5.0, step=1.0, key="sim_grid_trigger") / 100.0

        comparison_prices = pd.to_numeric(df_chart["Close"], errors="coerce")
        comparison_index = pd.DatetimeIndex(pd.to_datetime(comparison_prices.index, errors="coerce"))
        price_values = comparison_prices.to_numpy(dtype=float, na_value=np.nan)
        if len(price_values) < 5 or not np.isfinite(price_values).all() or (price_values <= 0).any():
            st.info("完整比較需要至少 5 筆有效且大於零的收盤價。")
            return
        if comparison_index.hasnans or not comparison_index.is_monotonic_increasing or not comparison_index.is_unique:
            st.info("比較需要依時間排序且每個時間點唯一的行情。")
            return
        comparison_months = comparison_index.tz_localize(None).to_period("M").nunique()
        st.caption(f"比較期間 {comparison_index[0]:%Y/%m/%d} 至 {comparison_index[-1]:%Y/%m/%d}，共 {len(comparison_prices)} 筆價格。")
        default_dca_monthly = float(sim_capital) / comparison_months
        previous_default = st.session_state.get("sim_dca_default")
        saved_monthly = st.session_state.get("sim_dca_monthly_input")
        if saved_monthly is None or (previous_default is not None and np.isclose(saved_monthly, previous_default)):
            st.session_state["sim_dca_monthly_input"] = default_dca_monthly
        st.session_state["sim_dca_default"] = default_dca_monthly
        dca_cols = st.columns([2, 1])
        with dca_cols[1]:
            if st.button("恢復相同總預算", key="sim_dca_reset"):
                st.session_state["sim_dca_monthly_input"] = default_dca_monthly
        with dca_cols[0]:
            dca_monthly = st.number_input(
                "💵 定期定額每月投入金額",
                min_value=0.0, step=1_000.0,
                key="sim_dca_monthly_input",
                help="預設把上方模擬本金平均分配至回測期間的每個月份；可自行更改。"
            )
        dca_budget_preview = float(dca_monthly) * comparison_months
        st.caption(f"期間有 {comparison_months} 個資料月份。定期定額累計投入約 {dca_budget_preview:,.0f}。")
        if not np.isclose(dca_budget_preview, float(sim_capital)):
            st.info("目前定期定額的總投入不同。要比較期末金額，可先按「恢復相同總預算」。")
        if selected_tf in ("1分鐘", "5分鐘", "30分鐘", "60分鐘"):
            st.info("目前是短線圖表，資料期間不到完整長期月扣款週期；定期定額只會在資料涵蓋月份的第一筆價格投入。若要比較長期月扣款，請切換日線、週線或月線。")
        elif selected_tf in ("1季", "半年", "1年"):
            st.info("目前圖表是季線或更長週期，沒有每個月的價格；定期投入只能使用每個有資料月份的第一筆價格，並非逐月扣款。若要比較月扣款，請切換日線、週線或月線。")

        closes_arr = price_values

    with report_tab:
        if len(closes_arr) >= 5:
            initial_p = closes_arr[0]
            bh_shares = (sim_capital * (1.0 - buy_cost_rate)) / initial_p
            bh_asset_curve = bh_shares * closes_arr
            bh_market_value = float(bh_asset_curve[-1])
            bh_asset_curve[-1] *= 1.0 - sell_cost_rate
            bh_final_val = float(bh_asset_curve[-1])

            stock_val = sim_capital * 0.5 * (1.0 - buy_cost_rate)
            cash_val = sim_capital * 0.5
            grid_shares = stock_val / initial_p
            grid_asset_curve = []

            last_trade_p = initial_p
            fixed_trade_shares = grid_shares * 0.1 # 策略2：固定交易初始持股的 10%

            for p in closes_arr:
                curr_stock_val = grid_shares * p
                total_val = curr_stock_val + cash_val

                # 網格價格波動觸發邏輯
                if p >= last_trade_p * (1.0 + trigger_pct) or p <= last_trade_p * (1.0 - trigger_pct):
                    if "固定比例" in grid_sim_strategy:
                        target_stock_val = total_val * 0.5
                        diff = target_stock_val - curr_stock_val

                        if diff > 0 and cash_val >= diff: # 買進
                            buy_amt = diff * (1.0 - buy_cost_rate)
                            cash_val -= diff
                            grid_shares += buy_amt / p
                        elif diff < 0 and grid_shares >= abs(diff)/p: # 賣出
                            sell_amt = abs(diff) * (1.0 - sell_cost_rate)
                            cash_val += sell_amt
                            grid_shares -= abs(diff) / p

                    elif "固定股數" in grid_sim_strategy:
                        trade_val = fixed_trade_shares * p
                        if p >= last_trade_p * (1.0 + trigger_pct): # 上漲賣出
                            if grid_shares >= fixed_trade_shares:
                                cash_val += trade_val * (1.0 - sell_cost_rate)
                                grid_shares -= fixed_trade_shares
                        elif p <= last_trade_p * (1.0 - trigger_pct): # 下跌買進
                            if cash_val >= trade_val:
                                cash_val -= trade_val
                                grid_shares += (trade_val * (1.0 - buy_cost_rate)) / p

                    elif "庫存百分比" in grid_sim_strategy:
                        if p >= last_trade_p * (1.0 + trigger_pct): # 上漲賣出現有 10%
                            sell_shares = grid_shares * 0.1
                            if sell_shares > 0:
                                trade_val = sell_shares * p
                                cash_val += trade_val * (1.0 - sell_cost_rate)
                                grid_shares -= sell_shares
                        elif p <= last_trade_p * (1.0 - trigger_pct): # 下跌動用現金 10% 買進
                            use_cash = cash_val * 0.1
                            if use_cash > 0:
                                cash_val -= use_cash
                                grid_shares += (use_cash * (1.0 - buy_cost_rate)) / p

                    last_trade_p = p # 更新最新交易參考價

                grid_asset_curve.append(grid_shares * p + cash_val)

            grid_market_value = float(grid_asset_curve[-1])
            grid_asset_curve[-1] = cash_val + grid_shares * closes_arr[-1] * (1.0 - sell_cost_rate)
            grid_final_val = float(grid_asset_curve[-1])
            grid_span_years = max(
                (pd.Timestamp(df_chart.index[-1]) - pd.Timestamp(df_chart.index[0])).total_seconds()
                / (365.25 * 24 * 60 * 60),
                0.0,
            )
            # Use actual timestamp span for every chart interval; fixed bars/year assumptions
            # badly distorted intraday, weekly, monthly and long-interval CAGR values.
            bh_cagr = (
                ((bh_final_val / float(sim_capital)) ** (1.0 / grid_span_years) - 1.0) * 100.0
                if grid_span_years >= 0.5 and bh_final_val > 0 and float(sim_capital) > 0 else None
            )
            grid_cagr = (
                ((grid_final_val / float(sim_capital)) ** (1.0 / grid_span_years) - 1.0) * 100.0
                if grid_span_years >= 0.5 and grid_final_val > 0 and float(sim_capital) > 0 else None
            )

            dca_result = None
            try:
                dca_series = pd.Series(closes_arr, index=df_chart.index, name="Close")
                dca_result = simulate_monthly_dca(
                    dca_series, float(dca_monthly), buy_cost_rate, sell_cost_rate,
                    deduct_cost_from_contribution=True,
                )
            except Exception as dca_error:
                st.warning(f"定期定額暫時無法計算：{dca_error}")

            def calc_mdd(curve):
                arr = np.array(curve)
                peak = np.maximum.accumulate(arr)
                drawdown = (arr - peak) / peak
                return float(np.min(drawdown)) * 100.0

            bh_mdd = calc_mdd(bh_asset_curve)
            grid_mdd = calc_mdd(grid_asset_curve)
            st.markdown("#### 三種方式結果摘要")
            summary_cols = st.columns(4)
            summary_cols[0].metric("買進持有｜期末扣費後資產", f"{bh_final_val:,.0f} 元", delta=f"損益 {bh_final_val - sim_capital:+,.0f} 元",
                                   help="以最後一筆價格估值，扣除估算的期末賣出費用。損益為扣費後資產減期初本金。")
            summary_cols[1].metric("網格｜期末扣費後資產", f"{grid_final_val:,.0f} 元", delta=f"損益 {grid_final_val - sim_capital:+,.0f} 元",
                                   help="剩餘現金加上持股扣除估算賣出成本後的金額。")
            if dca_result is not None:
                dca_metrics = dca_result["metrics"]
                summary_cols[2].metric("定期定額｜期末扣費後資產", f"{dca_metrics['ending_value']:,.0f} 元", delta=f"損益 {dca_metrics['profit_loss']:+,.0f} 元",
                                       help="期末持股市值扣除估算賣出成本，損益已減去累計投入。")
                summary_cols[3].metric("定期定額累計投入", f"{dca_metrics['total_contributed']:,.0f} 元", delta=f"{dca_metrics['contribution_count']} 次投入",
                                       help="回測期間所有月投入金額的加總；比較期末資產時，請同時考慮累計投入及投入時點。")

            comparison_rows = [
                {"投入方式": "一開始買進並持有", "累計投入(元)": float(sim_capital), "期末市值加現金": bh_market_value, "期末扣費後資產": bh_final_val,
                 "總損益(元)": bh_final_val - float(sim_capital), "年化報酬(%)": bh_cagr,
                 "最大跌幅(%)": bh_mdd},
                {"投入方式": f"網格｜{grid_sim_strategy.split(' ')[1]}", "累計投入(元)": float(sim_capital), "期末市值加現金": grid_market_value, "期末扣費後資產": grid_final_val,
                 "總損益(元)": grid_final_val - float(sim_capital), "年化報酬(%)": grid_cagr,
                 "最大跌幅(%)": grid_mdd},
            ]
            if dca_result is not None:
                dca_metrics = dca_result["metrics"]
                comparison_rows.append({
                    "投入方式": "每月定期定額",
                    "累計投入(元)": dca_metrics["total_contributed"],
                    "期末市值加現金": dca_metrics["ending_value_before_exit_cost"],
                    "期末扣費後資產": dca_metrics["ending_value"],
                    "總損益(元)": dca_metrics["profit_loss"],
                    "年化報酬(%)": dca_metrics["money_weighted_return"] * 100 if dca_metrics["money_weighted_return"] is not None else None,
                    "最大跌幅(%)": dca_metrics["mdd"] * 100,
                })
            st.dataframe(pd.DataFrame(comparison_rows).style.format({
                "累計投入(元)": "{:,.0f}", "期末市值加現金": "{:,.0f}", "期末扣費後資產": "{:,.0f}", "總損益(元)": "{:+,.0f}",
                "年化報酬(%)": "{:+.2f}%", "最大跌幅(%)": "{:.2f}%",
            }, na_rep="—"), use_container_width=True, hide_index=True)
            st.caption("期末市值加現金是扣期末賣出費用前的估值。期末扣費後資產只對剩餘持股估算賣出的手續費、交易稅和滑價。表中的總損益、年化報酬和最大跌幅採扣費後口徑，幣別依標的。定期定額的年化考慮每次投入日期，最大跌幅扣除新增投入的影響。三者均未另計股息。")
            st.caption("🧠 心理角度：買進持有考驗承受回撤的耐心；定期定額可減少一次押時點的壓力，但不會消除下跌；網格提供規則化操作，遇到長期單邊行情卻可能太早賣出或持續加碼。結果要配合自己能承受的過程，而不只挑期末數字最高的做法。")

            strat_display_name = grid_sim_strategy.split(" ")[1]

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x=date_strings, y=bh_asset_curve, mode='lines', name='單純買進持有 (Buy & Hold)', line=dict(color='#00CC66', width=2)))
            fig_compare.add_trace(go.Scatter(x=date_strings, y=grid_asset_curve, mode='lines', name=strat_display_name, line=dict(color='#FF9900', width=2, dash='dash')))
            if dca_result is not None:
                if "分鐘" in selected_tf:
                    dca_axis_dates = dca_result["history"].index.strftime("%m-%d %H:%M").tolist()
                else:
                    dca_axis_dates = dca_result["history"].index.strftime("%Y-%m-%d").tolist()
                fig_compare.add_trace(go.Scatter(
                    x=dca_axis_dates,
                    y=dca_result["history"]["portfolio_value"],
                    mode="lines", name="每月定期定額",
                    line=dict(color="#4C78A8", width=2, dash="dot"),
                ))
            fig_compare.update_layout(
                title=f"📈 【{c_name}】網格、買進持有與定期定額資產變化（期末計入估算賣出成本）",
                height=440,
                xaxis=dict(type='category', showgrid=True, gridcolor="rgba(128,128,128,0.2)", nticks=12),
                yaxis=dict(title="資產總價值 (元)", showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                margin=dict(l=10, r=20, t=40, b=10)
            )
            st.plotly_chart(fig_compare, use_container_width=True, config={"responsive": True})
