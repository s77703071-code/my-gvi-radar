"""Dynamic grid planning controls and scenarios."""

import numpy as np
import pandas as pd
import streamlit as st

from .grid import (
    recommend_dynamic_grid,
)
from .market_data import (
    fetch_grid_daily_history,
)


def render_grid(price_val, selected_stock):
    st.markdown("### 🎛️ 智慧型動態網格策略與資金自動規劃器 (3.12 操盤手教學版)")

    with st.expander("📚 點擊展開：網格交易 3 大資金管理機制與適用場景解析", expanded=False):
        st.markdown("""
        ##### **1. 固定比例再平衡 (1:1 / 自訂比例)**
        * **核心機制**：動態維持「股票價值 : 現金」的固定的比例（如 50%:50%）。股票上漲過高即賣出補回現金；股票下跌過深即動用現金買進股票。
        * **優點**：紀律化控制曝險，下跌時永遠有預備現金可用。
        * **缺點**：持續單邊大漲時可能過早賣出；持續單邊大跌時會持續消耗預備現金。
        * **適合場景**：**控制風險**、偏好長期穩定波動、不希望承受全額持股回撤的投資人。

        ##### **2. 固定股數交易**
        * **核心機制**：每次觸發目標價時，買進或賣出「完全相同的股數」（如每次 1,000 股）。
        * **優點**：規則最直觀、容易進行歷史回測與預估手續費。
        * **缺點**：股價越高時每次交易金額越大，資金壓力呈線性成長；股價極低時交易金額過小。
        * **適合場景**：**建立簡單易執行的自動化交易規則**。

        ##### **3. 庫存百分比管理**
        * **核心機制**：每次買賣當前庫存的一定比例（如每次交易當前庫存的 10%）。
        * **優點**：庫存部位較大時調整幅度自動放大；庫存縮小（低位）時調整幅度隨之減少，降低爆倉風險。
        * **缺點**：庫存極低時賣出數量會指數型遞減，可能較難完全出清清倉。
        * **適合場景**：讓交易規模跟隨現有庫存規模動態調節。
        """)

    st.markdown("---")

    grid_p = round(float(price_val), 2)
    is_tw = ".TW" in selected_stock or ".TWO" in selected_stock
    unit_label = "股" if not is_tw else "股 (台股預設)"

    grid_history = fetch_grid_daily_history(selected_stock)
    recommended_lower, recommended_upper, recommended_count, recommendation_reason = recommend_dynamic_grid(grid_p, grid_history)
    if st.session_state.get("_grid_default_stock") != selected_stock:
        st.session_state["_grid_default_stock"] = selected_stock
        st.session_state["grid_lower"] = recommended_lower
        st.session_state["grid_upper"] = recommended_upper
        st.session_state["grid_num"] = recommended_count

    st.info(f"**本檔建議預設理由：**{recommendation_reason}")
    if st.button("恢復本檔建議預設值", key="grid_restore_recommended"):
        st.session_state["grid_lower"] = recommended_lower
        st.session_state["grid_upper"] = recommended_upper
        st.session_state["grid_num"] = recommended_count

    g_top1, g_top2, g_top3 = st.columns(3)
    with g_top1:
        total_capital = st.number_input("💵 投入總資金 (元/$)", value=100000, step=10000, key="grid_capital")
    with g_top2:
        grid_strategy = st.selectbox(
            "🎯 選擇網格核心策略機制",
            options=["1. 固定比例再平衡 (例如 1:1)", "2. 固定股數交易", "3. 庫存百分比管理"],
            key="grid_strat_select"
        )
    with g_top3:
        init_stock_ratio = st.slider("⚖️ 初始股票部位佔比 (%)", min_value=10, max_value=90, value=50, step=5, key="grid_init_ratio")

    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        input_lower = st.number_input("網格下限價格", min_value=0.01, step=0.01, format="%.2f", key="grid_lower")
    with g_col2:
        input_upper = st.number_input("網格上限價格", min_value=0.01, step=0.01, format="%.2f", key="grid_upper")
    with g_col3:
        input_num = st.number_input("規劃總格數", min_value=3, max_value=50, step=1, key="grid_num")

    init_stock_cash = total_capital * (init_stock_ratio / 100.0)
    init_shares = int(init_stock_cash // grid_p) if grid_p > 0 else 0
    actual_init_stock_val = init_shares * grid_p
    actual_init_cash = total_capital - actual_init_stock_val

    st.info(
        f"💡 **初始建倉試算**：總資金 {total_capital:,.0f} 元 | "
        f"建倉購買 **{init_shares:,}** {unit_label} (約 {actual_init_stock_val:,.0f} 元，佔 {actual_init_stock_val/total_capital*100:.1f}%) | "
        f"保留預備現金 **{actual_init_cash:,.0f}** 元"
    )

    levels = np.linspace(input_lower, input_upper, input_num)
    grid_details = []

    buy_level_count = int(np.sum(levels < grid_p))
    sell_level_count = int(np.sum(levels > grid_p))
    reference_level_count = int(input_num - buy_level_count - sell_level_count)
    if input_upper <= grid_p:
        st.warning(f"現價 {grid_p:.2f} 高於或等於網格上限 {input_upper:.2f}，目前沒有賣出價位；這通常是上下限沿用舊股票設定所致，請提高上限或恢復本檔建議預設值。")
    elif input_lower >= grid_p:
        st.warning(f"現價 {grid_p:.2f} 低於或等於網格下限 {input_lower:.2f}，目前沒有買進價位；請降低下限或恢復本檔建議預設值。")
    else:
        st.caption(f"依目前設定，現價下方有 {buy_level_count} 個買進價位、上方有 {sell_level_count} 個賣出價位；規劃格數包含 {reference_level_count} 個現價參考價位（若有）。")

    for idx, level in enumerate(levels):
        action = "🟢 買進掛單" if level < grid_p else ("🔴 賣出掛單" if level > grid_p else "⚪ 現價參考價")

        if "固定比例" in grid_strategy:
            est_stock_val = init_shares * level
            est_total = est_stock_val + actual_init_cash
            target_stock_val = est_total * (init_stock_ratio / 100.0)
            diff_val = target_stock_val - est_stock_val

            if diff_val > 0:
                trade_desc = f"動用現金買進約 {abs(diff_val):,.0f} 元股票"
            elif diff_val < 0:
                trade_desc = f"賣出約 {abs(diff_val):,.0f} 元股票補回現金"
            else:
                trade_desc = "平衡狀態"

        elif "固定股數" in grid_strategy:
            fixed_shares = max(1, int((init_shares or 1000) / input_num))
            trade_desc = f"固定交易 {fixed_shares:,} {unit_label} (約 {fixed_shares * level:,.0f} 元)"

        else:
            pct = round(100.0 / input_num, 1)
            trade_desc = f"交易當前庫存之 {pct}% (隨庫存規模動態增減)"

        grid_details.append({
            "網格層級": f"第 {idx+1:02d} 格",
            "目標觸發價": round(level, 2),
            "偏離現價 (%)": f"{((level - grid_p) / grid_p * 100):+.2f}%",
            "預計動作": action,
            "資金/部位管理機制": trade_desc
        })

    st.dataframe(pd.DataFrame(grid_details), use_container_width=True, height=280)
    st.caption("「網格層級」是價格分段；「目標觸發價」是策略參考價；「偏離現價」顯示距離；「預計動作」與資金管理方式是依上方假設試算，並非實際掛單或保證成交。")
    st.caption("🧠 心理角度：網格把低買高賣變成固定規則，有助減少臨場猶豫；但若價格持續單向移動，使用者可能因錨定原本區間而不願調整，或因害怕錯過而追著改參數。先訂好資金上限和檢視頻率。")

    st.warning(
        "⚠️ **操盤手複利與風險特別提示：**\n"
        "1. **網格獲利 ≠ 長期投資正期望值**：網格在「箱型震盪行情」中最能發揮低買高賣優勢；但在「強勢單邊大漲」時，會因持續賣出股票而落後於買進持有 (Buy & Hold)；在「持續單邊下跌」時，則會逐步買進並累積帳面虧損。\n"
        "2. **交易成本與稅費影響**：頻繁再平衡會產生手續費與證券交易稅，長期運作下會侵蝕部分複利效益。\n"
        "3. **驗證建言**：若目標是建立長期複利系統，請切換至 **「Tab 4 策略自訂回測與網格複利對比」**，檢視歷史 **年化報酬率 (CAGR)**、**最大回撤 (MDD)** 與 **扣除稅費後的淨報酬率**。"
    )
