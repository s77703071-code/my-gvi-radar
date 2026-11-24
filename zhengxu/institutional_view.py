"""Institutional stock screening and streak display."""

import pandas as pd
import streamlit as st

from .market_data import (
    build_institutional_flow_streaks,
    build_institutional_screen,
    dataframe_to_excel_compatible_csv_bytes,
    dataframe_to_excel_workbook_bytes,
    fetch_institutional_market_data,
    fetch_institutional_price_volume_context,
    get_trader_watchlist_symbols,
    interpret_institutional_price_volume,
    record_institutional_strategy_snapshot,
)


def render_institutional():
    st.markdown("---")
    st.markdown("### 🏦 台股三大法人獨立選股策略")
    st.caption("只掃描自訂觀察名單中的台股普通股。百分比是把法人當日交易量放到公司已發行股數的規模中比較；正值代表買進多於賣出，負值代表賣出多於買進。這是當日交易流量，不是法人持股比例，也不代表後續方向。")
    filter_tab, history_tab = st.tabs(("策略篩選", "連續進出"))
    with filter_tab:
        inst_col1, inst_col2 = st.columns([1, 2])
        with inst_col1:
            inst_investor = st.selectbox("篩選法人", ['外資', '投信', '自營商'], key='inst_screen_investor')
            inst_min_net_pct = st.number_input("淨買賣超佔股本至少 (%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.01, format='%.2f', key='inst_min_net_pct')
        with inst_col2:
            st.write("篩選條件：所選法人的當日淨買進規模需達到你設定的門檻。表格仍列出外資、投信、自營商三方的買進、賣出與淨流向規模，方便比較交易方向和強弱。")
            st.caption("🧠 **心理讀法：**單日法人買超容易引發跟隨心理，但它只代表已發生的交易。觀察資料日期、連續性、股價與成交量；不能由買超直接推論法人正在吸籌或股價必漲。")
        if st.button("🏦 執行法人策略篩選", type='secondary', key='run_inst_screen'):
            st.session_state.pop('inst_strategy_flow_streak', None)
            watchlist_symbols = get_trader_watchlist_symbols()
            taiwan_tickers = [t for t in watchlist_symbols if t.endswith(('.TW', '.TWO'))]
            if not taiwan_tickers:
                st.warning("目前自訂觀察清單沒有台股代碼。請先加入上市代碼（例如 2330.TW）或上櫃代碼（例如 3293.TWO）。")
            else:
                with st.spinner("正在取得 TWSE／TPEx 官方法人交易與股數資料…"):
                    inst_trades, inst_shares, inst_errors = fetch_institutional_market_data()
                saved_flow_count = record_institutional_strategy_snapshot(inst_trades, inst_shares, taiwan_tickers)
                inst_result = build_institutional_screen(inst_trades, inst_shares, taiwan_tickers, inst_investor, float(inst_min_net_pct))
                flow_streaks = build_institutional_flow_streaks(
                    taiwan_tickers, inst_trades, inst_shares, inst_investor
                )
                if not flow_streaks.empty:
                    price_volume_contexts = []
                    with st.spinner("補上連續進出個股的近期價量資料…"):
                        for _, streak_row in flow_streaks.iterrows():
                            price_context = fetch_institutional_price_volume_context(
                                streak_row['股票代碼'], streak_row['最新資料日']
                            )
                            price_context['價量心理重點'] = interpret_institutional_price_volume(
                                streak_row['連續方向'],
                                price_context.get('近5日股價變化(%)'),
                                price_context.get('近5日均量變化(%)'),
                            )
                            price_volume_contexts.append(price_context)
                    flow_streaks = pd.concat(
                        [flow_streaks.reset_index(drop=True), pd.DataFrame(price_volume_contexts)],
                        axis=1,
                    )
                st.session_state['inst_strategy_flow_streak'] = {
                    'investor': inst_investor,
                    'watchlist_count': len(set(taiwan_tickers)),
                    'saved_count': saved_flow_count,
                    'data': flow_streaks,
                }
                if inst_result.empty:
                    st.info("此條件下沒有符合標的，或官方資料暫時缺漏。可調低淨買超門檻或檢查代碼格式。")
                else:
                    inst_display_cols = ['股票代碼', '股票名稱', '市場', '資料日期',
                        '外資買進佔股本比(%)', '外資賣出佔股本比(%)', '外資買賣超佔股本比(%)',
                        '投信買進佔股本比(%)', '投信賣出佔股本比(%)', '投信買賣超佔股本比(%)',
                        '自營商買進佔股本比(%)', '自營商賣出佔股本比(%)', '自營商買賣超佔股本比(%)']
                    pct_cols = [c for c in inst_display_cols if c.endswith('(%)')]
                    st.dataframe(inst_result[inst_display_cols].style.format({c: '{:+.4f}%' for c in pct_cols}), use_container_width=True)
                    export_frame = inst_result[inst_display_cols].copy()
                    csv_col, excel_col = st.columns(2)
                    with csv_col:
                        st.download_button(
                            "下載 CSV（UTF-8 中文相容）",
                            data=dataframe_to_excel_compatible_csv_bytes(export_frame),
                            file_name='taiwan_institutional_screen.csv',
                            mime='text/csv; charset=utf-8',
                            key='download_inst_screen',
                        )
                    with excel_col:
                        try:
                            excel_bytes = dataframe_to_excel_workbook_bytes(export_frame)
                            st.download_button(
                                "下載 Excel（直接開啟推薦）",
                                data=excel_bytes,
                                file_name='taiwan_institutional_screen.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                key='download_inst_screen_excel',
                            )
                        except Exception:
                            st.caption("目前環境無法產生 Excel 檔；若 CSV 欄位未自動分開，可用試算表的「匯入文字／CSV」功能選逗號分隔。")
                if inst_errors:
                    st.caption("有官方來源未能連線：" + "；".join(inst_errors[:2]))
    with history_tab:
        saved_flow_streak = st.session_state.get('inst_strategy_flow_streak')
        if isinstance(saved_flow_streak, dict):
            st.markdown("#### 🔁 法人連續進出觀察")
            st.caption(
                f"目前依自訂觀察名單追蹤「{saved_flow_streak.get('investor', '所選法人')}」，"
                f"本次清單 {saved_flow_streak.get('watchlist_count', 0)} 檔，其中 {saved_flow_streak.get('saved_count', 0)} 檔成功保存官方資料。"
                "同一股票同一資料日重跑會更新資料、不增加次數；名單只列出連續至少 2 個已記錄資料日方向相同的股票。"
                "只計系統實際保存的官方資料日，未記錄的交易日不會補造，因此不代表中間每個交易日都連續進出。"
            )
            st.caption("價量補充：股價比較近5個交易日收盤變化，成交量比較最近5日平均與前5日平均；日期未對齊或資料不足時會留白，不以缺值推論方向。")
            streak_table = saved_flow_streak.get('data')
            if not isinstance(streak_table, pd.DataFrame) or streak_table.empty:
                st.info("目前尚無符合條件的連續淨買超或淨賣超個股；請在後續不同官方資料日再次執行，系統才會累積新的觀察。")
            else:
                streak_display = streak_table.copy()
                for column in ('最新單日淨額（股）', '連續期間累計淨額（股）'):
                    streak_display[column] = streak_display[column].map(
                        lambda value: f"{value:+,.0f}" if pd.notna(value) else "—"
                    )
                for column in ('連續期間累計佔股本（%）', '近5日股價變化(%)', '近5日均量變化(%)'):
                    if column in streak_display.columns:
                        streak_display[column] = streak_display[column].map(
                            lambda value: f"{value:+.2f}%" if pd.notna(value) else "—"
                        )
                st.dataframe(streak_display, use_container_width=True, hide_index=True)
