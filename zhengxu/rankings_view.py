"""Institutional market flow rankings and historical appearances."""

import pandas as pd
import streamlit as st

from .market_data import (
    STOCK_NAME_MAP,
    build_institutional_rankings,
    fetch_institutional_market_data,
    institutional_top10_sort_specs,
    load_institutional_top10_consecutive_appearances,
    save_institutional_top10_snapshot,
)


def render_rankings(_render_stock_actions):
    st.markdown("#### 當日全市場三大法人佔股本比 Top 10")
    st.caption("表格的百分比把法人當日買進、賣出或淨流向，換成相對公司已發行股數的規模，並非法人持股比例。正值代表買進多於賣出，負值代表賣出多於買進。榜單列出淨買超較高、淨賣超較高，以及買進或賣出較活躍的前 10 檔；上市與上櫃採各自官方最新資料日。點選股票列，可選擇即時診斷或加入自訂觀察代碼。")
    st.info("🧠 **市場心理重點：**榜單名次容易讓人把「排前面」誤認成「比較會漲」。單日交易規模不代表持倉變化或未來方向；先看資料日期，再觀察是否連續出現，以及價格有沒有同步反應。")
    if st.button("📊 載入全市場法人 Top 10", key='load_inst_top10') or st.session_state.get('inst_top10_loaded', False):
        st.session_state['inst_top10_loaded'] = True
        with st.spinner("讀取上市、上櫃法人資料及官方已發行股數…"):
            market_trades, market_shares, market_errors = fetch_institutional_market_data()
        ranking_data = build_institutional_rankings(market_trades, market_shares)
        if ranking_data.empty:
            st.info("目前無法取得可排序的法人資料與已發行股數。")
        else:
            ranking_data['股票名稱'] = ranking_data.apply(
                lambda row: row.get('股票名稱') if row.get('股票名稱') and row.get('股票名稱') != row.get('股票代碼')
                else STOCK_NAME_MAP.get(f"{row.get('股票代碼')}.TWO" if row.get('市場') == '上櫃' else f"{row.get('股票代碼')}.TW", row.get('股票代碼')),
                axis=1)
            top10_snapshot_date, top10_saved_count = save_institutional_top10_snapshot(ranking_data)
            if top10_snapshot_date and top10_saved_count:
                st.caption(
                    f"連續上榜觀察已記錄至 {top10_snapshot_date}；同一資料日重載不會重複計數。"
                    "追蹤從本功能開始載入後逐步累積。"
                )
            elif top10_snapshot_date is None:
                st.warning("本次上市或上櫃資料不完整，為避免把不完整榜單算成全市場連續紀錄，本次未新增追蹤快照。")
            else:
                st.warning("本次全市場榜單快照未能保存；連續上榜統計暫不更新。")
            investor_tabs = st.tabs(['外資', '投信', '自營商'])
            for investor, investor_tab in zip(('外資', '投信', '自營商'), investor_tabs):
                with investor_tab:
                    sort_specs = institutional_top10_sort_specs(investor)
                    ranking_cols = st.columns(2)
                    for index, (column, ascending, title) in enumerate(sort_specs):
                        ranked = ranking_data.dropna(subset=[column]).sort_values(column, ascending=ascending).head(10)
                        show = ranked[['股票代碼', '股票名稱', '市場', '資料日期', column]].rename(columns={column:'佔股本比(%)'})
                        show_display = show.copy()
                        show_display['佔股本比(%)'] = show_display['佔股本比(%)'].apply(
                            lambda value: f"{value:+.4f}%" if pd.notna(value) else "—"
                        )
                        with ranking_cols[index % 2]:
                            st.markdown(f"**{title} Top 10**")
                            table_key = f"inst_top10_{investor}_{index}"
                            top_event = st.dataframe(
                                show_display,
                                use_container_width=True,
                                hide_index=True,
                                key=table_key,
                                on_select="rerun",
                                selection_mode="single-row",
                            )
                            top_rows = list(top_event.selection.rows)
                            if top_rows:
                                chosen = show.iloc[int(top_rows[0])]
                                ticker_suffix = ".TWO" if str(chosen["市場"]) == "上櫃" else ".TW"
                                selected_ticker = f"{chosen['股票代碼']}{ticker_suffix}"
                                action_key = f"inst_action_{investor}_{index}"
                                _render_stock_actions(selected_ticker, str(chosen["股票名稱"]), action_key)
                    st.markdown("#### 🔁 連續出現在 Top 10 的個股")
                    st.caption(
                        "同一法人、同一榜單類型，連續出現在本系統已保存的不同全市場資料快照中才列出。"
                        "只計已記錄快照，不代表每個交易日都完整記錄；本功能啟用前的日期不會補造。"
                    )
                    streak_rows = load_institutional_top10_consecutive_appearances(investor, top10_snapshot_date)
                    if streak_rows.empty:
                        st.info("目前尚無連續 2 次已記錄快照仍上榜的個股；系統會從本次及後續成功載入的完整資料逐步累積。")
                    else:
                        streak_display = streak_rows.copy()
                        streak_display['佔股本比(%)'] = streak_display['佔股本比(%)'].map(
                            lambda value: f"{value:+.4f}%" if pd.notna(value) else "—"
                        )
                        st.dataframe(streak_display, use_container_width=True, hide_index=True)
            if market_errors:
                st.caption("部分官方來源未能連線：" + "；".join(market_errors[:2]))
