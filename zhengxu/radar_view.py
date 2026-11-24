"""Three-radar stock scanning controls and ranked results."""

import numpy as np
import pandas as pd
import sqlite3
import streamlit as st
import yfinance as yf

from .radar_risk import evaluate_radar_risk, risk_status
from .radars import (
    calculate_fcf_yield,
    calculate_gvi,
    calculate_momentum,
    filter_radar_matches,
    rank_radar_matches,
)

from .chart_tools import (
    _format_optional_cell,
)
from .dividend import (
    _finite_number,
    _free_cash_flow_from_statements,
    _peg_from_company_data,
)
from .market_data import (
    AUTO_TW_UNIVERSE,
    AUTO_US_UNIVERSE,
    STOCK_NAME_MAP,
    TRADER_WATCHLIST_DEFAULTS,
    calculate_chip_and_backtest,
    is_cache_valid,
    parse_roe,
    signature_save_to_db,
)
from .storage import (
    CACHE_TTL,
    DB_FILE,
)


def render_radar(_render_stock_actions, cfg, explanation_style, selected_tf):
    st.markdown("---")
    st.markdown("### 🚀 華爾街機構級三核心策略雷達")
    scan_tab, watch_tab, validation_tab = st.tabs(("掃描與結果", "自訂觀察", "因子驗證"))
    with scan_tab:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            strat_gvi = st.checkbox("開啟 GVI 價值雷達", value=True, key="strat_gvi_check")
            st.caption("研究假設：以 ROE 與淨值資料計算 GVI，並假設目前 ROE 未來五年維持不變；財報更新較慢，產業間可比性有限。")
        with col_s2:
            strat_momentum = st.checkbox("開啟動能突破雷達", value=False, key="strat_momentum_check")
            st.caption("研究假設：以可用歷史高點計算距離。歷史資料少於 50 根時不評分、不納入篩選。")
        with col_s3:
            strat_qarp = st.checkbox("開啟 QARP 現金流雷達", value=False, key="strat_qarp_check")
            st.caption("研究假設：以自由現金流收益率與 PEG 作條件；缺少任一資料時不計入。")
        st.caption(
            f"動能比值比較最近最多 252 根「{selected_tf}」K 棒的最高收盤價；時間軸切換會改變比較期間，"
            "至少 50 根有效收盤資料才計算，不一定等同近一年。"
        )

    with validation_tab:
        st.warning(
            "這些門檻目前是研究假設，尚未證明具有穩定優勢。本版本沒有逐因子樣本外績效報告，選股分數不代表正期望值或買賣建議。"
        )
        validation_status = pd.DataFrame([
            {"因子": "GVI", "驗證狀態": "未驗證", "資料限制": "目前只有當期 ROE／淨值，沒有歷史可得日期的財報快照。"},
            {"因子": "動能", "驗證狀態": "未驗證", "資料限制": "可取歷史價格，但尚未執行獨立 walk-forward 樣本外比較。"},
            {"因子": "QARP", "驗證狀態": "未驗證", "資料限制": "目前只有當期自由現金流／PEG，沒有歷史點時資料。"},
        ])
        st.dataframe(validation_status, use_container_width=True, hide_index=True)
        st.caption(
            "有效比較需以相同股票池與期間，比較未加條件的基準組和逐一加入單一因子的組合；先鎖定參數，再用未參與調參的後段資料檢查超額報酬、回撤、換手與交易成本。"
            "缺少歷史點時財報時不會以當期基本面回填過去，避免前視偏誤。動能採最近最多 252 根所選週期 K 棒，切換時間軸會改變區間。"
        )

        st.info(
            "🧠 **三核心的市場心理讀法**\n\n"
            "**價值（GVI）：**可能吸引重視公司獲利與帳面價值的投資者，但低估值不等於安全。\n\n"
            "**趨勢（動能）：**接近近期高位可能吸引順勢買盤，也要留意追價過度與交易擁擠。\n\n"
            "**現金流與估值（QARP）：**可能吸引重視現金創造力和價格合理性的投資者，仍要查證現金流是否持續、成長預估是否可靠。\n\n"
            "條件分歧時先找出原因，不要只挑支持原有看法的指標；分數只表示本次條件比對，不是勝率或績效保證。"
        )

    with watch_tab:
        st.markdown("#### ✍️ 操盤手自訂觀察代碼掃描區")
        st.caption("每個欄位輸入一檔股票代碼。台股上市請加 .TW、上櫃請加 .TWO；美股輸入代碼即可。下方掃描結果沿用台股／美股核心池的明細欄位與策略評分方式。")
        watch_defaults = TRADER_WATCHLIST_DEFAULTS
        saved_watch_codes = list(st.session_state.get("trader_watchlist_values", watch_defaults))
        sync_watch_widgets = bool(st.session_state.pop("trader_watchlist_sync_needed", False))
        for watch_index in range(12):
            watch_key = f"trader_watch_ticker_{watch_index + 1}"
            if sync_watch_widgets or watch_key not in st.session_state:
                st.session_state[watch_key] = saved_watch_codes[watch_index]
        watch_columns = st.columns(3)
        custom_scan_list = []
        duplicate_watch_codes = []
        for watch_index in range(12):
            with watch_columns[watch_index % 3]:
                watch_code = st.text_input(
                    f"觀察代碼 {watch_index + 1}",
                    value=watch_defaults[watch_index],
                    placeholder="例如 2330.TW 或 NVDA",
                    key=f"trader_watch_ticker_{watch_index + 1}",
                ).strip().upper()
            if watch_code:
                if watch_code in custom_scan_list:
                    duplicate_watch_codes.append(watch_code)
                else:
                    custom_scan_list.append(watch_code)
        st.session_state["trader_watchlist_values"] = [
            str(st.session_state.get(f"trader_watch_ticker_{index + 1}", watch_defaults[index])).strip().upper()
            for index in range(12)
        ]
        if duplicate_watch_codes:
            st.caption("重複代碼已自動略過：" + "、".join(dict.fromkeys(duplicate_watch_codes)))
        st.caption(f"目前輸入 {len(custom_scan_list)} 檔；掃描結果會列出排名、股票代碼／名稱、股價、策略評分及已啟用指標明細。")

        coach_history = st.session_state.get("trader_backtest_history", [])
        if coach_history:
            matched_history = [item for item in coach_history if item.get("ticker") in custom_scan_list]
            if not matched_history:
                matched_history = coach_history[-20:]
                st.caption("目前觀察代碼沒有對應的回測紀錄，以下先顯示最近測試過的條件；加入該股票代碼後會一併列入觀察名單。")
            if matched_history:
                st.markdown("#### 🧭 技術指標回測條件觀察紀錄")
                st.caption("僅整理您在回測頁手動執行過的條件；不會自動改動選股分數或替您下單。")
                status_rank = {"可優先列入觀察": 0, "繼續觀察": 1, "樣本不足": 2, "暫不建議直接採用": 3}
                matched_history.sort(key=lambda item: (
                    status_rank.get(item.get("status"), 4),
                    -(item.get("win_rate_pct") if item.get("win_rate_pct") is not None else -1),
                    -(item.get("total_return_pct") if item.get("total_return_pct") is not None else -1),
                ))
                coach_view = pd.DataFrame(matched_history[:20])
                coach_columns = ["ticker", "stock_name", "condition", "timeframe", "period", "trade_count", "win_rate_pct",
                                 "total_return_pct", "buy_hold_pct", "dca_pct", "grid_pct", "oos_pct",
                                 "profit_factor", "status", "advice"]
                coach_view = coach_view[[column for column in coach_columns if column in coach_view.columns]].rename(columns={
                    "ticker": "股票代碼", "stock_name": "股票名稱", "condition": "回測條件", "timeframe": "週期", "period": "資料期間",
                    "trade_count": "交易筆數", "win_rate_pct": "勝率(%)", "total_return_pct": "整段報酬(%)",
                    "buy_hold_pct": "長期持有(%)", "dca_pct": "定期定額(%)", "grid_pct": "網格(%)",
                    "oos_pct": "較晚期間(%)", "profit_factor": "賺賠金額比", "status": "操盤手提示", "advice": "建議摘要",
                })
                coach_formats = {
                    "勝率(%)": "{:.1f}%",
                    "整段報酬(%)": "{:+.2f}%",
                    "長期持有(%)": "{:+.2f}%",
                    "定期定額(%)": "{:+.2f}%",
                    "網格(%)": "{:+.2f}%",
                    "較晚期間(%)": "{:+.2f}%",
                    "賺賠金額比": "{:.2f}",
                }
                coach_formatters = {
                    column: (lambda value, spec=spec: _format_optional_cell(value, spec))
                    for column, spec in coach_formats.items()
                }
                st.dataframe(coach_view.style.format(coach_formatters, na_rep="—"),
                             use_container_width=True, hide_index=True)
            else:
                st.caption("目前觀察代碼中尚無技術指標回測紀錄；請先在回測頁執行條件測試。")
        else:
            st.caption("技術指標回測完成後，條件勝率與報酬建議會出現在此處；紀錄保留於目前工作階段。")

    def render_stock_screen_results(result_view, title):
        st.markdown(f"#### {title}（依已啟用研究條件排名，最多 20 檔）")
        if result_view.empty:
            st.info("本次無符合標的。已完整套用目前勾選的條件，沒有放寬條件或回退顯示未通過的股票。")
            return
        if explanation_style.startswith("生動童趣"):
            st.caption("分數像這次班上的相對排名，不是考試及格率；換一批股票或勾選條件，名次就可能改變。GVI 假設目前 ROE 五年不變，像假設樹每年都結一樣多果實，並不是未來保證。")
        else:
            st.caption("GVI、動能、自由現金流與成長估值是研究用估算；策略評分只表示本次股票池中的相對排序，不是勝率或預期報酬。股票池或勾選條件改變時會重新排名，不宜跨次比較。GVI 假設目前 ROE 未來五年不變，尚未證明能穩定改善績效。價格接近高位可能吸引順勢買盤，也可能增加追價風險；價量動能不是實際籌碼或法人資料。")
        st.caption("點選明細表中的股票列，再選擇個股即時診斷或加入操盤手自訂觀察代碼。")
        if "風險狀態" in result_view:
            st.caption("共用風險關卡顯示查核原因與來源。原有篩選與排名依勾選的雷達條件計算。")
        table_event = st.dataframe(
            result_view,
            use_container_width=True,
            hide_index=True,
            key="stock_screen_results_table",
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_rows = list(table_event.selection.rows)
        if selected_rows:
            selected_row = int(selected_rows[0])
            if 0 <= selected_row < len(result_view):
                selected_result = result_view.iloc[selected_row]
                st.session_state["stock_screen_selected_ticker"] = str(selected_result["股票代碼"])
                selected_name = str(selected_result.get("股票名稱", ""))
                selected_ticker = str(selected_result["股票代碼"])
                _render_stock_actions(selected_ticker, selected_name, "stock_screen_selected")
        else:
            st.session_state.pop("stock_screen_selected_ticker", None)

    def load_data_from_sqlite_and_render(filter_list, title, is_custom_mode=False):
        st.session_state["stock_screen_result"] = None
        st.session_state.pop("stock_screen_selected_ticker", None)
        st.session_state.pop("stock_screen_results_table", None)
        if not any((strat_gvi, strat_momentum, strat_qarp)):
            st.info("請至少開啟一項研究條件，再執行掃描。")
            return
        conn = sqlite3.connect(DB_FILE)
        try: df_db = pd.read_sql_query("SELECT * FROM market_data", conn)
        except Exception: df_db = pd.DataFrame()
        conn.close()

        for t in filter_list:
            row_matches = df_db[df_db['ticker'] == t] if not df_db.empty else pd.DataFrame()
            if row_matches.empty or not is_cache_valid(row_matches.iloc[0].get('updated_at')):
                signature_save_to_db(t)

        conn = sqlite3.connect(DB_FILE)
        df_db = pd.read_sql_query("SELECT * FROM market_data", conn)
        conn.close()

        if df_db.empty:
            st.info("目前沒有可供篩選的快取資料。")
            return

        df_filtered = df_db[df_db['ticker'].isin(filter_list)].copy()
        raw_portfolio = []

        for idx, row in df_filtered.iterrows():
            t = row['ticker']
            try:
                ch_name = STOCK_NAME_MAP.get(t, t)
                df_hist = yf.download(t, period=cfg["p"], interval=cfg["i"], progress=False)
                if isinstance(df_hist.columns, pd.MultiIndex): df_hist.columns = df_hist.columns.get_level_values(0)
                if df_hist.index.tz is not None: df_hist.index = df_hist.index.tz_localize(None)
                risk_history = df_hist.copy(deep=False)
                df_hist = df_hist.dropna(subset=['Close', 'Volume']).copy()
                L = len(df_hist)

                close_today = float(row['close']) if (L == 0 or 'Close' not in df_hist.columns) else float(df_hist['Close'].to_numpy().flatten()[-1])
                close_series = pd.to_numeric(df_hist['Close'], errors='coerce').dropna() if 'Close' in df_hist.columns else pd.Series(dtype=float)
                momentum_value = calculate_momentum(close_today, close_series)
                if momentum_value is None:
                    momentum_value = np.nan

                bv_val = float(row['book_value']) if (pd.notnull(row['book_value']) and float(row['book_value']) > 0) else None
                roe_val = parse_roe(row['roe'])

                gvi_live = calculate_gvi(close_today, bv_val, roe_val) if (close_today > 0 and bv_val is not None and roe_val is not None) else None

                stock_obj = yf.Ticker(t)
                info_obj = stock_obj.info if hasattr(stock_obj, 'info') and stock_obj.info else {}

                fcf = _finite_number(info_obj.get('freeCashflow'))
                fcf_basis = "yfinance 公司資料欄位"
                if fcf is None:
                    fcf, fcf_basis = _free_cash_flow_from_statements(stock_obj)
                mcap = _finite_number(info_obj.get('marketCap')) or _finite_number(row.get('mcap'))
                fcf_yield = calculate_fcf_yield(fcf, mcap)

                peg_val, _peg_basis = _peg_from_company_data(info_obj)
                risk_findings = evaluate_radar_risk(
                    risk_history, timeframe=selected_tf, cache_updated=row.get("updated_at"),
                    cache_ttl=CACHE_TTL, roe=roe_val, roe_source=row.get("roe_source") or "",
                    free_cash_flow=fcf, fcf_source=fcf_basis, gvi=gvi_live,
                    momentum=momentum_value, fcf_yield=fcf_yield, peg=peg_val,
                )

                df_hist, chip_status_text, _ = calculate_chip_and_backtest(t, df_hist, selected_tf)

                raw_portfolio.append({
                    '股票代碼': t, '股票名稱': ch_name, '目前股價': close_today,
                    'GVI值': gvi_live,
                    '動能分數': momentum_value,
                    '自由現金流收益': fcf_yield,
                    '本益成長比(PEG)': peg_val,
                    '⚡ 價量動能估算': chip_status_text,
                    '風險狀態': risk_status(risk_findings),
                    '風險原因': "；".join(finding.reason for finding in risk_findings),
                    '查核來源': "；".join(dict.fromkeys(finding.source for finding in risk_findings)),
                    'gvi_raw': gvi_live,
                    'mo_raw': momentum_value,
                    'fcf_raw': fcf_yield
                })
            except Exception as e_item:
                st.sidebar.caption(f"標的跳過 [{t}]: {e_item}")

        if raw_portfolio:
            df_res = pd.DataFrame(raw_portfolio)

            df_res = filter_radar_matches(df_res, strat_gvi, strat_momentum, strat_qarp)

            if df_res.empty:
                empty_view = pd.DataFrame(columns=['排名', '股票代碼', '股票名稱', '目前股價', '策略評分'])
                st.session_state["stock_screen_result"] = {"view": empty_view, "title": title}
                render_stock_screen_results(empty_view, title)
                return

            df_res = rank_radar_matches(df_res, strat_gvi, strat_momentum, strat_qarp)
            df_res = df_res.sort_values(by='綜合分數', ascending=False).head(20).reset_index(drop=True)

            df_res['排名'] = [f"第 {i+1} 名" for i in range(len(df_res))]
            df_res['策略評分'] = df_res['策略評分_num'].astype(str) + " 分"
            df_res['GVI價值指標'] = df_res['GVI值'].apply(lambda x: f"{x:.4f}" if (pd.notnull(x) and x is not None) else "資料不足")
            df_res['動能(創高距離)'] = df_res['動能分數'].apply(lambda value: f"{value:.1f}%" if pd.notna(value) else "資料不足")
            df_res['自由現金流收益率'] = df_res['自由現金流收益'].apply(lambda x: f"{x:.2f}%" if (pd.notnull(x) and x is not None) else "資料不足")
            df_res['本益成長比(PEG)'] = df_res['本益成長比(PEG)'].apply(lambda x: f"{x:.2f}" if (pd.notnull(x) and x is not None) else "資料不足")

            cols = ['排名', '股票代碼', '股票名稱', '目前股價', '策略評分']
            if strat_gvi: cols.append('GVI價值指標')
            if strat_momentum: cols.append('動能(創高距離)')
            if strat_qarp: cols.extend(['自由現金流收益率', '本益成長比(PEG)'])
            if not strat_gvi and not strat_momentum and not strat_qarp:
                cols.extend(['GVI價值指標', '動能(創高距離)', '自由現金流收益率', '本益成長比(PEG)'])
            cols.extend(('⚡ 價量動能估算', '風險狀態', '風險原因', '查核來源'))

            df_final_view = df_res[cols]
            st.session_state["stock_screen_result"] = {
                "view": df_final_view.copy(),
                "title": title,
            }
            render_stock_screen_results(df_final_view, title)
        else:
            st.info("沒有取得可計算的股票資料，本次無符合標的。")

    scan_triggered_this_run = False
    with scan_tab:
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            if st.button("🇹🇼 一鍵執行：全自動過濾全台股核心池", type="secondary", use_container_width=True):
                scan_triggered_this_run = True
                load_data_from_sqlite_and_render(AUTO_TW_UNIVERSE, "🇹🇼 台灣股市核心策略篩選結果")
        with col_btn2:
            if st.button("🇺🇸 一鍵執行：全自動過濾全美股核心池", type="secondary", use_container_width=True):
                scan_triggered_this_run = True
                load_data_from_sqlite_and_render(AUTO_US_UNIVERSE, "🇺🇸 美國股市核心策略篩選結果")
        with col_btn3:
            if st.button("🚀 執行：自訂名單多因子本地精準過濾", type="primary", use_container_width=True):
                scan_triggered_this_run = True
                load_data_from_sqlite_and_render(custom_scan_list, "🎯 操盤手自訂名單策略篩選結果", is_custom_mode=True)
        if not scan_triggered_this_run:
            saved_screen_result = st.session_state.get("stock_screen_result")
            if isinstance(saved_screen_result, dict) and isinstance(saved_screen_result.get("view"), pd.DataFrame):
                render_stock_screen_results(saved_screen_result["view"], saved_screen_result.get("title", "選股策略篩選結果"))
