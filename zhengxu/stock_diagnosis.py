"""Selected-stock research, technical displays and AI reports."""

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import streamlit as st
import time
import yfinance as yf

from .radar_risk import evaluate_radar_risk, risk_status
from .radars import (
    calculate_fcf_yield,
    calculate_gvi,
    calculate_momentum,
    score_gvi,
    score_momentum,
    score_qarp,
)

from .backtest import (
    run_ai_candidate_backtests,
)
from .dividend import (
    _finite_number,
    _free_cash_flow_from_statements,
    _peg_from_company_data,
)
from .explanations import (
    build_price_volume_market_psychology,
    build_three_core_market_psychology_notes,
)
from .market_data import (
    _market_overview_snapshots,
    STOCK_NAME_MAP,
    _broker_branch_ai_state_key,
    build_broker_branch_ai_interpretation,
    calculate_chip_and_backtest,
    format_broker_branch_ai_context,
    is_cache_valid,
    parse_roe,
    signature_save_to_db,
)
from .research_ai import (
    build_ai_investment_prompt,
    collect_stock_research_context,
    generate_content_with_retry,
    generate_openai_research,
    get_gmail_secret_values,
    send_gmail_report,
)
from .research_quality import RadarEvidence, build_research_evidence, compose_research_report
from .official_close import render_official_close
from .chart_quotes import chart_time_label, load_chart_quote
from .storage import (
    CACHE_TTL,
    DB_FILE,
)


def _render_radar_risk(findings):
    with st.expander(f"共用風險關卡｜{risk_status(findings)}", expanded=bool(findings)):
        for finding in findings:
            st.write(f"{finding.category}｜{finding.reason}")
            st.caption(f"查核來源 {finding.source}")
        st.caption("這些提示與雷達分數分開。快取更新時間用於檢查快取，財報公告日需另外查核。")


def render_stock_diagnosis(
    api_key_input, auth_user, cfg, explanation_style, market_indexes,
    openai_api_key_input, openai_model_input, selected_stock, selected_tf, *,
    show_tactical_panel=True, show_official_close=True, show_radar_comparison=True,
):
    c_name = change_pct = date_strings = vol_ratio = price_val = is_tw_stock = None
    tabs_created = False
    grid_advice_metrics = None
    df_chart = None
    if selected_stock:
        if show_official_close:
            render_official_close(selected_stock)
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT book_value, roe, gvi, close, bv_source, roe_source, updated_at FROM market_data WHERE ticker=?", (selected_stock,))
            db_row = c.fetchone()
            conn.close()

            needs_sync = False
            was_missing_in_db = False

            if not db_row:
                needs_sync = True
                was_missing_in_db = True
            elif not is_cache_valid(db_row[6]):
                needs_sync = True

            if needs_sync:
                with st.spinner(f"📥 資料庫未發現【{selected_stock}】或數據已逾期，正自動啟動線上下載並更新資料庫..."):
                    signature_save_to_db(selected_stock)

                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("SELECT book_value, roe, gvi, close, bv_source, roe_source, updated_at FROM market_data WHERE ticker=?", (selected_stock,))
                db_row = c.fetchone()
                conn.close()

            df_chart, quote_failed = load_chart_quote(selected_stock, cfg)
            if df_chart is not None:
                if quote_failed:
                    st.warning("Yahoo Finance 行情未更新，以下沿用此標的與週期最後有效資料。")
                quote_caption = "Yahoo Finance · " + chart_time_label(df_chart.index[-1], cfg["i"], selected_stock)
                st.caption(quote_caption)
                risk_history = df_chart.copy(deep=False)
                df_chart, chip_status_text, _ = calculate_chip_and_backtest(selected_stock, df_chart, selected_tf)
                df_chart.attrs["quote_caption"] = quote_caption + (" · 未更新，沿用最後有效資料" if quote_failed else "")

                # 📅 日期格式適應性調整
                if "分鐘" in selected_tf:
                    date_strings = df_chart.index.strftime('%m-%d %H:%M').tolist()
                else:
                    date_strings = df_chart.index.strftime('%Y-%m-%d').tolist()

                stock_obj_main = yf.Ticker(selected_stock)
                try:
                    info_obj_main = stock_obj_main.info or {}
                except Exception:
                    info_obj_main = {}
                c_name = (
                    STOCK_NAME_MAP.get(selected_stock)
                    or info_obj_main.get('longName')
                    or info_obj_main.get('shortName')
                    or selected_stock
                )
                current_price_label = f"最新 K 棒價格 ({c_name} {selected_stock})"
                price_val = float(df_chart['Close'].to_numpy().flatten()[-1])
                is_tw_stock = ".TW" in selected_stock or ".TWO" in selected_stock

                bv_val = float(db_row[0]) if (db_row and db_row[0] is not None and db_row[0] > 0) else None
                roe_val = parse_roe(db_row[1]) if (db_row and db_row[1] is not None) else None
                bv_src = db_row[4] if (db_row and db_row[4]) else "無數據"
                roe_src = db_row[5] if (db_row and db_row[5]) else "無數據"
                cache_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(db_row[6])) if (db_row and db_row[6]) else "未設定"

                # 計算動能指標 (安全防呆)
                prev_close = float(df_chart['Close'].iloc[-2]) if len(df_chart) > 1 else price_val
                change = price_val - prev_close
                change_pct = (change / prev_close) * 100 if prev_close > 0 else 0.0

                vol_today = float(df_chart['Volume'].iloc[-1]) if len(df_chart) > 0 else 0
                vol_ma5 = float(df_chart['Volume'].rolling(5).mean().iloc[-1]) if len(df_chart) >= 5 else vol_today
                vol_ratio = (vol_today / vol_ma5) if vol_ma5 > 0 else 1.0

                ma5_val = float(df_chart['Close'].rolling(5).mean().iloc[-1]) if len(df_chart) >= 5 else price_val
                bias5 = ((price_val - ma5_val) / ma5_val) * 100 if ma5_val > 0 else 0.0

                high_today = float(df_chart['High'].iloc[-1]) if len(df_chart) > 0 else price_val
                low_today = float(df_chart['Low'].iloc[-1]) if len(df_chart) > 0 else price_val

                # 💡 修正：將變數計算獨立移出 UI 判斷區塊外，確保下方雷達圖程式永遠能讀取到 gvi_val
                if price_val > 0 and bv_val is not None and roe_val is not None:
                    pb_val = price_val / bv_val
                    gvi_val = calculate_gvi(price_val, bv_val, roe_val)
                    gvi_str = f"{gvi_val:.4f}"
                    roe_str = f"{roe_val * 100:.2f}%"
                    bv_str = f"{bv_val:,.2f} 元" if is_tw_stock else f"${bv_val:,.2f}"
                    pb_str = f"{pb_val:.2f} 倍"
                else:
                    pb_val = None
                    gvi_val = None
                    gvi_str = "資料不足"
                    roe_str = "資料不足"
                    bv_str = "資料不足"
                    pb_str = "資料不足"

                if show_tactical_panel:
                    st.markdown("### 🎛️ 戰術面板模式切換")
                    panel_mode = st.radio("選擇看盤模式：", ["📈 操盤/動能模式 (專注量價與短線動能)", "🏦 存股/價值模式 (專注基本面與估值)"], horizontal=True, label_visibility="collapsed")

                    if "價值" in panel_mode:
                        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                        with gc1:
                            st.metric(label=f"💰 {current_price_label}", value=f"{price_val:,.2f} 元" if is_tw_stock else f"${price_val:,.2f}", help="以所選週期最後一根可用 K 棒為準，盤中數值可能變動。")
                            st.caption(f"基本面快取更新 {cache_time_str}")

                        if bv_val is not None:
                            with gc2: st.metric(label="👑 GVI 成長價值", value=gvi_str, help="把帳面價值、目前股價與獲利能力放在一起估算的研究指標；公式假設目前 ROE 未來五年維持不變，這不是預測，且尚未證明能穩定改善績效。"); st.caption("📢 最新重算")
                            with gc3: st.metric(label="📊 ROE 股東權益報酬率", value=roe_str, help="公司以股東權益創造盈餘的比率；需搭配產業特性及財報期間判讀。"); st.caption(f"📢 {roe_src}")
                            with gc4: st.metric(label="📖 每股淨值", value=bv_str, help="公司帳面資產扣除負債後，平均分配到每一股的金額；不等於股票的市場價格。"); st.caption(f"📢 {bv_src}")
                            with gc5: st.metric(label="⚖️ 股價淨值比 (PB)", value=pb_str, help="顯示目前股價相對每股帳面價值的倍數；產業間差異大，跨產業比較要小心。"); st.caption("📢 溢價程度")
                        else:
                            with gc2: st.info("⚠️ 缺乏財報淨值數據，系統自動隱藏 PB 與相關估值欄位。建議切換至「操盤/動能模式」觀看。")
                    else:
                        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                        with gc1: st.metric(label=f"⚡ {current_price_label}", value=f"{price_val:,.2f}", delta=f"{change:+.2f} ({change_pct:+.2f}%)" if len(df_chart) > 1 else None, help="價格與漲跌以圖表目前選取的週期為準，最新一筆資料可能有延遲。")
                        with gc2: st.metric(label="🌊 最近一根 K 棒成交量", value=f"{vol_today:,.0f}", delta=f"量比: {vol_ratio:.2f}x", delta_color="normal" if vol_ratio >= 1 else "off", help="量比將最近一根 K 棒的成交量和近 5 根平均量比較；高於 1 表示比近期平均活躍，無法單獨辨別買賣方向。")
                        with gc3: st.metric(label="🎯 5根均線乖離率", value=f"{bias5:+.2f}%", delta="偏離大留意回檔" if abs(bias5) > 5 else "乖離正常", delta_color="inverse" if abs(bias5) > 5 else "off", help="最新收盤價相對近 5 根收盤均價的百分比差距；用來觀察短期偏離，不是反轉訊號。")
                        with gc4: st.metric(label="🔺 最近一根 K 棒最高價", value=f"{high_today:,.2f}", help="所選圖表週期中最新一根 K 棒的最高價。")
                        with gc5: st.metric(label="🔻 最近一根 K 棒最低價", value=f"{low_today:,.2f}", help="所選圖表週期中最新一根 K 棒的最低價。")

                # ======= 升級：計算三大策略所需數值與交叉分析 =======
                L = len(df_chart)
                close_today = price_val
                close_history = pd.to_numeric(df_chart['Close'], errors='coerce').dropna() if 'Close' in df_chart.columns else pd.Series(dtype=float)
                momentum_window_text = f"最近最多 252 根{selected_tf} K 棒"
                momentum_score = calculate_momentum(close_today, close_history)

                fcf_main = _finite_number(info_obj_main.get('freeCashflow'))
                fcf_basis_main = "yfinance 公司資料欄位"
                if fcf_main is None:
                    fcf_main, fcf_basis_main = _free_cash_flow_from_statements(stock_obj_main)
                mcap_main = _finite_number(info_obj_main.get('marketCap'))
                fcf_yield_main = calculate_fcf_yield(fcf_main, mcap_main)

                peg_val_main, peg_basis_main = _peg_from_company_data(info_obj_main)
                risk_findings = evaluate_radar_risk(
                    risk_history, timeframe=selected_tf, cache_updated=db_row[6] if db_row else None,
                    cache_ttl=CACHE_TTL, roe=roe_val, roe_source=roe_src,
                    free_cash_flow=fcf_main, fcf_source=fcf_basis_main, gvi=gvi_val,
                    momentum=momentum_score, fcf_yield=fcf_yield_main, peg=peg_val_main,
                )
                if show_radar_comparison:
                    _render_radar_risk(risk_findings)

                scores = {}
                reasons = {}
                scores["GVI 價值雷達"], reasons["GVI 價值雷達"] = score_gvi(gvi_val)
                scores["動能突破雷達"], reasons["動能突破雷達"] = score_momentum(momentum_score, momentum_window_text)
                scores["QARP 現金流雷達"], reasons["QARP 現金流雷達"] = score_qarp(fcf_yield_main, peg_val_main)

                three_core_psychology = build_three_core_market_psychology_notes(
                    gvi_val, momentum_score, fcf_yield_main, peg_val_main,
                    momentum_window_text, explanation_style,
                )

                # 顯示目前門檻的符合度，不把它解讀為實證優勢或進場建議。
                valid_scores = {name: score for name, score in scores.items() if score is not None}
                best_strategy = max(valid_scores, key=valid_scores.get) if valid_scores else "資料不足"
                best_score = valid_scores.get(best_strategy, 0)

                if best_score >= 90:
                    valuation_color = "#00cc66"
                elif best_score >= 75:
                    valuation_color = "#2baf2b"
                elif best_score >= 60:
                    valuation_color = "#ff9900"
                else:
                    valuation_color = "#cc0000"

                fin_src_display = f"每股淨值來源：{bv_src}；ROE 來源：{roe_src}"
                psychology_key_by_strategy = {
                    "GVI 價值雷達": "GVI 價值面",
                    "動能突破雷達": "動能趨勢面",
                    "QARP 現金流雷達": "QARP 現金流與估值面",
                }
                detail_sources = {
                    "GVI 價值雷達": f"每股淨值：{bv_src}；ROE：{roe_src}；股價：Yahoo Finance 行情資料。",
                    "動能突破雷達": f"Yahoo Finance 歷史收盤資料；比較{momentum_window_text}，至少需 50 根有效收盤資料。",
                    "QARP 現金流雷達": (
                        f"自由現金流依據：{fcf_basis_main}；搭配 Yahoo Finance 市值計算參考值。若使用年度財報，期間可能較近四季資料更舊。"
                        f"PEG 依據：{peg_basis_main}；估算只採年度盈餘成長，不以單季成長替代。"
                    ),
                }
                compact_psychology = {
                    "GVI 價值面": {
                        True: "可能吸引價值型關注；仍要查獲利能否延續。",
                        False: "價值條件支持較弱；不等於一定高估。",
                        None: "資料不足，暫不解讀市場關注。",
                    },
                    "動能趨勢面": {
                        True: "趨勢較受關注；留意追價與擁擠風險。",
                        False: "趨勢支持較弱；觀察整理或轉弱。",
                        None: "歷史價格不足，暫不判讀趨勢。",
                    },
                    "QARP 現金流與估值面": {
                        True: "現金流與估值條件達標；確認能否持續。",
                        False: "至少一項未達標；查現金創造與成長預期。",
                        None: "資料不足，暫不解讀現金流與估值。",
                    },
                }
                if explanation_style.startswith("生動童趣"):
                    compact_psychology = {
                        "GVI 價值面": {
                            True: "像果樹目前有成果；還要看能否持續。",
                            False: "這次沒達研究線，不等於公司一定差。",
                            None: "資料缺幾片，先不猜。",
                        },
                        "動能趨勢面": {
                            True: "像跑得快的選手；留意會不會喘。",
                            False: "趨勢支持較弱；不代表價格便宜。",
                            None: "歷史資料太少，先不判跑勢。",
                        },
                        "QARP 現金流與估值面": {
                            True: "兩項都達研究線；確認現金流能否續航。",
                            False: "至少一題未達標；先找出原因。",
                            None: "現金流或成長資料不夠。",
                        },
                    }
                factor_details = [
                    {
                        "strategy": "GVI 價值雷達",
                        "psych_key": "GVI 價值面",
                        "threshold": "GVI 達 0.20 以上",
                        "highlight": f"GVI {gvi_val:.4f}" if gvi_val is not None else "GVI 資料不足",
                    },
                    {
                        "strategy": "動能突破雷達",
                        "psych_key": "動能趨勢面",
                        "threshold": f"{momentum_window_text}最高收盤價比值達 80% 以上；至少需 50 根有效收盤資料",
                        "highlight": f"高點比值 {momentum_score:.1f}%" if momentum_score is not None else "動能資料不足",
                    },
                    {
                        "strategy": "QARP 現金流雷達",
                        "psych_key": "QARP 現金流與估值面",
                        "threshold": "自由現金流收益參考達 2.5% 以上，且 PEG 不高於 1.5",
                        "highlight": (
                            f"現金流收益 {fcf_yield_main:.2f}%｜PEG {peg_val_main:.2f}"
                            if fcf_yield_main is not None and peg_val_main is not None
                            else "現金流或 PEG 資料不足"
                        ),
                    },
                ]
                status_style = {
                    True: ("達門檻", "#176746"),
                    False: ("未達門檻", "#875218"),
                    None: ("資料不足", "#486657"),
                }
                available_core_checks = [
                    three_core_psychology["checks"].get(item["psych_key"])
                    for item in factor_details
                ]
                known_core_checks = [value for value in available_core_checks if value is not None]
                if not known_core_checks:
                    compact_overall = "三項資料不足，先補齊數據再比較。"
                elif all(value is True for value in known_core_checks) and len(known_core_checks) == 3:
                    compact_overall = "三項條件目前一致達標，可能提高市場研究關注；這仍未證明有績效優勢。"
                elif all(value is False for value in known_core_checks):
                    compact_overall = "目前可評估的條件都未達門檻，避免為了看多而放寬標準。"
                else:
                    compact_overall = "訊號有分歧；先查明價值、趨勢與現金流差異，避免只挑支持原看法的資料。"
                if explanation_style.startswith("生動童趣"):
                    if not known_core_checks:
                        compact_overall = "三張考卷資料都不夠，先補齊再比較。"
                    elif all(value is True for value in known_core_checks) and len(known_core_checks) == 3:
                        compact_overall = "三個角度都達研究線，像三張考卷都過線；不代表下一次一定考得好。"
                    elif all(value is False for value in known_core_checks):
                        compact_overall = "目前已評估的條件都沒過線，先別為了想看多就降低標準。"
                    else:
                        compact_overall = "三個答案不一樣，像拼圖還沒拼好；先找出差異原因。"

                compact_cards = []
                for item in factor_details:
                    check = three_core_psychology["checks"].get(item["psych_key"])
                    status_label, status_color = status_style[check]
                    score_value = scores.get(item["strategy"])
                    score_label = f"比對 {score_value} 分" if score_value is not None else "未評分"
                    brief = compact_psychology[item["psych_key"]][check]
                    compact_cards.append(
                        "<div style=\"min-width:150px;flex:1 1 150px;border:1px solid rgba(148,163,184,.28);"
                        "border-radius:9px;padding:9px 10px;background:#FFFFFF;\">"
                        f"<div style=\"font-weight:700;color:#173B2C;\">{html.escape(item['strategy'])}</div>"
                        f"<div style=\"margin:3px 0;color:{status_color};font-size:.88rem;\">{status_label}｜{score_label}</div>"
                        f"<div style=\"font-size:.84rem;line-height:1.45;color:#486657;\">🧠 {html.escape(brief)}</div>"
                        "</div>"
                    )
                compact_best = (
                    f"目前比對分數最高：{best_strategy}（{best_score} 分；分數只代表門檻相符程度）"
                    if valid_scores else "目前沒有可評分的研究條件。"
                )
                compact_core_html = (
                    "<div style=\"border-left:4px solid " + valuation_color + ";padding:11px 13px;"
                    "border-radius:8px;background:#E6F0E9;\">"
                    "<div style=\"font-size:1.03rem;font-weight:750;color:#173B2C;margin-bottom:5px;\">⚖️ 三核心研究條件比對</div>"
                    f"<div style=\"font-size:.88rem;color:#486657;margin-bottom:9px;\">{html.escape(compact_best)}</div>"
                    "<div style=\"display:flex;flex-wrap:wrap;gap:7px;\">"
                    + "".join(compact_cards)
                    + "</div>"
                    f"<div style=\"margin-top:9px;font-size:.88rem;color:#173B2C;\">🧭 心理合併觀察：{html.escape(compact_overall)}</div>"
                    "</div>"
                )

                if show_radar_comparison:
                    # ======= 視覺化圖表整合 (Radar Chart) =======
                    diag_c1, diag_c2 = st.columns([1.5, 1])
                    with diag_c1:
                        st.markdown(compact_core_html, unsafe_allow_html=True)
                        st.caption("研究比對分數不是勝率、報酬預測或買賣建議。")
                        with st.expander("🖍️ 查看三核心詳細條件與畫線重點", expanded=False):
                            st.markdown(
                                f"**目前比對分數最高：** {best_strategy}（{best_score} 分）  \n"
                                f"**每股淨值與 ROE 資料來源：** {fin_src_display}"
                            )
                            if was_missing_in_db:
                                st.info("財報快取原先沒有可用資料，系統已啟動備援來源補齊。")
                            st.caption(
                                "下列條件分數只表示符合目前研究門檻的程度；門檻尚未證明具有穩定的樣本外績效。"
                            )
                            for item in factor_details:
                                strategy_name = item["strategy"]
                                psych_key = item["psych_key"]
                                check = three_core_psychology["checks"].get(psych_key)
                                status_label, status_color = status_style[check]
                                detail_text = three_core_psychology["notes"].get(psych_key, "資料不足。")
                                score_value = scores.get(strategy_name)
                                score_text = f"{score_value} 分" if score_value is not None else "未評分"
                                highlight_text = html.escape(item["highlight"])
                                st.markdown(
                                    "<div style=\"margin:9px 0;padding:11px 13px;border:1px solid rgba(148,163,184,.25);"
                                    "border-radius:9px;background:#FFFFFF;\">"
                                    f"<div style=\"font-weight:750;color:#173B2C;\">{html.escape(strategy_name)} "
                                    f"<span style=\"color:{status_color};font-size:.88rem;\">{status_label}｜比對 {score_text}</span></div>"
                                    f"<div style=\"margin:7px 0;color:#173B2C;line-height:1.65;\">"
                                    f"<span style=\"text-decoration:underline;text-decoration-color:#FBBF24;"
                                    f"text-decoration-thickness:3px;text-underline-offset:3px;\">🖍️ 畫線重點：{highlight_text}</span></div>"
                                    f"<div style=\"font-size:.9rem;color:#486657;line-height:1.6;\"><b>門檻：</b>{html.escape(item['threshold'])}<br>"
                                    f"<b>數據比對：</b>{html.escape(reasons[strategy_name])}<br>"
                                    f"<b>市場心理：</b>{html.escape(detail_text)}<br>"
                                    f"<b>資料來源：</b>{html.escape(detail_sources[strategy_name])}</div>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown(
                                "**🧭 三核心合併解讀**：" + compact_overall + "  \n"
                                "**解讀限制**：可能的市場行為線索，不是投資人心理的直接測量；分數不代表績效優勢或部位建議。"
                            )
                    with diag_c2:
                        categories = ['GVI 價值雷達', '動能突破雷達', 'QARP 現金流雷達']
                        radar_scores = [scores['GVI 價值雷達'], scores['動能突破雷達'], scores['QARP 現金流雷達']]

                        fig_radar = go.Figure()
                        fig_radar.add_trace(go.Scatterpolar(
                            r=radar_scores,
                            theta=categories,
                            fill='toself',
                            name=c_name,
                            line_color=valuation_color,
                            fillcolor=valuation_color.replace(')', ', 0.2)').replace('rgb', 'rgba') if 'rgb' in valuation_color else f"{valuation_color}33"
                        ))
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(128,128,128,0.2)"),
                                angularaxis=dict(gridcolor="rgba(128,128,128,0.2)")
                            ),
                            showlegend=False,
                            margin=dict(l=30, r=30, t=30, b=30),
                            height=360,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_radar, use_container_width=True, config={"responsive": True})

                st.info(f"🔮 【{c_name}】{selected_tf} 價量動能估算：{chip_status_text}")

                report_scope = f"{auth_user['id']}:{selected_stock}"
                report_key = f"account_ai_report:{report_scope}"
                with st.expander(f"🧠 AI 雙核心個股研究報告與 Email 傳送【{c_name}】", expanded=report_key in st.session_state):
                    branch_ai_snapshot_key = _broker_branch_ai_state_key(auth_user.get("id", "guest"), selected_stock)
                    current_branch_ai_context = format_broker_branch_ai_context(
                        st.session_state.get(branch_ai_snapshot_key), selected_stock
                    )
                    current_branch_snapshot = st.session_state.get(branch_ai_snapshot_key)
                    current_branch_ai_interpretation = ""
                    if current_branch_ai_context:
                        current_branch_ai_interpretation = build_broker_branch_ai_interpretation(
                            current_branch_snapshot,
                            pd.to_numeric(df_chart["Close"], errors="coerce").dropna().tail(20),
                            selected_tf,
                        )
                    if current_branch_ai_context:
                        st.info(
                            "目前登入工作階段已有這檔股票的分點摘要。重新產生 AI 報告時會將家數、淨額與集中度摘要傳送給你選擇的 AI 服務；"
                            "不會傳送完整分點明細。若目前已經有舊報告，請重新分析才會納入這份分點資料。"
                        )
                    else:
                        st.caption("若先在「籌碼流向參考」匯入這檔股票的分點資料，重新產生報告時會一併分析。")
                    ai_provider = st.radio(
                        "AI 分析來源",
                        ["Gemini", "ChatGPT（含網路搜尋）"],
                        horizontal=True,
                        index=1 if openai_api_key_input else 0,
                        key=f"ai_provider_{report_scope}",
                        help="Gemini 使用本系統行情與可取得的新聞；ChatGPT 會額外搜尋近期市場、產業題材與同業資料並附來源。",
                    )
                    if st.button(f"🧠 啟動 AI 投資分析報告【{c_name}】", use_container_width=True, type="primary"):
                        provider_name = "ChatGPT" if ai_provider.startswith("ChatGPT") else "Gemini"
                        provider_key = openai_api_key_input if provider_name == "ChatGPT" else api_key_input
                        if not provider_key:
                            label = "ChatGPT API Key" if provider_name == "ChatGPT" else "Gemini API Key"
                            st.error(f"⚠️ 請先在左側邊欄設定 {label}，或改選另一個分析來源。")
                        else:
                            with st.spinner(f"🤖 正在取得歷史日線、執行樣本外回測並整理 {c_name} 研究報告…"):
                                try:
                                    daily_history = yf.download(
                                        selected_stock, period="5y", interval="1d", auto_adjust=False,
                                        actions=False, progress=False, threads=False,
                                    )
                                    ai_backtest = run_ai_candidate_backtests(daily_history, is_tw_stock)
                                    market_snapshot = []
                                    index_data = market_indexes
                                    if index_data is None:
                                        index_data = _market_overview_snapshots()
                                    for label, (value, _, index_change_pct, trading_day) in index_data.items():
                                        detail = f"{label} {value:,.2f}，日線資料日 {trading_day:%Y-%m-%d}"
                                        if index_change_pct is not None:
                                            detail += f"，較前一筆可用資料 {index_change_pct:+.2f}%"
                                        market_snapshot.append(detail)
                                    company_context = collect_stock_research_context(
                                        stock_obj_main, info_obj_main, market_snapshot
                                    )
                                    branch_snapshot = st.session_state.get(branch_ai_snapshot_key)
                                    broker_branch_ai_context = format_broker_branch_ai_context(
                                        branch_snapshot, selected_stock
                                    )
                                    broker_branch_ai_interpretation = ""
                                    if broker_branch_ai_context:
                                        ai_close_window = pd.to_numeric(df_chart["Close"], errors="coerce").dropna().tail(20)
                                        broker_branch_ai_interpretation = build_broker_branch_ai_interpretation(
                                            branch_snapshot, ai_close_window, selected_tf
                                        )
                                    ai_psychology = build_three_core_market_psychology_notes(
                                        gvi_val, momentum_score, fcf_yield_main, peg_val_main,
                                        momentum_window_text, explanation_style,
                                    )
                                    ai_psychology_lines = [
                                        f"• {factor}：{interpretation}"
                                        for factor, interpretation in ai_psychology["notes"].items()
                                    ]
                                    ai_psychology_lines.extend([
                                        f"> 🖍️ **畫線重點｜三核心合併觀察：** {ai_psychology['overall']}",
                                        f"• 價格與成交量：{build_price_volume_market_psychology(change_pct, vol_ratio, selected_tf)}",
                                        "• 解讀限制：以上是市場行為線索，不是對投資人身分或真實情緒的判定，也不是買賣訊號。",
                                    ])
                                    if broker_branch_ai_interpretation:
                                        ai_psychology_lines.insert(
                                            -1,
                                            "• 分點家數、淨額、流向集中度與近 20 筆價格：\n"
                                            + broker_branch_ai_interpretation,
                                        )
                                    market_psychology_brief = "\n".join(ai_psychology_lines)
                                    quote_index = df_chart.index[-1]
                                    quote_time = (quote_index.isoformat(sep=" ", timespec="minutes")
                                                  if isinstance(quote_index, pd.Timestamp) and pd.notna(quote_index)
                                                  else None)
                                    evidence_summary = build_research_evidence(
                                        price=price_val, quote_time=quote_time, timeframe=selected_tf,
                                        radar_evidence=[
                                            RadarEvidence(
                                                strategy, ai_psychology["checks"][psychology_key_by_strategy[strategy]],
                                                reason, detail_sources[strategy],
                                            )
                                            for strategy, reason in reasons.items()
                                        ],
                                        risk_findings=risk_findings,
                                    )
                                    score_summary = (
                                        f"{best_strategy}（{best_score}/100，這是條件比對分數，不是勝率）"
                                        if valid_scores else "資料不足，未評分"
                                    )
                                    change_summary = (
                                        f"{change_pct:+.2f}%" if len(df_chart) > 1 and prev_close > 0
                                        and _finite_number(change_pct) is not None else "資料不足"
                                    )
                                    price_summary = (
                                        f"{price_val:,.2f}" if _finite_number(price_val) is not None
                                        and price_val > 0 and quote_time else "資料不足"
                                    )
                                    quant_summary = (
                                        f"成長價值研究分數={gvi_str}；股東權益報酬率={roe_str}；股價淨值比={pb_str}；"
                                        f"目前研究比對分數最高的單項條件={score_summary}；"
                                        f"動能比值區間={momentum_window_text}；"
                                        f"價量動能估算={chip_status_text}；研究採用價格={price_summary}；最近收盤變化={change_summary}\n"
                                        f"三核心白話心理解讀：\n{market_psychology_brief}"
                                    )
                                    if broker_branch_ai_context:
                                        quant_summary += (
                                            f"\n\n{broker_branch_ai_context}\n\n"
                                            "以下是本次分析必須逐項解讀的分點與價格資料；請在市場心理觀察中引用具體數字，"
                                            "不得略過家數、淨額、集中度或近 20 筆價格方向：\n"
                                            f"{broker_branch_ai_interpretation}"
                                        )
                                    ai_prompt = build_ai_investment_prompt(
                                        selected_stock, c_name, provider_name, quant_summary,
                                        company_context,
                                        ai_backtest["text"]
                                        + "\n\n系統依樣本外門檻與最新日線整理的條件式建議：\n"
                                        + str(ai_backtest.get("advice", "回測資料不足，暫不提出策略建議。")),
                                        explanation_style=explanation_style,
                                        evidence_summary=evidence_summary,
                                    )
                                    if provider_name == "ChatGPT":
                                        success_report, ai_sources = generate_openai_research(
                                            ai_prompt, openai_api_key_input, openai_model_input
                                        )
                                    else:
                                        success_report = generate_content_with_retry(ai_prompt, api_key_input)
                                        ai_sources = []
                                    success_report = compose_research_report(
                                        evidence_summary, provider_name, success_report, ai_sources
                                    )
                                    if broker_branch_ai_interpretation:
                                        success_report = (
                                            str(success_report).rstrip()
                                            + "\n\n### 分點流向與近 20 筆價格方向｜資料補充解讀\n"
                                            + broker_branch_ai_interpretation
                                        )
                                    success_report = (
                                        str(success_report).rstrip()
                                        + "\n\n### 技術指標回測｜條件式建議\n"
                                        + "> 🖍️ **畫線重點｜回測後方向：** "
                                        + str(ai_backtest.get(
                                            "advice",
                                            "回測資料不足，暫不依技術指標回測提出策略建議。",
                                        ))
                                    )
                                    st.session_state[report_key] = {
                                        "text": success_report,
                                        "provider": provider_name,
                                        "backtest": ai_backtest,
                                        "sources": ai_sources,
                                        "psychology": market_psychology_brief,
                                        "branch_context": broker_branch_ai_context,
                                        "style": explanation_style,
                                        "evidence_version": 1,
                                    }
                                except Exception as ai_e:
                                    st.error(f"{provider_name} 投資研究報告產生失敗：{ai_e}")

                    saved_report = st.session_state.get(report_key)
                    if saved_report is not None:
                        report_provider = saved_report["provider"]
                        st.success(f"📋 {report_provider}【{c_name}】市場與策略研究報告")
                        if not saved_report.get("evidence_version"):
                            st.caption("這份報告早於查核摘要功能。重新產生即可套用目前的資料與來源標示。")
                        report_style = saved_report["style"]
                        if report_style != explanation_style:
                            st.caption(
                                f"此份報告以「{report_style}」產生；上方已切換為「{explanation_style}」。"
                                "如要更新報告文字，請重新執行 AI 分析。"
                            )
                        saved_branch_context = saved_report["branch_context"]
                        ai_report = saved_report["text"]
                        branch_report_heading = "### 分點流向與近 20 筆價格方向｜資料補充解讀"
                        report_has_branch_interpretation = branch_report_heading in ai_report
                        if current_branch_ai_interpretation and (
                            saved_branch_context != current_branch_ai_context
                            or not report_has_branch_interpretation
                        ):
                            st.warning("目前這份 AI 報告尚未納入目前工作階段的分點資料。請重新產生報告；以下先列出本次分點與價格的資料補充解讀。")
                            st.markdown("##### 分點流向與近 20 筆價格方向｜資料補充解讀")
                            st.markdown(current_branch_ai_interpretation)
                        saved_psychology_brief = saved_report["psychology"]
                        if saved_psychology_brief:
                            st.markdown("#### 🧠 市場心理重點")
                            st.markdown(saved_psychology_brief)
                        if saved_branch_context:
                            with st.expander("查看本次 AI 分析使用的分點摘要", expanded=False):
                                st.markdown(saved_branch_context)
                        if explanation_style.startswith("生動童趣"):
                            with st.expander("🧩 生活比喻版研究筆記", expanded=True):
                                st.markdown(ai_report)
                        else:
                            st.markdown(ai_report)
                        saved_backtest = saved_report["backtest"]
                        saved_backtest_table = saved_backtest.get("table") if isinstance(saved_backtest, dict) else None
                        if isinstance(saved_backtest_table, pd.DataFrame) and not saved_backtest_table.empty:
                            with st.expander("查看 AI 報告使用的全指標樣本外回測比較", expanded=False):
                                display_backtest = saved_backtest_table.copy()
                                for metric_column in (
                                    "前70%平均每筆損益(%)", "後30%平均每筆損益(%)",
                                    "後30%最大回撤(%)",
                                ):
                                    if metric_column in display_backtest:
                                        display_backtest[metric_column] = pd.to_numeric(
                                            display_backtest[metric_column], errors="coerce"
                                        ).round(2)
                                if "後30%賺賠金額比" in display_backtest:
                                    display_backtest["後30%賺賠金額比"] = pd.to_numeric(
                                        display_backtest["後30%賺賠金額比"], errors="coerce"
                                    ).round(2)
                                st.dataframe(display_backtest, use_container_width=True, hide_index=True)
                                st.caption(saved_backtest.get("assumptions", ""))
                                st.caption(
                                    "投資方向提示是依固定樣本外門檻與最新訊號產生的研究參考，未考量個人財務狀況、"
                                    "持倉成本或風險承受度；不代表個人化投資建議、買賣指令或建議部位。"
                                )
                        elif isinstance(saved_backtest, dict) and saved_backtest.get("text"):
                            st.info(saved_backtest["text"])
                        recipient_email = st.text_input(
                            "寄送分析報告到 Email",
                            key=f"ai_report_email_recipient_{report_scope}",
                            placeholder="請輸入收件信箱",
                        )
                        if st.button(
                            "✉️ 寄出這份分析",
                            key=f"send_ai_report_email_{report_scope}",
                            use_container_width=True,
                        ):
                            if not recipient_email.strip() or "@" not in recipient_email:
                                st.error("請輸入有效的收件 Email 地址。")
                            else:
                                try:
                                    gmail_sender, gmail_app_password = get_gmail_secret_values(st.secrets)
                                except Exception:
                                    gmail_sender, gmail_app_password = "", ""
                                if not gmail_sender or not gmail_app_password:
                                    st.error(
                                        "尚未設定 Gmail 寄件資料。請在 Streamlit Cloud 的 App Settings → Secrets "
                                        "加入 [gmail] 區段（sender_email、app_password），儲存後重新啟動。"
                                    )
                                else:
                                    email_subject = f"AI 個股分析｜{selected_stock} {c_name}"
                                    email_psychology_brief = saved_psychology_brief
                                    email_branch_context = current_branch_ai_context or saved_branch_context
                                    email_branch_interpretation = ""
                                    email_branch_is_current = bool(
                                        current_branch_ai_interpretation
                                        and saved_branch_context == current_branch_ai_context
                                        and report_has_branch_interpretation
                                    )
                                    if current_branch_ai_interpretation and not email_branch_is_current:
                                        email_branch_interpretation = (
                                            "【目前分點資料補充解讀｜AI 報告尚未重新產生】\n"
                                            + current_branch_ai_interpretation
                                            + "\n\n"
                                        )
                                    email_branch_section = (
                                        f"【本次 AI 分析使用的分點摘要】\n{email_branch_context}\n\n"
                                        if email_branch_context else ""
                                    )
                                    email_backtest_section = (
                                        "【技術指標樣本外回測比較與條件式建議】\n"
                                        + str(saved_backtest.get("text", "回測資料不足。"))
                                        + "\n\n【依回測條件整理的建議】\n"
                                        + str(saved_backtest.get("advice", "目前沒有可用的條件式建議。"))
                                        + "\n\n"
                                        if isinstance(saved_backtest, dict) else ""
                                    )
                                    email_body = (
                                        f"個股：{selected_stock} {c_name}\n"
                                        f"寄送時間：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                                        f"解說風格：{report_style}\n\n"
                                        f"{email_branch_section}"
                                        f"{email_branch_interpretation}"
                                        f"{email_backtest_section}"
                                        f"【市場心理重點】\n{email_psychology_brief}\n\n"
                                        f"{report_provider} 分析結果：\n\n{ai_report}"
                                    )
                                    try:
                                        with st.spinner("正在寄送分析報告…"):
                                            send_gmail_report(
                                                recipient_email,
                                                email_subject,
                                                email_body,
                                                gmail_sender,
                                                gmail_app_password,
                                            )
                                        st.success(f"分析報告已寄送至 {recipient_email.strip()}。")
                                    except Exception as mail_error:
                                        st.error(f"寄信失敗：{mail_error}")
                st.markdown("---")
            else:
                st.warning("Yahoo Finance 行情資料暫缺，這個標的與週期目前沒有可顯示的資料。")
                if show_radar_comparison:
                    _render_radar_risk(evaluate_radar_risk(
                        pd.DataFrame(), timeframe=selected_tf, cache_updated=db_row[6] if db_row else None,
                        cache_ttl=CACHE_TTL,
                    ))
        except Exception as e:
            st.error(f"數據載入異常：{e}")
    return df_chart, c_name, change_pct, date_strings, vol_ratio, price_val, grid_advice_metrics, is_tw_stock
