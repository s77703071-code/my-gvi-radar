"""Institutional flows and broker data research views."""

import pandas as pd
import streamlit as st

from .dividend import (
    _finite_number,
)
from .market_data import (
    TDCC_DISTRIBUTION_URL,
    _normalize_chip_trade_date,
    _render_tdcc_concentration,
    build_broker_branch_price_breadth_reading,
    build_chip_flow_summary,
    fetch_finmind_broker_branch,
    fetch_institutional_market_data,
    get_finmind_api_token,
    load_chip_flow_observations,
    save_broker_branch_ai_summary,
    save_chip_flow_observation,
    summarize_broker_branch_csv,
    summarize_broker_branch_frame,
)
from .workbench import (
    _clear_finmind_session_token,
)


def render_chip_flow(auth_user, df_chart, selected_stock, selected_tf):
    st.markdown("### 📦 市場走勢與資金流向參考")
    st.caption("觀察價格趨勢、成交熱度、三大法人當日動向，以及集保持股級距的週變化。沒有資料時會明確標示，不把缺漏當成零。")
    price_history = df_chart[[column for column in ("Close", "Volume") if column in df_chart.columns]].copy()
    institutional_trade = None
    issued_shares = None
    inst_errors = []
    stock_code = selected_stock.replace('.TWO', '').replace('.TW', '')
    is_taiwan_stock = selected_stock.endswith(('.TW', '.TWO'))
    if selected_stock.endswith(('.TW', '.TWO')):
        with st.spinner('讀取官方法人交易與已發行股數資料…'):
            inst_trades, inst_shares, inst_errors = fetch_institutional_market_data()
        institutional_trade = inst_trades.get(stock_code)
        issued_shares = inst_shares.get(stock_code)
    daily_volume = None
    trade_date = _normalize_chip_trade_date(institutional_trade.get('日期')) if institutional_trade else None
    if selected_tf == "1日" and trade_date and "Volume" in price_history.columns:
        for index_value, volume_value in pd.to_numeric(price_history["Volume"], errors="coerce").dropna().items():
            if _normalize_chip_trade_date(index_value) == trade_date:
                daily_volume = _finite_number(volume_value)
                break
    if is_taiwan_stock:
        save_chip_flow_observation(stock_code, institutional_trade, issued_shares, daily_volume)

    flow_summary = build_chip_flow_summary(price_history, institutional_trade, issued_shares, selected_tf)
    flow_cards = [
        ("trend", "最近價格走勢"),
        ("volume", "最近成交熱度"),
        ("institutions", "法人當日方向"),
    ]
    if selected_tf == "1日":
        flow_cards.append(("high_120", "近120日日線位置"))
    flow_cols = st.columns(len(flow_cards))
    available_statuses = []
    for col, (key, label) in zip(flow_cols, flow_cards):
        item = flow_summary.get(key)
        with col:
            st.metric(label, item["status"] if item else "資料暫缺")
            st.caption(item["detail"] if item else "目前資料不足或來源沒有回傳；不以零代替。")
        if item:
            available_statuses.append(item["status"].split("｜")[0])

    if available_statuses:
        st.info("簡單看：" + "；".join(available_statuses) + "。這些是近期市場狀況描述，不是單獨的買進或賣出訊號。")
    if selected_stock.endswith(('.TW', '.TWO')) and inst_errors:
        st.caption("法人來源目前未完整回傳；上方只顯示實際取得的資料。")
    st.caption("價格趨勢使用近 20 筆收盤價；成交熱度比較近 5 筆與前段平均成交量；法人方向使用官方當日三大法人資料。各項期間不同，請分開解讀。")

    if institutional_trade:
        investor_rows = []
        for investor_name in ("外資", "投信", "自營商"):
            net_shares = _finite_number(institutional_trade.get(f"{investor_name}買賣超股數"))
            net_pct = (net_shares / issued_shares * 100.0
                       if net_shares is not None and issued_shares and issued_shares > 0 else None)
            investor_rows.append({
                "法人類別": investor_name,
                "當日買賣超（股）": net_shares,
                "占已發行股數（%）": net_pct,
            })
        st.markdown("#### 🏦 三大法人當日明細")
        st.caption(f"資料日期：{institutional_trade.get('日期', '未提供')}；不同法人分開列示，缺漏資料保留空白。")
        st.dataframe(pd.DataFrame(investor_rows), hide_index=True, use_container_width=True)

    st.markdown("#### 📚 法人累積觀察（本系統開始記錄後）")
    history = load_chip_flow_observations(stock_code) if is_taiwan_stock else pd.DataFrame()
    if not history.empty:
        complete_history = history.dropna(subset=["foreign_net", "trust_net", "dealer_net"])
        st.caption(
            f"目前記錄 {len(history)} 個交易日，其中 {len(complete_history)} 筆含完整三大法人資料。"
            "只保存本頁曾成功取得的官方資料；未造補缺漏日期，也不代表連續交易日。"
        )
        history_cols = st.columns(2)
        for col, window in zip(history_cols, (5, 20)):
            label = f"最近 {window} 筆完整觀察合計"
            with col:
                if len(complete_history) >= window:
                    sample = complete_history.head(window)
                    total_net = float(sample[["foreign_net", "trust_net", "dealer_net"]].to_numpy().sum())
                    share_values = pd.to_numeric(sample["issued_shares"], errors="coerce").dropna()
                    share_pct = total_net / float(share_values.iloc[0]) * 100.0 if not share_values.empty and share_values.iloc[0] > 0 else None
                    st.metric(label, f"{total_net:+,.0f} 股", delta_color="off")
                    pct_text = f"（約 {share_pct:+.4f}% 已發行股數）" if share_pct is not None else ""
                    st.caption(
                        f"觀察區間：{sample.iloc[-1]['trade_date']} 至 {sample.iloc[0]['trade_date']}{pct_text}。"
                        "這是法人買賣超加總，不等於法人持股增減。"
                    )
                else:
                    st.metric(label, f"累積中 {len(complete_history)}/{window} 筆", delta_color="off")
                    st.caption("尚未累積足夠的完整官方資料，不顯示期間合計。")
        investor_period_rows = []
        for window in (5, 20):
            for column, investor_name in (
                ("foreign_net", "外資"),
                ("trust_net", "投信"),
                ("dealer_net", "自營商"),
            ):
                available = history.dropna(subset=[column]).head(window)
                row = {"觀察期間": f"最近 {window} 筆", "法人類別": investor_name}
                if len(available) >= window:
                    net_total = float(available[column].sum())
                    share_values = pd.to_numeric(available["issued_shares"], errors="coerce").dropna()
                    net_pct = (net_total / float(share_values.iloc[0]) * 100.0
                               if not share_values.empty and share_values.iloc[0] > 0 else None)
                    row["累計買賣超（股）"] = f"{net_total:+,.0f}"
                    row["約占已發行股數（%）"] = f"{net_pct:+.4f}%" if net_pct is not None else "資料暫缺"
                    volume_sample = history.dropna(subset=[column, "volume_shares"]).head(window)
                    if (len(volume_sample) >= window
                            and set(volume_sample["trade_date"]) == set(available["trade_date"])):
                        total_volume = float(pd.to_numeric(volume_sample["volume_shares"], errors="coerce").sum())
                        row["占同期成交量（%）"] = f"{net_total / total_volume * 100.0:+.2f}%" if total_volume > 0 else "資料暫缺"
                    else:
                        row["占同期成交量（%）"] = "日線成交量資料不足"
                    row["狀態"] = "已累積"
                else:
                    row["累計買賣超（股）"] = "—"
                    row["約占已發行股數（%）"] = "—"
                    row["占同期成交量（%）"] = "—"
                    row["狀態"] = f"累積中 {len(available)}/{window} 筆"
                investor_period_rows.append(row)
        st.dataframe(pd.DataFrame(investor_period_rows), hide_index=True, use_container_width=True)
        st.caption("上表按法人分開計算；成交量占比只在資料庫有同日線成交量時顯示。買賣超是交易流量，不能直接解讀為持股增加或未來漲跌。")
    elif is_taiwan_stock:
        st.info("資料累積中：成功取得每日官方資料後會依股票與交易日保存，足夠筆數前不顯示 5／20 筆合計。")
    else:
        st.info("法人累積觀察僅適用台股。")

    st.markdown("#### 👥 集保持股級距變化（週資料）")
    st.caption("以持股級距觀察小額與大額持有比例變化。這是持有分布，不代表買賣方向，也無法直接判定持有人是散戶或主力。")
    if is_taiwan_stock:
        st.caption(f"資料來源：[臺灣集中保管結算所]({TDCC_DISTRIBUTION_URL})")
        _render_tdcc_concentration(selected_stock)
    else:
        st.info("集保持股級距資料目前僅適用台股。")

    st.caption(
        f"個股分點參考目前對應 {stock_code}；要查看其他股票，請先在左側更換分析標的，"
        "並匯入該股票的券商分點 CSV。"
    )
    with st.expander("🔎 選用：券商分點買賣明細", expanded=False):
        st.caption("分點資料反映券商帳戶交易彙整，無法辨認最終投資人身分，也不等於主力或散戶流向。")
        if not is_taiwan_stock:
            st.info("券商分點資料目前僅適用台股。")
        else:
            def _render_branch_summary(branch_summary, source_label):
                if not branch_summary:
                    return
                if branch_summary.get('error'):
                    st.warning(branch_summary['error'])
                    return
                branch_cols = st.columns(3)
                count_suffix = "家" if branch_summary["count_unit"] == "分點" else "列"
                with branch_cols[0]:
                    st.metric("分點淨額合計（核對）", f"{branch_summary['net_shares']:+,.0f} 股", delta_color="off")
                with branch_cols[1]:
                    st.metric(f"淨買超{branch_summary['count_unit']}家數", f"{branch_summary['buyers']:,} {count_suffix}", delta_color="off")
                with branch_cols[2]:
                    st.metric(f"淨賣超{branch_summary['count_unit']}家數", f"{branch_summary['sellers']:,} {count_suffix}", delta_color="off")
                buy_share = branch_summary.get("top_net_buy_share_pct")
                sell_share = branch_summary.get("top_net_sell_share_pct")
                buy_share_text = f"，約占全部淨買超 {buy_share:.1f}%" if buy_share is not None else ""
                sell_share_text = f"，約占全部淨賣超 {sell_share:.1f}%" if sell_share is not None else ""
                st.caption(
                    f"{source_label}；淨買超前 15 個{branch_summary['count_unit']}合計 "
                    f"{branch_summary['top_net_buy']:+,.0f} 股{buy_share_text}；淨賣超前 15 個"
                    f"{branch_summary['count_unit']}合計 {branch_summary['top_net_sell']:+,.0f} 股{sell_share_text}。"
                )
                if buy_share is not None and sell_share is not None:
                    if buy_share > sell_share:
                        concentration_label = "買方分點相對集中"
                        concentration_delta = f"買方高 {buy_share - sell_share:.1f} 個百分點"
                    elif sell_share > buy_share:
                        concentration_label = "賣方分點相對集中"
                        concentration_delta = f"賣方高 {sell_share - buy_share:.1f} 個百分點"
                    else:
                        concentration_label = "買賣兩側相同"
                        concentration_delta = "前 15 個分點占比相同"
                    concentration_value = f"買 {buy_share:.1f}%｜賣 {sell_share:.1f}%"
                elif buy_share is not None:
                    concentration_label = "賣方集中度資料不足"
                    concentration_delta = "僅有淨買超占比"
                    concentration_value = f"買 {buy_share:.1f}%｜賣—"
                elif sell_share is not None:
                    concentration_label = "買方集中度資料不足"
                    concentration_delta = "僅有淨賣超占比"
                    concentration_value = f"買—｜賣 {sell_share:.1f}%"
                else:
                    concentration_label = "資料不足"
                    concentration_delta = None
                    concentration_value = "—"
                st.metric(
                    "分點淨流向集中度（前 15 個）",
                    concentration_value,
                    delta=concentration_label if concentration_delta is None else concentration_delta,
                    delta_color="off",
                    help="比較淨買超與淨賣超各自前 15 個分點，占該方向全部淨額的比例。這只表示券商交易渠道的相對集中程度，不能判定背後是主力或散戶。",
                )
                st.caption(
                    f"集中度比較：{concentration_label}。這表示淨買／賣超金額在少數分點的分布差異，"
                    "不是持股集中度，也無法由券商分點辨認主力或散戶身分。"
                )
                st.caption(
                    "讀法：買超／賣超家數只表示各分點淨額為正或負的數量，不代表散戶／大戶人數。"
                    "在資料涵蓋完整且口徑一致時，分點淨額合計接近 0 常是買賣相抵的核對結果，並非方向訊號；"
                    "前 15 比例反映分點淨額集中程度，也不能識別最終投資人。"
                )
                trend_item = flow_summary.get("trend")
                trend_status = trend_item.get("status", "") if trend_item else ""
                if "偏強" in trend_status:
                    trend_label = "偏強"
                elif "偏弱" in trend_status:
                    trend_label = "偏弱"
                elif "整理" in trend_status:
                    trend_label = "整理"
                else:
                    trend_label = "資料不足"
                net_shares = branch_summary["net_shares"]
                branch_direction = "淨買超" if net_shares > 0 else "淨賣超" if net_shares < 0 else "買賣接近平衡"
                st.info(
                    f"個股參考（{stock_code}）：近期價格走勢{trend_label}；"
                    f"該分點資料期間合計{branch_direction} {abs(net_shares):,.0f} 股。"
                    "價格趨勢與分點資料涵蓋期間可能不同，請分開解讀；這是市場觀察參考，不代表因果或買賣訊號。"
                )
                if trend_item:
                    price_change_pct = trend_item.get("period_change_pct")
                    period_bars = int(trend_item.get("period_bars", 20) or 20)
                    st.markdown(
                        "**📈 股價方向 × 分點家數：** "
                        + build_broker_branch_price_breadth_reading(branch_summary, price_change_pct)
                    )
                    st.caption(
                        f"價格方向取最近 {period_bars} 筆「{selected_tf}」收盤資料；分點家數取{source_label}。"
                        "若 CSV 沒有資料日期，兩者無法確認為同日或同期間；這是方向搭配觀察，不是統計相關係數。"
                        "要判斷是否穩定同向，需累積多個日期且日期對齊的分點與股價資料。"
                    )
                st.markdown("##### 🧠 市場心理觀察：怎麼理解這些分點數字？")
                buyer_count = int(branch_summary["buyers"])
                seller_count = int(branch_summary["sellers"])
                if buyer_count > seller_count:
                    st.markdown(
                        f"**參與廣度：** 淨買超分點有 {buyer_count} 個，淨賣超分點有 {seller_count} 個，"
                        "表示這份資料中淨買方分點參與面較廣。這可以作為買方參與較廣的線索，"
                        "但不能推論買方是散戶，或賣方是主力。"
                    )
                elif seller_count > buyer_count:
                    st.markdown(
                        f"**參與廣度：** 淨賣超分點有 {seller_count} 個，淨買超分點有 {buyer_count} 個，"
                        "表示這份資料中淨賣方分點參與面較廣。這可以作為賣方參與較廣的線索，"
                        "但不能推論賣方是散戶，或買方是主力。"
                    )
                else:
                    st.markdown(
                        f"**參與廣度：** 淨買超與淨賣超分點數相同（各 {buyer_count} 個）；"
                        "家數本身沒有顯示哪一方參與面較廣。"
                    )

                concentration_notes = []
                if buy_share is not None:
                    concentration_notes.append(
                        f"前 15 個淨買超分點占全部淨買超 {buy_share:.1f}%"
                    )
                if sell_share is not None:
                    concentration_notes.append(
                        f"前 15 個淨賣超分點占全部淨賣超 {sell_share:.1f}%"
                    )
                if concentration_notes:
                    st.markdown(
                        "**淨流向集中度：** " + "；".join(concentration_notes) +
                        "。比例較高表示該方向的淨額較集中在排名靠前的分點；它不是持股集中度，也不能辨認分點背後的投資人。"
                    )
                if trend_label != "資料不足":
                    st.markdown(
                        f"**價格情緒線索：** 近 20 筆收盤價相對平均價呈{trend_label}；"
                        "可與分點流向一起觀察市場是否同向，但兩者期間若不一致，就不能據此判定背離或趨勢延續。"
                    )
                st.caption(
                    "市場心理不是這些欄位直接測得的結果。單日或單份 CSV 只能描述該資料範圍的成交分布；"
                    "請勿只憑分點家數判定吸籌、出貨或預測漲跌。"
                )
                breakdown = branch_summary.get("breakdown")
                if isinstance(breakdown, pd.DataFrame) and not breakdown.empty:
                    top_rows = pd.concat([breakdown.head(10), breakdown.nsmallest(10, "買賣超")]).drop_duplicates()
                    st.dataframe(top_rows, hide_index=True, use_container_width=True)

            st.markdown("##### 選擇分點資料來源")
            st.caption("可輸入自己的 FinMind Token 自動帶入，或改用 CSV；FinMind 資料通常於交易日盤後更新。")
            st.warning(
                "授權提醒：FinMind 將商業 web／app 使用列為 Sponsor Pro 授權範圍，並禁止原始資料再散布或建立鏡像 API。"
                "即使使用者自行輸入 Token，也不會自動取得本網站展示資料的授權；公開提供前請管理者向 FinMind 確認。"
            )
            st.markdown("[查看 FinMind 方案與資料授權](https://finmind.github.io/en/Pricing/)")

            account_id = str(auth_user.get("id", "guest"))
            user_token_key = f"finmind_user_token_{account_id}"
            source_key = f"branch_data_source_{account_id}"
            shared_token = get_finmind_api_token()
            source_options = ["輸入自己的 FinMind Token（自動帶入）", "手動上傳 CSV"]
            if shared_token:
                source_options.insert(1, "使用網站管理者設定的共用 Token")
            if st.session_state.get(source_key) not in source_options:
                st.session_state[source_key] = source_options[0]
            st.caption("自填 Token 需要每位使用者自己的 FinMind Sponsor 以上方案，且方案需包含券商分點資料。")
            branch_source = st.radio(
                "請選擇一種方式",
                source_options,
                key=source_key,
                help="自填 Token 只保留在目前登入工作階段；CSV 不需要 API Token。",
            )

            if branch_source == "輸入自己的 FinMind Token（自動帶入）":
                st.caption(
                    "請使用你自己的 FinMind Token。Token 會由本 App 伺服器代為呼叫 FinMind；"
                    "本版本不會寫入帳號資料庫，登出時會清除。請只在你信任的服務輸入 Token。"
                )
                finmind_token = st.text_input(
                    "你的 FinMind API Token",
                    type="password",
                    key=user_token_key,
                    help="券商分點 API 需要你的 FinMind 帳號方案包含此資料集。",
                ).strip()
                token_ack_key = f"finmind_token_ack_{account_id}"
                st.button(
                    "清除目前輸入的 Token",
                    key=f"clear_finmind_token_{account_id}",
                    on_click=_clear_finmind_session_token,
                    args=(user_token_key, token_ack_key),
                )
                if finmind_token:
                    token_acknowledged = st.checkbox(
                        "我確認此 Token 屬於我、我的方案包含券商分點資料，並同意本次由 App 伺服器代為查詢。",
                        key=token_ack_key,
                    )
                else:
                    token_acknowledged = False
                    st.info("請輸入自己的 FinMind Token；若不使用 API，請改選「手動上傳 CSV」。")
                active_finmind_token = finmind_token
                active_finmind_ack = token_acknowledged
                source_label = "FinMind（個人 Token）"

            elif branch_source == "使用網站管理者設定的共用 Token":
                st.info("此 Token 由網站管理者設定，所有選用此方式的使用者會共用同一資料方案與查詢額度。")
                token_ack_key = f"finmind_shared_ack_{account_id}"
                active_finmind_ack = st.checkbox(
                    "我了解這是網站共用資料來源，並同意本次查詢。",
                    key=token_ack_key,
                )
                active_finmind_token = shared_token
                source_label = "FinMind（網站共用 Token）"

            else:
                active_finmind_token = ""
                active_finmind_ack = False
                source_label = ""

            if branch_source == "手動上傳 CSV":
                st.info(
                    f"目前分析標的是 {stock_code}。分點 CSV 僅用來分析該檔股票；如要分析其他股票，"
                    "請先在左側更換股票代碼，再匯入該股票的 CSV。"
                )
                branch_file = st.file_uploader(
                    "選擇券商分點 CSV 檔",
                    type=["csv"],
                    key=f"chip_branch_upload_{account_id}_{selected_stock}",
                )
                if branch_file is not None:
                    branch_summary = summarize_broker_branch_csv(branch_file, stock_code)
                    saved_branch_context = save_broker_branch_ai_summary(
                        st.session_state, account_id, stock_code, branch_summary, "使用者上傳 CSV"
                    )
                    if saved_branch_context:
                        st.caption("已把這檔股票的分點摘要暫存於目前登入工作階段；啟動 AI 報告時會一併使用摘要，不傳送完整分點明細。")
                    _render_branch_summary(branch_summary, "依上傳檔案所含資料期間彙總")
            elif active_finmind_token and active_finmind_ack:
                from datetime import datetime as _datetime
                from zoneinfo import ZoneInfo as _ZoneInfo
                default_branch_date = pd.Timestamp(trade_date).date() if trade_date else _datetime.now(_ZoneInfo("Asia/Taipei")).date()
                branch_trade_date = st.date_input(
                    "分點資料日期",
                    value=default_branch_date,
                    key=f"chip_branch_date_{account_id}_{stock_code}",
                )
                branch_date_text = branch_trade_date.strftime("%Y-%m-%d")
                with st.spinner(f"自動讀取 {stock_code} 的券商分點資料…"):
                    branch_frame, branch_error = fetch_finmind_broker_branch(
                        stock_code, branch_date_text, active_finmind_token, account_scope=account_id
                    )
                if branch_error:
                    save_broker_branch_ai_summary(
                        st.session_state, account_id, stock_code,
                        {"error": branch_error}, f"{source_label}｜{branch_date_text}",
                    )
                    st.info(f"FinMind：{branch_error}")
                else:
                    auto_summary = summarize_broker_branch_frame(branch_frame, stock_code)
                    saved_branch_context = save_broker_branch_ai_summary(
                        st.session_state, account_id, stock_code, auto_summary,
                        f"{source_label}｜{branch_date_text}",
                    )
                    if saved_branch_context:
                        st.caption("已把此日期的分點摘要暫存於目前登入工作階段；啟動 AI 報告時會一併使用摘要，不傳送完整分點明細。")
                    _render_branch_summary(auto_summary, f"{source_label}｜{branch_date_text}")
            elif branch_source != "手動上傳 CSV" and active_finmind_token and not active_finmind_ack:
                st.info("請先完成上方確認，才會使用所選 Token 發出查詢。")
