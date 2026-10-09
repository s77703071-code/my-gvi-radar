# ---------------------------------------------------------
                # 華爾街智慧估值雷達與操盤手報告動態診斷邏輯
                # ---------------------------------------------------------
                if gvi_val is not None and pb_val is not None:
                    if gvi_val >= 0.35 and pb_val <= 1.5:
                        valuation_color = "#00cc66"
                        valuation_status = "🔥 極度便宜（有安全邊際，機構瘋狂撿便宜區）"
                        valuation_desc = "內在價值強勁但估值嚴重低估！屬於下檔風險鎖死、長線大送分的黃金買點。"
                    elif gvi_val >= 0.20 or (pb_val > 1.5 and pb_val <= 3.5 and roe_val >= 0.12):
                        valuation_color = "#2baf2b"
                        valuation_status = "🟢 合理甜美（體質估值相稱，長線穩健布局期）"
                        valuation_desc = "股價完美對位體質，沒有嚴重泡沫或主力刻意打壓，屬長線基金安全期。"
                    elif pb_val > 3.5 and pb_val <= 7.0:
                        valuation_color = "#ff9900"
                        valuation_status = "⚠️ 偏貴溢價（樂觀情緒透支，操盤手需嚴格風控）"
                        valuation_desc = "股價已提前預支未來 1-2 年的獲利。追高性價比低，進場必須嚴守破均線短線停損。"
                    else:
                        valuation_color = "#cc0000"
                        valuation_status = "🚨 泡沫嚴重（全面避開提款機，估值嚴重偏離）"
                        valuation_desc = "投機情緒沸騰！PB極高且ROE無法支撐，主力隨時可能倒貨提款，切勿盲目進場當接盤俠。"
                else:
                    valuation_color = "#888888"
                    valuation_status = "⚠️ 基本面資料不足（系統停止計算 GVI）"

                    # 🔍 動態排查缺值具體原因
                    reasons = []
                    if price_val is None or price_val <= 0:
                        reasons.append("❌ <b>股價異常</b>：查無即時報價或標的停牌 ($P \\le 0$)")
                    
                    if bv_val is None:
                        reasons.append("❌ <b>每股淨值 (BV) 缺失</b>：可能為 ETF/債券/REITs 等非普通股，或財報未揭露")
                    elif bv_val <= 0:
                        reasons.append("❌ <b>每股淨值為負 ($BV \\le 0$)</b>：公司長期虧損導致淨值侵蝕（財務危機/全額交割）")

                    if roe_val is None:
                        reasons.append("❌ <b>ROE 缺失/無法推算</b>：當期連續虧損 (EPS < 0) 導致交易所 PE 標示 N/A 無法推算代用值，或財報缺值")

                    reason_details = "<br/>&nbsp;&nbsp;&nbsp;&nbsp;• " + "<br/>&nbsp;&nbsp;&nbsp;&nbsp;• ".join(reasons)
                    
                    valuation_desc = (
                        f"<b>【基本面缺值動態診斷】</b>{reason_details}<br/>"
                        f"💡 <b>操盤手安全提醒：</b>根據防護機制，系統嚴禁使用隨機數或預設假值補缺。若標的為連續虧損個股或 ETF，出現「資料不足」屬於防範錯估的正常避險現象。"
                    )

                # 渲染操盤手報告卡片
                st.markdown(
                    f"<div style='background-color:rgba(30,30,30,0.7); padding:14px 18px; border-left:6px solid {valuation_color}; border-radius:4px; margin-bottom:15px;'>"
                    f"<h5 style='margin:0; color:white;'>⚖️ 華爾街智慧估值雷達：<span style='color:{valuation_color}; font-weight:bold;'>{valuation_status}</span></h5>"
                    f"<p style='margin:6px 0 0 0; font-size:14px; color:#cccccc;'>{valuation_desc}<br/><small style='color:#aaaaaa;'>註：GVI 門檻 (0.35/0.20) 係依據 5 年淨值複合成長經驗模型設定之篩選條件，非通用會計絕對標準。</small></p>"
                    f"</div>", 
                    unsafe_allow_html=True
                )