"""Market psychology explanations used by multiple strategies."""

from .dividend import (
    _finite_number,
)


def build_three_core_market_psychology_notes(gvi_value, momentum_pct, fcf_yield_pct, peg_value,
                                            momentum_window="近一年",
                                            explanation_style="一般白話｜專業易讀"):
    """Translate three research factors into cautious, plain-language behavior cues."""
    gvi = _finite_number(gvi_value)
    momentum = _finite_number(momentum_pct)
    fcf_yield = _finite_number(fcf_yield_pct)
    peg = _finite_number(peg_value)
    momentum_window = str(momentum_window or "目前可用資料區間")
    notes = {}
    checks = {}

    if gvi is None:
        notes["GVI 價值面"] = "估值資料不足，暫時無法判讀價值條件可能如何影響市場關注。"
        checks["GVI 價值面"] = None
    elif gvi >= 0.20:
        notes["GVI 價值面"] = (
            "目前達到價值研究門檻，可能吸引重視獲利能力與帳面價值的投資者進一步研究；"
            "仍要核對財報日期、產業差異及獲利是否持續，不能只因分數較高就認定便宜。"
        )
        checks["GVI 價值面"] = True
    else:
        notes["GVI 價值面"] = (
            "目前未達價值研究門檻，表示這項價值假設支持較弱；不代表市場一定高估，"
            "也可能是獲利、成長或產業風險需要進一步查證。"
        )
        checks["GVI 價值面"] = False

    if momentum is None:
        notes["動能趨勢面"] = "有效價格資料不足，暫時不能判讀趨勢延續或追價情緒。"
        checks["動能趨勢面"] = None
    elif momentum >= 80:
        notes["動能趨勢面"] = (
            f"價格接近{momentum_window}可用收盤高位，表示趨勢延續較受關注，也可能吸引順勢買盤；"
            "同時要留意追價與交易過度擁擠的風險，不能把強勢直接當成後續上漲保證。"
        )
        checks["動能趨勢面"] = True
    else:
        notes["動能趨勢面"] = (
            f"價格離{momentum_window}可用收盤高位較遠，順勢追價的支持相對較弱；"
            "這不等於股票便宜，可能是整理或趨勢轉弱，需搭配其他資料判讀。"
        )
        checks["動能趨勢面"] = False

    if fcf_yield is None or peg is None:
        notes["QARP 現金流與估值面"] = "現金流或成長估值資料不足，暫時不能判讀這項條件。"
        checks["QARP 現金流與估值面"] = None
    elif fcf_yield >= 2.5 and peg <= 1.5:
        notes["QARP 現金流與估值面"] = (
            "目前現金流與成長估值條件同時達標，可能吸引重視現金創造力及價格合理性的投資者；"
            "仍須確認現金流能否延續、成長估計是否可靠，避免把低估值直接當成安全。"
        )
        checks["QARP 現金流與估值面"] = True
    else:
        notes["QARP 現金流與估值面"] = (
            "目前至少一項現金流或成長估值條件未達標；這可能反映現金創造力、成長預期或價格要求需要再查證，"
            "不能單憑未達標就說市場過度悲觀或股票太貴。"
        )
        checks["QARP 現金流與估值面"] = False

    available_checks = [value for value in checks.values() if value is not None]
    passed_count = sum(value is True for value in available_checks)
    if not available_checks:
        overall = "三個研究角度的有效資料不足，先補齊資料再比較。"
    elif passed_count == len(available_checks):
        overall = (
            "目前可評估的研究條件方向一致，可能讓投資者對基本面或趨勢較有信心；"
            "一致不等於已證明有優勢，仍要檢查共同資料偏差與樣本外結果。"
        )
    elif passed_count == 0:
        overall = (
            "目前可評估的研究條件都未達門檻，代表本次篩選支持較少；"
            "不要為了得到看多結論而臨時放寬條件。"
        )
    else:
        overall = (
            "三個研究角度出現分歧，這是提醒使用者避免確認偏誤的時點："
            "不要只挑支持原有看法的指標，應先找出價值、趨勢與現金流訊號不一致的原因。"
        )
    if str(explanation_style).startswith("生動童趣"):
        notes = {
            "GVI 價值面": (
                "像看公司這棵樹最近結了多少果實：目前達到設定的研究線，可能有人想多看一眼；"
                "還要確認果實年年都有，不能只憑這次分數說股票便宜。"
                if checks["GVI 價值面"] is True else
                "這次價值條件沒有達到設定線，像考卷有一題還不夠好；不代表公司一定不好或股票一定貴。"
                if checks["GVI 價值面"] is False else
                "價值資料缺幾片拼圖，先不猜它便宜還是昂貴。"
            ),
            "動能趨勢面": (
                f"價格接近{momentum_window}的可用高點，像跑者跑得很快，容易吸引人跟上；"
                "也可能突然喘一下，不能把強勢當成保證。"
                if checks["動能趨勢面"] is True else
                f"價格離{momentum_window}的可用高點還有距離，趨勢支持比較弱；"
                "這不等於便宜，可能是在休息，也可能正在轉弱。"
                if checks["動能趨勢面"] is False else
                "歷史價格資料太少，像比賽只看了一小段影片，先不判斷跑勢。"
            ),
            "QARP 現金流與估值面": (
                "現金流和成長估值兩題都達到設定線，可能吸引重視真實現金的投資人；"
                "還要確認公司未來能不能繼續產生現金。"
                if checks["QARP 現金流與估值面"] is True else
                "現金流或成長估值至少一題沒達到設定線；先找出是哪一題，不能直接說公司很差或太貴。"
                if checks["QARP 現金流與估值面"] is False else
                "現金流或成長資料不夠，像少了幾題考卷，先不要急著打分數。"
            ),
        }
        if not available_checks:
            overall = "三張考卷的資料都不夠，先補資料再比較。"
        elif passed_count == len(available_checks) and len(available_checks) == 3:
            overall = "三個角度目前都達到研究線，像三張考卷都過了設定分數；這不代表下一次一定考得好。"
        elif passed_count == 0:
            overall = "目前已評估的條件都沒達到研究線；不要為了得到想要的答案而偷偷降低標準。"
        else:
            overall = "三個角度的答案不一樣，像拼圖還沒拼好；先找出價值、趨勢和現金流為什麼不一致。"
    return {"notes": notes, "checks": checks, "overall": overall}


def build_price_volume_market_psychology(change_pct, volume_ratio, timeframe_name):
    """Describe price/volume behavior without claiming to know trader identities or emotions."""
    change = _finite_number(change_pct)
    ratio = _finite_number(volume_ratio)
    interval = str(timeframe_name or "目前圖表週期")
    if change is None or ratio is None:
        return f"{interval}的價格或成交量資料不足，暫時不解讀市場參與熱度。"
    if change > 0 and ratio >= 1.0:
        return (
            f"{interval}價格上升且成交量不低於近期平均，表示這段走勢的參與度較高；"
            "可能有趨勢認同或追價行為，但成交量不能辨別買賣雙方身分。"
        )
    if change < 0 and ratio >= 1.0:
        return (
            f"{interval}價格下跌且成交量不低於近期平均，表示這段期間交易較活躍並伴隨價格走弱；"
            "可留意賣壓或風險重新評估，但不能單憑這點判定恐慌出場。"
        )
    if change > 0:
        return (
            f"{interval}價格上升但成交量低於近期平均，走勢偏強而參與度沒有同步增加；"
            "避免把少量成交的上漲直接當成市場普遍認同。"
        )
    if change < 0:
        return (
            f"{interval}價格下跌但成交量低於近期平均，表示價格走弱時成交參與相對較淡；"
            "可能是觀望或買盤不足，仍需觀察後續變化。"
        )
    return f"{interval}價格變化接近持平；可視為目前方向不明，避免只憑單一週期推斷市場情緒。"


def _render_market_psychology_guide(st, selected_module, explanation_style="一般白話｜專業易讀"):
    """Show a behavior-focused guide in the selected language and layout style."""
    guides = {
        "📊 彩色 K 線圖與成交量": "K 線和成交量呈現已發生的交易結果。價升量增可能代表參與度提高，也可能伴隨追價；單根 K 棒不能辨認投資人身分或確定情緒。",
        "📦 籌碼流向參考": "分點家數反映淨買方或淨賣方參與的分點廣度；分點不是投資人身分。再看淨額集中程度、價格和資料期間，避免直接貼上主力吸籌或散戶接盤標籤。",
        "🤖 網格自動生成器": "網格較依賴價格來回波動；人在震盪時容易想低買高賣，但遇到單邊行情可能太早賣出強勢股，或在下跌中不斷加碼。先確認資金上限和價格區間，不要把近期高低點當成必然邊界。",
        "⚖️ 網格／持有／定期定額比較": "買進持有考驗承受回撤的耐心；定期定額降低一次押時點的壓力，但不會消除虧損；網格提供規則化操作，卻可能在單邊行情落後。比較時先看自己能否承受過程中的下跌與錯過上漲。",
        "🧪 技術指標自訂策略回測": "回測容易讓人過度相信剛好適合歷史行情的參數。先看較晚期間、交易筆數、成本和最大跌幅，再確認規則能否照做；不要只挑最好看的回測結果。",
        "💸 配息複利策略": "高殖利率容易讓人只盯著現金入帳，忽略股價下跌或配息不持續。配息不是額外免費報酬；先看現金流和發放能力，再看總資產變化。",
        "🚀 華爾街機構級三核心策略雷達": "價值條件容易讓人錨定帳面價值；動能條件可能放大追漲或錯過恐懼；現金流與估值條件則要防止把便宜誤當安全。三者不一致時，先查原因，不要只挑符合原有看法的指標。",
        "🏦 台股三大法人獨立選股策略": "法人買賣超是已公布的交易流量，不等於持股變化，也不是未來方向保證。看到法人買超容易產生跟隨心理；應同時看日期、連續性、股價和成交量。",
        "📊 當日全市場三大法人佔股本比 Top 10": "排行容易讓人把『名次高』誤當成『勝率高』。單日法人流量只說明當日交易相對股本的大小；避免追逐榜單，先確認資料日期和是否連續出現。",
        "🧭 投資組合理財": "成本價容易成為心理錨點，虧損時也可能讓人不願重新評估。預先設定配置和風險檢查門檻，有助於降低情緒化決策；門檻仍須符合個人承受能力，不代表最佳配置。",
    }
    playful_guides = {
        "📊 彩色 K 線圖與成交量": "把 K 線想成價格的天氣紀錄，成交量像是今天有多少人來操場。人變多不代表明天一定漲，還要看價格往哪裡走。",
        "📦 籌碼流向參考": "券商分點像很多個售票窗口；窗口數量多，不代表背後的人一定是散戶或大戶。還要一起看買賣金額、集中程度、日期和股價。",
        "🤖 網格自動生成器": "網格像在價格範圍裡放一排階梯，漲到一格就照規則賣、跌到一格就照規則買。若價格一直往同一方向跑，階梯可能不再適合。",
        "⚖️ 網格／持有／定期定額比較": "三種方法像三種走路方式：一次出發、分次出發、或照價格階梯調整。哪一種比較適合，要看途中跌多少、花多少成本，不能只看終點誰最高。",
        "🧪 技術指標自訂策略回測": "回測像拿舊比賽錄影練習：以前這樣做的結果可以參考，但下一場可能不一樣。要看練習次數、扣掉費用後的結果和最深跌幅。",
        "💸 配息複利策略": "股息像公司把一部分成果分給股東，不是憑空多出一筆錢。若股價下跌或公司少發股息，收到的錢可能補不回去。",
        "🚀 華爾街機構級三核心策略雷達": "三核心像三張不同科目的考卷：價值、趨勢、現金流。三張答案不一樣時，要先找原因；高分只是符合研究條件，不是保證賺錢。",
        "🏦 台股三大法人獨立選股策略": "法人買賣超像公告的當日收據，不是他們手上還剩多少，也不能預告明天。先看日期、連續幾天，再和股價、成交量一起看。",
        "📊 當日全市場三大法人佔股本比 Top 10": "Top 10 像今天的排行榜，只告訴我們今天誰的交易比較大，不代表名次高就比較會漲。先看日期，再看後面幾天有沒有延續。",
        "🧭 投資組合理財": "投資組合像把積木分成幾盒，不要只盯著其中一盒。要看每盒比例、總共可能跌多少，還要留意成本價會不會讓自己捨不得重新檢查。",
    }
    note = guides.get(selected_module)
    if note:
        if explanation_style.startswith("生動童趣"):
            with st.expander(f"🧩 生活比喻版｜{selected_module}", expanded=True):
                st.success(playful_guides.get(selected_module, note))
                st.caption("這些比喻只幫助理解資料，不代表明天走勢，也不是買賣訊號。")
        else:
            st.info(f"🧠 市場心理重點｜{selected_module}\n\n{note}")
            st.caption("這些是可能的行為線索，不是對投資人心理的直接測量，也不是買賣訊號。")
