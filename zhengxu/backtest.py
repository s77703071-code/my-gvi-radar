"""Technical indicator and AI research backtest calculations."""

import numpy as np
import pandas as pd

from .dividend import (
    _finite_number,
)


def run_ai_candidate_backtests(history, is_taiwan_stock):
    """Compare the full built-in indicator strategy set using chronological holdout data."""
    if history is None or history.empty:
        no_data = "無法下載可用日線歷史資料，因此沒有進行策略回測，也不提供依回測條件產生的策略建議。"
        return {"table": pd.DataFrame(), "best": None, "text": no_data, "advice": no_data}
    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in required):
        no_data = "日線資料缺少必要的開高低收或成交量欄位，沒有進行策略回測，也不提供依回測條件產生的策略建議。"
        return {"table": pd.DataFrame(), "best": None, "text": no_data, "advice": no_data}
    frame = frame[required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).sort_index()
    frame = frame.loc[frame["Close"] > 0]
    if len(frame) < 504:
        return {
            "table": pd.DataFrame(), "best": None,
            "text": f"目前只有 {len(frame)} 筆有效日線（至少需要約 2 年資料），不足以分成前 70% 與後 30% 做基本樣本外比較。",
            "advice": "目前有效日線不足以做樣本外檢查，因此不提供策略進場建議；請補足歷史資料或先觀察。",
        }

    split = int(len(frame) * 0.70)
    indicator_frame = compute_kline_indicator_frame(frame)
    fee = 0.1425 if is_taiwan_stock else 0.0
    tax = 0.3 if is_taiwan_stock else 0.0
    slip = 0.05
    candidates = [
        {
            "name": "五線開花（MA5／10／20／60／120）",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "five_ma_mode": "五線開花", "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "five_ma_mode": "排列失序", "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用"},
        },
        {
            "name": "MACD 黃金／死亡交叉",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "黃金交叉", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "死亡交叉", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用"},
        },
        {
            "name": "KD 黃金／死亡交叉",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "黃金交叉", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "死亡交叉", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用"},
        },
        {
            "name": "RSI 超賣回升／超買回落",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "超賣回升 (RSI向上突破30)", "bb_mode": "停用", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "超買警戒 (RSI跌破70)", "bb_mode": "停用"},
        },
        {
            "name": "布林通道下軌反彈／回到中軌",
            "entry": {"ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "觸及布林下軌反彈", "match_mode": "ALL"},
            "exit": {"stop_loss_pct": 8.0, "trailing_stop_pct": 12.0, "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "站上布林中軌"},
        },
    ]

    def add_candidate(name, conditions, entry_updates, exit_updates):
        entry = {
            "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [], "five_ma_mode": "停用",
            "kd_mode": "停用", "macd_mode": "停用", "vol_mode": "停用",
            "rsi_mode": "停用", "bb_mode": "停用", "bias_mode": "停用",
            "atr_mode": "停用", "atr_pct_limit": 4.0, "obv_mode": "停用",
            "cci_mode": "停用", "adx_mode": "停用", "adx_threshold": 20.0,
            "ema_mode": "停用", "match_mode": "ALL",
        }
        exit_rules = {
            "stop_loss_pct": 8.0, "trailing_stop_pct": 12.0,
            "ma_mode": "停用", "ma_p": 20, "ma_cross_pairs": [],
            "five_ma_mode": "停用", "kd_mode": "停用", "macd_mode": "停用",
            "vol_mode": "停用", "rsi_mode": "停用", "bb_mode": "停用",
            "bias_mode": "停用", "atr_stop_multiple": 0.0, "obv_mode": "停用",
            "cci_mode": "停用", "adx_mode": "停用", "adx_threshold": 20.0,
            "ema_mode": "停用",
        }
        entry.update(entry_updates)
        exit_rules.update(exit_updates)
        candidates.append({
            "name": name, "conditions": conditions,
            "entry": entry, "exit": exit_rules,
        })

    # The AI report uses standard, explainable presets to compare every
    # indicator available in the custom backtest panel. Users can still tune
    # their own periods and thresholds in that panel.
    add_candidate(
        "MA20 突破／跌破",
        "收盤向上突破 MA20 進場；收盤向下跌破 MA20 離場。",
        {"ma_mode": "突破均線"}, {"ma_mode": "跌破均線"},
    )
    add_candidate(
        "MA20 站穩／跌破",
        "收盤站在 MA20 上方進場；收盤向下跌破 MA20 離場。",
        {"ma_mode": "站穩均線上"}, {"ma_mode": "跌破均線"},
    )
    add_candidate(
        "MA5／MA20 黃金與死亡交叉",
        "MA5 向上穿越 MA20 進場；MA5 向下跌破 MA20 離場。",
        {"ma_cross_pairs": [(5, 20)]}, {"ma_cross_pairs": [(5, 20)]},
    )
    add_candidate(
        "成交量放大 1.5 倍＋MA20 突破",
        "收盤向上突破 MA20，且成交量達 20 日均量 1.5 倍時進場；跌破 MA20 或成交量達均量 2.5 倍時離場。",
        {"ma_mode": "突破均線", "vol_mode": "爆量 (大於20日均量1.5倍)"},
        {"ma_mode": "跌破均線", "vol_mode": "極端爆量倒貨 (大於20日均量2.5倍)"},
    )
    add_candidate(
        "成交量放大 2 倍＋MA20 突破",
        "收盤向上突破 MA20，且成交量達 20 日均量 2 倍時進場；跌破 MA20 或成交量達均量 2.5 倍時離場。",
        {"ma_mode": "突破均線", "vol_mode": "爆量 (大於20日均量2.0倍)"},
        {"ma_mode": "跌破均線", "vol_mode": "極端爆量倒貨 (大於20日均量2.5倍)"},
    )
    add_candidate(
        "RSI 強勢突破 50／跌破 70",
        "RSI 向上突破 50 進場；RSI 從 70 以上跌回 70 以下離場。",
        {"rsi_mode": "強勢突破 (RSI向上突破50)"},
        {"rsi_mode": "超買警戒 (RSI跌破70)"},
    )
    add_candidate(
        "布林上軌突破／上軌拉回",
        "收盤向上突破布林上軌進場；觸及上軌後收回上軌內側離場。",
        {"bb_mode": "突破布林上軌"}, {"bb_mode": "觸及上軌拉回"},
    )
    add_candidate(
        "BIAS 自 −5% 回升／跌破 0%",
        "BIAS 從 −5% 以下回升並向上穿越 −5% 時進場；BIAS 向下跌破 0% 離場。",
        {"bias_mode": "自 −5% 向上回升"}, {"bias_mode": "跌破 0%"},
    )
    add_candidate(
        "BIAS 突破 0%／高檔回落",
        "BIAS 向上穿越 0% 時進場；BIAS 從 +5% 以上回落並跌破 +5% 時離場。",
        {"bias_mode": "向上突破 0%"}, {"bias_mode": "高檔回落並跌破 +5%"},
    )
    add_candidate(
        "ATR 低波動篩選＋MA20 突破",
        "收盤向上突破 MA20，且 ATR 不高於收盤價的 4% 時進場；跌破 MA20 或觸發進場時 ATR 的 2 倍停損時離場。",
        {"ma_mode": "突破均線", "atr_mode": "低波動篩選", "atr_pct_limit": 4.0},
        {"ma_mode": "跌破均線", "atr_stop_multiple": 2.0},
    )
    add_candidate(
        "OBV 上升／轉弱",
        "OBV 高於前一根 K 棒時進場；OBV 低於前一根 K 棒時離場。",
        {"obv_mode": "OBV 高於前一根"}, {"obv_mode": "OBV 低於前一根"},
    )
    add_candidate(
        "CCI 上穿 −100／跌破 +100",
        "CCI 向上穿越 −100 時進場；CCI 從 +100 以上向下跌破 +100 時離場。",
        {"cci_mode": "上穿 −100"}, {"cci_mode": "跌破 +100"},
    )
    add_candidate(
        "CCI 上穿 0／跌破 0",
        "CCI 向上穿越 0 時進場；CCI 向下跌破 0 時離場。",
        {"cci_mode": "上穿 0"}, {"cci_mode": "跌破 0"},
    )
    add_candidate(
        "DMI／ADX 趨勢增強＋方向反轉離場",
        "ADX 達 20 且 +DI 高於 −DI 時進場；+DI 向下跌破 −DI 時離場。",
        {"adx_mode": "趨勢增強且 +DI 領先", "adx_threshold": 20.0},
        {"adx_mode": "+DI 跌破 −DI"},
    )
    add_candidate(
        "DMI／ADX 趨勢增強＋ADX 轉弱離場",
        "ADX 達 20 且 +DI 高於 −DI 時進場；ADX 低於 20 時離場。",
        {"adx_mode": "趨勢增強且 +DI 領先", "adx_threshold": 20.0},
        {"adx_mode": "ADX 低於門檻", "adx_threshold": 20.0},
    )
    add_candidate(
        "EMA12／EMA26 黃金與死亡交叉",
        "EMA12 向上穿越 EMA26 進場；EMA12 向下跌破 EMA26 離場。",
        {"ema_mode": "EMA12 黃金交叉 EMA26"},
        {"ema_mode": "EMA12 死亡交叉 EMA26"},
    )
    # Add readable rule descriptions to the five legacy candidates as well.
    legacy_conditions = {
        "五線開花（MA5／10／20／60／120）": "MA5＞MA10＞MA20＞MA60＞MA120 且五條均線同時上升時進場；排列失序離場。",
        "MACD 黃金／死亡交叉": "DIF 向上穿越訊號線進場；DIF 向下跌破訊號線離場。",
        "KD 黃金／死亡交叉": "K 線向上穿越 D 線進場；K 線向下跌破 D 線離場。",
        "RSI 超賣回升／超買回落": "RSI 向上穿越 30 進場；RSI 從 70 以上跌回 70 以下離場。",
        "布林通道下軌反彈／回到中軌": "觸及布林下軌後收回軌道內進場；收盤向上穿越中軌離場。",
    }
    for candidate in candidates:
        candidate.setdefault("conditions", legacy_conditions.get(candidate["name"], "依該指標的交叉或門檻觸發；另套用風險控制。"))

    def score_period(part):
        dates = pd.DatetimeIndex(part.index).strftime("%Y-%m-%d").tolist()
        trades = run_strategy_backtest(part, entry, exit_rules, dates, risk_reward_ratio=3.0)
        analysis = analyze_indicator_backtest(part, trades, 100000.0, fee, tax, slip, "1日", is_taiwan_stock)
        return analysis

    rows = []
    for candidate in candidates:
        entry = candidate["entry"]
        exit_rules = candidate["exit"]
        train_part = indicator_frame.iloc[:split].copy()
        # Retain the prior candle for crossover detection; it is pre-test data.
        test_part = indicator_frame.iloc[max(0, split - 1):].copy()
        train_result = score_period(train_part)
        test_result = score_period(test_part)
        full_dates = pd.DatetimeIndex(indicator_frame.index).strftime("%Y-%m-%d").tolist()
        full_trades = run_strategy_backtest(
            indicator_frame, entry, exit_rules, full_dates, risk_reward_ratio=3.0
        )
        latest_triggered = bool(
            not full_trades.empty
            and "_entry_index" in full_trades.columns
            and (pd.to_numeric(full_trades["_entry_index"], errors="coerce") == len(indicator_frame) - 1).any()
        )
        train = train_result["strategy_metrics"] if train_result else {}
        test = test_result["strategy_metrics"] if test_result else {}
        test_count = int(test.get("trade_count", 0) or 0)
        test_expectancy = _finite_number(test.get("expectancy_pct"))
        test_pf = _finite_number(test.get("profit_factor"))
        test_mdd = _finite_number(test.get("mdd"))
        passed = (
            test_count >= 5 and test_expectancy is not None and test_expectancy > 0
            and (test_pf is None or test_pf > 1.0)
            and test_mdd is not None and test_mdd >= -0.30
        )
        rows.append({
            "策略": candidate["name"],
            "進出場條件": candidate.get("conditions", ""),
            "前70%交易數": int(train.get("trade_count", 0) or 0),
            "前70%平均每筆損益(%)": _finite_number(train.get("expectancy_pct")),
            "後30%交易數": test_count,
            "後30%平均每筆損益(%)": test_expectancy,
            "後30%賺賠金額比": test_pf,
            "後30%最大回撤(%)": test_mdd * 100.0 if test_mdd is not None else None,
            "最新一根是否觸發": "有觸發" if latest_triggered else "未觸發",
            "樣本外檢查": "通過初步門檻" if passed else "未通過／資料不足",
            "_passed": passed,
            "_latest_triggered": latest_triggered,
        })
    result_frame = pd.DataFrame(rows)
    def candidate_direction(test_count, expectancy, profit_factor, max_drawdown, passed, latest_triggered):
        """Turn each candidate's holdout evidence into a cautious investor-facing stance."""
        if passed:
            if latest_triggered:
                return (
                    "偏多觀察｜樣本外初步通過且最新資料觸發；先按風險規則驗證，"
                    "不視為買進指令。"
                )
            return (
                "中性觀望｜歷史樣本外初步通過，但最新資料未觸發；等待條件出現，"
                "不提前進場。"
            )

        failed_checks = []
        if test_count < 5:
            failed_checks.append(f"樣本外僅 {test_count} 筆，交易數不足")
        if expectancy is None:
            failed_checks.append("成本後平均損益資料不足")
        elif expectancy <= 0:
            failed_checks.append("成本後平均每筆損益未為正")
        if profit_factor is not None and profit_factor <= 1.0:
            failed_checks.append("賺賠金額比未高於 1")
        if max_drawdown is None:
            failed_checks.append("最大回撤資料不足")
        elif max_drawdown < -0.30:
            failed_checks.append(f"最大回撤 {max_drawdown:.0%} 超過 30% 門檻")
        reason_text = "；".join(failed_checks) if failed_checks else "未通過樣本外門檻"
        return f"暫不採用｜{reason_text}；方向以觀望為主，不依此策略進場。"

    result_frame["投資方向提示"] = [
        candidate_direction(
            int(row["後30%交易數"]),
            _finite_number(row["後30%平均每筆損益(%)"]),
            _finite_number(row["後30%賺賠金額比"]),
            (_finite_number(row["後30%最大回撤(%)"]) / 100.0)
            if _finite_number(row["後30%最大回撤(%)"]) is not None else None,
            bool(row["_passed"]), bool(row["_latest_triggered"]),
        )
        for _, row in result_frame.iterrows()
    ]
    passed_rows = result_frame.loc[result_frame["_passed"]].copy()
    if passed_rows.empty:
        best = None
        recommendation = "沒有策略同時通過樣本外交易數、成本後正期望、賺賠金額比及回撤門檻；本次不指定可執行策略，建議觀望或先用模擬交易。"
        advice = (
            "目前沒有任何內建指標策略通過樣本外檢查，因此不提供進場策略建議。即使最新一根 K 棒出現個別技術條件，"
            "也不足以支持把它當成有效策略；先觀察、補充資料或用模擬交易，不要為了得到結論而放寬門檻。"
        )
    else:
        currently_triggered_passed = passed_rows.loc[passed_rows["_latest_triggered"]]
        recommendation_pool = currently_triggered_passed if not currently_triggered_passed.empty else passed_rows
        best_row = recommendation_pool.sort_values(
            ["後30%平均每筆損益(%)", "後30%最大回撤(%)"],
            ascending=[False, False],
        ).iloc[0]
        best = str(best_row["策略"])
        best_pf = _finite_number(best_row["後30%賺賠金額比"])
        best_pf_text = f"{best_pf:.2f}" if best_pf is not None else "樣本外尚無虧損交易"
        recommendation = (
            f"初步通過門檻的策略為「{best}」；後 30% 樣本外平均每筆成本後損益 "
            f"{best_row['後30%平均每筆損益(%)']:.2f}%、交易 {int(best_row['後30%交易數'])} 筆、"
            f"賺賠金額比 {best_pf_text}、最大回撤 {best_row['後30%最大回撤(%)']:.2f}%。"
        )
        if bool(best_row.get("_latest_triggered", False)):
            advice = (
                f"優先列入進一步研究的是「{best}」：它達到本系統的樣本外初步門檻，且最新可用日線"
                f"（{frame.index[-1]:%Y-%m-%d}）觸發了該策略的進場條件。這只代表條件同時成立，不是買進指令；"
                "仍須核對資料日期、流動性、公司事件與風險承受度，並依回測中的停損與停利規則先做模擬或小額驗證。"
            )
        else:
            advice = (
                f"「{best}」是樣本外表現相對較佳且通過初步門檻的研究候選，但最新可用日線"
                f"（{frame.index[-1]:%Y-%m-%d}）尚未觸發其進場條件。建議列入觀察，等待規則訊號出現；"
                "不要因歷史回測通過就提前追價或把它當成目前買進建議。"
            )
        other_passed = passed_rows.loc[passed_rows["策略"] != best, "策略"].astype(str).tolist()
        if other_passed:
            advice += " 其他通過初步門檻的候選策略：" + "、".join(other_passed[:5]) + ("等。" if len(other_passed) > 5 else "。")
    period = f"{frame.index[0]:%Y-%m-%d} 至 {frame.index[-1]:%Y-%m-%d}；前70%訓練、後30%樣本外測試"
    assumptions = (
        f"共 {len(frame)} 筆日線；成本假設：每邊手續費 {fee:.4f}%、賣出交易稅 {tax:.3f}%、每邊滑價 {slip:.2f}%；"
        + ("美股回測手續費與交易稅預設為 0，實際費用依券商、商品與交易方式而異，結果可能偏樂觀；請再用自訂回測填入實際費率。"
           if not is_taiwan_stock else "台股回測以一般個股費率估算；券商折扣與 ETF 等商品稅率可能不同，請依實際交易成本調整。")
        + "訊號在收盤資料確認並以同一根收盤價模擬成交，屬理想化假設；真實下單價格可能不同。"
        "回測逐項比較 MA20 突破與站穩、MA5／20 交叉、五線排列、KD、MACD、成交量、RSI、布林通道、BIAS、ATR、OBV、CCI、DMI／ADX 與 EMA12／26。"
        "各策略使用表列的明確進出場條件；指標週期採常用預設（RSI 14、布林 20、BIAS／CCI 20、ATR／ADX 14、成交量均線 20）。"
        "各策略套用固定停損 8%、移動停損 12%、預設風險報酬 1:3；ATR 候選另依進場時 ATR 設定 2 倍波動停損。"
        "通過門檻為後段至少 5 筆交易、成本後平均單筆報酬大於 0、"
        "賺賠金額比大於 1（無虧損時不以缺值判失敗）且最大回撤不超過 30%。"
    )
    def backtest_value(value, suffix=""):
        number = _finite_number(value)
        return f"{number:+.2f}{suffix}" if number is not None else "資料不足"

    strategy_lines = []
    for _, row in result_frame.iterrows():
        pf_value = _finite_number(row.get("後30%賺賠金額比"))
        pf_text = f"{pf_value:.2f}" if pf_value is not None else "無虧損或資料不足"
        strategy_lines.append(
            f"- {row['策略']}｜條件：{row['進出場條件']}｜前段 {int(row['前70%交易數'])} 筆；後段 {int(row['後30%交易數'])} 筆；"
            f"後段平均每筆損益 {backtest_value(row.get('後30%平均每筆損益(%)'), '%')}；"
            f"賺賠金額比 {pf_text}；最大回撤 {backtest_value(row.get('後30%最大回撤(%)'), '%')}；"
            f"最新資料日 {frame.index[-1]:%Y-%m-%d}：{row['最新一根是否觸發']}；"
            f"{row['樣本外檢查']}。\n"
            f"  🖍️ 畫線重點｜投資方向：{row['投資方向提示']}"
        )
    results_text = "\n".join(strategy_lines)
    return {"table": result_frame.drop(columns=["_passed", "_latest_triggered"]), "best": best,
            "text": f"回測期間：{period}\n{assumptions}\n各策略樣本外結果：\n{results_text}\n整體結論：{recommendation}",
            "period": period, "assumptions": assumptions, "recommendation": recommendation,
            "advice": advice}


def describe_backtest_conditions(entry_conds, exit_conds, risk_reward_ratio=None):
    """Create a concise, readable label for a manually tested indicator setup."""
    entry_labels = {
        "ma_mode": "均線", "kd_mode": "KD", "macd_mode": "MACD",
        "vol_mode": "成交量", "rsi_mode": "RSI", "bb_mode": "布林通道",
    }
    exit_labels = {
        "ma_mode": "均線離場", "kd_mode": "KD離場", "macd_mode": "MACD離場",
        "vol_mode": "成交量離場", "rsi_mode": "RSI離場", "bb_mode": "布林通道離場",
    }
    entries = []
    for key, label in entry_labels.items():
        value = entry_conds.get(key)
        if value and value != "停用":
            suffix = f"{entry_conds.get('ma_p')}日" if key == "ma_mode" else ""
            entries.append(f"{label}{suffix}:{value}")
    entry_cross_pairs = _normalize_ma_cross_pairs(entry_conds.get("ma_cross_pairs", []))
    if entry_cross_pairs:
        pairs_text = "、".join(f"{fast}/{slow}日" for fast, slow in entry_cross_pairs)
        entries.append(f"均線黃金交叉{pairs_text}")
    exits = []
    for key, label in exit_labels.items():
        value = exit_conds.get(key)
        if value and value != "停用":
            suffix = f"{exit_conds.get('ma_p')}日" if key == "ma_mode" else ""
            exits.append(f"{label}{suffix}:{value}")
    exit_cross_pairs = _normalize_ma_cross_pairs(exit_conds.get("ma_cross_pairs", []))
    if exit_cross_pairs:
        pairs_text = "、".join(f"{fast}/{slow}日" for fast, slow in exit_cross_pairs)
        exits.append(f"均線死亡交叉{pairs_text}")
    stop_loss = float(entry_conds.get("stop_loss_pct", exit_conds.get("stop_loss_pct", 0)) or 0)
    trailing = float(exit_conds.get("trailing_stop_pct", 0) or 0)
    if stop_loss > 0:
        exits.append(f"固定停損{stop_loss:g}%")
    if trailing > 0:
        exits.append(f"移動停損{trailing:g}%")
    if risk_reward_ratio is not None and (stop_loss > 0 or trailing > 0):
        exits.append(f"風險報酬1:{float(risk_reward_ratio):g}")
    elif risk_reward_ratio is not None:
        exits.append("風險報酬停利未生效（沒有價格型停損）")
    entry_text = "、".join(entries) if entries else "未啟用買進條件"
    if entries:
        entry_text += "（" + ("全部符合" if entry_conds.get("match_mode") == "ALL" else "任一符合") + "）"
    exit_text = "、".join(exits) if exits else "未啟用離場條件"
    return f"買進：{entry_text}；離場：{exit_text}"


def build_backtest_coach_advice(strategy_metrics, benchmark_metrics, dca_metrics=None,
                                grid_metrics=None, out_of_sample_metrics=None):
    """Turn historical metrics into cautious, plain-language review notes."""
    trade_count = int(strategy_metrics.get("trade_count") or 0)
    cumulative = _finite_number(strategy_metrics.get("cumulative_return"))
    win_rate = _finite_number(strategy_metrics.get("win_rate"))
    profit_factor = _finite_number(strategy_metrics.get("profit_factor"))
    expectancy = _finite_number(strategy_metrics.get("expectancy_pct"))
    notes = []
    if trade_count < 20:
        status = "暫不建議直接採用" if (
            cumulative is not None and cumulative <= 0
            or expectancy is not None and expectancy <= 0
            or profit_factor is not None and profit_factor < 1.0
        ) else "樣本不足"
        notes.append(f"目前只有 {trade_count} 筆完成交易，勝率容易被少數交易影響；先累積更多紀錄，不據此判定策略好壞。")
    elif cumulative is None or expectancy is None or cumulative <= 0 or expectancy <= 0 or (profit_factor is not None and profit_factor < 1.0):
        status = "暫不建議直接採用"
        notes.append("扣除設定費用後，整段報酬或每筆平均損益不理想；先調整條件並重新檢查較晚期間的結果。")
    elif trade_count >= 30 and profit_factor is not None and profit_factor >= 1.2 and win_rate is not None and win_rate >= 0.55:
        status = "可優先列入觀察"
        notes.append("交易筆數、勝率、整體報酬與賺賠金額比目前達到初步觀察條件；這是回測候選，不代表適合直接投入。")
    else:
        status = "繼續觀察"
        notes.append("目前回測有正向跡象，但交易筆數或賺賠金額比仍不足以列為優先候選。")

    if win_rate is not None:
        if win_rate < 0.40:
            status = "暫不建議直接採用"
            notes.append("勝率低於 40%，目前不列為優先使用條件；即使總報酬為正，也要確認賺錢交易平均幅度明顯大於賠錢交易，並以更多樣本和樣本外結果再驗證。")
        elif win_rate >= 0.55:
            if profit_factor is not None and profit_factor < 1.0:
                notes.append("勝率雖高，但賺賠金額比低於 1；少數較大的虧損可能抵銷多筆小獲利。")
            else:
                notes.append("勝率達到 55% 的參考線；仍須同時看賺賠金額比、總報酬及回撤，不能只看勝率。")
        else:
            notes.append("勝率未達 55% 參考線；請搭配賺賠金額比和每筆平均損益判讀。")

    compare_values = [("一開始買進並持有", benchmark_metrics.get("cumulative_return"))]
    if dca_metrics:
        total_contributed = _finite_number(dca_metrics.get("total_contributed"))
        ending_value = _finite_number(dca_metrics.get("ending_value"))
        dca_return = (ending_value / total_contributed - 1.0) if total_contributed and ending_value is not None else None
        compare_values.append(("定期定額", dca_return))
    if grid_metrics:
        compare_values.append(("網格（依比較頁設定）", grid_metrics.get("cumulative_return")))
    underperformed = []
    for label, value in compare_values:
        value = _finite_number(value)
        if cumulative is not None and value is not None and cumulative < value:
            underperformed.append(label)
    if underperformed:
        if status == "可優先列入觀察" and compare_values and len(underperformed) == len(compare_values):
            status = "繼續觀察"
        notes.append("同期間報酬低於「" + "、".join(underperformed) + "」；若目標是追求資產成長，先比較簡單做法並確認風險、投入時點與費用設定一致。")
    elif compare_values and cumulative is not None:
        notes.append("目前策略整段報酬沒有落後可用的同期比較項目；仍需留意最大跌幅及不同市場期間的表現。")

    oos_return = _finite_number(out_of_sample_metrics.get("cumulative_return")) if out_of_sample_metrics else None
    oos_trades = int(out_of_sample_metrics.get("trade_count") or 0) if out_of_sample_metrics else 0
    if out_of_sample_metrics:
        if oos_trades < 10:
            if status == "可優先列入觀察":
                status = "繼續觀察"
            notes.append(f"較晚 30% 資料只有 {oos_trades} 筆交易，樣本仍少，無法確認條件是否穩定。")
        elif oos_return is not None and oos_return <= 0:
            status = "暫不建議直接採用"
            notes.append("較晚 30% 資料為負報酬；整段回測的好結果可能集中在較早期間。")
        else:
            notes.append("較晚 30% 資料仍為正報酬，提供一項穩定性參考，但不代表未來能維持。")
    else:
        if status == "可優先列入觀察":
            status = "繼續觀察"
        notes.append("歷史長度不足以切出較晚期間檢查；不要只憑整段回測選用條件。")

    return {"status": status, "notes": notes}


def _normalize_ma_cross_pairs(pairs):
    """Return unique (fast, slow) periods where the fast average is shorter."""
    normalized = []
    for pair in pairs or []:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        try:
            fast, slow = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, OverflowError):
            continue
        if fast < 1 or slow < 1 or fast >= slow:
            continue
        if (fast, slow) not in normalized:
            normalized.append((fast, slow))
    return normalized


def _count_ma_cross_events(df, pairs, direction="golden"):
    """Count actual moving-average crosses in the supplied backtest rows."""
    normalized = _normalize_ma_cross_pairs(pairs)
    if not normalized or df is None or df.empty or "Close" not in df.columns:
        return [(fast, slow, 0) for fast, slow in normalized]

    close = pd.to_numeric(df["Close"], errors="coerce")
    counts = []
    for fast_period, slow_period in normalized:
        fast_column, slow_column = f"MA_CROSS_{fast_period}", f"MA_CROSS_{slow_period}"
        fast_ma = (
            pd.to_numeric(df[fast_column], errors="coerce").to_numpy(dtype=float)
            if fast_column in df.columns else close.rolling(fast_period).mean().to_numpy(dtype=float)
        )
        slow_ma = (
            pd.to_numeric(df[slow_column], errors="coerce").to_numpy(dtype=float)
            if slow_column in df.columns else close.rolling(slow_period).mean().to_numpy(dtype=float)
        )
        events = np.zeros(len(df), dtype=bool)
        if len(df) > 1:
            valid = (
                np.isfinite(fast_ma[:-1]) & np.isfinite(slow_ma[:-1])
                & np.isfinite(fast_ma[1:]) & np.isfinite(slow_ma[1:])
            )
            if direction == "golden":
                events[1:] = valid & (fast_ma[:-1] <= slow_ma[:-1]) & (fast_ma[1:] > slow_ma[1:])
            elif direction == "death":
                events[1:] = valid & (fast_ma[:-1] >= slow_ma[:-1]) & (fast_ma[1:] < slow_ma[1:])
            else:
                raise ValueError("direction must be 'golden' or 'death'")
        counts.append((fast_period, slow_period, int(events.sum())))
    return counts


def _slice_history_for_date_range(history, start_date, end_date):
    """Return bars inside an inclusive calendar-date range."""
    if history is None or history.empty:
        return history.copy() if history is not None else pd.DataFrame()
    index = pd.DatetimeIndex(pd.to_datetime(history.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_localize(None)
    start = pd.Timestamp(start_date)
    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    mask = (index >= start) & (index < end_exclusive)
    return history.iloc[np.asarray(mask)].copy()


def compute_backtest_indicators(df, ma_entry_p=20, ma_exit_p=20, rsi_p=14, bb_p=20,
                                entry_ma_cross_pairs=None, exit_ma_cross_pairs=None):
    df_calc = df.copy()
    df_calc['MA_entry'] = df_calc['Close'].rolling(window=ma_entry_p).mean()
    df_calc['MA_exit'] = df_calc['Close'].rolling(window=ma_exit_p).mean()

    cross_periods = {
        period
        for pair in (
            _normalize_ma_cross_pairs(entry_ma_cross_pairs)
            + _normalize_ma_cross_pairs(exit_ma_cross_pairs)
        )
        for period in pair
    }
    for period in cross_periods:
        df_calc[f'MA_CROSS_{period}'] = df_calc['Close'].rolling(window=period).mean()
    for period in (5, 10, 20, 60, 120):
        df_calc[f'MA_FIVE_{period}'] = df_calc['Close'].rolling(window=period).mean()

    low_min = df_calc['Low'].rolling(window=9).min()
    high_max = df_calc['High'].rolling(window=9).max()
    rsv = np.where(high_max == low_min, 50.0, (df_calc['Close'] - low_min) / (high_max - low_min) * 100.0)

    k_list, d_list = [50.0], [50.0]
    for r in rsv[1:]:
        if np.isnan(r): r = 50.0
        k_val = (2.0/3.0) * k_list[-1] + (1.0/3.0) * r
        d_val = (2.0/3.0) * d_list[-1] + (1.0/3.0) * k_val
        k_list.append(k_val)
        d_list.append(d_val)

    df_calc['K'] = k_list
    df_calc['D'] = d_list

    ema12 = df_calc['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_calc['Close'].ewm(span=26, adjust=False).mean()
    df_calc['DIF'] = ema12 - ema26
    df_calc['DEM'] = df_calc['DIF'].ewm(span=9, adjust=False).mean()

    df_calc['VOL_MA20'] = df_calc['Volume'].rolling(window=20).mean()

    delta = df_calc['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_p).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_p).mean()
    rs = gain / (loss + 1e-9)
    df_calc['RSI'] = 100 - (100 / (1 + rs))

    bb_middle = df_calc['Close'].rolling(window=bb_p).mean()
    bb_std = df_calc['Close'].rolling(window=bb_p).std()
    df_calc['BB_Middle'] = bb_middle
    df_calc['BB_Upper'] = bb_middle + (bb_std * 2)
    df_calc['BB_Lower'] = bb_middle - (bb_std * 2)

    return df_calc


def compute_kline_indicator_frame(df, **indicator_kwargs):
    """Build the commonly used chart indicators without changing backtest rules."""
    frame = compute_backtest_indicators(df, **indicator_kwargs)
    high = pd.to_numeric(frame["High"], errors="coerce")
    low = pd.to_numeric(frame["Low"], errors="coerce")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    volume = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0)

    frame["EMA12"] = close.ewm(span=12, adjust=False).mean()
    frame["EMA26"] = close.ewm(span=26, adjust=False).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    frame["ATR"] = atr

    average_20 = close.rolling(window=20, min_periods=20).mean()
    frame["BIAS"] = (close / average_20 - 1) * 100

    typical_price = (high + low + close) / 3
    typical_average = typical_price.rolling(window=20, min_periods=20).mean()
    mean_deviation = typical_price.rolling(window=20, min_periods=20).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))), raw=True
    )
    frame["CCI"] = (typical_price - typical_average) / (0.015 * mean_deviation.replace(0, np.nan))

    direction = np.sign(close.diff()).fillna(0)
    frame["OBV"] = (direction * volume).cumsum()

    upward_move = high.diff()
    downward_move = -low.diff()
    plus_dm = upward_move.where((upward_move > downward_move) & (upward_move > 0), 0.0)
    minus_dm = downward_move.where((downward_move > upward_move) & (downward_move > 0), 0.0)
    smoothed_atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / smoothed_atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / smoothed_atr.replace(0, np.nan)
    directional_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / directional_sum
    frame["PLUS_DI"] = plus_di
    frame["MINUS_DI"] = minus_di
    frame["ADX"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    return frame

def run_strategy_backtest(df_calc, entry_conds, exit_conds, date_chinese_list,
                         risk_reward_ratio=None, *, next_open=False):
    if next_open and any(exit_conds.get(key, 0) for key in (
        "stop_loss_pct", "trailing_stop_pct", "atr_stop_multiple",
    )):
        raise ValueError("次日開盤比較僅適用收盤確認的指標訊號。")
    trades = []
    unfilled_signal = None
    position = False
    entry_price = 0.0
    entry_date = ""
    entry_index = None
    highest_price = 0.0
    initial_stop_price = None
    take_profit_price = None
    risk_ratio = _finite_number(risk_reward_ratio)

    closes = df_calc['Close'].to_numpy()
    opens = df_calc['Open'].to_numpy() if 'Open' in df_calc.columns else closes
    highs = df_calc['High'].to_numpy()
    lows = df_calc['Low'].to_numpy()
    vols = df_calc['Volume'].to_numpy()
    vol_ma20 = df_calc['VOL_MA20'].to_numpy()

    ma_entry = df_calc['MA_entry'].to_numpy()
    ma_exit = df_calc['MA_exit'].to_numpy()
    entry_ma_cross_pairs = _normalize_ma_cross_pairs(entry_conds.get('ma_cross_pairs', []))
    exit_ma_cross_pairs = _normalize_ma_cross_pairs(exit_conds.get('ma_cross_pairs', []))
    cross_periods = {
        period for pair in entry_ma_cross_pairs + exit_ma_cross_pairs for period in pair
    }
    cross_ma_values = {}
    for period in cross_periods:
        column = f'MA_CROSS_{period}'
        if column in df_calc.columns:
            cross_ma_values[period] = df_calc[column].to_numpy()
        else:
            # Keep direct callers compatible even if they did not request the
            # extra moving-average columns when computing indicators.
            cross_ma_values[period] = df_calc['Close'].rolling(window=period).mean().to_numpy()
    k_arr = df_calc['K'].to_numpy()
    d_arr = df_calc['D'].to_numpy()
    dif_arr = df_calc['DIF'].to_numpy()
    dem_arr = df_calc['DEM'].to_numpy()
    rsi_arr = df_calc['RSI'].to_numpy()
    def indicator_values(column_name):
        if column_name not in df_calc.columns:
            return np.full(len(df_calc), np.nan, dtype=float)
        return pd.to_numeric(df_calc[column_name], errors='coerce').to_numpy(dtype=float)

    bias_arr = indicator_values('BIAS')
    atr_arr = indicator_values('ATR')
    obv_arr = indicator_values('OBV')
    cci_arr = indicator_values('CCI')
    adx_arr = indicator_values('ADX')
    plus_di_arr = indicator_values('PLUS_DI')
    minus_di_arr = indicator_values('MINUS_DI')
    ema12_arr = indicator_values('EMA12')
    ema26_arr = indicator_values('EMA26')
    bb_upper = df_calc['BB_Upper'].to_numpy()
    bb_lower = df_calc['BB_Lower'].to_numpy()
    bb_middle = df_calc['BB_Middle'].to_numpy() if 'BB_Middle' in df_calc.columns else np.full(len(df_calc), np.nan)
    five_ma_periods = (5, 10, 20, 60, 120)
    five_ma_arrays = {
        period: df_calc[f'MA_FIVE_{period}'].to_numpy()
        if f'MA_FIVE_{period}' in df_calc.columns
        else np.full(len(df_calc), np.nan)
        for period in five_ma_periods
    }

    def five_line_bullish(index, require_rising=False):
        if index < 0:
            return False
        values = [five_ma_arrays[period][index] for period in five_ma_periods]
        if not np.isfinite(values).all() or not all(values[i] > values[i + 1] for i in range(4)):
            return False
        if require_rising:
            if index == 0:
                return False
            previous = [five_ma_arrays[period][index - 1] for period in five_ma_periods]
            if not np.isfinite(previous).all() or not all(values[i] > previous[i] for i in range(5)):
                return False
        return True

    def execution_price(index):
        if not next_open:
            return float(closes[index])
        opening = _finite_number(opens[index]) if "Open" in df_calc.columns else None
        if opening is None or opening <= 0:
            raise ValueError(f"{date_chinese_list[index]} 缺少有效開盤價，無法完成次日開盤比較。")
        return opening

    for i in range(1, len(df_calc)):
        fill_index = i + int(next_open)
        if not position:
            buy_checks = []

            if entry_conds['ma_mode'] == '突破均線':
                buy_checks.append(not np.isnan(ma_entry[i-1]) and closes[i-1] <= ma_entry[i-1] and closes[i] > ma_entry[i])
            elif entry_conds['ma_mode'] == '站穩均線上':
                buy_checks.append(not np.isnan(ma_entry[i]) and closes[i] > ma_entry[i])

            if entry_conds.get('five_ma_mode') == '五線開花':
                buy_checks.append(five_line_bullish(i, require_rising=True))

            for fast_period, slow_period in entry_ma_cross_pairs:
                fast_ma = cross_ma_values[fast_period]
                slow_ma = cross_ma_values[slow_period]
                buy_checks.append(
                    np.isfinite(fast_ma[i-1]) and np.isfinite(slow_ma[i-1])
                    and np.isfinite(fast_ma[i]) and np.isfinite(slow_ma[i])
                    and fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]
                )

            if entry_conds['kd_mode'] == '黃金交叉':
                buy_checks.append(k_arr[i-1] <= d_arr[i-1] and k_arr[i] > d_arr[i])

            if entry_conds['macd_mode'] == '黃金交叉':
                buy_checks.append(dif_arr[i-1] <= dem_arr[i-1] and dif_arr[i] > dem_arr[i])

            if entry_conds['vol_mode'] == '爆量 (大於20日均量1.5倍)':
                buy_checks.append(not np.isnan(vol_ma20[i]) and vols[i] >= vol_ma20[i] * 1.5)
            elif entry_conds['vol_mode'] == '爆量 (大於20日均量2.0倍)':
                buy_checks.append(not np.isnan(vol_ma20[i]) and vols[i] >= vol_ma20[i] * 2.0)

            if entry_conds['rsi_mode'] == '超賣回升 (RSI向上突破30)':
                buy_checks.append(rsi_arr[i-1] <= 30 and rsi_arr[i] > 30)
            elif entry_conds['rsi_mode'] == '強勢突破 (RSI向上突破50)':
                buy_checks.append(rsi_arr[i-1] <= 50 and rsi_arr[i] > 50)

            if entry_conds['bb_mode'] == '突破布林上軌':
                buy_checks.append(closes[i-1] <= bb_upper[i-1] and closes[i] > bb_upper[i])
            elif entry_conds['bb_mode'] == '觸及布林下軌反彈':
                buy_checks.append(lows[i-1] <= bb_lower[i-1] and closes[i] > bb_lower[i])

            bias_mode = entry_conds.get('bias_mode', '停用')
            if bias_mode == '向上突破 0%':
                buy_checks.append(np.isfinite(bias_arr[i-1:i+1]).all() and bias_arr[i-1] <= 0 <= bias_arr[i])
            elif bias_mode == '自 −5% 向上回升':
                buy_checks.append(np.isfinite(bias_arr[i-1:i+1]).all() and bias_arr[i-1] <= -5 < bias_arr[i])

            if entry_conds.get('atr_mode') == '低波動篩選':
                atr_pct = (atr_arr[i] / closes[i] * 100) if closes[i] > 0 and np.isfinite(atr_arr[i]) else np.nan
                buy_checks.append(np.isfinite(atr_pct) and atr_pct <= float(entry_conds.get('atr_pct_limit', 4.0)))

            obv_mode = entry_conds.get('obv_mode', '停用')
            if obv_mode == 'OBV 高於前一根':
                buy_checks.append(np.isfinite(obv_arr[i-1:i+1]).all() and obv_arr[i] > obv_arr[i-1])

            cci_mode = entry_conds.get('cci_mode', '停用')
            if cci_mode == '上穿 −100':
                buy_checks.append(np.isfinite(cci_arr[i-1:i+1]).all() and cci_arr[i-1] <= -100 < cci_arr[i])
            elif cci_mode == '上穿 0':
                buy_checks.append(np.isfinite(cci_arr[i-1:i+1]).all() and cci_arr[i-1] <= 0 < cci_arr[i])

            if entry_conds.get('adx_mode') == '趨勢增強且 +DI 領先':
                adx_floor = float(entry_conds.get('adx_threshold', 20.0))
                buy_checks.append(
                    np.isfinite([adx_arr[i], plus_di_arr[i], minus_di_arr[i]]).all()
                    and adx_arr[i] >= adx_floor and plus_di_arr[i] > minus_di_arr[i]
                )

            if entry_conds.get('ema_mode') == 'EMA12 黃金交叉 EMA26':
                buy_checks.append(
                    np.isfinite([ema12_arr[i-1], ema26_arr[i-1], ema12_arr[i], ema26_arr[i]]).all()
                    and ema12_arr[i-1] <= ema26_arr[i-1] and ema12_arr[i] > ema26_arr[i]
                )

            if buy_checks and (all(buy_checks) if entry_conds['match_mode'] == 'ALL' else any(buy_checks)):
                if fill_index >= len(df_calc):
                    unfilled_signal = "買進"
                    break
                position = True
                entry_price = execution_price(fill_index)
                entry_date = date_chinese_list[fill_index]
                entry_index = fill_index
                # The entry is modeled at the close, so earlier highs from
                # that same candle are not part of the position's trailing peak.
                highest_price = entry_price
                # Lock the initial risk at entry. If both percentage stops are
                # enabled, use the closer price stop (the smaller loss distance).
                configured_risk_pcts = [
                    float(exit_conds.get('stop_loss_pct', 0.0)),
                    float(exit_conds.get('trailing_stop_pct', 0.0)),
                ]
                atr_multiple_for_risk = float(exit_conds.get('atr_stop_multiple', 0.0) or 0.0)
                entry_atr_for_risk = atr_arr[i] if i < len(atr_arr) else np.nan
                if atr_multiple_for_risk > 0 and np.isfinite(entry_atr_for_risk) and entry_price > 0:
                    configured_risk_pcts.append(
                        float(entry_atr_for_risk) * atr_multiple_for_risk / entry_price * 100.0
                    )
                configured_risk_pcts = [value for value in configured_risk_pcts if value > 0]
                if configured_risk_pcts:
                    initial_risk_pct = min(configured_risk_pcts)
                    initial_stop_price = entry_price * (1.0 - initial_risk_pct / 100.0)
                    take_profit_price = (
                        entry_price + (entry_price - initial_stop_price) * risk_ratio
                        if risk_ratio is not None and risk_ratio > 0 else None
                    )
                else:
                    initial_stop_price = None
                    take_profit_price = None
        else:
            exit_reasons = []
            stop_hits = []

            if exit_conds['stop_loss_pct'] > 0:
                sl_price = entry_price * (1.0 - exit_conds['stop_loss_pct'] / 100.0)
                if lows[i] <= sl_price or closes[i] <= sl_price:
                    exit_reasons.append(f"觸發停損 (-{exit_conds['stop_loss_pct']}%)")
                    stop_hits.append((sl_price, exit_reasons[-1]))

            if exit_conds['trailing_stop_pct'] > 0:
                trail_price = highest_price * (1.0 - exit_conds['trailing_stop_pct'] / 100.0)
                if lows[i] <= trail_price or closes[i] <= trail_price:
                    exit_reasons.append(f"最高點回撤 (-{exit_conds['trailing_stop_pct']}%)")
                    stop_hits.append((trail_price, exit_reasons[-1]))

            atr_stop_multiple = float(exit_conds.get('atr_stop_multiple', 0.0) or 0.0)
            entry_atr_value = float(atr_arr[entry_index]) if entry_index is not None and np.isfinite(atr_arr[entry_index]) else np.nan
            if atr_stop_multiple > 0 and np.isfinite(entry_atr_value):
                atr_stop_price = entry_price - entry_atr_value * atr_stop_multiple
                if atr_stop_price > 0 and (lows[i] <= atr_stop_price or closes[i] <= atr_stop_price):
                    exit_reasons.append(f"ATR 停損（進場時波動幅度 × {atr_stop_multiple:g}）")
                    stop_hits.append((atr_stop_price, exit_reasons[-1]))

            if exit_conds['ma_mode'] == '跌破均線' and not np.isnan(ma_exit[i-1]):
                if closes[i-1] >= ma_exit[i-1] and closes[i] < ma_exit[i]:
                    exit_reasons.append(f"跌破 MA{exit_conds['ma_p']}")

            for fast_period, slow_period in exit_ma_cross_pairs:
                fast_ma = cross_ma_values[fast_period]
                slow_ma = cross_ma_values[slow_period]
                if (
                    np.isfinite(fast_ma[i-1]) and np.isfinite(slow_ma[i-1])
                    and np.isfinite(fast_ma[i]) and np.isfinite(slow_ma[i])
                    and fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]
                ):
                    exit_reasons.append(
                        f"均線死亡交叉 MA{fast_period}/MA{slow_period}"
                    )

            if exit_conds['kd_mode'] == '死亡交叉':
                if k_arr[i-1] >= d_arr[i-1] and k_arr[i] < d_arr[i]:
                    exit_reasons.append("KD 死亡交叉")

            if exit_conds['macd_mode'] == '死亡交叉':
                if dif_arr[i-1] >= dem_arr[i-1] and dif_arr[i] < dem_arr[i]:
                    exit_reasons.append("MACD 死亡交叉")

            if exit_conds['vol_mode'] == '極端爆量倒貨 (大於20日均量2.5倍)':
                if not np.isnan(vol_ma20[i]) and vols[i] >= vol_ma20[i] * 2.5:
                    exit_reasons.append("觸發極端爆量離場")

            if exit_conds['rsi_mode'] == '超買警戒 (RSI跌破70)':
                if rsi_arr[i-1] >= 70 and rsi_arr[i] < 70:
                    exit_reasons.append("RSI 70 死亡交叉離場")

            if exit_conds['bb_mode'] == '跌破布林下軌':
                if closes[i-1] >= bb_lower[i-1] and closes[i] < bb_lower[i]:
                    exit_reasons.append("跌破布林下軌離場")
            elif exit_conds['bb_mode'] == '觸及上軌拉回':
                if highs[i-1] >= bb_upper[i-1] and closes[i] < bb_upper[i]:
                    exit_reasons.append("布林上軌受阻離場")
            elif exit_conds['bb_mode'] == '站上布林中軌':
                if np.isfinite(bb_middle[i-1]) and np.isfinite(bb_middle[i]) and closes[i-1] <= bb_middle[i-1] and closes[i] > bb_middle[i]:
                    exit_reasons.append("站上布林中軌停利")

            bias_mode = exit_conds.get('bias_mode', '停用')
            if bias_mode == '跌破 0%':
                if np.isfinite(bias_arr[i-1:i+1]).all() and bias_arr[i-1] >= 0 > bias_arr[i]:
                    exit_reasons.append("BIAS 跌破 0%")
            elif bias_mode == '高檔回落並跌破 +5%':
                if np.isfinite(bias_arr[i-1:i+1]).all() and bias_arr[i-1] >= 5 > bias_arr[i]:
                    exit_reasons.append("BIAS 高檔回落至 +5% 以下")

            if exit_conds.get('obv_mode') == 'OBV 低於前一根':
                if np.isfinite(obv_arr[i-1:i+1]).all() and obv_arr[i] < obv_arr[i-1]:
                    exit_reasons.append("OBV 低於前一根")

            cci_mode = exit_conds.get('cci_mode', '停用')
            if cci_mode == '跌破 +100':
                if np.isfinite(cci_arr[i-1:i+1]).all() and cci_arr[i-1] >= 100 > cci_arr[i]:
                    exit_reasons.append("CCI 跌破 +100")
            elif cci_mode == '跌破 0':
                if np.isfinite(cci_arr[i-1:i+1]).all() and cci_arr[i-1] >= 0 > cci_arr[i]:
                    exit_reasons.append("CCI 跌破 0")

            adx_mode = exit_conds.get('adx_mode', '停用')
            if adx_mode == '+DI 跌破 −DI':
                if np.isfinite([plus_di_arr[i-1], minus_di_arr[i-1], plus_di_arr[i], minus_di_arr[i]]).all() and plus_di_arr[i-1] >= minus_di_arr[i-1] and plus_di_arr[i] < minus_di_arr[i]:
                    exit_reasons.append("DMI +DI 跌破 −DI")
            elif adx_mode == 'ADX 低於門檻':
                if np.isfinite(adx_arr[i]) and adx_arr[i] < float(exit_conds.get('adx_threshold', 20.0)):
                    exit_reasons.append("ADX 趨勢強度低於設定門檻")

            if exit_conds.get('ema_mode') == 'EMA12 死亡交叉 EMA26':
                if np.isfinite([ema12_arr[i-1], ema26_arr[i-1], ema12_arr[i], ema26_arr[i]]).all() and ema12_arr[i-1] >= ema26_arr[i-1] and ema12_arr[i] < ema26_arr[i]:
                    exit_reasons.append("EMA12 死亡交叉 EMA26")

            if exit_conds.get('five_ma_mode') == '排列失序' and not five_line_bullish(i):
                exit_reasons.append("五線多頭排列失序")

            target_hit = take_profit_price is not None and (
                float(highs[i]) >= take_profit_price or float(closes[i]) >= take_profit_price
            )
            if stop_hits:
                if take_profit_price is not None:
                    # If a daily bar crosses both a stop and target, use the
                    # stop first because OHLC data cannot reveal intraday order.
                    # A gap through a stop is modeled at the opening price.
                    stop_level = max(level for level, _reason in stop_hits)
                    opening_price = _finite_number(opens[i]) or float(closes[i])
                    exit_price = min(opening_price, float(stop_level))
                else:
                    # Keep the original close-based stop execution when the
                    # risk/reward target is disabled or cannot be derived.
                    exit_price = float(closes[i])
            elif target_hit:
                # A favorable gap through the target is modeled at the open.
                opening_price = _finite_number(opens[i]) or float(closes[i])
                exit_price = max(opening_price, float(take_profit_price))
                exit_reasons = [
                    f"達成風險報酬停利目標（1:{risk_ratio:g}）"
                ]
            elif exit_reasons:
                if fill_index >= len(df_calc):
                    unfilled_signal = "賣出"
                    break
                exit_price = execution_price(fill_index)
            else:
                # Update the trailing peak only after this bar's exit checks;
                # daily OHLC data does not reveal whether its high came first.
                highest_price = max(highest_price, float(highs[i]))
                continue

            if stop_hits and target_hit:
                exit_reasons.append("同根資料也觸及停利目標，依保守方式先計停損")
            if stop_hits or target_hit or exit_reasons:
                exit_date = date_chinese_list[fill_index]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append({
                    '買進日期': entry_date,
                    '買進價格 (元)': round(entry_price, 2),
                    '初始停損參考價 (元)': round(initial_stop_price, 2) if initial_stop_price is not None else None,
                    '目標停利價 (元)': round(take_profit_price, 2) if take_profit_price is not None else None,
                    '設定風險報酬比': f"1:{risk_ratio:g}" if take_profit_price is not None else None,
                    '賣出日期': exit_date,
                    '賣出價格 (元)': round(exit_price, 2),
                    '平倉報酬率 (%)': round(pnl_pct, 2),
                    '離場觸發原因': " | ".join(exit_reasons),
                    '持有K棒數': max(0, fill_index - entry_index),
                    '_entry_index': entry_index,
                    '_exit_index': fill_index,
                    '_entry_fill_price': entry_price,
                    '_exit_fill_price': exit_price,
                })
                position = False

    # Include a still-open position at the final close so the reported result
    # and exposure do not silently omit its unrealized gain or loss.
    if position and len(df_calc) > 0 and entry_index is not None:
        exit_index = len(df_calc) - 1
        exit_price = float(closes[exit_index])
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 if entry_price else 0.0
        trades.append({
            '買進日期': entry_date,
            '買進價格 (元)': round(entry_price, 2),
            '初始停損參考價 (元)': round(initial_stop_price, 2) if initial_stop_price is not None else None,
            '目標停利價 (元)': round(take_profit_price, 2) if take_profit_price is not None else None,
            '設定風險報酬比': f"1:{risk_ratio:g}" if take_profit_price is not None else None,
            '賣出日期': date_chinese_list[exit_index],
            '賣出價格 (元)': round(exit_price, 2),
            '平倉報酬率 (%)': round(pnl_pct, 2),
            '離場觸發原因': '期末未平倉（以最後收盤價估值）',
            '持有K棒數': max(0, exit_index - entry_index),
            '_entry_index': entry_index,
            '_exit_index': exit_index,
            '_entry_fill_price': entry_price,
            '_exit_fill_price': exit_price,
        })

    ledger = pd.DataFrame(trades)
    if next_open:
        ledger.attrs["unfilled_signal"] = unfilled_signal
    return ledger


def compare_ma_execution_timing(history, cost_settings, is_taiwan_stock):
    """Compare the same close-confirmed MA5/20 signals under two execution rules."""
    if len(history) < 21:
        raise ValueError("比較需要至少 21 筆日線。")
    pair = [(5, 20)]
    disabled = {key: "停用" for key in ("ma_mode", "kd_mode", "macd_mode", "vol_mode", "rsi_mode", "bb_mode")}
    entry = dict(disabled, ma_cross_pairs=pair, match_mode="ALL")
    exit_rules = dict(disabled, ma_cross_pairs=pair, stop_loss_pct=0, trailing_stop_pct=0)
    indicators = compute_kline_indicator_frame(
        history, entry_ma_cross_pairs=pair, exit_ma_cross_pairs=pair,
    )
    dates = indicators.index.strftime("%Y-%m-%d").tolist()
    results = []
    for label, next_open in (("同日收盤", False), ("次日開盤", True)):
        trades = run_strategy_backtest(indicators, entry, exit_rules, dates, next_open=next_open)
        outcome = analyze_indicator_backtest(indicators, trades, *cost_settings, "1日", is_taiwan_stock)
        if outcome is None:
            raise ValueError("收盤價有缺漏，無法完成兩種成交方式的比較。")
        results.append((label, trades, outcome["strategy_metrics"]))
    return results


def _annualization_bars(timeframe_name, is_taiwan_stock):
    session_minutes = 270 if is_taiwan_stock else 390
    mapping = {
        "1分鐘": 252 * session_minutes,
        "5分鐘": 252 * (session_minutes // 5),
        "30分鐘": 252 * (session_minutes // 30),
        "60分鐘": 252 * (session_minutes // 60),
        "1日": 252,
        "1週": 52,
        "1個月": 12,
        # The current data mapping fetches quarterly bars for these choices.
        "1季": 4,
        "半年": 2,
        "1年": 1,
    }
    return mapping.get(timeframe_name, 252)


def analyze_indicator_backtest(df_calc, trades, initial_capital, commission_pct,
                               sell_tax_pct, slippage_pct, timeframe_name,
                               is_taiwan_stock):
    """Build a marked-to-market equity curve and comparable buy-and-hold metrics."""
    if df_calc is None or df_calc.empty or 'Close' not in df_calc.columns:
        return None

    close = pd.to_numeric(df_calc['Close'], errors='coerce').astype(float)
    close = close.replace([np.inf, -np.inf], np.nan)
    if close.isna().any() or len(close) < 2 or (close <= 0).any():
        return None

    n = len(close)
    price = close.to_numpy()
    entry_fee_factor = max(0.0, 1.0 - (float(commission_pct) + float(slippage_pct)) / 100.0)
    exit_fee_factor = max(0.0, 1.0 - (float(commission_pct) + float(sell_tax_pct) + float(slippage_pct)) / 100.0)
    strategy_factors = np.ones(n, dtype=float)
    in_market = np.zeros(n, dtype=bool)
    trade_rows = []

    if trades is not None and not trades.empty:
        ordered = trades.sort_values('_entry_index') if '_entry_index' in trades.columns else trades
        for _, trade in ordered.iterrows():
            if '_entry_index' not in trade or '_exit_index' not in trade:
                continue
            entry_i, exit_i = int(trade['_entry_index']), int(trade['_exit_index'])
            if not (0 <= entry_i < n and entry_i <= exit_i < n):
                continue
            entry_fill = _finite_number(trade.get('_entry_fill_price')) or float(price[entry_i])
            exit_fill = _finite_number(trade.get('_exit_fill_price')) or float(price[exit_i])
            strategy_factors[entry_i] *= entry_fee_factor
            for i in range(entry_i + 1, exit_i):
                strategy_factors[i] *= price[i] / price[i - 1]
            if exit_i > entry_i:
                strategy_factors[entry_i] *= price[entry_i] / entry_fill
                strategy_factors[exit_i] *= exit_fill / price[exit_i - 1]
            else:
                strategy_factors[exit_i] *= exit_fill / entry_fill
            strategy_factors[exit_i] *= exit_fee_factor
            in_market[entry_i:exit_i + 1] = True
            gross_ratio = exit_fill / entry_fill
            net_return = entry_fee_factor * gross_ratio * exit_fee_factor - 1.0
            trade_rows.append({
                '買進日期': trade.get('買進日期'),
                '賣出日期': trade.get('賣出日期'),
                '成本後報酬率 (%)': net_return * 100.0,
                '持有K棒數': max(0, exit_i - entry_i),
                '目標停利價 (元)': trade.get('目標停利價 (元)'),
            })

    strategy_equity = pd.Series(float(initial_capital) * np.cumprod(strategy_factors), index=close.index, name='技術策略資產')
    strategy_returns = strategy_equity.pct_change().dropna()
    bars_per_year = _annualization_bars(timeframe_name, is_taiwan_stock)
    span_years = max((close.index[-1] - close.index[0]).total_seconds() / (365.25 * 24 * 60 * 60), 0.0)

    def summarize(equity, returns):
        cumulative = float(equity.iloc[-1] / float(initial_capital) - 1.0)
        drawdown = equity / equity.cummax() - 1.0
        mdd = float(drawdown.min())
        min_observations = min(20, max(2, int(round(bars_per_year * 0.5))))
        enough_history = span_years >= 0.5 and len(returns) >= min_observations
        cagr = float((equity.iloc[-1] / float(initial_capital)) ** (1.0 / span_years) - 1.0) if enough_history and span_years > 0 and equity.iloc[-1] > 0 else None
        ann_vol = float(returns.std(ddof=1) * np.sqrt(bars_per_year)) if enough_history and len(returns) > 1 else None
        sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(bars_per_year)) if enough_history and len(returns) > 1 and returns.std(ddof=1) > 0 else None
        downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
        downside_dev = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
        sortino = float(returns.mean() / downside_dev * np.sqrt(bars_per_year)) if enough_history and downside_dev > 0 else None
        calmar = float(cagr / abs(mdd)) if cagr is not None and mdd < 0 else None
        underwater = drawdown.to_numpy() < 0
        longest_underwater = run = 0
        for below_peak in underwater:
            run = run + 1 if below_peak else 0
            longest_underwater = max(longest_underwater, run)
        return {
            'cumulative_return': cumulative, 'cagr': cagr, 'mdd': mdd,
            'annualized_volatility': ann_vol, 'sharpe': sharpe,
            'sortino': sortino, 'calmar': calmar,
            'longest_drawdown_bars': longest_underwater,
            'ending_value': float(equity.iloc[-1]), 'drawdown': drawdown,
        }

    strategy_metrics = summarize(strategy_equity, strategy_returns)
    strategy_metrics['exposure'] = float(in_market.mean()) if n else 0.0

    benchmark_factors = np.ones(n, dtype=float)
    benchmark_factors[0] *= entry_fee_factor
    for i in range(1, n):
        benchmark_factors[i] *= price[i] / price[i - 1]
    benchmark_factors[-1] *= exit_fee_factor
    benchmark_equity = pd.Series(float(initial_capital) * np.cumprod(benchmark_factors), index=close.index, name='同標的買進持有')
    benchmark_metrics = summarize(benchmark_equity, benchmark_equity.pct_change().dropna())

    trade_frame = pd.DataFrame(trade_rows)
    net_returns = trade_frame['成本後報酬率 (%)'] if not trade_frame.empty else pd.Series(dtype=float)
    wins = net_returns[net_returns > 0]
    losses = net_returns[net_returns < 0]
    # Profit factor is computed from sequential compounded dollar P&L.
    winning_amounts, losing_amounts = [], []
    capital_before = float(initial_capital)
    if not trade_frame.empty:
        for trade_return in net_returns:
            pnl_amount = capital_before * (float(trade_return) / 100.0)
            if pnl_amount > 0:
                winning_amounts.append(pnl_amount)
            elif pnl_amount < 0:
                losing_amounts.append(pnl_amount)
            capital_before += pnl_amount
    profit_factor = (sum(winning_amounts) / abs(sum(losing_amounts))) if losing_amounts else None
    consecutive_losses = longest_loss_streak = 0
    for value in net_returns:
        if value < 0:
            consecutive_losses += 1
            longest_loss_streak = max(longest_loss_streak, consecutive_losses)
        else:
            consecutive_losses = 0
    strategy_metrics.update({
        'trade_count': int(len(net_returns)),
        'win_rate': float((net_returns > 0).mean()) if len(net_returns) else None,
        'profit_factor': float(profit_factor) if profit_factor is not None else None,
        'expectancy_pct': float(net_returns.mean()) if len(net_returns) else None,
        'average_win_pct': float(wins.mean()) if len(wins) else None,
        'average_loss_pct': float(losses.mean()) if len(losses) else None,
        'payoff_ratio': float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else None,
        'average_holding_bars': float(trade_frame['持有K棒數'].mean()) if len(trade_frame) else None,
        'longest_loss_streak': longest_loss_streak,
    })

    if timeframe_name == "1季":
        period_key = close.index.to_period("Q")
        period_label, period_count, period_type = "季報酬", 4, "quarter"
    elif timeframe_name == "半年":
        period_key = pd.MultiIndex.from_arrays([close.index.year, (close.index.month - 1) // 6 + 1])
        period_label, period_count, period_type = "半年報酬", 2, "half_year"
    elif timeframe_name == "1年":
        period_key = close.index.to_period("Y")
        period_label, period_count, period_type = "年報酬", 1, "year"
    else:
        period_key = close.index.to_period("M")
        period_label, period_count, period_type = "月報酬", 12, "month"
    last_positions = pd.Series(np.arange(n), index=close.index).groupby(period_key).last().to_numpy(dtype=int)
    period_ends = strategy_equity.iloc[last_positions]
    period_returns = period_ends.pct_change()
    if len(period_returns):
        period_returns.iloc[0] = period_ends.iloc[0] / float(initial_capital) - 1.0
    period_returns.name = period_label
    return {
        'strategy_equity': strategy_equity,
        'strategy_drawdown': strategy_metrics.pop('drawdown'),
        'strategy_metrics': strategy_metrics,
        'benchmark_equity': benchmark_equity,
        'benchmark_drawdown': benchmark_metrics.pop('drawdown'),
        'benchmark_metrics': benchmark_metrics,
        'trade_analysis': trade_frame,
        'period_returns': period_returns,
        'period_label': period_label,
        'period_count': period_count,
        'period_type': period_type,
        'span_years': span_years,
        'bars_per_year': bars_per_year,
    }
