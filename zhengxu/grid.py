"""Grid strategy calculations and periodic investment helpers."""

import numpy as np
import pandas as pd


def recommend_dynamic_grid(price, history):
    """Suggest a balanced grid around the current price using recent daily range."""
    price = float(price) if price is not None else 0.0
    if not np.isfinite(price) or price <= 0:
        return 0.01, 0.02, 9, "目前價格無效，請先確認股票報價。"

    atr = None
    observation_count = 0
    if isinstance(history, pd.DataFrame) and {"High", "Low", "Close"}.issubset(history.columns):
        recent = history[["High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna().tail(40)
        if len(recent) >= 10:
            previous_close = recent["Close"].shift(1)
            true_range = pd.concat([
                recent["High"] - recent["Low"],
                (recent["High"] - previous_close).abs(),
                (recent["Low"] - previous_close).abs(),
            ], axis=1).max(axis=1).dropna().tail(20)
            if len(true_range) >= 9:
                atr = float(true_range.mean())
                observation_count = len(true_range)

    if atr is None or not np.isfinite(atr) or atr <= 0:
        half_range = round(price * 0.10, 2)
        half_range = max(half_range, 0.01)
        lower = max(round(price - half_range, 2), 0.01)
        upper = round(price + half_range, 2)
        grid_count = 9
        reason = ("最近日線資料不足，先採用通用起始值：現價上下約 10%、共 9 個價位，讓買進與賣出價位分布在現價兩側。"
                  "這只是方便開始規劃的預設，不保證適合每檔股票。")
        return lower, upper, grid_count, reason

    half_range = min(max(3.0 * atr, price * 0.05), price * 0.30)
    half_range = max(round(half_range, 2), 0.01)
    lower = max(round(price - half_range, 2), 0.01)
    upper = max(round(price + half_range, 2), round(lower + 0.02, 2))
    target_spacing = max(1.25 * atr, price * 0.015)
    interval_count = int(round((upper - lower) / target_spacing))
    grid_count = max(3, min(15, interval_count + 1))
    if grid_count % 2 == 0:
        grid_count = grid_count + 1 if grid_count < 15 else grid_count - 1

    actual_spacing_pct = (upper - lower) / max(grid_count - 1, 1) / price * 100.0
    atr_pct = atr / price * 100.0
    down_pct = (price - lower) / price * 100.0
    up_pct = (upper - price) / price * 100.0
    reason = (
        f"依最近約 {observation_count} 個日線資料估算，平均單日波動約 {atr_pct:.2f}%。"
        f"上下限各距現價約 {down_pct:.1f}%／{up_pct:.1f}%：通常取約 3 倍近期波動，至少保留 5% 空間，最多放寬至 30%。"
        f"建議 {grid_count} 個價位，平均間距約 {actual_spacing_pct:.2f}%，並盡量讓現價上下都有規劃價位。"
        "此設定是兼顧交易機會與格距的起點，不代表最佳或保證獲利；可依自己的持股和風險承受度手動調整。"
    )
    return lower, upper, grid_count, reason


def _dual_grid_analyze(ticker, history):
    """Rule-based technical classification. Growth pool is a price/quality proxy, not fundamental advice."""
    if not isinstance(history, pd.DataFrame) or history.empty or "Close" not in history:
        return None
    close = history["Close"].dropna().astype(float)
    if len(close) < 80 or close.iloc[-1] <= 0:
        return None
    last = float(close.iloc[-1])
    high = history["High"].astype(float) if "High" in history else close
    low = history["Low"].astype(float) if "Low" in history else close
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if tr.rolling(14).mean().notna().any() else last * .03
    atr_pct = (atr / last * 100) if last else 0.0
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.rolling(min(100, len(close))).mean().iloc[-1])
    ret20 = float((last / close.iloc[-21] - 1) * 100) if len(close) > 21 else 0.0
    ret60 = float((last / close.iloc[-61] - 1) * 100) if len(close) > 61 else ret20
    ret120 = float((last / close.iloc[-121] - 1) * 100) if len(close) > 121 else ret60
    high60 = float(high.tail(60).max())
    low60 = float(low.tail(60).min())
    range_pct = ((high60 - low60) / last * 100) if last else 0.0
    range_position = ((last - low60) / (high60 - low60)) if high60 > low60 else .5
    peak = float(close.tail(252).max())
    drawdown_from_peak = (last / peak - 1) * 100 if peak else 0.0
    volume_dollar = None
    if "Volume" in history:
        volume_dollar = float((history["Volume"].tail(20) * close.tail(20)).median())
    # Lightweight ADX proxy: smoothed directional movement divided by smoothed true range.
    if "High" in history and "Low" in history:
        up = history["High"].astype(float).diff()
        down = -history["Low"].astype(float).diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        atr14 = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(14).mean() / atr14
        minus_di = 100 * minus_dm.rolling(14).mean() / atr14
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = float(dx.rolling(14).mean().iloc[-1]) if dx.rolling(14).mean().notna().any() else 25.0
    else:
        adx = 25.0
    adx = adx if np.isfinite(adx) else 25.0

    # Range-bound pool: modest trend, usable volatility, and sufficient recent price movement.
    range_score = 0
    range_score += 25 if 1.0 <= atr_pct <= 5.0 else (12 if 0.5 <= atr_pct <= 7.0 else 0)
    range_score += 25 if adx < 22 else (12 if adx < 30 else 0)
    range_score += 20 if -12 <= ret60 <= 12 else (8 if -20 <= ret60 <= 20 else 0)
    range_score += 15 if 8 <= range_pct <= 35 else (8 if 5 <= range_pct <= 45 else 0)
    range_score += 15 if volume_dollar is None or volume_dollar > 1_000_000 else 0

    # Growth pool: technical trend/momentum proxy only. Fundamentals are not guaranteed by this score.
    growth_score = 0
    growth_score += 25 if last > ma200 else 0
    growth_score += 20 if ma50 > ma200 else 0
    growth_score += 20 if ret120 > 8 else (10 if ret120 > 0 else 0)
    growth_score += 15 if ret60 > 0 else 0
    growth_score += 10 if drawdown_from_peak > -25 else (5 if drawdown_from_peak > -40 else 0)
    growth_score += 10 if volume_dollar is None or volume_dollar > 1_000_000 else 0

    # Assign a single primary pool to avoid duplicated capital/stock allocations.
    if max(range_score, growth_score) < 35:
        pool = "暫不納入"
    elif growth_score > range_score:
        pool = "成長型網格"
    else:
        pool = "震盪型網格"
    lower = max(0.01, float(close.tail(60).quantile(.10)))
    upper = float(close.tail(60).quantile(.90))
    if upper <= lower:
        lower, upper = max(0.01, last * .9), last * 1.1
    return {
        "股票代碼": ticker, "現價": last, "主要策略池": pool,
        "震盪分數": int(range_score), "成長分數": int(growth_score),
        "ATR波動率%": atr_pct, "ADX參考值": adx, "近60日報酬%": ret60,
        "近120日報酬%": ret120, "距近一年高點%": drawdown_from_peak,
        "60日區間位置": range_position, "參考下界": lower, "參考上界": upper,
        "20日成交金額中位數": volume_dollar,
    }


def _dual_grid_backtest(close_series, capital, fee_rate, sell_tax_rate, slippage_rate, grid_levels=7,
                        *, dca_monthly_fraction=0.10):
    """Calibrate on 60 closes, then compare three strategies over the remaining dates."""
    prices = pd.to_numeric(pd.Series(close_series), errors="coerce")
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices[np.isfinite(prices) & (prices > 0) & prices.index.notna()]
    prices = prices.sort_index(kind="stable")
    prices = prices[~prices.index.duplicated(keep="last")]
    if len(prices) <= 60 or not np.isfinite(capital) or capital <= 0 or not 0 < dca_monthly_fraction <= 1:
        return None

    calibration = prices.iloc[:60]
    prices = prices.iloc[60:]
    lower = float(calibration.quantile(.10))
    upper = float(calibration.quantile(.90))
    if upper <= lower * 1.02:
        anchor = float(calibration.iloc[0])
        lower, upper = anchor * .85, anchor * 1.15
    levels = np.linspace(lower, upper, int(max(3, grid_levels)))
    start_price = float(prices.iloc[0])
    cost_rate = max(0.0, float(fee_rate)) + max(0.0, float(slippage_rate))
    sell_cost_rate = cost_rate + max(0.0, float(sell_tax_rate))
    cash = float(capital) * .5
    shares = (float(capital) * .5) / (start_price * (1 + cost_rate))
    # Start with a half-cash/half-stock mix; buys/sells are fixed notional slices.
    trade_budget = float(capital) / max(2 * (len(levels) - 1), 2)
    equity_curve = []
    trades = 0
    prev_price = float(prices.iloc[0])
    for idx, p0 in enumerate(prices):
        p = float(p0)
        if idx > 0:
            crossed_down = [lv for lv in levels if prev_price > lv >= p]
            crossed_up = [lv for lv in levels if prev_price < lv <= p]
            for _lv in crossed_down:
                unit_price = p * (1 + cost_rate)
                qty = min(cash / unit_price, trade_budget / unit_price)
                if qty > 0:
                    cash -= qty * unit_price
                    shares += qty
                    trades += 1
            for _lv in crossed_up:
                qty = min(shares, trade_budget / p)
                if qty > 0:
                    cash += qty * p * (1 - sell_cost_rate)
                    shares -= qty
                    trades += 1
        equity_curve.append(cash + shares * p * (1 - sell_cost_rate))
        prev_price = p

    # Buy-and-hold: buy as many shares as possible at start, value remaining cash too.
    hold_shares = capital / (start_price * (1 + cost_rate))
    hold_cash = capital - hold_shares * start_price * (1 + cost_rate)
    hold_curve = hold_cash + hold_shares * prices * (1 - sell_cost_rate)

    # The planned amount is independent of how many months are in the data.
    month_groups = pd.Series(np.arange(len(prices)), index=prices.index).groupby(prices.index.to_period("M")).first().values
    monthly_totals = capital * np.minimum(1.0, dca_monthly_fraction * np.arange(1, len(month_groups) + 1))
    contributions = np.diff(monthly_totals, prepend=0.0)
    invested = pd.Series(0.0, index=prices.index)
    invested.iloc[month_groups] = contributions
    dca_shares = (invested / (prices * (1 + cost_rate))).cumsum()
    dca_curve = (capital - invested.cumsum()).clip(lower=0) + dca_shares * prices * (1 - sell_cost_rate)

    years = (prices.index[-1] - prices.index[0]).days / 365.25
    strategies = (
        ("雙邊動態網格（示意）", pd.Series(equity_curve, index=prices.index), trades),
        ("買進持有", hold_curve, 1),
        ("每月定期定額（示意）", dca_curve, int(np.count_nonzero(contributions))),
    )
    rows = []
    for label, curve, trade_count in strategies:
        end = float(curve.iloc[-1])
        drawdown = curve / curve.cummax().clip(lower=capital) - 1
        rows.append({
            "策略": label,
            "期末資產": end,
            "總報酬%": (end / capital - 1) * 100,
            "年化報酬CAGR%": ((end / capital) ** (1 / years) - 1) * 100 if years > 0 else np.nan,
            "最大回撤%": float(drawdown.min() * 100),
            "成交次數": trade_count,
        })
    chart = pd.DataFrame({label: curve for label, curve, _ in strategies}).rename_axis("日期").reset_index()
    return pd.DataFrame(rows), chart, {
        "lower": lower, "upper": upper, "trades": trades, "years": years,
        "calibration_start": calibration.index[0].date(), "calibration_end": calibration.index[-1].date(),
        "evaluation_start": prices.index[0].date(), "evaluation_end": prices.index[-1].date(),
        "dca_monthly_amount": float(capital * dca_monthly_fraction),
    }

def calculate_money_weighted_return(cash_flows):
    """Solve annualized return for dated cash flows; return None if no valid root exists."""
    dated = {}
    for date_value, amount in cash_flows:
        date_value = pd.Timestamp(date_value)
        if date_value.tzinfo is not None:
            date_value = date_value.tz_localize(None)
        dated[date_value.normalize()] = dated.get(date_value.normalize(), 0.0) + float(amount)
    dated = sorted((day, amount) for day, amount in dated.items() if amount != 0)
    if len(dated) < 2 or not any(amount < 0 for _, amount in dated) or not any(amount > 0 for _, amount in dated):
        return None
    start = dated[0][0]
    years = np.asarray([(day - start).days / 365.25 for day, _ in dated], dtype=float)
    amounts = np.asarray([amount for _, amount in dated], dtype=float)

    def npv(rate):
        try:
            return float(np.sum(amounts / np.power(1.0 + rate, years)))
        except (OverflowError, FloatingPointError, ZeroDivisionError):
            return float("inf") if rate < 0 else float("-inf")

    low, high = -0.9999, 1.0
    low_value = npv(low)
    high_value = npv(high)
    while high_value > 0 and high < 1_000_000:
        high = high * 2.0 + 1.0
        high_value = npv(high)
    if not np.isfinite(low_value) or not np.isfinite(high_value) or low_value * high_value > 0:
        return None
    for _ in range(120):
        middle = (low + high) / 2.0
        middle_value = npv(middle)
        if abs(middle_value) < 1e-8:
            return float(middle)
        if middle_value > 0:
            low = middle
        else:
            high = middle
    return float((low + high) / 2.0)


def simulate_monthly_dca(
    price_series, monthly_contribution, commission_rate=0.0, exit_cost_rate=0.0,
    *, deduct_cost_from_contribution=False,
):
    """Buy monthly and report cash-flow-aware returns.

    Comparison pages deduct an estimated buy cost from the allocated cash,
    matching the buy-and-hold and grid cash-budget convention.
    """
    if isinstance(price_series, pd.DataFrame):
        if "Close" not in price_series.columns:
            raise ValueError("定期定額需要收盤價資料。")
        price_series = price_series["Close"]
    prices = pd.to_numeric(price_series, errors="coerce").copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.loc[~prices.index.isna()].dropna().sort_index()
    prices = prices[prices > 0]
    monthly_contribution = float(monthly_contribution)
    commission_rate = max(0.0, float(commission_rate))
    exit_cost_rate = min(1.0, max(0.0, float(exit_cost_rate)))
    if prices.empty or monthly_contribution <= 0:
        raise ValueError("請提供有效股價與大於零的每月投入金額。")

    allocation_factor = (
        max(0.0, 1.0 - commission_rate)
        if deduct_cost_from_contribution else 1.0 / (1.0 + commission_rate)
    )
    first_sessions = set(prices.groupby(prices.index.to_period("M"), sort=True).head(1).index)
    shares = 0.0
    total_contributed = 0.0
    total_fees = 0.0
    history_rows = []
    cash_flows = []
    time_weighted_growth = 1.0
    previous_value = 0.0
    for day, price in prices.items():
        contribution = monthly_contribution if day in first_sessions else 0.0
        if contribution > 0:
            added_shares = contribution * allocation_factor / float(price)
            shares += added_shares
            total_contributed += contribution
            total_fees += contribution - added_shares * float(price)
            cash_flows.append((day, -contribution))
        portfolio_value = shares * float(price)
        denominator = previous_value + contribution
        period_growth = portfolio_value / denominator if denominator > 0 else 1.0
        time_weighted_growth *= period_growth
        history_rows.append({
            "date": day,
            "portfolio_value": portfolio_value,
            "contribution": contribution,
            "time_weighted_growth": time_weighted_growth,
        })
        previous_value = portfolio_value

    history = pd.DataFrame(history_rows).set_index("date")
    ending_value_before_exit_cost = float(history["portfolio_value"].iloc[-1])
    exit_fee = ending_value_before_exit_cost * exit_cost_rate
    if exit_fee:
        history.iloc[-1, history.columns.get_loc("portfolio_value")] = ending_value_before_exit_cost - exit_fee
        total_fees += exit_fee
    ending_value = float(history["portfolio_value"].iloc[-1])
    cash_flows.append((history.index[-1], ending_value))
    money_weighted_return = calculate_money_weighted_return(cash_flows)
    twr_curve = history["time_weighted_growth"].copy()
    twr_curve.iloc[-1] *= 1.0 - exit_cost_rate
    drawdown = twr_curve / twr_curve.cummax() - 1.0
    return {
        "history": history,
        "metrics": {
            "contribution_count": len(first_sessions),
            "total_contributed": total_contributed,
            "ending_value": ending_value,
            "ending_value_before_exit_cost": ending_value_before_exit_cost,
            "profit_loss": ending_value - total_contributed,
            "money_weighted_return": money_weighted_return,
            "mdd": float(drawdown.min()),
            "fees": total_fees,
            "exit_fee": exit_fee,
        },
    }
