"""Market sources, cached market data and market analysis helpers."""

from math import isfinite
from typing import Mapping
import csv
import io
import numpy as np
import os
import pandas as pd
import re
import requests
import sqlite3
import streamlit as st
import time
import yfinance as yf

from .radars import (
    calculate_gvi,
)

from .dividend import (
    _finite_number,
)
from .storage import (
    CACHE_TTL,
    DB_FILE,
)


def parse_roe(raw_roe):
    value = _finite_number(raw_roe)
    if value is None:
        return None
    val = value
    if abs(val) > 3.0:
        val = val / 100.0
    return val

def is_cache_valid(updated_at):
    if not updated_at or np.isnan(updated_at):
        return False
    return (time.time() - float(updated_at)) < CACHE_TTL


def fetch_tw_official_pb_pe(ticker):
    clean_code = ticker.replace(".TW", "").replace(".TWO", "")
    is_tpex = ".TWO" in ticker
    try:
        if is_tpex:
            url = "https://www.tpex.org.tw/web/stock/aftertrading/peratio_analysis/pera_result.php?l=zh-tw&response=json"
            res = requests.get(url, timeout=4)
            data = res.json()
            for row in data.get('aaData', []):
                if row[0] == clean_code:
                    pb = float(row[5]) if len(row) > 5 and row[5] not in ['N/A', '-', ''] else None
                    pe = float(row[2]) if len(row) > 2 and row[2] not in ['N/A', '-', ''] else None
                    return pb, pe
        else:
            url = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d?response=json"
            res = requests.get(url, timeout=4)
            data = res.json()
            for row in data.get('data', []):
                if row[0] == clean_code:
                    pb = float(row[5]) if len(row) > 5 and row[5] not in ['N/A', '-', ''] else None
                    pe = float(row[4]) if len(row) > 4 and row[4] not in ['N/A', '-', ''] else None
                    return pb, pe
    except Exception as e:
        st.sidebar.caption(f"TWSE/TPEx API 擷取提醒 ({ticker}): {e}")
    return None, None

def _to_number(value):
    """Parse TWSE/TPEx comma-formatted numbers without turning missing values into zero."""
    if value is None:
        return None
    text = str(value).strip().replace(',', '').replace(' ', '')
    if text in ('', '-', '--', 'N/A', 'nan', 'None'):
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_grid_daily_history(ticker):
    """Fetch cached daily prices used only to suggest initial grid settings."""
    try:
        frame = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _dual_grid_fetch_history(ticker, period="2y", *, start=None, end=None):
    """Cached daily OHLCV for the dual-pool screener and illustrative backtest."""
    try:
        date_args = {"period": period} if start is None else {"start": start, "end": end}
        frame = yf.download(ticker, **date_args, interval="1d", auto_adjust=True, progress=False, threads=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame()
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
        frame = frame[needed].copy()
        for col in needed:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def build_chip_flow_summary(price_history, institutional_trade=None, issued_shares=None, timeframe_name=None):
    """Summarize accessible trend, volume activity, and official institutional flow in plain language."""
    summary = {}
    if isinstance(price_history, pd.DataFrame) and "Close" in price_history.columns:
        close = pd.to_numeric(price_history["Close"], errors="coerce").dropna()
        if len(close) >= 20 and float(close.iloc[-1]) > 0:
            average = float(close.tail(20).mean())
            change = float(close.iloc[-1] / average - 1.0) if average > 0 else None
            recent_closes = close.tail(20)
            period_change = (
                float(recent_closes.iloc[-1] / recent_closes.iloc[0] - 1.0)
                if len(recent_closes) >= 2 and float(recent_closes.iloc[0]) > 0 else None
            )
            if change is not None:
                tone = "偏強" if change >= 0.01 else "偏弱" if change <= -0.01 else "整理"
                relation = "高於" if change > 0 else "低於" if change < 0 else "接近"
                summary["trend"] = {
                    "status": f"{tone}｜現價{relation}20日平均價",
                    "detail": f"近20筆收盤價由首筆至末筆變化 {period_change:+.2%}；現價較近20筆平均價 {change:+.2%}。這描述近期走勢，不代表未來方向。",
                    "period_change_pct": period_change,
                    "period_bars": len(recent_closes),
                }
        if isinstance(price_history, pd.DataFrame) and "Volume" in price_history.columns:
            volume = pd.to_numeric(price_history["Volume"], errors="coerce").dropna()
            if len(volume) >= 10:
                recent = float(volume.tail(5).mean())
                baseline_series = volume.iloc[-25:-5] if len(volume) >= 25 else volume.iloc[:-5]
                baseline = float(baseline_series.mean()) if not baseline_series.empty else 0.0
                if baseline > 0:
                    ratio = recent / baseline
                    tone = "交易變熱" if ratio >= 1.2 else "交易轉淡" if ratio <= 0.8 else "大致持平"
                    summary["volume"] = {
                        "status": tone,
                        "detail": f"近5筆平均成交量是前段參考平均的 {ratio:.2f} 倍；只代表成交活躍度。",
                    }
        if timeframe_name == "1日" and len(close) >= 120 and float(close.iloc[-1]) > 0:
            high_close_120 = float(close.tail(120).max())
            distance = float(close.iloc[-1] / high_close_120 - 1.0) if high_close_120 > 0 else None
            if distance is not None:
                summary["high_120"] = {
                    "status": f"距120日日線最高收盤 {distance:.2%}",
                    "detail": f"現價相對近120日日線最高收盤價變化 {distance:+.2%}；這是價格位置描述，不是突破或買賣訊號。",
                }

    net_keys = ["外資買賣超股數", "投信買賣超股數", "自營商買賣超股數"]
    if institutional_trade and issued_shares and issued_shares > 0:
        net_values = [_finite_number(institutional_trade.get(key)) for key in net_keys]
        if all(value is not None for value in net_values):
            net_pct = sum(net_values) / float(issued_shares) * 100.0
            direction = "法人當日偏買" if net_pct > 0 else "法人當日偏賣" if net_pct < 0 else "法人當日買賣接近"
            summary["institutions"] = {
                "status": direction,
                "detail": f"三大法人合計買賣超佔已發行股數 {net_pct:+.4f}%；只看當日，單日變化不等於持股集中。",
            }
    return summary


def build_broker_branch_price_breadth_reading(branch_summary, price_change_pct):
    """Pair recent price direction with net-buy/net-sell branch breadth cautiously."""
    if not isinstance(branch_summary, Mapping):
        return "分點家數資料不足，暫時無法比較價格方向與買賣參與廣度。"
    change = _finite_number(price_change_pct)
    if change is None:
        return "價格資料不足，暫時無法配對股價方向與分點家數。"

    buyers = int(branch_summary.get("buyers", 0) or 0)
    sellers = int(branch_summary.get("sellers", 0) or 0)
    unit = "分點" if branch_summary.get("count_unit") == "分點" else "資料列"
    if change > 0 and buyers > sellers:
        reading = (
            f"近20筆價格上漲 {change:+.2%}，淨買超{unit}數也多於淨賣超{unit}數；"
            "兩個方向暫時一致，可作為買方參與較廣的線索。"
        )
    elif change > 0 and sellers > buyers:
        reading = (
            f"近20筆價格上漲 {change:+.2%}，但淨賣超{unit}數較多，方向出現分歧；"
            "可能同時有賣方參與較廣與其他買盤支撐，不能只靠家數判定出貨。"
        )
    elif change < 0 and sellers > buyers:
        reading = (
            f"近20筆價格下跌 {change:+.2%}，淨賣超{unit}數也較多；"
            "可留意賣方參與較廣的線索，但不能據此斷定恐慌或預測續跌。"
        )
    elif change < 0 and buyers > sellers:
        reading = (
            f"近20筆價格下跌 {change:+.2%}，但淨買超{unit}數較多，方向出現分歧；"
            "可能有分點承接，也可能只是買進參與面較廣，價格仍偏弱，不能直接解讀成吸籌。"
        )
    else:
        reading = (
            f"近20筆價格變化為 {change:+.2%}，淨買超與淨賣超{unit}數相同；"
            "目前家數沒有顯示明確的買賣參與廣度差異。"
        )
    return reading + " 分點家數不是投資人數，也不等於交易金額。"


def _normalize_chip_trade_date(value):
    """Normalize TWSE Gregorian or TPEx ROC trade dates without guessing unknown labels."""
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r'\d{8}', text):
        return f'{text[:4]}-{text[4:6]}-{text[6:8]}'
    roc_match = re.fullmatch(r'(\d{2,3})[./年-](\d{1,2})[./月-](\d{1,2})日?', text)
    if roc_match:
        year, month, day = map(int, roc_match.groups())
        if year < 1911:
            year += 1911
        try:
            return pd.Timestamp(year=year, month=month, day=day).strftime('%Y-%m-%d')
        except ValueError:
            return None
    parsed = pd.to_datetime(text, errors='coerce')
    return parsed.strftime('%Y-%m-%d') if not pd.isna(parsed) else None


def save_chip_flow_observation(symbol, institutional_trade, issued_shares, volume_shares=None):
    """Persist one observed official daily row; missing investor values remain NULL, never zero."""
    if not institutional_trade:
        return False
    trade_date = _normalize_chip_trade_date(institutional_trade.get('日期'))
    if not trade_date:
        return False
    investor_values = (
        _finite_number(institutional_trade.get('外資買賣超股數')),
        _finite_number(institutional_trade.get('投信買賣超股數')),
        _finite_number(institutional_trade.get('自營商買賣超股數')),
    )
    if all(value is None for value in investor_values):
        return False
    try:
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            conn.execute('''INSERT INTO chip_flow_observations
                (symbol, trade_date, market, foreign_net, trust_net, dealer_net, issued_shares, volume_shares, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, trade_date) DO UPDATE SET
                    market=excluded.market,
                    foreign_net=excluded.foreign_net,
                    trust_net=excluded.trust_net,
                    dealer_net=excluded.dealer_net,
                    issued_shares=excluded.issued_shares,
                    volume_shares=COALESCE(excluded.volume_shares, chip_flow_observations.volume_shares),
                    updated_at=excluded.updated_at''',
                (str(symbol), trade_date, str(institutional_trade.get('市場') or ''),
                 *investor_values, _finite_number(issued_shares), _finite_number(volume_shares),
                 pd.Timestamp.now(tz='Asia/Taipei').isoformat()))
        return True
    except Exception:
        return False


def load_chip_flow_observations(symbol, limit=20):
    """Load only dates actually captured from official sources, newest first."""
    try:
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            return pd.read_sql_query(
                '''SELECT trade_date, market, foreign_net, trust_net, dealer_net, issued_shares, volume_shares
                   FROM chip_flow_observations WHERE symbol=?
                   ORDER BY trade_date DESC LIMIT ?''',
                conn, params=(str(symbol), int(limit)))
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_institutional_market_data():
    """Fetch official latest daily institutional trades and issued share counts for TWSE/TPEx."""
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.twse.com.tw/'}
    result = {}
    errors = []
    # TWSE T86 is date based. Walk back over weekends/holidays to the latest report.
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo('Asia/Taipei')).date()
    twse_payload = None
    twse_date = None
    for offset in range(8):
        query_date = (today - timedelta(days=offset)).strftime('%Y%m%d')
        try:
            response = requests.get('https://www.twse.com.tw/rwd/zh/fund/T86',
                                    params={'date': query_date, 'selectType': 'ALLBUT0999', 'response': 'json'},
                                    headers=headers, timeout=12)
            response.raise_for_status()
            payload = response.json()
            if payload.get('data') and payload.get('fields'):
                twse_payload, twse_date = payload, query_date
                break
        except Exception as exc:
            errors.append(f'TWSE: {exc}')
    if twse_payload:
        fields = twse_payload.get('fields', [])
        for row in twse_payload.get('data', []):
            values = dict(zip(fields, row))
            code = str(values.get('證券代號', '')).strip()
            if not code:
                continue
            def pick(prefix):
                for key, val in values.items():
                    if prefix in key:
                        return _to_number(val)
                return None
            foreign_buy = pick('外陸資買進股數(不含外資自營商)')
            foreign_sell = pick('外陸資賣出股數(不含外資自營商)')
            trust_buy, trust_sell = pick('投信買進股數'), pick('投信賣出股數')
            # Domestic dealer buy/sell includes both proprietary and hedge categories.
            dealer_buys = [pick('自營商買進股數(自行買賣)'), pick('自營商買進股數(避險)')]
            dealer_sells = [pick('自營商賣出股數(自行買賣)'), pick('自營商賣出股數(避險)')]
            result[code] = {'日期': twse_date, '市場': '上市', '股票名稱': str(values.get('證券名稱', '')).strip(),
                '外資買進股數': foreign_buy, '外資賣出股數': foreign_sell,
                '外資買賣超股數': (foreign_buy - foreign_sell) if None not in (foreign_buy, foreign_sell) else None,
                '投信買進股數': trust_buy, '投信賣出股數': trust_sell,
                '投信買賣超股數': (trust_buy - trust_sell) if None not in (trust_buy, trust_sell) else None,
                '自營商買進股數': sum(x for x in dealer_buys if x is not None) if any(x is not None for x in dealer_buys) else None,
                '自營商賣出股數': sum(x for x in dealer_sells if x is not None) if any(x is not None for x in dealer_sells) else None}
            if result[code]['自營商買進股數'] is not None and result[code]['自營商賣出股數'] is not None:
                result[code]['自營商買賣超股數'] = result[code]['自營商買進股數'] - result[code]['自營商賣出股數']
            else:
                result[code]['自營商買賣超股數'] = None
    else:
        errors.append('TWSE 最近 8 日沒有可用的 T86 資料。')

    try:
        response = requests.get('https://www.tpex.org.tw/openapi/v1/tpex_3insti_daily_trading',
                                headers=headers, timeout=12)
        response.raise_for_status()
        payload = response.json()
        for values in payload if isinstance(payload, list) else []:
            code = str(values.get('SecuritiesCompanyCode', '')).strip()
            if not code:
                continue
            def find_value(*needles):
                for key, val in values.items():
                    if all(n.lower() in key.lower() for n in needles):
                        return _to_number(val)
                return None
            fb, fs = find_value('Foreign Investors', 'Total Buy'), find_value('Foreign Investors', 'Total Sell')
            tb, ts = find_value('SecuritiesInvestmentTrustCompanies', 'TotalBuy'), find_value('SecuritiesInvestmentTrustCompanies', 'TotalSell')
            db, ds = find_value('Dealers', 'TotalBuy'), find_value('Dealers', 'TotalSell')
            # If endpoint labels use spaces in place of camel case, retry common variants.
            if tb is None: tb = find_value('Securities Investment Trust Companies', 'Total Buy')
            if ts is None: ts = find_value('Securities Investment Trust Companies', 'Total Sell')
            if db is None: db = find_value('Dealers', 'Total Buy')
            if ds is None: ds = find_value('Dealers', 'Total Sell')
            result[code] = {'日期': str(values.get('Date', '最新交易日')), '市場': '上櫃',
                '股票名稱': str(values.get('CompanyName') or values.get('SecuritiesCompanyName') or values.get('公司名稱') or '').strip(),
                '外資買進股數': fb, '外資賣出股數': fs, '外資買賣超股數': fb-fs if None not in (fb, fs) else None,
                '投信買進股數': tb, '投信賣出股數': ts, '投信買賣超股數': tb-ts if None not in (tb, ts) else None,
                '自營商買進股數': db, '自營商賣出股數': ds, '自營商買賣超股數': db-ds if None not in (db, ds) else None}
    except Exception as exc:
        errors.append(f'TPEx: {exc}')

    shares_by_code = {}
    names_by_code = {}
    for url in ('https://openapi.twse.com.tw/v1/opendata/t187ap03_L',
                'https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O'):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            profiles = response.json()
            for profile in profiles if isinstance(profiles, list) else []:
                code = str(profile.get('公司代號') or profile.get('SecuritiesCompanyCode') or '').strip()
                shares = _to_number(profile.get('已發行普通股數或TDR原股發行股數') or profile.get('已發行普通股數') or profile.get('IssuedShares'))
                company_name = str(profile.get('公司名稱') or profile.get('CompanyName') or profile.get('公司簡稱') or '').strip()
                if code and shares and shares > 0:
                    shares_by_code[code] = shares
                if code and company_name:
                    names_by_code[code] = company_name
        except Exception as exc:
            errors.append(f'股數資料: {exc}')
    for code, trade in result.items():
        if not trade.get('股票名稱'):
            trade['股票名稱'] = names_by_code.get(code, code)
    return result, shares_by_code, errors

def build_institutional_screen(trades, shares_by_code, tickers, investor, min_net_pct):
    rows = []
    for ticker in tickers:
        code = ticker.replace('.TWO', '').replace('.TW', '').strip()
        trade = trades.get(code)
        shares = shares_by_code.get(code)
        if not trade or not shares:
            continue
        row = {'股票代碼': ticker, '股票名稱': trade.get('股票名稱') or ticker, '市場': trade['市場'], '資料日期': trade['日期'], '已發行股數': shares}
        for name in ('外資', '投信', '自營商'):
            for measure, label in (('買進股數', '買進'), ('賣出股數', '賣出'), ('買賣超股數', '買賣超')):
                value = trade.get(f'{name}{measure}')
                row[f'{name}{label}佔股本比(%)'] = value / shares * 100 if value is not None else None
        if row.get(f'{investor}買賣超佔股本比(%)') is not None and row[f'{investor}買賣超佔股本比(%)'] >= min_net_pct:
            rows.append(row)
    return pd.DataFrame(rows)


TRADER_WATCHLIST_DEFAULTS = ("2330.TW", "3293.TWO", "8069.TWO", "NVDA", "AAPL") + ("",) * 7


def get_trader_watchlist_symbols():
    """Read the watchlist across module reruns without depending on another page's local variable."""
    saved_values = st.session_state.get("trader_watchlist_values")
    if isinstance(saved_values, (list, tuple)):
        source_values = saved_values
    else:
        widget_keys = [f"trader_watch_ticker_{index}" for index in range(1, 13)]
        if any(key in st.session_state for key in widget_keys):
            source_values = [st.session_state.get(key, "") for key in widget_keys]
        else:
            source_values = TRADER_WATCHLIST_DEFAULTS

    normalized = []
    for value in source_values:
        ticker = str(value or "").strip().upper()
        if ticker and ticker not in normalized:
            normalized.append(ticker)
    return normalized


def dataframe_to_excel_compatible_csv_bytes(frame):
    """Serialize a comma-delimited CSV with an explicit UTF-8 BOM for Excel."""
    csv_text = frame.to_csv(index=False, sep=',', lineterminator='\r\n')
    return b'\xef\xbb\xbf' + csv_text.encode('utf-8')


def dataframe_to_excel_workbook_bytes(frame, sheet_name='法人策略結果'):
    """Build an Excel workbook with readable widths, frozen headers, and filters."""
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes = 'A2'
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_index, column_name in enumerate(frame.columns, start=1):
            values = [str(column_name)] + [
                str(value) for value in frame[column_name].dropna().tolist()
            ]
            display_width = max((len(value) for value in values), default=10)
            worksheet.column_dimensions[get_column_letter(column_index)].width = min(
                max(display_width + 2, 12), 42
            )
            if str(column_name).endswith('(%)'):
                for row_index in range(2, len(frame) + 2):
                    worksheet.cell(row=row_index, column=column_index).number_format = '+0.0000"%";-0.0000"%";0.0000"%'
    return output.getvalue()


def record_institutional_strategy_snapshot(trades, shares_by_code, tickers):
    """Save official daily institutional flows for the strategy's Taiwan watchlist."""
    saved = 0
    for ticker in dict.fromkeys(str(value) for value in tickers):
        code = ticker.replace('.TWO', '').replace('.TW', '').strip()
        trade = trades.get(code)
        if trade and save_chip_flow_observation(code, trade, shares_by_code.get(code)):
            saved += 1
    return saved


def build_institutional_flow_streaks(tickers, trades, shares_by_code, investor, minimum_observations=2):
    """Summarize same-direction net-flow streaks from captured official report dates."""
    net_columns = {
        '外資': 'foreign_net',
        '投信': 'trust_net',
        '自營商': 'dealer_net',
    }
    net_column = net_columns.get(investor)
    if not net_column:
        return pd.DataFrame()

    rows = []
    for ticker in dict.fromkeys(str(value) for value in tickers):
        code = ticker.replace('.TWO', '').replace('.TW', '').strip()
        trade = trades.get(code)
        current_date = _normalize_chip_trade_date(trade.get('日期')) if trade else None
        if not current_date:
            continue

        history = load_chip_flow_observations(code, limit=120)
        if history.empty:
            continue
        history = history.sort_values('trade_date', ascending=False).reset_index(drop=True)
        # Only report a streak if the latest observation belongs to this scan's data date.
        if str(history.iloc[0].get('trade_date')) != current_date:
            continue
        latest_net = _finite_number(history.iloc[0].get(net_column))
        if latest_net is None or latest_net == 0:
            continue

        direction = 1 if latest_net > 0 else -1
        streak_values = []
        streak_dates = []
        for _, observation in history.iterrows():
            net_value = _finite_number(observation.get(net_column))
            if net_value is None or net_value == 0 or (1 if net_value > 0 else -1) != direction:
                break
            streak_values.append(net_value)
            streak_dates.append(str(observation.get('trade_date') or ''))
        if len(streak_values) < int(minimum_observations):
            continue

        shares = _finite_number(shares_by_code.get(code))
        if not shares or shares <= 0:
            shares = _finite_number(history.iloc[0].get('issued_shares'))
        cumulative_net = sum(streak_values)
        rows.append({
            '股票代碼': ticker,
            '股票名稱': str(trade.get('股票名稱') or ticker),
            '市場': str(trade.get('市場') or ''),
            '連續方向': '連續淨買超' if direction > 0 else '連續淨賣超',
            '連續已記錄資料日': len(streak_values),
            '最新單日淨額（股）': latest_net,
            '連續期間累計淨額（股）': cumulative_net,
            '連續期間累計佔股本（%）': cumulative_net / shares * 100 if shares else None,
            '起始資料日': streak_dates[-1],
            '最新資料日': current_date,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result['_累計流量絕對值'] = pd.to_numeric(
            result['連續期間累計佔股本（%）'], errors='coerce'
        ).abs()
        result = result.sort_values(
            ['連續已記錄資料日', '_累計流量絕對值'],
            ascending=[False, False],
            na_position='last',
        ).drop(columns=['_累計流量絕對值']).reset_index(drop=True)
    return result


def interpret_institutional_price_volume(flow_direction, price_change_pct, volume_change_pct):
    """Explain whether recent price/volume behavior agrees with observed institutional flow."""
    price_state = (
        '上漲' if price_change_pct is not None and price_change_pct >= 0.5
        else '下跌' if price_change_pct is not None and price_change_pct <= -0.5
        else '接近平盤' if price_change_pct is not None else None
    )
    volume_state = (
        '量增' if volume_change_pct is not None and volume_change_pct >= 10
        else '量縮' if volume_change_pct is not None and volume_change_pct <= -10
        else '量能變化不大' if volume_change_pct is not None else None
    )
    is_buying = flow_direction == '連續淨買超'

    if price_state == '上漲' and volume_state == '量增':
        if is_buying:
            return '法人連續買超、價升量增，流向與近期價格反應一致；仍可能伴隨追價，不能當成續漲保證。'
        return '法人連續賣超，但股價上漲且量增，流向與價格不同步；可能有其他買盤承接，不能只憑此判定出貨。'
    if price_state == '下跌' and volume_state == '量增':
        if is_buying:
            return '法人連續買超，但股價下跌且量增，價格尚未配合，賣壓可能較強；不能直接解讀為吸籌。'
        return '法人連續賣超、價跌量增，流向與近期價格反應一致，賣壓值得留意；仍不是未來跌勢保證。'
    if is_buying and price_state == '下跌':
        return '法人連續買超但股價走弱，價格尚未配合；先觀察賣壓是否減輕，不要直接視為吸籌。'
    if not is_buying and price_state == '上漲':
        return '法人連續賣超但股價走強，流向與價格不同步；可能有其他買盤承接，勿只憑法人流向判定出貨。'
    if price_state == '上漲' and volume_state == '量縮':
        return '股價上漲但成交量縮，價格有走高、參與度未同步增加；價量確認仍有限。'
    if price_state == '下跌' and volume_state == '量縮':
        return '股價下跌但成交量縮，走勢偏弱但成交未擴大；可觀察後續是否止穩。'
    if price_state and volume_state:
        return f'近5日股價{price_state}、成交{volume_state}；與法人連續流向的配合尚不明確，避免單一指標下結論。'
    return '股價或成交量資料不足，暫不判斷與法人流向是否配合。'


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_institutional_price_volume_context(ticker, as_of_date):
    """Fetch recent daily closes/volumes ending on the institutional report date."""
    try:
        normalized_as_of = _normalize_chip_trade_date(as_of_date)
        if not normalized_as_of:
            return {'股價資料日': None, '近5日股價變化(%)': None, '近5日均量變化(%)': None}
        as_of = pd.Timestamp(normalized_as_of)
        start = (as_of - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        end = (as_of + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        history = yf.download(
            str(ticker), start=start, end=end, interval='1d',
            auto_adjust=False, progress=False, threads=False,
        )
        if history is None or history.empty:
            return {'股價資料日': None, '近5日股價變化(%)': None, '近5日均量變化(%)': None}
        if isinstance(history.columns, pd.MultiIndex):
            matched_level = next((
                level for level in range(history.columns.nlevels)
                if str(ticker) in history.columns.get_level_values(level).astype(str)
            ), None)
            if matched_level is not None:
                history = history.xs(str(ticker), axis=1, level=matched_level)
            if isinstance(history.columns, pd.MultiIndex):
                history.columns = history.columns.get_level_values(0)
        if 'Close' not in history.columns or 'Volume' not in history.columns:
            return {'股價資料日': None, '近5日股價變化(%)': None, '近5日均量變化(%)': None}

        history = history[['Close', 'Volume']].copy()
        history.index = pd.to_datetime(history.index, errors='coerce')
        if getattr(history.index, 'tz', None) is not None:
            history.index = history.index.tz_localize(None)
        history = history.loc[history.index <= as_of]
        history['Close'] = pd.to_numeric(history['Close'], errors='coerce')
        history['Volume'] = pd.to_numeric(history['Volume'], errors='coerce')
        history = history.dropna(subset=['Close', 'Volume']).sort_index()
        if history.empty:
            return {'股價資料日': None, '近5日股價變化(%)': None, '近5日均量變化(%)': None}

        price_change_pct = None
        if len(history) >= 6 and float(history['Close'].iloc[-6]) != 0:
            price_change_pct = (float(history['Close'].iloc[-1]) / float(history['Close'].iloc[-6]) - 1) * 100
        volume_change_pct = None
        if len(history) >= 10:
            previous_volume = float(history['Volume'].iloc[-10:-5].mean())
            recent_volume = float(history['Volume'].iloc[-5:].mean())
            if previous_volume > 0:
                volume_change_pct = (recent_volume / previous_volume - 1) * 100
        return {
            '股價資料日': history.index[-1].strftime('%Y-%m-%d'),
            '近5日股價變化(%)': price_change_pct,
            '近5日均量變化(%)': volume_change_pct,
        }
    except Exception:
        return {'股價資料日': None, '近5日股價變化(%)': None, '近5日均量變化(%)': None}

TDCC_DISTRIBUTION_URL = "https://openapi.tdcc.com.tw/v1/opendata/1-5"
TDCC_HOLDING_GROUPS = {"retail": ("1", "2"), "big": ("12", "13", "14", "15")}


def _parse_tdcc_concentration(payload, ticker):
    """Summarize complete TDCC tier groups without treating missing rows as zero."""
    if not isinstance(payload, list):
        return {"status": "format_error"}
    if not payload:
        return {"status": "missing"}
    if not isinstance(payload[0], dict) or "證券代號" not in payload[0]:
        return {"status": "format_error"}

    code = ticker.strip().upper().removesuffix(".TWO").removesuffix(".TW")
    target_tiers = {tier for tiers in TDCC_HOLDING_GROUPS.values() for tier in tiers}
    weeks = {}
    for row in payload:
        if not isinstance(row, dict) or str(row.get("證券代號", "")).strip() != code:
            continue
        raw_date = str(row.get("資料日期", row.get("\ufeff資料日期", ""))).strip()
        date = pd.to_datetime(raw_date, format="%Y%m%d", errors="coerce") if re.fullmatch(r"[0-9]{8}", raw_date) else pd.NaT
        if pd.isna(date):
            return {"status": "format_error"}
        week = weeks.setdefault(date.date().isoformat(), {})
        tier = str(row.get("持股分級", "")).strip()
        if tier not in target_tiers:
            continue
        shares = str(row.get("股數", "")).strip()
        ratio = _to_number(row.get("占集保庫存數比例%", row.get("佔集保庫存數比例%")))
        if tier in week or not re.fullmatch(r"[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+", shares) or ratio is None or not 0 <= ratio <= 100:
            week[tier] = None
        else:
            week[tier] = (int(shares.replace(",", "")), ratio)

    if not weeks:
        return {"status": "missing"}

    dates = sorted(weeks, reverse=True)[:2]
    result = {"date": dates[0], "previous_date": dates[1] if len(dates) == 2 else None}
    for name, tiers in TDCC_HOLDING_GROUPS.items():
        totals = []
        for date in dates:
            values = [weeks[date].get(tier) for tier in tiers]
            totals.append(
                (sum(value[0] for value in values), round(sum(value[1] for value in values), 8))
                if all(value is not None for value in values) else (None, None)
            )
        result[f"{name}_shares"], result[f"{name}_pct"] = totals[0]
        current = totals[0][1]
        previous = totals[1][1] if len(totals) == 2 else None
        result[f"{name}_change_pp"] = round(current - previous, 8) if current is not None and previous is not None else None

    valid_groups = sum(result[f"{name}_pct"] is not None for name in TDCC_HOLDING_GROUPS)
    result["status"] = {0: "format_error", 1: "partial", 2: "ok"}[valid_groups]
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tdcc_concentration(ticker):
    response = requests.get(TDCC_DISTRIBUTION_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return _parse_tdcc_concentration(response.json(), ticker)


def _render_tdcc_concentration(ticker):
    try:
        with st.spinner("讀取集保中心持股級距資料…"):
            result = fetch_tdcc_concentration(ticker)
    except ValueError:
        result = {"status": "format_error"}
    except requests.RequestException:
        result = {"status": "source_error"}

    state_key = f"tdcc_last_valid_{ticker}"
    status = result["status"]
    if status == "ok":
        st.session_state[state_key] = result
    elif status != "partial":
        messages = {
            "missing": "官方資料未包含這檔證券。",
            "format_error": "集保資料格式異常或級距不完整。",
            "source_error": "集保資料暫時無法更新。",
        }
        message = messages[status]
        last_valid = st.session_state.get(state_key)
        if last_valid is None:
            st.warning(message + (f" 資料日期：{result['date']}。" if result.get("date") else ""))
            return
        st.warning(message + " 以下為本次工作階段最後有效資料，未更新。")
        result = last_valid
    else:
        st.warning("部分集保持股級距資料不完整，缺少的欄位以「資料不足」顯示。")

    labels = {"retail": "小額持股級距（≤5,000 股）", "big": "大額持股級距（≥400,001 股）"}
    for (name, label), column in zip(labels.items(), st.columns(2)):
        ratio = result[f"{name}_pct"]
        change = result[f"{name}_change_pp"]
        column.metric(
            label, f"{ratio:.2f}%" if ratio is not None else "資料不足",
            delta=f"{change:+.2f} 個百分點" if change is not None else None,
            delta_color="off",
        )
        shares = result[f"{name}_shares"]
        if shares is not None:
            column.caption(f"級距合計 {shares:,} 股／單位")
    if any(result[f"{name}_change_pp"] is not None for name in labels):
        st.caption(f"比較日期：{result['previous_date']} → {result['date']}。占比變化以百分點計。")
    else:
        st.caption(f"資料日期：{result['date']}。目前沒有可比較的前期級距資料。")

def fetch_mops_insider_holding(ticker):
    """Read latest MOPS monthly director/insider holding table for a single Taiwan company."""
    code = ticker.replace('.TWO', '').replace('.TW', '').strip()
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo('Asia/Taipei'))
    headers = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/x-www-form-urlencoded'}
    url = 'https://mopsov.twse.com.tw/mops/web/ajax_stapap1'
    for month_offset in range(0, 4):
        month_date = now.replace(day=1) - timedelta(days=month_offset * 28)
        roc_year = month_date.year - 1911
        payload = {'encodeURIComponent':'1','step':'1','firstin':'1','off':'1','keyword4':'','code1':'','TYPEK2':'',
                   'checkbtn':'1','queryName':'co_id','inpuType':'co_id','TYPEK':'all','isnew':'false',
                   'co_id':code,'year':str(roc_year),'month':f'{month_date.month:02d}'}
        try:
            response = requests.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            tables = pd.read_html(io.StringIO(response.text))
            for table in tables:
                columns = [' '.join(str(part) for part in col if str(part) != 'nan') if isinstance(col, tuple) else str(col) for col in table.columns]
                normalized = [c.replace(' ', '').replace('\n', '') for c in columns]
                table.columns = normalized
                insider_cols = [c for c in normalized if '內部人關係人目前持股合計' in c]
                direct_cols = [c for c in normalized if ('目前持股' in c or '持有股數' in c) and '設質' not in c]
                if insider_cols:
                    amount = _to_number(table[insider_cols[0]].iloc[-1])
                elif direct_cols:
                    amount = pd.to_numeric(table[direct_cols[0]].astype(str).str.replace(',', ''), errors='coerce').sum()
                else:
                    continue
                if amount and amount > 0:
                    return {'shares': float(amount), 'date': f'{roc_year}/{month_date.month:02d}', 'source':'MOPS 董監事持股餘額'}
        except Exception:
            continue
    return None

def get_finmind_api_token():
    """Read provider credentials from deployment secrets or environment, never from source text."""
    token = str(os.environ.get("FINMIND_TOKEN", "")).strip()
    if token:
        return token
    try:
        return str(st.secrets.get("FINMIND_TOKEN", "")).strip()
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finmind_broker_branch(stock_code, trade_date, api_token, account_scope=""):
    """Fetch one Taiwan stock's daily branch details from FinMind's authenticated endpoint."""
    # account_scope is included in the cache key so different app users do not share cached results.
    if not api_token:
        return pd.DataFrame(), "尚未設定 FinMind API Token。"
    try:
        response = requests.get(
            "https://api.finmindtrade.com/api/v4/taiwan_stock_trading_daily_report",
            params={"data_id": str(stock_code), "date": str(trade_date)},
            headers={"Authorization": f"Bearer {api_token}"},
            timeout=25,
        )
        if response.status_code in (401, 402, 403):
            return pd.DataFrame(), "FinMind 拒絕此請求；請確認 Token 有效且帳號方案包含台股分點資料。"
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data", []) if isinstance(payload, dict) else []
        if not records:
            message = str(payload.get("msg") or payload.get("detail") or "該日期尚無可用分點資料。")
            return pd.DataFrame(), message
        return pd.DataFrame(records), None
    except Exception:
        return pd.DataFrame(), "自動取得分點資料失敗；請稍後重試，或使用 CSV 匯入。"


def _broker_branch_csv_column_kind(column):
    """Identify required branch CSV columns across common Chinese and API headers."""
    name = str(column).replace("\ufeff", "").strip().strip("\"'")
    compact = re.sub(r"\s+", "", name).lower()
    if compact in {"date", "trade_date", "trading_date", "資料日期", "交易日期", "成交日期", "日期"}:
        return "date"
    if compact in {"stock_id", "data_id", "stock_code", "ticker", "symbol", "股票代碼", "證券代號"}:
        return "code"
    if "代號" in compact or "代碼" in compact:
        if "證券" in compact or "股票" in compact:
            return "code"
    if compact in {"securities_trader_id", "broker_id", "branch_id"} or (
        "券商" in compact and ("代號" in compact or "代碼" in compact)
    ):
        return "branch_id"
    if compact in {"securities_trader", "broker", "broker_branch", "branch", "branch_name"}:
        return "branch"
    if "券商" in compact or "分點" in compact:
        return "branch"
    if compact in {"buy", "buy_volume", "buy_shares"}:
        return "buy"
    if "買進" in compact and any(unit in compact for unit in ("股數", "數量", "股")):
        return "buy"
    if compact in {"sell", "sell_volume", "sell_shares"}:
        return "sell"
    if "賣出" in compact and any(unit in compact for unit in ("股數", "數量", "股")):
        return "sell"
    return None


def _normalize_broker_stock_code(value):
    """Normalize Taiwan stock codes such as 3293, 3293.0, and 3293.TWO."""
    text = str(value).strip().upper()
    match = re.search(r"(?<!\d)(\d{4,6})(?!\d)", text)
    return match.group(1) if match else text.replace(".TW", "").replace(".TWO", "")


def _broker_csv_metadata_stock_code(text, delimiter, header_line):
    """Read a stock code from metadata rows like `證券代號,3293` before the header."""
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            if reader.line_num - 1 >= header_line:
                break
            if row and _broker_branch_csv_column_kind(row[0]) == "code":
                for cell in row[1:]:
                    code = _normalize_broker_stock_code(cell)
                    if code:
                        return code
    except csv.Error:
        return None
    return None


def _normalize_broker_csv_metadata(text):
    """Normalize Excel stock-code cells and harmless spaces after quoted CSV fields."""
    lines = text.splitlines(keepends=True)
    metadata_labels = ("股票代號", "股票代碼", "證券代號", "證券代碼")
    for index, line in enumerate(lines[:100]):
        if any(label in line for label in metadata_labels):
            lines[index] = re.sub(r'(?<![^,\t;|])="([^"]*)"', r"\1", line)
    text = "".join(lines)
    # Some Taiwan broker CSV exports leave a space between the final quote and
    # the delimiter/newline. Python's strict CSV parser rejects that harmless padding.
    return re.sub(r'"[ \t]+(?=[,;\t|]|\r?\n|$)', '"', text)


def _broker_branch_csv_header_groups(header_fields):
    """Find one or more side-by-side broker tables separated by blank columns."""
    sections = []
    current = []
    for index, value in enumerate(header_fields):
        if not str(value).strip():
            if current:
                sections.append(current)
                current = []
        else:
            current.append(index)
    if current:
        sections.append(current)

    def is_broker_table(indices):
        kinds = {_broker_branch_csv_column_kind(header_fields[index]) for index in indices}
        return {"buy", "sell"}.issubset(kinds) and bool({"code", "branch", "branch_id"}.intersection(kinds))

    matching_sections = [indices for indices in sections if is_broker_table(indices)]
    if matching_sections:
        return matching_sections
    all_columns = list(range(len(header_fields)))
    return [all_columns] if is_broker_table(all_columns) else []


def _combine_broker_branch_csv_tables(frame, header_fields):
    """Normalize repeated side-by-side tables into one row-oriented broker table."""
    groups = _broker_branch_csv_header_groups(header_fields)
    if not groups:
        return frame

    canonical_names = {
        "code": "stock_id",
        "branch_id": "securities_trader_id",
        "branch": "securities_trader",
        "buy": "buy_shares",
        "sell": "sell_shares",
        "date": "trade_date",
    }
    normalized_frames = []
    for indices in groups:
        role_indices = {}
        for index in indices:
            kind = _broker_branch_csv_column_kind(header_fields[index])
            if kind in canonical_names and kind not in role_indices and index < frame.shape[1]:
                role_indices[kind] = index
        if not {"buy", "sell"}.issubset(role_indices):
            continue
        group_frame = pd.DataFrame(index=frame.index)
        for kind, index in role_indices.items():
            group_frame[canonical_names[kind]] = frame.iloc[:, index]
        normalized_frames.append(group_frame)

    if not normalized_frames:
        return frame
    return pd.concat(normalized_frames, ignore_index=True, sort=False)


def summarize_broker_branch_frame(frame, stock_code, *, strict_shares=False):
    """Aggregate price-level branch rows into one buy/sell total per broker branch."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {"error": "沒有可分析的分點資料。"}
    code_col = next((c for c in frame.columns if _broker_branch_csv_column_kind(c) == "code"), None)
    buy_col = next((c for c in frame.columns if _broker_branch_csv_column_kind(c) == "buy"), None)
    sell_col = next((c for c in frame.columns if _broker_branch_csv_column_kind(c) == "sell"), None)
    if not all((code_col, buy_col, sell_col)):
        return {"error": "資料需包含股票代號、買進股數、賣出股數欄位。"}
    target_code = _normalize_broker_stock_code(stock_code)
    frame_codes = frame[code_col].map(_normalize_broker_stock_code)
    selected = frame[frame_codes == target_code].copy()
    if selected.empty:
        return {"error": f"目前分析標的是 {target_code}，但這份 CSV 找不到該股票的資料。請切換至 CSV 所屬股票，或匯入 {target_code} 的分點 CSV。"}

    branch_id_col = next((c for c in selected.columns if str(c).lower() == "securities_trader_id"
                          or ('券商' in str(c) and '代號' in str(c))), None)
    branch_name_col = next((c for c in selected.columns if _broker_branch_csv_column_kind(c) == "branch"
                            or str(c).lower() == "securities_trader"), None)
    group_cols = [column for column in (branch_id_col, branch_name_col) if column]
    selected["_buy"] = pd.to_numeric(selected[buy_col].astype(str).str.replace(',', ''), errors='coerce')
    selected["_sell"] = pd.to_numeric(selected[sell_col].astype(str).str.replace(',', ''), errors='coerce')
    if strict_shares:
        amounts = selected[["_buy", "_sell"]]
        valid = np.isfinite(amounts) & (amounts >= 0) & (amounts % 1 == 0)
        formatted = r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.0+)?"
        valid["_buy"] &= selected[buy_col].astype(str).str.strip().str.fullmatch(formatted)
        valid["_sell"] &= selected[sell_col].astype(str).str.strip().str.fullmatch(formatted)
        bad = int((~valid.all(axis=1)).sum())
        if bad:
            return {"error": f"有 {bad} 列的買進或賣出股數缺漏或無效。請確認兩欄都是大於或等於零的整數。"}
    selected = selected.dropna(subset=["_buy", "_sell"])
    if selected.empty:
        return {"error": "買進或賣出欄位沒有可讀取的數值。"}
    if group_cols:
        grouped = selected.groupby(group_cols, dropna=False, as_index=False)[["_buy", "_sell"]].sum()
        grouped["分點"] = grouped[branch_name_col].fillna("").astype(str) if branch_name_col else ""
        if branch_id_col:
            ids = grouped[branch_id_col].fillna("").astype(str)
            grouped["分點"] = ids + grouped["分點"].map(lambda value: f"｜{value}" if value else "")
        count_unit = "分點"
    else:
        grouped = selected[["_buy", "_sell"]].copy().reset_index(drop=True)
        grouped["分點"] = [f"資料列 {i + 1}" for i in range(len(grouped))]
        count_unit = "資料列"
    grouped["買賣超"] = grouped["_buy"] - grouped["_sell"]
    buyers = int((grouped["買賣超"] > 0).sum())
    sellers = int((grouped["買賣超"] < 0).sum())
    breakdown = grouped[["分點", "_buy", "_sell", "買賣超"]].rename(
        columns={"_buy": "買進股數", "_sell": "賣出股數"}
    ).sort_values("買賣超", ascending=False).reset_index(drop=True)
    positive_net_total = float(grouped.loc[grouped["買賣超"] > 0, "買賣超"].sum())
    negative_net_total = float(-grouped.loc[grouped["買賣超"] < 0, "買賣超"].sum())
    top_net_buy = float(grouped.nlargest(15, "買賣超")["買賣超"].clip(lower=0).sum())
    top_net_sell = float(grouped.nsmallest(15, "買賣超")["買賣超"].clip(upper=0).sum())
    return {
        "net_shares": float(grouped["買賣超"].sum()),
        "buyers": buyers,
        "sellers": sellers,
        "count_unit": count_unit,
        "top_net_buy": top_net_buy,
        "top_net_sell": top_net_sell,
        "top_net_buy_share_pct": top_net_buy / positive_net_total * 100.0 if positive_net_total > 0 else None,
        "top_net_sell_share_pct": abs(top_net_sell) / negative_net_total * 100.0 if negative_net_total > 0 else None,
        "breakdown": breakdown,
    }


def _broker_branch_ai_state_key(user_id, stock_code):
    normalized_code = _normalize_broker_stock_code(stock_code)
    return f"broker_branch_ai_{str(user_id)}_{normalized_code}"


def save_broker_branch_ai_summary(session_state, user_id, stock_code, branch_summary, source_label):
    """Keep only a compact, account-scoped branch summary for this session's AI report."""
    key = _broker_branch_ai_state_key(user_id, stock_code)
    if not isinstance(branch_summary, Mapping) or branch_summary.get("error"):
        session_state.pop(key, None)
        return None
    snapshot = {
        "stock_code": _normalize_broker_stock_code(stock_code),
        "source_label": str(source_label or "使用者提供的分點資料"),
        "trade_period": branch_summary.get("trade_period"),
        "net_shares": _finite_number(branch_summary.get("net_shares")),
        "buyers": int(branch_summary.get("buyers", 0) or 0),
        "sellers": int(branch_summary.get("sellers", 0) or 0),
        "count_unit": str(branch_summary.get("count_unit", "分點")),
        "top_net_buy": _finite_number(branch_summary.get("top_net_buy")),
        "top_net_sell": _finite_number(branch_summary.get("top_net_sell")),
        "top_net_buy_share_pct": _finite_number(branch_summary.get("top_net_buy_share_pct")),
        "top_net_sell_share_pct": _finite_number(branch_summary.get("top_net_sell_share_pct")),
        "saved_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    session_state[key] = snapshot
    return snapshot


def format_broker_branch_ai_context(snapshot, stock_code):
    """Prepare a short, cautious branch-flow context for an AI report/email."""
    if not isinstance(snapshot, Mapping):
        return ""
    target_code = _normalize_broker_stock_code(stock_code)
    if snapshot.get("stock_code") != target_code:
        return ""
    unit = "分點" if snapshot.get("count_unit") == "分點" else "資料列"
    net_shares = snapshot.get("net_shares")
    net_text = f"{net_shares:+,.0f} 股" if net_shares is not None else "資料不足"
    buy_share = snapshot.get("top_net_buy_share_pct")
    sell_share = snapshot.get("top_net_sell_share_pct")
    buy_share_text = f"前15個淨買超{unit}占全部淨買超約 {buy_share:.1f}%" if buy_share is not None else "淨買超集中度資料不足"
    sell_share_text = f"前15個淨賣超{unit}占全部淨賣超約 {sell_share:.1f}%" if sell_share is not None else "淨賣超集中度資料不足"
    saved_at = str(snapshot.get("saved_at") or "")
    period = snapshot.get("trade_period")
    return "\n".join([
        f"【分點資料摘要｜{target_code}】",
        f"資料來源：{snapshot.get('source_label', '使用者提供的分點資料')}；載入時間：{saved_at or '未提供'}。"
        + (f" 交易日期：{period}。" if period else ""),
        f"分點淨額合計：{net_text}；淨買超{unit}：{int(snapshot.get('buyers', 0) or 0)}；淨賣超{unit}：{int(snapshot.get('sellers', 0) or 0)}。",
        f"淨流向集中度：{buy_share_text}；{sell_share_text}。",
        "資料限制：CSV 若未標示交易日期，不能假設分點統計與目前股價是同日或同期間；淨額合計接近零通常是買賣相抵，不是方向訊號。",
        "解讀原則：分點家數表示交易渠道的參與廣度，不代表投資人數或散戶／主力身分；請將家數、淨額、集中度和日期對齊後的價格變化分開觀察，不據此宣稱吸籌、出貨或預測漲跌。",
    ])


def build_broker_branch_ai_interpretation(snapshot, price_history, timeframe_name):
    """Build a deterministic, user-facing branch-flow and recent-price interpretation."""
    if not isinstance(snapshot, Mapping):
        return ""
    close = pd.to_numeric(pd.Series(price_history), errors="coerce").dropna().tail(20)
    buyers = int(snapshot.get("buyers", 0) or 0)
    sellers = int(snapshot.get("sellers", 0) or 0)
    unit = "分點" if snapshot.get("count_unit") == "分點" else "資料列"
    net_shares = _finite_number(snapshot.get("net_shares"))
    buy_share = _finite_number(snapshot.get("top_net_buy_share_pct"))
    sell_share = _finite_number(snapshot.get("top_net_sell_share_pct"))

    lines = [
        f"觀察到的分點資料：淨買超{unit} {buyers:,} 個、淨賣超{unit} {sellers:,} 個；"
        f"這份資料的分點淨額合計 {net_shares:+,.0f} 股。" if net_shares is not None
        else f"觀察到的分點資料：淨買超{unit} {buyers:,} 個、淨賣超{unit} {sellers:,} 個；淨額資料不足。"
    ]

    concentration_parts = []
    if buy_share is not None:
        top_buy = _finite_number(snapshot.get("top_net_buy"))
        amount_text = f"（合計 {top_buy:+,.0f} 股）" if top_buy is not None else ""
        concentration_parts.append(f"前 15 個淨買超{unit}{amount_text}約占全部淨買超 {buy_share:.1f}%")
    if sell_share is not None:
        top_sell = _finite_number(snapshot.get("top_net_sell"))
        amount_text = f"（合計 {top_sell:+,.0f} 股）" if top_sell is not None else ""
        concentration_parts.append(f"前 15 個淨賣超{unit}{amount_text}約占全部淨賣超 {sell_share:.1f}%")
    if concentration_parts:
        lines.append("流向集中度：" + "；".join(concentration_parts) + "。比例越高，代表該方向的淨額越集中在排名前面的分點；不代表持股集中度。")
    else:
        lines.append("流向集中度：目前無法計算前 15 個淨買超／淨賣超分點占比。")

    if len(close) >= 2 and float(close.iloc[0]) > 0:
        price_change_pct = float(close.iloc[-1] / close.iloc[0] - 1.0)
        price_direction = "上漲" if price_change_pct > 0 else "下跌" if price_change_pct < 0 else "持平"
        date_range = ""
        try:
            start_date = pd.Timestamp(close.index[0]).strftime("%Y-%m-%d")
            end_date = pd.Timestamp(close.index[-1]).strftime("%Y-%m-%d")
            date_range = f"（{start_date} 至 {end_date}）"
        except Exception:
            pass
        lines.append(
            f"價格觀察：近 {len(close)} 筆「{timeframe_name or '目前圖表'}」收盤由 {float(close.iloc[0]):,.2f} 元"
            f"變為 {float(close.iloc[-1]):,.2f} 元，期間{price_direction} {price_change_pct:+.2%}{date_range}。"
        )
        lines.append("家數與價格搭配：" + build_broker_branch_price_breadth_reading(snapshot, price_change_pct))
    else:
        lines.append("價格觀察：可用收盤資料少於 2 筆，無法計算近 20 筆價格方向。")

    if buyers > sellers:
        breadth_note = f"淨買超{unit}比淨賣超多 {buyers - sellers:,} 個，表示淨買方參與面較廣。"
    elif sellers > buyers:
        breadth_note = f"淨賣超{unit}比淨買超多 {sellers - buyers:,} 個，表示淨賣方參與面較廣。"
    else:
        breadth_note = f"淨買超與淨賣超{unit}數相同，家數沒有顯示哪一方參與面較廣。"
    lines.append("可能的市場心理解讀：" + breadth_note + "若價格方向與買賣參與廣度一致，可視為當下行為線索；若不一致，代表訊號分歧，宜觀察後續資料，不宜直接貼上吸籌或出貨標籤。")

    source_label = str(snapshot.get("source_label") or "使用者提供的分點資料")
    source_date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", source_label)
    if snapshot.get("trade_period"):
        date_note = f"分點交易日期為 {snapshot['trade_period']}；近 20 筆價格為另一觀察期間，請核對兩者是否對齊。"
    elif source_date_match:
        date_note = f"分點資料標示日期為 {source_date_match.group(1)}；近 20 筆價格涵蓋的期間較長，不能把兩者當成完全相同期間。"
    else:
        date_note = "這份分點資料沒有可確認的交易日期；無法確認它與近 20 筆價格期間對齊，因此只能作方向參考。"
    lines.append("資料限制：" + date_note + "分點家數不是投資人數，資料也無法辨認主力或散戶身分；淨額接近零可能只是買賣相抵，不代表沒有資金流動。")
    return "\n".join(f"- {line}" for line in lines)


def _broker_csv_trade_period(frame, target_code, metadata_rows):
    date_column = next((c for c in frame.columns if _broker_branch_csv_column_kind(c) == "date"), None)
    if date_column is not None:
        code_column = next(c for c in frame.columns if _broker_branch_csv_column_kind(c) == "code")
        values = frame.loc[frame[code_column].map(_normalize_broker_stock_code) == target_code, date_column]
    else:
        values = []
        for row in metadata_rows:
            if not row:
                continue
            cells = re.split("[：:]", row[0], maxsplit=1)
            if _broker_branch_csv_column_kind(cells[0]) == "date":
                values.extend(value for value in cells[1:] + row[1:] if value.strip())

    dates = []
    for value in values:
        text = str(value).strip().strip('"')
        parts = re.fullmatch(r"([0-9]{3,4})[-/]([0-9]{1,2})[-/]([0-9]{1,2})"
                             r"|([0-9]{3,4})([0-9]{2})([0-9]{2})", text)
        if parts is None:
            return None
        year, month, day = parts.groups()[:3] if parts[1] else parts.groups()[3:]
        date_text = f"{int(year) + (1911 if len(year) == 3 else 0):04d}-{int(month):02d}-{int(day):02d}"
        parsed = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
        if pd.isna(parsed):
            return None
        dates.append(parsed.date().isoformat())
    if not dates or (date_column is None and len(set(dates)) != 1):
        return None
    first, last = min(dates), max(dates)
    return first if first == last else f"{first} 至 {last}"


def summarize_broker_branch_csv(uploaded_file, stock_code):
    """Read common broker CSV encodings/delimiters and locate the actual header row."""
    if uploaded_file is None:
        return None
    try:
        raw_content = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        if isinstance(raw_content, str):
            raw_content = raw_content.encode("utf-8")
    except Exception as exc:
        return {"error": f"無法讀取上傳檔案：{exc}"}
    if not raw_content:
        return {"error": "上傳的 CSV 是空檔案。"}

    delimiters = (",", "\t", ";", "|")
    encodings = ("utf-8-sig", "utf-8", "cp950", "big5", "utf-16")
    last_parse_error = None
    for encoding in encodings:
        try:
            text = _normalize_broker_csv_metadata(raw_content.decode(encoding))
        except (UnicodeDecodeError, LookupError):
            continue

        for delimiter in delimiters:
            header_line = None
            header_fields = None
            try:
                reader = csv.reader(io.StringIO(text), delimiter=delimiter)
                for row in reader:
                    if reader.line_num > 100:
                        break
                    kinds = {_broker_branch_csv_column_kind(cell) for cell in row}
                    has_trade_columns = {"buy", "sell"}.issubset(kinds)
                    has_stock_or_branch = bool({"code", "branch", "branch_id"}.intersection(kinds))
                    if has_trade_columns and has_stock_or_branch:
                        # csv.reader.line_num is one-based; pandas skiprows is zero-based.
                        header_line = reader.line_num - 1
                        header_fields = row
                        break
            except csv.Error as exc:
                last_parse_error = str(exc)
                continue
            if header_line is None:
                continue

            try:
                frame = pd.read_csv(
                    io.StringIO(text),
                    sep=delimiter,
                    skiprows=header_line,
                    header=0,
                    engine="python",
                    dtype=str,
                    skip_blank_lines=True,
                    on_bad_lines="error",
                )
            except Exception as exc:
                last_parse_error = str(exc)
                continue
            frame = _combine_broker_branch_csv_tables(frame, header_fields or list(frame.columns)).dropna(how="all")
            code_col = next((column for column in frame.columns if _broker_branch_csv_column_kind(column) == "code"), None)
            if code_col is None:
                metadata_code = _broker_csv_metadata_stock_code(text, delimiter, header_line)
                if metadata_code is None:
                    return {"error": "CSV 標題列沒有股票代號，且前方資訊列也找不到「證券代號」；無法確認資料屬於哪一檔股票。"}
                if metadata_code != _normalize_broker_stock_code(stock_code):
                    return {"error": f"CSV 內的股票代號是 {metadata_code}，目前分析標的是 {_normalize_broker_stock_code(stock_code)}。請切換至 {metadata_code}，或匯入目前標的的分點 CSV。"}
                frame["stock_id"] = metadata_code
            else:
                csv_codes = frame[code_col].map(_normalize_broker_stock_code).dropna().unique().tolist()
                target_code = _normalize_broker_stock_code(stock_code)
                if csv_codes and target_code not in csv_codes:
                    shown_codes = ", ".join(csv_codes[:5])
                    more_codes = " 等" if len(csv_codes) > 5 else ""
                    return {"error": f"這份 CSV 包含 {shown_codes}{more_codes}；目前分析標的是 {target_code}。請切換至 CSV 所屬股票，或匯入 {target_code} 的分點 CSV。"}
            result = summarize_broker_branch_frame(frame, stock_code, strict_shares=True)
            if "error" not in result:
                metadata_rows = csv.reader(io.StringIO("\n".join(text.splitlines()[:header_line])),
                                           delimiter=delimiter)
                result["trade_period"] = _broker_csv_trade_period(
                    frame, _normalize_broker_stock_code(stock_code), metadata_rows
                )
            return result

    if last_parse_error:
        return {
            "error": "CSV 引號或欄位分隔格式不一致，無法安全匯入。請確認每列引號成對、欄位分隔一致。"
                     f"（{last_parse_error}）"
        }
    return {
        "error": "找不到可辨識的 CSV 標題列。請確認檔案包含股票代號、買進股數與賣出股數欄位；"
                 "系統會自動搜尋前 100 列的標題，也支援逗號、Tab、分號或直線分隔。"
    }

def build_institutional_rankings(trades, shares_by_code, limit=10):
    rows = []
    for code, trade in trades.items():
        shares = shares_by_code.get(code)
        if not shares:
            continue
        row = {'股票代碼':code, '股票名稱':trade.get('股票名稱') or code, '市場':trade.get('市場'), '資料日期':trade.get('日期')}
        for investor in ('外資','投信','自營商'):
            for measure, label in (('買進股數','買進'),('賣出股數','賣出'),('買賣超股數','買賣超')):
                value = trade.get(f'{investor}{measure}')
                row[f'{investor}{label}佔股本比(%)'] = value / shares * 100 if value is not None else None
        rows.append(row)
    return pd.DataFrame(rows)


def institutional_top10_sort_specs(investor):
    """Return the four stable Top 10 category definitions used for display and tracking."""
    return [
        (f'{investor}買賣超佔股本比(%)', False, '買賣超佔股本比最高'),
        (f'{investor}買賣超佔股本比(%)', True, '買賣超佔股本比最低'),
        (f'{investor}買進佔股本比(%)', False, '買進佔股本比最高'),
        (f'{investor}賣出佔股本比(%)', False, '賣出佔股本比最高'),
    ]


def save_institutional_top10_snapshot(ranking_data):
    """Persist current Top 10 membership once per latest available report date."""
    if not isinstance(ranking_data, pd.DataFrame) or ranking_data.empty or '資料日期' not in ranking_data:
        return None, 0
    if not {'上市', '上櫃'}.issubset(set(ranking_data.get('市場', pd.Series(dtype=str)).dropna().astype(str))):
        return None, 0
    normalized_dates = ranking_data['資料日期'].map(_normalize_chip_trade_date).dropna()
    if normalized_dates.empty:
        return None, 0
    snapshot_date = str(normalized_dates.max())
    captured_at = pd.Timestamp.now(tz='Asia/Taipei').isoformat(timespec='seconds')
    rows = []
    snapshot_days = []
    for investor in ('外資', '投信', '自營商'):
        for value_column, ascending, ranking_type in institutional_top10_sort_specs(investor):
            snapshot_days.append((snapshot_date, investor, ranking_type, captured_at))
            if value_column not in ranking_data:
                continue
            ranked = ranking_data.dropna(subset=[value_column]).sort_values(value_column, ascending=ascending).head(10)
            for _, item in ranked.iterrows():
                rows.append((
                    snapshot_date,
                    investor,
                    ranking_type,
                    str(item.get('股票代碼') or ''),
                    str(item.get('股票名稱') or ''),
                    str(item.get('市場') or ''),
                    _normalize_chip_trade_date(item.get('資料日期')) or '',
                    _finite_number(item.get(value_column)),
                    captured_at,
                ))
    try:
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            conn.execute('DELETE FROM institutional_top10_appearances WHERE snapshot_date=?', (snapshot_date,))
            conn.execute('DELETE FROM institutional_top10_snapshot_days WHERE snapshot_date=?', (snapshot_date,))
            conn.executemany('''INSERT INTO institutional_top10_snapshot_days
                (snapshot_date, investor, ranking_type, captured_at) VALUES (?, ?, ?, ?)''', snapshot_days)
            conn.executemany('''INSERT INTO institutional_top10_appearances
                (snapshot_date, investor, ranking_type, ticker, stock_name, market, data_date, rank_value, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', rows)
        return snapshot_date, len(rows)
    except Exception:
        return snapshot_date, 0


def load_institutional_top10_consecutive_appearances(investor, snapshot_date, lookback_snapshots=120):
    """List current Top 10 entries with consecutive captured snapshots in the same category."""
    if not snapshot_date:
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_FILE, timeout=30) as conn:
            history = pd.read_sql_query('''
                SELECT snapshot_date, investor, ranking_type, ticker, stock_name, market, data_date, rank_value
                FROM institutional_top10_appearances
                WHERE snapshot_date IN (
                    SELECT DISTINCT snapshot_date
                    FROM institutional_top10_snapshot_days
                    WHERE investor=? AND snapshot_date<=?
                    ORDER BY snapshot_date DESC LIMIT ?
                )
                AND investor=?
                ORDER BY snapshot_date DESC
            ''', conn, params=(str(investor), str(snapshot_date), int(lookback_snapshots), str(investor)))
            observed_days = pd.read_sql_query('''
                SELECT snapshot_date, ranking_type
                FROM institutional_top10_snapshot_days
                WHERE investor=? AND snapshot_date IN (
                    SELECT DISTINCT snapshot_date
                    FROM institutional_top10_snapshot_days
                    WHERE investor=? AND snapshot_date<=?
                    ORDER BY snapshot_date DESC LIMIT ?
                )
                ORDER BY snapshot_date DESC
            ''', conn, params=(str(investor), str(investor), str(snapshot_date), int(lookback_snapshots)))
    except Exception:
        return pd.DataFrame()
    if history.empty or observed_days.empty:
        return pd.DataFrame()

    output = []
    current_rows = history[history['snapshot_date'].astype(str) == str(snapshot_date)]
    for ranking_type in current_rows['ranking_type'].dropna().unique():
        category_history = history[history['ranking_type'] == ranking_type]
        dates = observed_days.loc[
            observed_days['ranking_type'] == ranking_type, 'snapshot_date'
        ].dropna().astype(str).unique().tolist()
        dates = sorted(dates, reverse=True)
        membership = {
            day: set(category_history.loc[category_history['snapshot_date'].astype(str) == day, 'ticker'].astype(str))
            for day in dates
        }
        latest_members = current_rows[current_rows['ranking_type'] == ranking_type]
        for _, item in latest_members.iterrows():
            ticker = str(item.get('ticker') or '')
            streak_dates = []
            for day in dates:
                if day not in membership or ticker not in membership[day]:
                    break
                streak_dates.append(day)
            if len(streak_dates) < 2:
                continue
            output.append({
                '榜單類型': str(ranking_type),
                '股票代碼': ticker,
                '股票名稱': str(item.get('stock_name') or ticker),
                '市場': str(item.get('market') or ''),
                '連續已記錄快照數': len(streak_dates),
                '起始快照日': streak_dates[-1],
                '最近快照日': streak_dates[0],
                '本次資料日期': str(item.get('data_date') or ''),
                '佔股本比(%)': _finite_number(item.get('rank_value')),
            })
    result = pd.DataFrame(output)
    if not result.empty:
        result = result.sort_values(
            ['連續已記錄快照數', '榜單類型', '佔股本比(%)'],
            ascending=[False, True, False],
        ).reset_index(drop=True)
    return result

def signature_save_to_db(t):
    try:
        stock = yf.Ticker(t)
        info = stock.info if hasattr(stock, 'info') and stock.info else {}

        p_raw = info.get('currentPrice') or info.get('previousClose')
        if not p_raw:
            hist = stock.history(period="5d")
            if not hist.empty:
                p_raw = float(hist['Close'].iloc[-1])
        price = float(p_raw) if (p_raw and float(p_raw) > 0) else None

        book_value = info.get('bookValue')
        bv_source = "yfinance 最新財報" if (book_value is not None and not np.isnan(book_value) and book_value > 0) else None

        raw_roe = info.get('returnOnEquity')
        roe_val = parse_roe(raw_roe)
        roe_source = "yfinance 最新財報" if roe_val is not None else None

        # 台股回退機制
        if (".TW" in t or ".TWO" in t) and (book_value is None or roe_val is None):
            official_pb, official_pe = fetch_tw_official_pb_pe(t)
            if (book_value is None or book_value <= 0) and price and official_pb and official_pb > 0:
                book_value = price / official_pb
                bv_source = "TWSE/TPEx PB估算值"
            if roe_val is None and official_pe and official_pe > 0 and official_pb and official_pb > 0:
                roe_val = parse_roe(official_pb / official_pe)
                roe_source = "TWSE/TPEx PB/PE代理值"

        # 美股/其他市場彈性回退機制 (1Q -> 半年 -> 1年)
        if (".TW" not in t and ".TWO" not in t) and (book_value is None or roe_val is None):
            try:
                bs_q = stock.quarterly_balance_sheet
                bs_a = stock.balance_sheet
                inc_q = stock.quarterly_financials
                inc_a = stock.financials

                tot_equity = None
                bv_source_temp = ""
                # 依序嘗試：季報 -> 年報
                for bs, src_name in [(bs_q, "yfinance 最新季報遞補"), (bs_a, "yfinance 最新年報遞補")]:
                    if not bs.empty:
                        for key in ['Total Stockholder Equity', 'Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest']:
                            if key in bs.index:
                                found_eq = bs.loc[key].iloc[0]
                                if pd.notnull(found_eq):
                                    tot_equity = float(found_eq)
                                    bv_source_temp = src_name
                                    break
                    if tot_equity is not None: break

                if book_value is None and tot_equity is not None:
                    shares_out = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
                    if shares_out and shares_out > 0:
                        calculated_bv = tot_equity / float(shares_out)
                        if calculated_bv > 0:
                            book_value = calculated_bv
                            bv_source = bv_source_temp
                        else:
                            bv_source = "美股負股東權益(庫藏股/虧損)"

                if roe_val is None and tot_equity is not None and tot_equity > 0:
                    roe_source_temp = ""
                    net_inc = None
                    if not inc_q.empty:
                        for net_key in ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests']:
                            if net_key in inc_q.index:
                                inc_series = inc_q.loc[net_key].dropna()
                                if len(inc_series) >= 4:
                                    net_inc = float(inc_series.iloc[:4].sum())
                                    roe_source_temp = "yfinance 近四季(TTM)推算"
                                elif len(inc_series) >= 2:
                                    net_inc = float(inc_series.iloc[:2].sum()) * 2
                                    roe_source_temp = "yfinance 近半年年化推算"
                                elif len(inc_series) > 0:
                                    net_inc = float(inc_series.iloc[0]) * 4
                                    roe_source_temp = "yfinance 單季年化推算"
                                break
                    if net_inc is None and not inc_a.empty:
                        for net_key in ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests']:
                            if net_key in inc_a.index:
                                inc_series = inc_a.loc[net_key].dropna()
                                if len(inc_series) > 0:
                                    net_inc = float(inc_series.iloc[0])
                                    roe_source_temp = "yfinance 最新年報遞補"
                                break

                    if net_inc is not None:
                        roe_val = parse_roe(net_inc / tot_equity)
                        roe_source = roe_source_temp
            except Exception as us_err:
                st.sidebar.caption(f"美股財報備援解析提醒 [{t}]: {us_err}")

        bv_val = float(book_value) if (book_value is not None and not np.isnan(book_value) and book_value > 0) else None
        roe_val = parse_roe(roe_val)

        gvi = calculate_gvi(price, bv_val, roe_val) if (price and price > 0 and bv_val and bv_val > 0 and roe_val is not None) else None

        mcap = info.get('marketCap')
        mcap_val = float(mcap) if (mcap and float(mcap) > 0) else None
        shares_val = (mcap_val / price) if (mcap_val and price and price > 0) else None
        updated_at = int(time.time())

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO market_data
                     (ticker, date, open, high, low, close, volume, gvi, roe, mcap, shares, book_value, roe_source, bv_source, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (t, "LATEST", price, price, price, price, 10000,
                   gvi, roe_val, mcap_val, shares_val, bv_val, roe_source, bv_source, updated_at))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"❌ 數據寫入資料庫失敗 [{t}]: {e}")


def _fetch_market_index_history(symbol, period):
    try:
        history = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    except Exception:
        return None
    if not isinstance(history, pd.DataFrame):
        return None
    if isinstance(history.columns, pd.MultiIndex):
        history = history.droplevel(list(range(1, history.columns.nlevels)), axis="columns")
    if "Close" not in history or not isinstance(history.index, pd.DatetimeIndex) or not history.columns.is_unique:
        return None
    closes = pd.to_numeric(history["Close"], errors="coerce")
    valid = history.index.notna() & np.isfinite(closes) & (closes > 0)
    return history.loc[valid].sort_index(kind="stable") if valid.any() else None


def _market_index_history(symbol, period="5d"):
    key = (symbol, period)
    cache = st.session_state.setdefault("public_market_index_cache", {})
    previous = cache.get(key)
    history = _fetch_market_index_history(symbol, period)
    stale = history is None or (previous is not None and history.index[-1].value < previous.index[-1].value)
    if stale:
        history = previous
    else:
        cache[key] = history
    if history is None:
        return None
    displayed = history.copy()
    displayed.attrs["quote_stale"] = stale
    return displayed


def _market_index_snapshot(symbol, *, include_status=False):
    history = _market_index_history(symbol)
    if history is None:
        return None
    closes = pd.to_numeric(history["Close"], errors="coerce")
    closes = closes[closes.index.notna() & np.isfinite(closes) & (closes > 0)]
    closes = closes.sort_index(kind="stable")
    closes = closes.groupby(closes.index.normalize()).last()
    if closes.empty:
        return None
    value = float(closes.iloc[-1])
    change = percentage = None
    if len(closes) > 1:
        previous = float(closes.iloc[-2])
        points = value - previous
        percent = (points / previous) * 100
        if isfinite(percent):
            change, percentage = points, percent
    snapshot = value, change, percentage, closes.index[-1].date()
    return (*snapshot, history.attrs.get("quote_stale", False)) if include_status else snapshot


WORKBENCH_INDEXES = (
    ("台灣加權", "^TWII", "追蹤台灣證券交易所上市股票的整體表現。"),
    ("標普 500", "^GSPC", "追蹤美國大型上市公司的整體股價表現。"),
    ("道瓊工業", "^DJI", "由 30 家美國大型企業組成的價格加權指數。"),
    ("那斯達克綜合", "^IXIC", "追蹤在那斯達克掛牌的股票。"),
)


def _market_overview_snapshots(*, include_status=False):
    snapshots = {}
    for label, symbol, _ in WORKBENCH_INDEXES:
        snapshot = _market_index_snapshot(symbol, include_status=include_status)
        if snapshot is not None:
            snapshots[label] = snapshot
    return snapshots


def _render_market_overview(compact=False):
    if not compact:
        st.subheader("台美市場概況")
        st.caption("來源 Yahoo Finance 日線。依各市場交易日顯示，可能延遲；變動相較前一筆可用資料。")
    index_data = _market_overview_snapshots(include_status=True)
    snapshots = {}
    for column, (label, symbol, description) in zip(st.columns(4), WORKBENCH_INDEXES):
        with column:
            snapshot = index_data.get(label)
            if snapshot is None:
                st.metric(label if compact else f"{label} ({symbol})", "資料暫缺", help=description)
                continue
            value, change, percentage, trading_day, stale = snapshot
            snapshots[label] = value, change, percentage, trading_day
            delta = f"{change:+.2f} ({percentage:+.2f}%)" if change is not None else None
            st.metric(label if compact else f"{label} ({symbol})", f"{value:,.2f}", delta, help=description)
            st.caption(f"資料日 {trading_day:%Y-%m-%d}" + (" · 未更新，沿用前值" if stale else "")
                       + (" · 缺前期資料" if change is None else ""))
    return snapshots


AUTO_TW_UNIVERSE = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2382.TW',
    '3008.TW', '2303.TW', '2881.TW', '2882.TW', '2891.TW',
    '1301.TW', '1303.TW', '2002.TW', '2207.TW', '2327.TW',
    '2357.TW', '2379.TW', '2395.TW', '2408.TW', '2603.TW',
    '3293.TWO', '6121.TWO', '8069.TWO', '5483.TWO', '8299.TWO',
    '3105.TWO', '3529.TWO', '6488.TWO', '4966.TWO', '3264.TWO'
]

AUTO_US_UNIVERSE = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN',
    'META', 'TSLA', 'TSM', 'AMD', 'AVGO',
    'QCOM', 'INTC', 'NFLX', 'SMCI', 'ASML',
    'COST', 'WMT', 'JPM', 'V', 'LLY'
]

STOCK_NAME_MAP = {
    '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '2308.TW': '台達電',
    '2382.TW': '廣達', '3008.TW': '大立光', '2303.TW': '聯電', '2881.TW': '富邦金',
    '2882.TW': '國泰金', '2891.TW': '中信金', '1301.TW': '台塑', '1303.TW': '南亞',
    '2002.TW': '中鋼', '2207.TW': '和泰車', '2327.TW': '國巨', '2357.TW': '華碩',
    '2379.TW': '瑞昱', '2395.TW': '研華', '2408.TW': '南亞科', '2603.TW': '長榮',
    '3293.TWO': '鈊象', '6121.TWO': '新普', '8069.TWO': '元太', '5483.TWO': '中美晶',
    '8299.TWO': '群聯', '3105.TWO': '穩懋', '3529.TWO': '力旺', '6488.TWO': '環球晶',
    '4966.TWO': '譜瑞-KY', '3264.TWO': '欣銓',
    'AAPL': '蘋果公司', 'NVDA': '輝達', 'MSFT': '微軟', 'GOOGL': '谷歌',
    'AMZN': '亞馬遜', 'META': '臉書META', 'TSLA': '特斯拉', 'TSM': '台積電ADR',
    'AMD': '超微半導體', 'AVGO': '博通', 'QCOM': '高通', 'INTC': '英特爾',
    'NFLX': '網飛', 'SMCI': '美超微', 'ASML': '艾司摩爾', 'COST': '好市多',
    'WMT': '沃爾瑪', 'JPM': '摩根大通', 'V': 'VISA卡', 'LLY': '禮來藥廠'
}


def calculate_chip_and_backtest(ticker_name, df, timeframe_name):
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df = df.dropna(subset=['Close', 'Low', 'Volume']).copy()

    # 避開不必要的 Resample，防止時間軸被擠壓成單一 K 棒
    if timeframe_name == "半年": df = df.resample('6ME').last().dropna()
    elif timeframe_name == "1年": df = df.resample('12ME').last().dropna()

    if df.empty or len(df) < 5: return df, "盤後快取中", 0.0

    closes = df['Close'].astype(float).to_numpy().flatten()
    vols = df['Volume'].astype(float).to_numpy().flatten()
    vol_ma20 = pd.Series(vols).rolling(20, min_periods=1).mean().to_numpy()

    current_trend = 0.0
    sim_net = np.zeros(len(vols))

    for i in range(len(vols)):
        pct_change = (closes[i] - closes[i-1]) / closes[i-1] if (i > 0 and closes[i-1] > 0) else 0.0
        vol_ratio = (vols[i] / vol_ma20[i]) if vol_ma20[i] > 0 else 1.0
        signal = pct_change * min(vol_ratio, 3.0) * 100.0

        current_trend = 0.85 * current_trend + signal
        sim_net[i] = np.clip(current_trend, -6.0, 6.0)

    df['價量動能估算'] = sim_net
    df['價量動能變化'] = df['價量動能估算'].diff(1)
    diff_values = df['價量動能變化'].dropna().to_numpy().flatten()
    latest_val = float(df['價量動能估算'].to_numpy().flatten()[-1])

    count = 0
    if len(diff_values) > 0:
        is_positive = diff_values[-1] > 0
        for val in reversed(diff_values):
            if (val > 0) == is_positive and val != 0: count += 1
            else: break
        status = f"🔥 連續 {count} 根上升" if is_positive else f"📉 連續 {count} 根下降"
    else: status = "🔄 區間震盪"

    return df, f"{latest_val:+.2f}% ({status})", latest_val
