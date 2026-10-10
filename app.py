# ==============================================================================
# 【機構級三核心策略雷達 3.11 全股票相容專業畫線與全指標回測版】 - app.py
# ==============================================================================
import sys, os, streamlit as st, yfinance as yf, pandas as pd, numpy as np, json, sqlite3, io, time, requests
import streamlit.components.v1 as components
import google.generativeai as genai
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="機構級三核心策略雷達 3.11", layout="wide", page_icon="📈")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "market_cache.db")
CACHE_TTL = 86400  # 快取有效期限：24 小時 (86400 秒)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS market_data 
                 (ticker TEXT PRIMARY KEY, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER, 
                  gvi REAL, roe REAL, mcap REAL, shares REAL, book_value REAL, 
                  roe_source TEXT, bv_source TEXT, updated_at INTEGER)''')
    
    existing_cols = [row[1] for row in c.execute("PRAGMA table_info(market_data)").fetchall()]
    cols_to_add = {
        "book_value": "REAL",
        "roe_source": "TEXT",
        "bv_source": "TEXT",
        "updated_at": "INTEGER"
    }
    for col, col_type in cols_to_add.items():
        if col not in existing_cols:
            try:
                c.execute(f"ALTER TABLE market_data ADD COLUMN {col} {col_type}")
            except Exception:
                pass
    conn.commit()
    conn.close()

init_db()

def parse_roe(raw_roe):
    if raw_roe is None or np.isnan(raw_roe) or not np.isfinite(raw_roe):
        return None
    val = float(raw_roe)
    if abs(val) > 3.0:
        val = val / 100.0
    return val

def is_cache_valid(updated_at):
    if not updated_at or np.isnan(updated_at):
        return False
    return (time.time() - float(updated_at)) < CACHE_TTL

def generate_content_with_retry(prompt_text, api_key_val):
    genai.configure(api_key=api_key_val)
    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.5-flash', 'gemini-2.0-flash']
    last_exception = None
    
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt_text)
                return response.text.strip()
            except Exception as e:
                last_exception = e
                if "503" in str(e) or "high demand" in str(e).lower() or "busy" in str(e).lower():
                    sleep_time = 2 ** (attempt + 1)
                    time.sleep(sleep_time)
                    continue
                break
    raise last_exception

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
        bv_source = "yfinance 財報" if (book_value is not None and not np.isnan(book_value) and book_value > 0) else None

        raw_roe = info.get('returnOnEquity')
        roe_val = parse_roe(raw_roe)
        roe_source = "yfinance 財報" if roe_val is not None else None

        # 🇹🇼 台灣股市備援邏輯
        if (".TW" in t or ".TWO" in t) and (book_value is None or roe_val is None):
            official_pb, official_pe = fetch_tw_official_pb_pe(t)
            if (book_value is None or book_value <= 0) and price and official_pb and official_pb > 0:
                book_value = price / official_pb
                bv_source = "TWSE/TPEx PB估算值"
            if roe_val is None and official_pe and official_pe > 0 and official_pb and official_pb > 0:
                roe_val = parse_roe(official_pb / official_pe)
                roe_source = "TWSE/TPEx PB/PE代理值"

        # 🇺🇸 美國股市備援邏輯
        if (".TW" not in t and ".TWO" not in t) and (book_value is None or roe_val is None):
            try:
                bs = stock.quarterly_balance_sheet
                if bs.empty: bs = stock.balance_sheet
                inc = stock.quarterly_financials
                if inc.empty: inc = stock.financials

                tot_equity = None
                if not bs.empty:
                    equity_keys = ['Total Stockholder Equity', 'Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest']
                    for key in equity_keys:
                        if key in bs.index:
                            found_eq = bs.loc[key].iloc[0]
                            if found_eq is not None and not np.isnan(found_eq):
                                tot_equity = float(found_eq)
                                break

                if book_value is None and tot_equity is not None:
                    shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding')
                    if shares and shares > 0:
                        calculated_bv = tot_equity / float(shares)
                        if calculated_bv > 0:
                            book_value = calculated_bv
                            bv_source = "yfinance 美股資產負債表精算"
                        else:
                            bv_source = "美股負股東權益(庫藏股/虧損)"

                if roe_val is None and tot_equity is not None and tot_equity > 0 and not inc.empty:
                    net_income_keys = ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests']
                    for net_key in net_income_keys:
                        if net_key in inc.index:
                            inc_series = inc.loc[net_key].dropna()
                            net_income_ttm = float(inc_series.iloc[:4].sum()) if len(inc_series) >= 4 else (float(inc_series.iloc[0]) * 4 if len(inc_series) > 0 else None)
                            if net_income_ttm is not None:
                                roe_val = parse_roe(net_income_ttm / tot_equity)
                                roe_source = "yfinance 美股財報 TTM 推算"
                            break
            except Exception as us_err:
                st.sidebar.caption(f"美股財報備援解析提醒 [{t}]: {us_err}")

        bv_val = float(book_value) if (book_value is not None and not np.isnan(book_value) and book_value > 0) else None
        roe_val = parse_roe(roe_val)

        gvi = (bv_val / price) * ((1 + roe_val) ** 5) if (price and price > 0 and bv_val and bv_val > 0 and roe_val is not None) else None

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

def compute_backtest_indicators(df, ma_entry_p=20, ma_exit_p=20, rsi_p=14, bb_p=20):
    df_calc = df.copy()
    df_calc['MA_entry'] = df_calc['Close'].rolling(window=ma_entry_p).mean()
    df_calc['MA_exit'] = df_calc['Close'].rolling(window=ma_exit_p).mean()

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
    df_calc['BB_Upper'] = bb_middle + (bb_std * 2)
    df_calc['BB_Lower'] = bb_middle - (bb_std * 2)

    return df_calc

def run_strategy_backtest(df_calc, entry_conds, exit_conds, date_chinese_list):
    trades = []
    position = False
    entry_price = 0.0
    entry_date = ""
    highest_price = 0.0

    closes = df_calc['Close'].to_numpy()
    highs = df_calc['High'].to_numpy()
    lows = df_calc['Low'].to_numpy()
    vols = df_calc['Volume'].to_numpy()
    vol_ma20 = df_calc['VOL_MA20'].to_numpy()

    ma_entry = df_calc['MA_entry'].to_numpy()
    ma_exit = df_calc['MA_exit'].to_numpy()
    k_arr = df_calc['K'].to_numpy()
    d_arr = df_calc['D'].to_numpy()
    dif_arr = df_calc['DIF'].to_numpy()
    dem_arr = df_calc['DEM'].to_numpy()
    rsi_arr = df_calc['RSI'].to_numpy()
    bb_upper = df_calc['BB_Upper'].to_numpy()
    bb_lower = df_calc['BB_Lower'].to_numpy()

    for i in range(1, len(df_calc)):
        if not position:
            buy_checks = []
            
            if entry_conds['ma_mode'] == '突破均線':
                buy_checks.append(not np.isnan(ma_entry[i-1]) and closes[i-1] <= ma_entry[i-1] and closes[i] > ma_entry[i])
            elif entry_conds['ma_mode'] == '站穩均線上':
                buy_checks.append(not np.isnan(ma_entry[i]) and closes[i] > ma_entry[i])

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

            if buy_checks and (all(buy_checks) if entry_conds['match_mode'] == 'ALL' else any(buy_checks)):
                position = True
                entry_price = float(closes[i])
                entry_date = date_chinese_list[i]
                highest_price = float(highs[i])
        else:
            highest_price = max(highest_price, float(highs[i]))
            exit_reasons = []

            if exit_conds['stop_loss_pct'] > 0:
                sl_price = entry_price * (1.0 - exit_conds['stop_loss_pct'] / 100.0)
                if lows[i] <= sl_price or closes[i] <= sl_price:
                    exit_reasons.append(f"觸發停損 (-{exit_conds['stop_loss_pct']}%)")

            if exit_conds['trailing_stop_pct'] > 0:
                trail_price = highest_price * (1.0 - exit_conds['trailing_stop_pct'] / 100.0)
                if lows[i] <= trail_price or closes[i] <= trail_price:
                    exit_reasons.append(f"最高點回撤 (-{exit_conds['trailing_stop_pct']}%)")

            if exit_conds['ma_mode'] == '跌破均線' and not np.isnan(ma_exit[i-1]):
                if closes[i-1] >= ma_exit[i-1] and closes[i] < ma_exit[i]:
                    exit_reasons.append(f"跌破 MA{exit_conds['ma_p']}")

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

            if exit_reasons:
                exit_price = float(closes[i])
                exit_date = date_chinese_list[i]
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0
                trades.append({
                    '買進日期': entry_date,
                    '買進價格 (元)': round(entry_price, 2),
                    '賣出日期': exit_date,
                    '賣出價格 (元)': round(exit_price, 2),
                    '平倉報酬率 (%)': round(pnl_pct, 2),
                    '離場觸發原因': " | ".join(exit_reasons)
                })
                position = False

    return pd.DataFrame(trades)

# 🔐 管理員操盤密碼固定為 7770
st.sidebar.markdown("### 🔒 操盤手安全密碼鎖")
input_password = st.sidebar.text_input("請輸入管理員操盤密碼：", type="password")

if input_password != "7770":
    st.title("🔒 華爾街機構級三核心策略雷達終端")
    st.warning("⚠️ 請於左側邊欄輸入正確的管理員操盤密碼以解鎖核心雷達。")
    st.stop()
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 全球個股即時診斷")
    st.sidebar.caption("💡 提示：上市請加 `.TW`，上櫃請加 `.TWO`（例如：3293.TWO）")
    selected_stock = st.sidebar.text_input("輸入台美股代碼：", value="3293.TWO").strip().upper()
    st.title("📈 機構級三核心策略雷達 3.11（專業畫線與年.月簡化版）")

    st.markdown("### 🌐 全球大盤即時看板")
    col1, col2, col3, col4 = st.columns(4)

    def get_market_index(ticker):
        try:
            df = yf.Ticker(ticker).history(period="5d")
            if not df.empty and len(df) >= 2:
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                close_series = df['Close'].dropna().to_numpy().flatten()
                if len(close_series) >= 2:
                    close_today = float(close_series[-1])
                    change = close_today - float(close_series[-2])
                    return close_today, change, (change / float(close_series[-2])) * 100
        except Exception: 
            pass
        return None, None, None

    tw_val, tw_chg, tw_pct = get_market_index("^TWII")
    sp_val, sp_chg, sp_pct = get_market_index("^GSPC")
    dj_val, dj_chg, dj_pct = get_market_index("^DJI")
    nas_val, nas_chg, nas_pct = get_market_index("^IXIC")

    with col1: st.metric(label="🇹🇼 台灣加權指數 (^TWII)", value=f"{tw_val:,.2f}" if tw_val else "更新中...", delta=f"{tw_chg:+.2f} ({tw_pct:+.2f}%)" if tw_val else "--")
    with col2: st.metric(label="🇺🇸 標普 500 指數 (^GSPC)", value=f"{sp_val:,.2f}" if sp_val else "載入中...", delta=f"{sp_chg:+.2f} ({sp_pct:+.2f}%)" if sp_val else "--")
    with col3: st.metric(label="🇺🇸 道瓊工業指數 (^DJI)", value=f"{dj_val:,.2f}" if dj_val else "載入中...", delta=f"{dj_chg:+.2f} ({dj_pct:+.2f}%)" if dj_val else "--")
    with col4: st.metric(label="🇺🇸 那斯達克指數 (^IXIC)", value=f"{nas_val:,.2f}" if nas_val else "載入中...", delta=f"{nas_chg:+.2f} ({nas_pct:+.2f}%)" if nas_val else "--")

    st.markdown("---")

    st.sidebar.markdown("### 🧠 系統 API 金鑰設定")
    api_key_input = ""
    try:
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"].strip() != "":
            api_key_input = st.secrets["GEMINI_API_KEY"].strip()
            st.sidebar.success("🔒 Gemini API 金鑰已從雲端安全盾自動加載")
    except Exception:
        pass

    if not api_key_input:
        api_key_input = st.sidebar.text_input("請輸入您的 Gemini API Key：", type="password")

    st.sidebar.markdown("### 🎛️ 個人自訂看盤面板")
    selected_tf = st.sidebar.selectbox("⏱️ 看盤 K 線時間軸級別", options=["1分鐘", "5分鐘", "30分鐘", "60分鐘", "1日", "1週", "1個月", "1季", "半年", "1年"], index=4)
    tf_mapping = {
        "1分鐘": {"p": "7d", "i": "1m"}, "5分鐘": {"p": "7d", "i": "5m"}, "30分鐘": {"p": "30d", "i": "30m"}, "60分鐘": {"p": "60d", "i": "60m"},
        "1日": {"p": "2y", "i": "1d"}, "1週": {"p": "5y", "i": "1wk"}, "1個月": {"p": "max", "i": "1mo"}, 
        "1季": {"p": "max", "i": "3mo"}, "半年": {"p": "max", "i": "3mo"}, "1年": {"p": "max", "i": "3mo"}
    }
    cfg = tf_mapping[selected_tf]

    with st.sidebar.expander("🛠️ 均線指標快速視窗設定", expanded=True):
        preset_ma = st.selectbox("📊 選擇均線快捷套組", options=["標準視窗 (5日 / 10日 / 20日)", "極短線 (3日 / 5日 / 10日)", "波段趨勢 (20日 / 60日 / 120日)", "自訂均線配置"], index=0)
        default_days = [5, 10, 20, 60, 240] if preset_ma == "標準視窗 (5日 / 10日 / 20日)" else ([3, 5, 10, 20, 60] if preset_ma == "極短線 (3日 / 5日 / 10日)" else [20, 60, 120, 240, 500])
        default_actives = [True, True, True, False, False]
        personal_ma_configs = []
        default_colors = ["#FF5733", "#33FF57", "#3357FF", "#F3FF33", "#FF33F3"]

        for i in range(1, 6):
            col_show, col_p, col_c = st.columns([0.8, 1.4, 0.8])
            with col_show: is_active = st.checkbox(f"MA{i}", value=default_actives[i-1], key=f"ma_active_{i}")
            with col_p: ma_p = st.number_input(f"天數{i}", min_value=1, max_value=500, value=int(default_days[i-1]), label_visibility="collapsed", key=f"personal_ma_p_{i}")
            with col_c: ma_c = str(st.color_picker(f"C{i}", value=default_colors[i-1], label_visibility="collapsed", key=f"personal_ma_c_{i}"))
            if is_active: personal_ma_configs.append({"period": int(ma_p), "color": ma_c})

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
            
        df['籌碼集中度'] = sim_net
        df['5日差值'] = df['籌碼集中度'].diff(1)
        diff_values = df['5日差值'].dropna().to_numpy().flatten()
        latest_val = float(df['籌碼集中度'].to_numpy().flatten()[-1])
        
        count = 0
        if len(diff_values) > 0:
            is_positive = diff_values[-1] > 0
            for val in reversed(diff_values):
                if (val > 0) == is_positive and val != 0: count += 1
                else: break
            status = f"🔥 連續 {count} 日上升" if is_positive else f"📉 連續 {count} 日下降"
        else: status = "🔄 區間震盪洗盤"
        
        return df, f"{latest_val:+.2f}% ({status})", latest_val

    df_chart = None
    if selected_stock:
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

            df_chart = yf.download(selected_stock, period=cfg["p"], interval=cfg["i"], progress=False)
            if len(df_chart) > 0:
                if isinstance(df_chart.columns, pd.MultiIndex): df_chart.columns = df_chart.columns.get_level_values(0)
                df_chart, chip_status_text, _ = calculate_chip_and_backtest(selected_stock, df_chart, selected_tf)
                
                # 💡【優化 1：日期改成年.月 (YYYY.MM) 格式】
                date_strings = df_chart.index.strftime('%Y.%m').tolist()
                c_name = STOCK_NAME_MAP.get(selected_stock, selected_stock)
                price_val = float(df_chart['Close'].to_numpy().flatten()[-1])
                is_tw_stock = ".TW" in selected_stock or ".TWO" in selected_stock

                bv_val = float(db_row[0]) if (db_row and db_row[0] is not None and db_row[0] > 0) else None
                roe_val = parse_roe(db_row[1]) if (db_row and db_row[1] is not None) else None
                bv_src = db_row[4] if (db_row and db_row[4]) else "無數據"
                roe_src = db_row[5] if (db_row and db_row[5]) else "無數據"
                cache_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(db_row[6])) if (db_row and db_row[6]) else "未設定"

                if price_val > 0 and bv_val is not None and roe_val is not None:
                    pb_val = price_val / bv_val
                    gvi_val = (bv_val / price_val) * ((1 + roe_val) ** 5)
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

                gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                with gc1: st.metric(label=f"💰 當前現價 ({selected_stock})", value=f"{price_val:,.2f} 元" if is_tw_stock else f"${price_val:,.2f}"); st.caption(f"📢 即時報價 ({cache_time_str})")
                with gc2: st.metric(label="👑 GVI 成長價值指標", value=gvi_str); st.caption("📢 即時重算 (最新股價對位)")
                with gc3: st.metric(label="📊 股東權益報酬率 ROE", value=roe_str); st.caption(f"📢 來源: {roe_src}")
                with gc4: st.metric(label="📖 每股淨值", value=bv_str); st.caption(f"📢 來源: {bv_src}")
                with gc5: st.metric(label="⚖️ 股價淨值比 (PB)", value=pb_str); st.caption("📢 溢價程度")

                if gvi_val is not None and pb_val is not None:
                    if gvi_val >= 0.35 and pb_val <= 1.5:
                        valuation_color = "#00cc66"
                        valuation_status = "🔥 極度便宜（有安全邊際，機構瘋狂撿便宜區）"
                        valuation_desc = "💡 <b>操盤手報告：</b>內在價值強勁但估值嚴重低估！屬下檔風險鎖死黃金買點。"
                    elif gvi_val >= 0.20 or (pb_val > 1.5 and pb_val <= 3.5 and roe_val >= 0.12):
                        valuation_color = "#2baf2b"
                        valuation_status = "🟢 合理甜美（體質估值相稱，長線穩健布局期）"
                        valuation_desc = "💡 <b>操盤手報告：</b>股價完美對位體質，屬長線基金安全期。"
                    elif pb_val > 3.5 and pb_val <= 7.0:
                        valuation_color = "#ff9900"
                        valuation_status = "⚠️ 偏貴溢價（樂觀情緒透支，操盤手需嚴格風控）"
                        valuation_desc = "💡 <b>操盤手報告：</b>股價已提前預支獲利，追高性價比低，嚴守停損。"
                    else:
                        valuation_color = "#cc0000"
                        valuation_status = "🚨 泡沫嚴重（全面避開提款機，估值嚴重偏離）"
                        valuation_desc = "💡 <b>操盤手報告：</b>投機情緒沸騰！PB極高且ROE無法支撐，切勿當接盤俠。"
                else:
                    valuation_color = "#888888"
                    valuation_status = "⚠️ 基本面資料不足（系統停止計算 GVI）"

                    reasons = []
                    if was_missing_in_db:
                        reasons.append("📡 <b>自動連線下載結果</b>：線上請求完成，但交易所未回傳完整數據。")
                    if bv_val is None:
                        if "負股東權益" in bv_src or "庫藏股" in bv_src:
                            reasons.append("📉 <b>美股負股東權益</b>：長期註銷庫藏股導至 $BV \\le 0$（如 AAPL），GVI不適用。")
                        else:
                            reasons.append("❌ <b>每股淨值 (BV) 缺失</b>：可能為 ETF、債券，或財報未揭露。")
                    if roe_val is None:
                        reasons.append("❌ <b>ROE 缺失/非正值</b>：當期公司虧損或尚無預估資料。")

                    reason_html = "<br/>&nbsp;&nbsp;&nbsp;&nbsp;• " + "<br/>&nbsp;&nbsp;&nbsp;&nbsp;• ".join(reasons)
                    
                    valuation_desc = (
                        f"<b>【自動排查與缺值診斷】</b>{reason_html}<br/><br/>"
                        f"💡 <b>給操盤手的應變建議：</b><br/>"
                        f"1. 若標的為美股庫藏股巨頭（如 AAPL），請改用 <b>「QARP 現金流雷達」</b>。<br/>"
                        f"2. 若標的為飆股或技術面突破股，請改用 <b>「動能突破雷達」</b> 搭配 K 線操作。"
                    )

                st.markdown(
                    f"<div style='background-color:rgba(30,30,30,0.7); padding:14px 18px; border-left:6px solid {valuation_color}; border-radius:4px; margin-bottom:15px;'>"
                    f"<h5 style='margin:0; color:white;'>⚖️ 華爾街智慧估值雷達：<span style='color:{valuation_color}; font-weight:bold;'>{valuation_status}</span></h5>"
                    f"<p style='margin:6px 0 0 0; font-size:14px; color:#cccccc;'>{valuation_desc}</p>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                st.info(f"🔮 【{c_name}】{selected_tf} 即時籌碼動能判定：{chip_status_text}")

                if st.button(f"🧠 啟動 Gemini AI 分析【{c_name}】個股綜合投資價值", use_container_width=True, type="primary"):
                    if not api_key_input: 
                        st.error("⚠️ 請先在左側邊欄輸入 Gemini API Key！")
                    else:
                        with st.spinner(f"🤖 精算 {c_name} 報告中..."):
                            try:
                                ai_prompt = f"請針對個股:{selected_stock}({c_name})，當前量化指標：GVI={gvi_str}, ROE={roe_str}, PB={pb_str}，估值狀態為{valuation_status}。300字內提供繁體中文投資報告。"
                                success_report = generate_content_with_retry(ai_prompt, api_key_input)
                                st.session_state[f"ai_report_{selected_stock}"] = success_report
                            except Exception as ai_e: 
                                st.error(f"Google 雲端機房繁忙或金鑰異常: {ai_e}")
                
                if f"ai_report_{selected_stock}" in st.session_state:
                    st.success(f"📋 AI 【{c_name}】核心投資分析報告")
                    st.markdown(st.session_state[f"ai_report_{selected_stock}"])
                    st.markdown("---")
        except Exception as e:
            st.error(f"數據載入異常：{e}")

    try:
        if df_chart is None or df_chart.empty:
            st.error(f"❌ 無此標的或無法取得數據：【{selected_stock}】，請檢查股票代碼是否正確。")
        else:
            tab1, tab2, tab3, tab4 = st.tabs(["📊 彩色 K 線圖與畫線工具箱", "💰 法人散戶流向報告", "🤖 網格自動生成器 (3.7 版)", "🧪 策略自訂回測器 (3.8 全指標選單版)"])
            
            # ==============================================================================
            # 【Tab 1: 3.11 Plotly 原生 K 線圖 + 年.月日期 + 畫圖選色移除/文字註記 + K棒寬度固定】
            # ==============================================================================
            with tab1:
                col_title, col_draw_color = st.columns([3, 1])
                with col_title:
                    ma_display_html = "<div style='background-color:rgba(20,20,20,0.8); padding:6px 12px; border:1px solid #444; border-radius:8px; display:inline-block; font-family:monospace; font-size:14px; color:white; vertical-align:middle;'>"
                    for ma in personal_ma_configs:
                        p, c = ma["period"], ma["color"]
                        ma_series = df_chart['Close'].rolling(window=p).mean().dropna()
                        if not ma_series.empty:
                            latest_ma_val = ma_series.to_numpy().flatten()[-1]
                            ma_display_html += f"<span style='color:{c}; font-weight:bold; margin-right:12px;'>■ {p}日均線: {latest_ma_val:,.2f}</span>"
                    ma_display_html += "</div>"
                    st.markdown(f"### 📊 【{c_name}】全功能互動 K 線圖 {ma_display_html}", unsafe_allow_html=True)
                
                with col_draw_color:
                    # 💡【優化 3：動態自訂畫線筆劃顏色】
                    draw_color = st.color_picker("🎨 自訂畫線與文字顏色", value="#FF3333", key="draw_line_color_picker")

                st.caption("💡 **繪圖工具列指南**：點擊右上方工具列「直線 (drawline)」、「筆刷」、「矩形 (drawrect)」、「📝 輸入文字 (drawtext)」即可在圖上繪製標記；點擊「橡皮擦 (eraseshape)」後選取畫線即可移除！")
                
                fig = make_subplots(rows=1, cols=1)
                
                # 繪製 K 線
                fig.add_trace(go.Candlestick(
                    x=date_strings, 
                    open=df_chart['Open'].to_numpy().flatten().tolist(), 
                    high=df_chart['High'].to_numpy().flatten().tolist(), 
                    low=df_chart['Low'].to_numpy().flatten().tolist(), 
                    close=df_chart['Close'].to_numpy().flatten().tolist(), 
                    name="K線",
                    hovertext=[f"日期：{d}" for d in date_strings]
                ), row=1, col=1)
                
                # 繪製自訂均線
                for ma in personal_ma_configs:
                    p, c = ma["period"], ma["color"]
                    ma_series = df_chart['Close'].rolling(window=p).mean().dropna()
                    if not ma_series.empty:
                        ma_list = ma_series.to_numpy().flatten().tolist()
                        fig.add_trace(go.Scatter(x=date_strings[-len(ma_list):], y=ma_list, mode='lines', name=f'{p}日均線 (MA{p})', line=dict(color=c, width=1.8)), row=1, col=1)
                
                fig.update_layout(
                    xaxis_rangeslider_visible=False, 
                    height=620, 
                    margin=dict(l=10, r=40, t=10, b=10), 
                    dragmode='pan', 
                    showlegend=True, 
                    # 💡【優化 2：將 X 軸類型設為 category 確保放大縮小時 K 棒不會變形放大/縮小】
                    xaxis=dict(
                        type='category',
                        showgrid=True, 
                        gridcolor="rgba(128,128,128,0.2)",
                        nticks=15  # 自動顯示適當數量的 YYYY.MM 日期標籤
                    ), 
                    yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
                    # 💡【優化 3 & 4：設定預設畫線樣式與自訂顏色】
                    newshape=dict(
                        line=dict(color=draw_color, width=2.5),
                        fillcolor=draw_color,
                        opacity=0.6
                    )
                )
                
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    config={
                        'modeBarButtonsToAdd': [
                            'drawline',       # 直線 / 趨勢線
                            'drawopenpath',   # 畫筆自由畫線
                            'drawrect',       # 繪製矩形 (壓力/支撐/網格區間)
                            'drawcircle',     # 繪製圓形
                            'drawtext',       # 💡【優化 4：新增圖上輸入文字標註功能】
                            'eraseshape'      # 💡【優化 3：橡皮擦點擊移除劃線】
                        ],
                        'displayModeBar': True,
                        'scrollZoom': True
                    }
                )
            
            with tab2:
                st.plotly_chart(go.Figure(data=[go.Bar(x=['主力買超', '主力賣超', '散戶買超', '散戶賣超'], y=[float(df_chart['Volume'].to_numpy().flatten()[-1])*0.3, float(df_chart['Volume'].to_numpy().flatten()[-1])*0.25, float(df_chart['Volume'].to_numpy().flatten()[-1])*0.2, float(df_chart['Volume'].to_numpy().flatten()[-1])*0.25])]), use_container_width=True)
            
            with tab3:
                st.markdown("### 🎛️ 智慧型動態網格策略與資金自動規劃器 (3.7 版)")
                
                grid_p = float(price_val)
                is_tw = ".TW" in selected_stock or ".TWO" in selected_stock
                unit_label = "股" if not is_tw else "股 (台股預設)"

                g_top1, g_top2, g_top3 = st.columns(3)
                with g_top1:
                    total_capital = st.number_input("💵 投入總資金 (元/$)", value=100000, step=10000, key="grid_capital")
                with g_top2:
                    grid_strategy = st.selectbox(
                        "🎯 選擇網格核心策略機制", 
                        options=["1. 固定比例再平衡 (例如 1:1)", "2. 固定股數交易", "3. 庫存百分比管理"],
                        key="grid_strat_select"
                    )
                with g_top3:
                    init_stock_ratio = st.slider("⚖️ 初始股票部位佔比 (%)", min_value=10, max_value=90, value=50, step=5, key="grid_init_ratio")

                st.markdown("---")

                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1: 
                    input_lower = st.number_input("網格下限價格", value=round(grid_p * 0.80, 2), key="grid_lower")
                with g_col2: 
                    input_upper = st.number_input("網格上限價格", value=round(grid_p * 1.20, 2), key="grid_upper")
                with g_col3: 
                    input_num = st.number_input("規劃總格數", min_value=3, max_value=50, value=10, step=1, key="grid_num")

                init_stock_cash = total_capital * (init_stock_ratio / 100.0)
                init_shares = int(init_stock_cash // grid_p) if grid_p > 0 else 0
                actual_init_stock_val = init_shares * grid_p
                actual_init_cash = total_capital - actual_init_stock_val

                st.info(
                    f"💡 **初始建倉試算**：總資金 {total_capital:,.0f} 元 | "
                    f"建倉購買 **{init_shares:,}** {unit_label} (約 {actual_init_stock_val:,.0f} 元，佔 {actual_init_stock_val/total_capital*100:.1f}%) | "
                    f"保留預備現金 **{actual_init_cash:,.0f}** 元"
                )

                levels = np.linspace(input_lower, input_upper, input_num)
                grid_details = []

                for idx, level in enumerate(levels):
                    action = "🟢 買進掛單" if level < grid_p else ("🔴 賣出掛單" if level > grid_p else "⚪ 當前基準價")
                    
                    if "固定比例" in grid_strategy:
                        est_stock_val = init_shares * level
                        est_total = est_stock_val + actual_init_cash
                        target_stock_val = est_total * (init_stock_ratio / 100.0)
                        diff_val = target_stock_val - est_stock_val
                        
                        if diff_val > 0:
                            trade_desc = f"動用現金買進約 {abs(diff_val):,.0f} 元股票"
                        elif diff_val < 0:
                            trade_desc = f"賣出約 {abs(diff_val):,.0f} 元股票補回現金"
                        else:
                            trade_desc = "平衡狀態"

                    elif "固定股數" in grid_strategy:
                        fixed_shares = max(1, int((init_shares or 1000) / input_num))
                        trade_desc = f"固定交易 {fixed_shares:,} {unit_label} (約 {fixed_shares * level:,.0f} 元)"

                    else:
                        pct = round(100.0 / input_num, 1)
                        trade_desc = f"交易當前庫存之 {pct}% (隨庫存規模動態增減)"

                    grid_details.append({
                        "網格層級": f"Grid #{idx+1:02d}",
                        "目標觸發價": round(level, 2),
                        "偏離現價 (%)": f"{((level - grid_p) / grid_p * 100):+.2f}%",
                        "預計動作": action,
                        "資金/部位管理機制": trade_desc
                    })

                st.dataframe(pd.DataFrame(grid_details), use_container_width=True, height=280)

            with tab4:
                st.markdown(f"### 🧪 【{c_name}】全指標多空量化策略與動態停損回測器 (3.8 版)")
                
                bt_col1, bt_col2 = st.columns(2)
                
                with bt_col1:
                    st.markdown("##### 🟢 進場 (買進) 策略下拉式選單")
                    entry_ma_mode = st.selectbox("1. 均線策略 (MA)", ["停用", "突破均線", "站穩均線上"], index=1, key="bt_entry_ma_sel")
                    entry_ma_p = st.number_input("買進均線天數 (MA)", min_value=1, max_value=240, value=20, key="bt_entry_ma_p")
                    
                    entry_kd_mode = st.selectbox("2. KD 指標策略", ["停用", "黃金交叉"], index=0, key="bt_entry_kd_sel")
                    entry_macd_mode = st.selectbox("3. MACD 指標策略", ["停用", "黃金交叉"], index=0, key="bt_entry_macd_sel")
                    entry_vol_mode = st.selectbox("4. 成交量策略", ["停用", "爆量 (大於20日均量1.5倍)", "爆量 (大於20日均量2.0倍)"], index=0, key="bt_entry_vol_sel")
                    entry_rsi_mode = st.selectbox("5. RSI 指標策略", ["停用", "超賣回升 (RSI向上突破30)", "強勢突破 (RSI向上突破50)"], index=0, key="bt_entry_rsi_sel")
                    entry_bb_mode = st.selectbox("6. 布林通道策略", ["停用", "突破布林上軌", "觸及布林下軌反彈"], index=0, key="bt_entry_bb_sel")
                    
                    entry_match_mode = st.radio("買進訊號判定邏輯", options=["同時滿足所有選定條件 (ALL)", "任一滿足即買進 (ANY)"], index=0, key="bt_match")

                with bt_col2:
                    st.markdown("##### 🔴 離場 (停損/停利) 策略下拉式選單")
                    stop_loss_pct = st.number_input("固定停損幅度 (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.5, key="bt_sl")
                    trailing_stop_pct = st.number_input("最高價移動回撤幅度 (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.5, key="bt_trail")
                    
                    exit_ma_mode = st.selectbox("1. 均線平倉策略 (MA)", ["停用", "跌破均線"], index=1, key="bt_exit_ma_sel")
                    exit_ma_p = st.number_input("賣出均線天數 (MA)", min_value=1, max_value=240, value=20, key="bt_exit_ma_p")
                    
                    exit_kd_mode = st.selectbox("2. KD 離場策略", ["停用", "死亡交叉"], index=0, key="bt_exit_kd_sel")
                    exit_macd_mode = st.selectbox("3. MACD 離場策略", ["停用", "死亡交叉"], index=0, key="bt_exit_macd_sel")
                    exit_vol_mode = st.selectbox("4. 成交量離場策略", ["停用", "極端爆量倒貨 (大於20日均量2.5倍)"], index=0, key="bt_exit_vol_sel")
                    exit_rsi_mode = st.selectbox("5. RSI 離場策略", ["停用", "超買警戒 (RSI跌破70)"], index=0, key="bt_exit_rsi_sel")
                    exit_bb_mode = st.selectbox("6. 布林通道離場策略", ["停用", "跌破布林下軌", "觸及上軌拉回"], index=0, key="bt_exit_bb_sel")

                st.markdown("---")

                if st.button(f"🚀 開始執行【{c_name}】全指標量化回測", type="primary", use_container_width=True, key="run_bt_btn"):
                    entry_conds = {
                        'ma_mode': entry_ma_mode,
                        'ma_p': entry_ma_p,
                        'kd_mode': entry_kd_mode,
                        'macd_mode': entry_macd_mode,
                        'vol_mode': entry_vol_mode,
                        'rsi_mode': entry_rsi_mode,
                        'bb_mode': entry_bb_mode,
                        'match_mode': 'ALL' if "同時" in entry_match_mode else 'ANY'
                    }
                    
                    exit_conds = {
                        'stop_loss_pct': stop_loss_pct,
                        'trailing_stop_pct': trailing_stop_pct,
                        'ma_mode': exit_ma_mode,
                        'ma_p': exit_ma_p,
                        'kd_mode': exit_kd_mode,
                        'macd_mode': exit_macd_mode,
                        'vol_mode': exit_vol_mode,
                        'rsi_mode': exit_rsi_mode,
                        'bb_mode': exit_bb_mode
                    }
                    
                    df_calc = compute_backtest_indicators(df_chart, ma_entry_p=entry_ma_p, ma_exit_p=exit_ma_p)
                    df_trades = run_strategy_backtest(df_calc, entry_conds, exit_conds, date_strings)

                    if not df_trades.empty:
                        st.dataframe(df_trades, use_container_width=True)
                    else:
                        st.warning("⚠️ 在選定區間內未有符合條件的完整交易紀錄。")

    except Exception as ex_tab:
        st.error(f"❌ 畫面渲染異常：{ex_tab}")

    st.markdown("---")
    st.markdown("### 🚀 華爾街機構級三核心策略雷達")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: strat_gvi = st.checkbox("開啟 GVI 價值雷達", value=True, key="strat_gvi_check")
    with col_s2: strat_momentum = st.checkbox("開啟動能突破雷達", value=False, key="strat_momentum_check")
    with col_s3: strat_qarp = st.checkbox("開啟 QARP 現金流雷達", value=False, key="strat_qarp_check")

    custom_input_pool = st.text_area("✍️ 操盤手自訂觀察代碼掃描區：", value="2330.TW, 3293.TWO, 8069.TWO, NVDA, AAPL")
    custom_scan_list = [c.strip().upper() for c in custom_input_pool.split(",") if c.strip()]

    def load_data_from_sqlite_and_render(filter_list, title, is_custom_mode=False):
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

        if df_db.empty: return
            
        df_filtered = df_db[df_db['ticker'].isin(filter_list)].copy()
        raw_portfolio = []
        
        for idx, row in df_filtered.iterrows():
            t = row['ticker']
            try:
                ch_name = STOCK_NAME_MAP.get(t, t)
                df_hist = yf.download(t, period=cfg["p"], interval=cfg["i"], progress=False)
                if isinstance(df_hist.columns, pd.MultiIndex): df_hist.columns = df_hist.columns.get_level_values(0)
                if df_hist.index.tz is not None: df_hist.index = df_hist.index.tz_localize(None)
                df_hist = df_hist.dropna(subset=['Close', 'Volume']).copy()
                L = len(df_hist)
                
                close_today = float(row['close']) if (L == 0 or 'Close' not in df_hist.columns) else float(df_hist['Close'].to_numpy().flatten()[-1])
                high_252d = close_today * 1.05 if L < 50 else float(df_hist['Close'].tail(min(252, L)).max())
                
                bv_val = float(row['book_value']) if (pd.notnull(row['book_value']) and float(row['book_value']) > 0) else None
                roe_val = parse_roe(row['roe'])

                gvi_live = (bv_val / close_today) * ((1 + roe_val) ** 5) if (close_today > 0 and bv_val is not None and roe_val is not None) else None

                stock_obj = yf.Ticker(t)
                info_obj = stock_obj.info if hasattr(stock_obj, 'info') and stock_obj.info else {}
                
                fcf = info_obj.get('freeCashflow')
                mcap = info_obj.get('marketCap') or row['mcap']

                if (fcf is None or np.isnan(fcf)) and mcap and mcap > 0:
                    try:
                        cf = stock_obj.quarterly_cashflow
                        if cf.empty: cf = stock_obj.cashflow
                        if not cf.empty:
                            ocf, capex = None, None
                            for ocf_k in ['Operating Cash Flow', 'Total Cash From Operating Activities', 'Cash Flow From Continuing Operating Activities']:
                                if ocf_k in cf.index:
                                    s = cf.loc[ocf_k].dropna()
                                    ocf = float(s.iloc[:4].sum()) if len(s) >= 4 else float(s.iloc[0]) * 4
                                    break
                            for cap_k in ['Capital Expenditure', 'Capital Expenditures']:
                                if cap_k in cf.index:
                                    s = cf.loc[cap_k].dropna()
                                    capex = float(s.iloc[:4].sum()) if len(s) >= 4 else float(s.iloc[0]) * 4
                                    break
                            if ocf is not None:
                                capex_val = abs(float(capex)) if capex is not None else 0.0
                                fcf = float(ocf) - capex_val
                    except Exception:
                        pass

                fcf_yield = (float(fcf) / float(mcap) * 100.0) if (fcf and mcap and float(mcap) > 0) else None

                peg_raw = info_obj.get('pegRatio')
                peg_val = float(peg_raw) if (peg_raw and not np.isnan(peg_raw) and peg_raw > 0) else None

                if peg_val is None:
                    try:
                        pe_val = info_obj.get('trailingPE') or info_obj.get('forwardPE')
                        growth_val = info_obj.get('earningsGrowth') or info_obj.get('earningsQuarterlyGrowth')
                        if pe_val and growth_val and float(pe_val) > 0 and float(growth_val) > 0:
                            peg_val = float(pe_val) / (float(growth_val) * 100.0)
                    except Exception:
                        pass

                df_hist, chip_status_text, _ = calculate_chip_and_backtest(t, df_hist, selected_tf)
                
                raw_portfolio.append({
                    '股票代碼': t, '股票名稱': ch_name, '目前股價': close_today, 
                    'GVI值': gvi_live, 
                    '動能分數': float((close_today / high_252d) * 100) if high_252d > 0 else 0.0, 
                    '自由現金流收益': fcf_yield, 
                    '本益成長比(PEG)': peg_val, 
                    '⚡ 即時籌碼動能': chip_status_text, 
                    'gvi_raw': gvi_live, 
                    'mo_raw': float((close_today / high_252d) * 100) if high_252d > 0 else 0.0, 
                    'fcf_raw': fcf_yield
                })
            except Exception as e_item: 
                st.sidebar.caption(f"標的跳過 [{t}]: {e_item}")
                
        if raw_portfolio:
            df_res = pd.DataFrame(raw_portfolio)
            
            if strat_gvi: df_res = df_res[df_res['gvi_raw'].notnull() & (df_res['gvi_raw'] >= 0.20)]
            if strat_momentum: df_res = df_res[df_res['mo_raw'] >= 80.0]
            if strat_qarp: df_res = df_res[df_res['fcf_raw'].notnull() & (df_res['fcf_raw'] >= 2.5) & df_res['本益成長比(PEG)'].notnull() & (df_res['本益成長比(PEG)'] <= 1.5)]
            
            if df_res.empty: df_res = pd.DataFrame(raw_portfolio)
                
            df_res['GVI_Rank'] = df_res['gvi_raw'].fillna(0).rank(pct=True)
            df_res['MO_Rank'] = df_res['mo_raw'].fillna(0).rank(pct=True)
            df_res['FCF_Rank'] = df_res['fcf_raw'].fillna(0).rank(pct=True)
            df_res['綜合分數'] = 0.0
            active_strats = 0
            
            if strat_gvi: df_res['綜合分數'] += df_res['GVI_Rank']; active_strats += 1
            if strat_momentum: df_res['綜合分數'] += df_res['MO_Rank']; active_strats += 1
            if strat_qarp: df_res['綜合分數'] += df_res['FCF_Rank']; active_strats += 1
            if active_strats == 0: df_res['綜合分數'] = df_res['GVI_Rank']; active_strats = 1
            
            df_res['策略評分_num'] = ((df_res['綜合分數'] / active_strats) * 100).round(1)
            df_res = df_res.sort_values(by='綜合分數', ascending=False).head(20).reset_index(drop=True)
            
            df_res['排名'] = [f"第 {i+1} 名" for i in range(len(df_res))]
            df_res['策略評分'] = df_res['策略評分_num'].astype(str) + " 分"
            df_res['GVI價值指標'] = df_res['GVI值'].apply(lambda x: f"{x:.4f}" if (pd.notnull(x) and x is not None) else "資料不足")
            df_res['動能(創高距離)'] = df_res['動能分數'].round(1).astype(str) + "%"
            df_res['自由現金流收益率'] = df_res['自由現金流收益'].apply(lambda x: f"{x:.2f}%" if (pd.notnull(x) and x is not None) else "資料不足")
            df_res['本益成長比(PEG)'] = df_res['本益成長比(PEG)'].apply(lambda x: f"{x:.2f}" if (pd.notnull(x) and x is not None) else "資料不足")
            
            cols = ['排名', '股票代碼', '股票名稱', '目前股價', '策略評分']
            if strat_gvi: cols.append('GVI價值指標')
            if strat_momentum: cols.append('動能(創高距離)')
            if strat_qarp: cols.extend(['自由現金流收益率', '本益成長比(PEG)'])
            if not strat_gvi and not strat_momentum and not strat_qarp:
                cols.extend(['GVI價值指標', '動能(創高距離)', '自由現金流收益率', '本益成長比(PEG)'])
            cols.append('⚡ 即時籌碼動能')
            
            df_final_view = df_res[cols]
            st.markdown(f"#### {title} (已依勾選核心策略評分動態排名 Top 20)")
            st.dataframe(df_final_view, use_container_width=True)

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("🇹🇼 一鍵執行：全自動過濾全台股核心池", type="secondary", use_container_width=True):
            load_data_from_sqlite_and_render(AUTO_TW_UNIVERSE, "🇹🇼 台灣股市核心策略篩選結果")
    with col_btn2:
        if st.button("🇺🇸 一鍵執行：全自動過濾全美股核心池", type="secondary", use_container_width=True):
            load_data_from_sqlite_and_render(AUTO_US_UNIVERSE, "🇺🇸 美國股市核心策略篩選結果")
    with col_btn3:
        if st.button("🚀 執行：自訂名單多因子本地精準過濾", type="primary", use_container_width=True):
            load_data_from_sqlite_and_render(custom_scan_list, "🎯 操盤手自訂名單策略篩選結果", is_custom_mode=True)