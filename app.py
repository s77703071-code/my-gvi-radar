# ==============================================================================
# 【華爾街機構級量化策略終端 4.0 - 模組化無隨機生成真實版】 - app.py
# ==============================================================================
import os, sqlite3, time, io
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="機構級量化策略雷達 4.0", layout="wide", page_icon="📈")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "market_cache.db")

# ==============================================================================
# 1. 資料庫基礎建設 (Database Setup)
# ==============================================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS market_data 
                 (ticker TEXT, date TEXT, close REAL, gvi REAL, roe REAL, pb REAL, 
                  fcf_yield REAL, peg REAL, PRIMARY KEY (ticker, date))''')
    conn.commit()
    conn.close()
init_db()

# ==============================================================================
# 2. 資料擷取模組 (Data Fetcher Layer)
# ==============================================================================
class DataFetcher:
    @staticmethod
    def get_stock_raw_data(ticker: str, period="2y", interval="1d"):
        """取得 K 線與成交量資料，並清理欄位名稱"""
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=['Close', 'Volume']).copy()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def fetch_fundamentals(ticker: str) -> dict:
        """從 yfinance 抓取真實基本面與財報資料，完全無隨機生成」"""
        res = {
            'price': None, 'book_value': None, 'roe': None, 
            'fcf_yield': None, 'peg': None, 'pe': None, 'shares': None, 'net_income_growth': None
        }
        try:
            stock = yf.Ticker(ticker)
            info = stock.info if hasattr(stock, 'info') and stock.info else {}
            
            # 股價與發行股數
            res['price'] = info.get('currentPrice') or info.get('previousClose')
            res['shares'] = info.get('sharesOutstanding')
            res['pe'] = info.get('trailingPE')
            
            # 每股淨值 (Book Value)
            res['book_value'] = info.get('bookValue')
            if res['book_value'] is None or np.isnan(res['book_value']):
                bs = stock.quarterly_balance_sheet if not stock.quarterly_balance_sheet.empty else stock.balance_sheet
                if not bs.empty:
                    for key in ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest']:
                        if key in bs.index:
                            equity = bs.loc[key].iloc[0]
                            if equity and res['shares'] and res['shares'] > 0:
                                res['book_value'] = float(equity) / float(res['shares'])
                            break

            # ROE (Return On Equity)
            res['roe'] = info.get('returnOnEquity')
            if res['roe'] is None or np.isnan(res['roe']):
                eps = info.get('trailingEps')
                if eps is not None and res['book_value'] and res['book_value'] > 0:
                    res['roe'] = float(eps) / float(res['book_value'])

            # 自由現金流收益率 (FCF Yield) = (Operating Cash Flow - CapEx) / MarketCap
            cf = stock.quarterly_cashflow if not stock.quarterly_cashflow.empty else stock.cashflow
            mcap = info.get('marketCap')
            if not cf.empty and mcap and mcap > 0:
                ocf_key = next((k for k in ['Operating Cash Flow', 'Total Cash From Operating Activities'] if k in cf.index), None)
                capex_key = next((k for k in ['Capital Expenditure', 'Capital Expenditures'] if k in cf.index), None)
                if ocf_key:
                    ocf = cf.loc[ocf_key].iloc[:4].sum()  # 近四季加總
                    capex = cf.loc[capex_key].iloc[:4].sum() if capex_key else 0
                    fcf = ocf + capex  # capex 通常為負值
                    res['fcf_yield'] = float(fcf) / float(mcap)

            # PEG 實算 = PE / Profit Growth Rate
            growth = info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth')
            if res['pe'] and growth and growth > 0:
                res['peg'] = float(res['pe']) / (float(growth) * 100)
                
        except Exception:
            pass
        return res

# ==============================================================================
# 3. 因子計算模組 (Factor Engine)
# ==============================================================================
class FactorEngine:
    @staticmethod
    def calculate_gvi(price: float, book_value: float, roe: float) -> float:
        """
        精算 GVI (Growth Value Indicator)
        邊界條件防護：
        - 淨值 <= 0 或 股價 <= 0：回傳 None
        - 1 + ROE <= 0：虧損極端，回傳 None
        """
        if price is None or book_value is None or roe is None:
            return None
        if price <= 0 or book_value <= 0:
            return None
        if (1 + roe) <= 0:
            return None
        
        try:
            gvi = (float(book_value) / float(price)) * ((1 + float(roe)) ** 5)
            return float(gvi) if not np.isnan(gvi) and not np.isinf(gvi) else None
        except Exception:
            return None

    @staticmethod
    def calculate_volume_momentum(df: pd.DataFrame) -> dict:
        """計算真實價量動能指標（非隨機生成籌碼）"""
        if df.empty or len(df) < 20:
            return {'vol_ma5_ratio': None, 'price_momentum_20d': None, 'status_text': "資料不足"}
        
        close = df['Close'].to_numpy()
        volume = df['Volume'].to_numpy()
        
        # 5日成交量 / 20日均量 比值
        vol_ma5 = np.mean(volume[-5:])
        vol_ma20 = np.mean(volume[-20:])
        vol_ratio = (vol_ma5 / vol_ma20) if vol_ma20 > 0 else 1.0
        
        # 20日價格動能
        p_mom = (close[-1] - close[-20]) / close[-20]
        
        if vol_ratio > 1.2 and p_mom > 0.03:
            status = "🔥 帶量突破增溫"
        elif vol_ratio < 0.8 and p_mom < -0.03:
            status = "📉 縮量陰跌壓制"
        else:
            status = "🔄 區間量價沉澱"
            
        return {
            'vol_ma5_ratio': round(vol_ratio, 2),
            'price_momentum_20d': round(p_mom * 100, 2),
            'status_text': status
        }

# ==============================================================================
# 4. 歷史回測引擎 (Backtest Engine Layer)
# ==============================================================================
class BacktestEngine:
    @staticmethod
    def run_strategy_backtest(df_price: pd.DataFrame, 
                              entry_signal: pd.Series, 
                              commission_rate=0.001425, 
                              tax_rate=0.003, 
                              slippage_rate=0.001) -> dict:
        """
        專業級單向多頭回測引擎
        含交易成本（手續費+印花稅）、滑價模型與回撤分析
        """
        if df_price.empty or len(df_price) < 30:
            return {}

        df = df_price.copy()
        df['signal'] = entry_signal.reindex(df.index).fillna(0)
        
        capital = 1000000.0  # 初始資金 100萬
        position = 0.0
        entry_price = 0.0
        
        equity_curve = []
        trades = []
        
        total_cost_factor_buy = 1.0 + commission_rate + slippage_rate
        total_cost_factor_sell = 1.0 - commission_rate - tax_rate - slippage_rate

        close_prices = df['Close'].to_numpy()
        signals = df['signal'].to_numpy()
        dates = df.index

        for i in range(len(df)):
            p = close_prices[i]
            sig = signals[i]
            
            # 持倉未買賣，更新資產
            if position > 0 and sig == 0:
                # 平倉訊號（例如訊號轉為 0）
                exit_price = p * total_cost_factor_sell
                revenue = position * exit_price
                capital += revenue
                trades.append((revenue - (position * entry_price)) / (position * entry_price))
                position = 0.0
            elif position == 0 and sig == 1:
                # 買進訊號
                buy_price = p * total_cost_factor_buy
                position = capital / buy_price
                entry_price = buy_price
                capital = 0.0

            # 計算當日市值
            current_equity = capital + (position * p * total_cost_factor_sell if position > 0 else 0)
            equity_curve.append(current_equity)

        equity_series = pd.Series(equity_curve, index=dates)
        
        # 績效指標計算
        total_return = (equity_series.iloc[-1] - 1000000.0) / 1000000.0
        num_years = max(len(df) / 252.0, 0.1)
        cagr = (equity_series.iloc[-1] / 1000000.0) ** (1.0 / num_years) - 1.0
        
        # 最大回撤 (Max Drawdown)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 勝率 (Win Rate)
        win_rate = np.mean([1 if r > 0 else 0 for r in trades]) if len(trades) > 0 else 0.0

        return {
            'equity_series': equity_series,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': len(trades)
        }

# ==============================================================================
# 5. 主應用程式與 UI 報表渲染 (Streamlit App)
# ==============================================================================
def main():
    st.sidebar.markdown("### 🔒 操盤手安全密碼鎖")
    input_password = st.sidebar.text_input("請輸入管理員密碼：", type="password")

    if input_password != "7770":
        st.title("🔒 華爾街機構級四核心量化終端")
        st.warning("⚠️ 請於左側邊欄輸入正確的管理員密碼 (7770) 以解鎖。")
        st.stop()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 全球個股即時診斷")
    selected_stock = st.sidebar.text_input("輸入個股代碼（如 2330.TW, 3293.TWO, NVDA）：", value="2330.TW").strip().upper()
    
    st.title("📈 機構級四核心量化策略雷達 4.0")
    st.caption("🛡️ 已完全排除偽數據生成，實現全鏈條真實基本面與歷史回測驗證")

    # 執行數據讀取與計算
    if selected_stock:
        f_data = DataFetcher.fetch_fundamentals(selected_stock)
        df_chart = DataFetcher.get_stock_raw_data(selected_stock)

        if not df_chart.empty:
            price = f_data['price'] if f_data['price'] else float(df_chart['Close'].iloc[-1])
            bv = f_data['book_value']
            roe = f_data['roe']
            gvi = FactorEngine.calculate_gvi(price, bv, roe)
            pb = (price / bv) if (price and bv and bv > 0) else None
            
            # 動能與指標
            mom_info = FactorEngine.calculate_volume_momentum(df_chart)

            # --- 指播看板 ---
            gc1, gc2, gc3, gc4, gc5 = st.columns(5)
            is_tw = ".TW" in selected_stock or ".TWO" in selected_stock
            
            with gc1: 
                st.metric("💰 當前現價", f"{price:,.2f} 元" if is_tw else f"${price:,.2f}")
            with gc2: 
                st.metric("👑 GVI 成長價值指標", f"{gvi:.4f}" if gvi else "N/A (邊界無效)")
            with gc3: 
                st.metric("📊 ROE (TTM)", f"{roe*100:.2f}%" if roe else "N/A")
            with gc4: 
                st.metric("📖 每股淨值 (BV)", f"{bv:,.2f}" if bv else "N/A")
            with gc5: 
                st.metric("⚖️ 股價淨值比 (PB)", f"{pb:.2f} 倍" if pb else "N/A")

            # 估值狀態判讀
            if gvi is not None and pb is not None:
                if gvi >= 0.35 and pb <= 1.5:
                    v_status, v_color = "🔥 極度便宜（高安全邊際）", "#00cc66"
                elif gvi >= 0.20 or (pb <= 3.5 and roe and roe >= 0.12):
                    v_status, v_color = "🟢 合理甜美區", "#2baf2b"
                elif pb > 3.5:
                    v_status, v_color = "⚠️ 偏貴溢價區", "#ff9900"
                else:
                    v_status, v_color = "🚨 泡沫偏離區", "#cc0000"
            else:
                v_status, v_color = "⚪ 數據缺失/淨值為負，無法計算判讀", "#888888"

            st.markdown(
                f"<div style='background-color:rgba(30,30,30,0.7); padding:12px 18px; border-left:6px solid {v_color}; border-radius:4px; margin:15px 0;'>"
                f"<h5 style='margin:0; color:white;'>⚖️ 華爾街估值狀態：<span style='color:{v_color};'>{v_status}</span></h5>"
                f"<p style='margin:4px 0 0 0; font-size:13px; color:#cccccc;'>💡 價量狀況：{mom_info['status_text']} | 20日漲跌幅：{mom_info['price_momentum_20d']}%</p>"
                f"</div>", 
                unsafe_allow_html=True
            )

            # --- 分頁：圖表與回測 ---
            tab1, tab2 = st.tabs(["📊 彩色 K 線與指標", "🧪 策略真實歷史回測引擎"])
            
            with tab1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                                             low=df_chart['Low'], close=df_chart['Close'], name="K線"), row=1, col=1)
                
                # MA 均線
                df_chart['MA20'] = df_chart['Close'].rolling(20).mean()
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1.5), name="20日均線"), row=1, col=1)
                
                # 成交量
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name="成交量", marker_color='gray'), row=2, col=1)
                fig.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.markdown("#### 🧪 20日均線突破歷史策略回測驗證")
                st.caption("模型設定：包含 0.1425% 手續費、0.3% 證券交易稅與 0.1% 單邊市場滑價模型")
                
                # 簡單均線突破邏輯作為範例策略訊號
                entry_sig = (df_chart['Close'] > df_chart['MA20']).astype(int)
                bt_res = BacktestEngine.run_strategy_backtest(df_chart, entry_sig)
                
                if bt_res:
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.metric("累積總報酬率", f"{bt_res['total_return']*100:.2f}%")
                    bc2.metric("年化報酬率 (CAGR)", f"{bt_res['cagr']*100:.2f}%")
                    bc3.metric("最大歷史回撤 (MDD)", f"{bt_res['max_drawdown']*100:.2f}%")
                    bc4.metric("交易勝率", f"{bt_res['win_rate']*100:.1f}% ({bt_res['trade_count']}次交易)")

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=bt_res['equity_series'].index, y=bt_res['equity_series'].values, mode='lines', name='權益曲線', line=dict(color='#00cc66', width=2)))
                    fig_bt.update_layout(title="淨值成長曲線 (Initial: $1,000,000)", height=350, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_bt, use_container_width=True)

if __name__ == "__main__":
    main()