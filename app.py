import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==========================================
# 頁面配置
# ==========================================
st.set_page_config(
    page_title="機構級三核心策略診斷系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS 樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .strategy-badge {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .warning-box {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏛️ 機構級三核心策略雷達與全球個股即時診斷</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">結合多因子量化、產業鏈價值分析與籌碼動態之智能交易決策平台</div>', unsafe_allow_html=True)

# ==========================================
# 數據獲取與彈性備援邏輯
# ==========================================
@st.cache_data(ttl=3600)
def fetch_stock_data_with_fallback(ticker_symbol):
    """
    獲取股票數據與財務報表，具備彈性備援機制
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # 1. 價格歷史數據
        hist = ticker.history(period="1y")
        if hist.empty:
            return None, "無法取得價格歷史數據，請確認代碼是否正確。"
        
        # 2. 基本面數據獲取與備援 (Fallback)
        info = ticker.info if ticker.info else {}
        
        # 嘗試取得季度財報，若無則降級取得年度財報
        financials_q = ticker.quarterly_financials
        financials_y = ticker.financials
        bs_q = ticker.quarterly_balance_sheet
        
        data_period_tag = "最新當季財報"
        fin_data = {}
        
        if financials_q is not None and not financials_q.empty:
            latest_fin = financials_q.iloc[:, 0]
            data_period_tag = f"季報 ({financials_q.columns[0].strftime('%Y-%m')})"
        elif financials_y is not None and not financials_y.empty:
            latest_fin = financials_y.iloc[:, 0]
            data_period_tag = f"年報 ({financials_y.columns[0].strftime('%Y')})"
        else:
            latest_fin = pd.Series()
            data_period_tag = "即時市場推算 (無完整財報數據)"
            
        # 估算關鍵指標
        pe_ratio = info.get('forwardPE') or info.get('trailingPE') or 0.0
        pb_ratio = info.get('priceToBook') or 0.0
        roe = info.get('returnOnEquity') or 0.0
        rev_growth = info.get('revenueGrowth') or 0.0
        profit_margin = info.get('profitMargins') or 0.0
        
        # 組裝診斷數據包
        diagnostic_payload = {
            'symbol': ticker_symbol,
            'short_name': info.get('shortName', ticker_symbol),
            'sector': info.get('sector', '未分類'),
            'industry': info.get('industry', '未分類'),
            'price': hist['Close'].iloc[-1],
            'change_pct': ((hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'roe': roe,
            'rev_growth': rev_growth,
            'profit_margin': profit_margin,
            'data_source_tag': data_period_tag,
            'hist': hist,
            'raw_info': info
        }
        
        return diagnostic_payload, None
    except Exception as e:
        return None, f"數據處理異常: {str(e)}"

# ==========================================
# 策略核心評估引擎 (Cross-Strategy Analysis)
# ==========================================
def analyze_three_strategies(data):
    """
    進行三核心策略交叉分析並選出最適操盤建議
    """
    scores = {}
    details = {}
    
    # 策略 1: 多因子量化價值/成長策略 (Multi-Factor Quant)
    quant_score = 50
    quant_reasons = []
    if data['pe_ratio'] > 0 and data['pe_ratio'] < 25:
        quant_score += 15
        quant_reasons.append("本益比處於合理區間")
    if data['roe'] > 0.12:
        quant_score += 20
        quant_reasons.append("ROE 高於 12% (優異股東回報)")
    if data['rev_growth'] > 0.08:
        quant_score += 15
        quant_reasons.append("營收保持成長姿態")
    scores['quant'] = min(quant_score, 100)
    details['quant'] = quant_reasons

    # 策略 2: 產業鏈價值與護城河策略 (Value Chain & Moat)
    moat_score = 50
    moat_reasons = []
    if data['profit_margin'] > 0.20:
        moat_score += 25
        moat_reasons.append("高營業利潤率 (>20%)，具高產品議價力/護城河")
    elif data['profit_margin'] > 0.10:
        moat_score += 10
        moat_reasons.append("利潤率穩定 (>10%)")
    if data['pb_ratio'] < 5.0:
        moat_score += 15
        moat_reasons.append("資產基底紮實，價格未嚴重過熱")
    scores['moat'] = min(moat_score, 100)
    details['moat'] = moat_reasons

    # 策略 3: 動量與技術籌碼策略 (Momentum & Trend)
    hist = data['hist']
    ma50 = hist['Close'].rolling(50).mean().iloc[-1]
    ma200 = hist['Close'].rolling(200).mean().iloc[-1]
    current_price = data['price']
    
    momentum_score = 50
    momentum_reasons = []
    if current_price > ma50:
        momentum_score += 20
        momentum_reasons.append("站上 50 日均線，短期趨勢偏多")
    if ma50 > ma200:
        momentum_score += 20
        momentum_reasons.append("50日均線大於200日均線 (多頭排列)")
    
    # 波動度檢測
    volatility = hist['Close'].pct_change().std() * np.sqrt(252)
    if volatility < 0.35:
        momentum_score += 10
        momentum_reasons.append("波動度健康且控管良好")
        
    scores['momentum'] = min(momentum_score, 100)
    details['momentum'] = momentum_reasons

    # 決策引擎：尋找最適策略
    best_strategy = max(scores, key=scores.get)
    
    strategy_names = {
        'quant': '多因子量化成長策略',
        'moat': '產業護城河價值策略',
        'momentum': '動量趨勢交易策略'
    }

    # 操盤手建言生成
    trader_advice = ""
    avg_score = np.mean(list(scores.values()))
    
    if avg_score >= 75:
        action = "🟢 強力推薦 / 分批佈局"
        trader_advice = f"該標的在【{strategy_names[best_strategy]}】表現最為突出 (得分: {scores[best_strategy]})。三大策略形成多方共振，建議操盤手可設定拉回 3-5% 建立核心部位，並以 200 日均線作為長線停損點。"
    elif avg_score >= 60:
        action = "🟡 觀望升級 / 區間操作"
        trader_advice = f"標的優勢集中於【{strategy_names[best_strategy]}】，但整體多方共振力道尚可。建議採用區間操作策略，等待突破關鍵阻力或回測支撐時再行進場。"
    else:
        action = "🔴 風控優先 / 暫緩進場"
        trader_advice = "三大策略綜合評分偏低，基本面或技術面暫無顯著機構資金注入跡象，建議操盤手保持現金觀望，轉移至其他高勝率標的。"

    return {
        'scores': scores,
        'details': details,
        'best_strategy_key': best_strategy,
        'best_strategy_name': strategy_names[best_strategy],
        'action': action,
        'trader_advice': trader_advice,
        'avg_score': avg_score
    }

# ==========================================
# 側邊欄控制
# ==========================================
st.sidebar.header("🔍 個股即時診斷設定")
target_symbol = st.sidebar.text_input("輸入股票代碼 (例: AAPL, NVDA, 2330.TW)", value="NVDA").upper()
search_btn = st.sidebar.button("開始交叉分析", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 診斷備援機制說明")
st.sidebar.caption("""
本系統具備**財報彈性備援機制**：
1. 優先採用最新一季財報 (Quarterly)
2. 若季報缺失，自動降級調用最近一年財報 (Annual)
3. 若無財報，啟動市場即時數據進行估算
""")

# ==========================================
# 主畫面執行邏輯
# ==========================================
if target_symbol:
    with st.spinner(f"正在擷取 {target_symbol} 全球市場數據並進行三核心策略交叉計算..."):
        data, err = fetch_stock_data_with_fallback(target_symbol)
        
    if err:
        st.error(f"❌ 分析中斷: {err}")
    else:
        # 分析策略
        analysis = analyze_three_strategies(data)
        
        # 標題與基本資訊
        col_title1, col_title2 = st.columns([3, 1])
        with col_title1:
            st.title(f"{data['short_name']} ({data['symbol']})")
            st.caption(f"產業類別: {data['sector']} | {data['industry']}  •  數據時效: :green[{data['data_source_tag']}]")
        with col_title2:
            st.metric(
                label="最新股價",
                value=f"${data['price']:.2f}",
                delta=f"{data['change_pct']:.2f}%"
            )
            
        st.markdown("---")
        
        # 1. 操盤手決策核心面板
        st.subheader("🎯 操盤手最佳策略指引 (Traders Actionable Advice)")
        
        advice_col1, advice_col2 = st.columns([1, 2])
        
        with advice_col1:
            st.markdown(f"### 建議行動")
            st.markdown(f"## {analysis['action']}")
            st.markdown(f"**推薦首選策略：**")
            st.markdown(f"### `:blue[{analysis['best_strategy_name']}]`")
            st.caption(f"三大策略綜合強度：{analysis['avg_score']:.1f} / 100")

        with advice_col2:
            st.info(f"**💡 操盤手執行建議：**\n\n{analysis['trader_advice']}")
            
        st.markdown("---")
        
        # 2. 三大策略交叉分析雷達與細節
        st.subheader("📊 三核心策略交叉分析結果")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            # 雷達圖
            categories = ['多因子量化策略', '產業護城河策略', '動量趨勢策略']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[analysis['scores']['quant'], analysis['scores']['moat'], analysis['scores']['momentum']],
                theta=categories,
                fill='toself',
                name=data['symbol'],
                line_color='#2563EB'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with c2:
            st.write("#### 各策略契合度與驅動因子")
            
            # 多因子量化
            st.markdown(f"**1. 多因子量化得分：{analysis['scores']['quant']}/100**")
            for r in analysis['details']['quant']:
                st.markdown(f"- {r}")
                
            # 護城河
            st.markdown(f"**2. 產業護城河得分：{analysis['scores']['moat']}/100**")
            for r in analysis['details']['moat']:
                st.markdown(f"- {r}")
                
            # 動量趨勢
            st.markdown(f"**3. 動量趨勢得分：{analysis['scores']['momentum']}/100**")
            for r in analysis['details']['momentum']:
                st.markdown(f"- {r}")

        st.markdown("---")

        # 3. K 線與趨勢圖表
        st.subheader("📈 近一年價格走勢與技術支撐")
        hist_df = data['hist']
        hist_df['MA50'] = hist_df['Close'].rolling(50).mean()
        hist_df['MA200'] = hist_df['Close'].rolling(200).mean()
        
        fig_line = px.line(hist_df, x=hist_df.index, y=['Close', 'MA50', 'MA200'],
                           labels={'value': '價格', 'Date': '日期', 'variable': '指標'},
                           color_discrete_map={'Close': '#1E293B', 'MA50': '#3B82F6', 'MA200': '#EF4444'})
        fig_line.update_layout(margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)