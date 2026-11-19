"""Chart drawing controls and indicator definitions for the interface."""

import json
import pandas as pd
import streamlit.components.v1 as components


def render_plotly_delete_hotkey(st, chart_key, clear_revision=0):
    """Persist drawn Plotly shapes and keep erase controls working after reruns."""
    encoded_key = json.dumps(str(chart_key))
    encoded_revision = json.dumps(int(clear_revision))
    components.html(f"""
    <style>
      body {{ margin: 0; font-family: sans-serif; color: #173B2C; background: transparent; }}
      #shape-delete-panel {{ display: flex; align-items: center; gap: 8px; overflow-x: auto;
        padding: 8px 2px; white-space: nowrap; -webkit-overflow-scrolling: touch; }}
      #shape-delete-panel button {{ min-height: 42px; padding: 8px 12px; border-radius: 8px;
        border: 1px solid #CADCD1; background: #FFFFFF; color: #173B2C; font-size: 14px;
        cursor: pointer; touch-action: manipulation; }}
      #shape-delete-panel button:disabled {{ opacity: .55; cursor: default; }}
      #shape-delete-status {{ padding: 3px 2px; font-size: 13px; color: #486657; }}
    </style>
    <div id="shape-delete-status">標線刪除工具載入中…</div>
    <div id="shape-delete-panel"></div>
    <script>
    (() => {{
      try {{
        const hostWindow = window.parent;
        const hostDocument = hostWindow.document;
        const chartKey = {encoded_key};
        const clearRevision = {encoded_revision};
        hostWindow.__plotlyShapeDeleteKeys = hostWindow.__plotlyShapeDeleteKeys || {{}};
        const state = hostWindow.__plotlyShapeDeleteKeys[chartKey] || {{ overChart: false, graph: null }};
        hostWindow.__plotlyShapeDeleteKeys[chartKey] = state;
        const panel = document.getElementById('shape-delete-panel');
        const status = document.getElementById('shape-delete-status');
        const storageKey = 'codex-plotly-shapes:' + chartKey;
        const revisionKey = storageKey + ':revision';
        let storage = null;
        try {{ storage = hostWindow.localStorage; }} catch (_error) {{}}
        if (storage) {{
          try {{
            const previousRevision = storage.getItem(revisionKey);
            if (previousRevision !== String(clearRevision)) {{
              storage.removeItem(storageKey);
              storage.setItem(revisionKey, String(clearRevision));
            }}
          }} catch (_error) {{ storage = null; }}
        }}
        const findGraph = () => {{
          return Array.from(hostDocument.querySelectorAll('.js-plotly-plot'))
            .find((graph) => graph._fullLayout && graph._fullLayout.meta
              && graph._fullLayout.meta.codex_chart_key === chartKey) || null;
        }};
        const persistShapes = (graph) => {{
          if (!storage) return;
          try {{ storage.setItem(storageKey, JSON.stringify(graph.layout.shapes || [])); }} catch (_error) {{}}
        }};
        const renderShapeControls = () => {{
          if (!panel || !status) return;
          panel.replaceChildren();
          const shapes = state.graph && state.graph.layout ? (state.graph.layout.shapes || []) : [];
          status.textContent = shapes.length
            ? `目前有 ${{shapes.length}} 條標線；可直接按下方按鈕刪除，不必點選圖上細線。`
            : '目前沒有標線。';
          if (!(hostWindow.Plotly && hostWindow.Plotly.relayout)) {{
            if (shapes.length) status.textContent += ' 若按鈕無法使用，請用圖表工具列橡皮擦。';
            return;
          }}
          shapes.forEach((shape, index) => {{
            const button = document.createElement('button');
            const kind = shape.type || 'shape';
            button.textContent = `刪除第 ${{index + 1}} 條（${{kind}}）`;
            button.setAttribute('aria-label', `刪除第 ${{index + 1}} 條標線`);
            button.addEventListener('click', () => {{
              const nextShapes = (state.graph.layout.shapes || []).slice();
              nextShapes.splice(index, 1);
              hostWindow.Plotly.relayout(state.graph, {{ shapes: nextShapes }}).then(() => {{
                persistShapes(state.graph);
                renderShapeControls();
              }});
            }});
            panel.appendChild(button);
          }});
        }};
        state.renderShapeControls = renderShapeControls;
        const attachGraph = () => {{
          const graph = findGraph();
          if (!graph) {{ state.graph = null; return false; }}
          if (state.graph === graph) {{ renderShapeControls(); return true; }}
          state.graph = graph;
          if (storage) {{
            try {{
              const saved = storage.getItem(storageKey);
              if (saved !== null && hostWindow.Plotly && hostWindow.Plotly.relayout) {{
                const shapes = JSON.parse(saved);
                if (Array.isArray(shapes)) hostWindow.Plotly.relayout(graph, {{ shapes }});
              }}
            }} catch (_error) {{}}
          }}
          if (graph.on) graph.on('plotly_relayout', () => hostWindow.requestAnimationFrame(() => {{
            persistShapes(graph);
            if (state.renderShapeControls) state.renderShapeControls();
          }}));
          if (state.renderShapeControls) state.renderShapeControls();
          return true;
        }};
        if (!state.listenersAttached) {{
          state.listenersAttached = true;
          hostDocument.addEventListener('pointermove', (event) => {{
            const graph = state.graph || findGraph();
            if (!graph) {{ state.overChart = false; return; }}
            const bounds = graph.getBoundingClientRect();
            state.overChart = event.clientX >= bounds.left && event.clientX <= bounds.right
              && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
          }}, true);
          hostDocument.addEventListener('keydown', (event) => {{
            if (!['Delete', 'Backspace'].includes(event.key) || !state.overChart) return;
            const target = event.target;
            if (target && target.closest && target.closest('input, textarea, select, [contenteditable="true"]')) return;
            const graph = findGraph();
            if (graph) state.graph = graph;
            const eraseButton = graph && Array.from(graph.querySelectorAll('.modebar-btn')).find((button) => {{
              const label = [button.getAttribute('data-title'), button.getAttribute('aria-label'), button.getAttribute('title')]
                .filter(Boolean).join(' ').toLowerCase();
              return label.includes('erase') && label.includes('shape');
            }});
            if (eraseButton) {{ event.preventDefault(); eraseButton.click(); }}
          }}, true);
        }}
        if (!attachGraph()) {{
          const retry = hostWindow.setInterval(() => {{ if (attachGraph()) hostWindow.clearInterval(retry); }}, 150);
          hostWindow.setTimeout(() => hostWindow.clearInterval(retry), 10000);
        }}
      }} catch (_error) {{ /* Built-in Plotly eraser remains available if browser isolation blocks this shortcut. */ }}
    }})();
    </script>
    """, height=90, scrolling=False)


def _format_optional_cell(value, format_string):
    """Format a numeric table cell without failing when the result is missing."""
    try:
        if value is None or pd.isna(value):
            return "—"
        return format_string.format(value)
    except (TypeError, ValueError, OverflowError):
        return "—"


def _render_metric_glossary(st):
    """Show short, plain-language definitions for calculated fields across panels."""
    with st.expander("📘 指標與計算欄位怎麼看？", expanded=False):
        st.caption("點開可查各面板常見數值的意思與用途；指標是分析參考，不是獲利保證或單獨的買賣訊號。")
        sections = [
            ("回測與風險", [
                "**整段／累積報酬**：期末資產相對投入本金增加或減少多少；含息欄位會把股息計入。",
                "**年化報酬 CAGR**：把整段期間的複利結果換算成一年，方便比較不同長度的期間；不代表每年都會得到這個報酬。",
                "**最大回撤 MDD／最大跌幅**：歷史資產曲線從高點到後續低點的最深跌幅，用來了解曾承受的最差下跌。",
                "**年化波動率**：報酬上下起伏的幅度估算；波動大不等於一定虧損，也不代表未來風險上限。",
                "**夏普比率 Sharpe**：比較歷史報酬與整體資金起伏。數值越高，表示過去每承受一份整體起伏所換到的報酬較多；目前以 0% 無風險利率估算，不保證未來。",
                "**索提諾比率 Sortino**：著重比較報酬與下跌時的起伏；數值較高代表過去承受下跌起伏時的報酬相對較好，不是虧損機率。",
                "**卡瑪比率 Calmar**：把平均每年報酬和歷史最深跌幅放在一起看；過去報酬相對跌幅較高，不代表未來也能維持。",
                "**勝率／賺錢交易比例**：扣除設定成本後賺錢的交易占比；高勝率仍可能被少數大虧損抵銷。",
                "**賺賠金額比／獲利因子**：比較歷史獲利交易與虧損交易的總金額；大於 1 代表過去獲利總額較高，不能單獨代表策略可靠。",
                "**每筆平均損益／期望值**：每筆已完成交易平均賺或賠多少，通常以扣除回測成本後計算；筆數太少時容易受個別交易影響。",
                "**持有時間比例／曝險**：回測中持有股票的時間占比；其餘時間通常是現金。",
                "**持有 K 棒數**：從模擬進場到出場經過的圖表資料根數；不同圖表週期下，一根代表的時間不同。",
                "**成本後報酬率**：單筆模擬交易扣除設定手續費、交易稅及滑價後的損益比例；成本仍是估算值。",
                "**風險報酬比／目標停利價**：依進場價、停損距離及設定倍數推算的目標價；是模擬規則，不是獲利機率。",
                "**回撤期間**：資產低於前次高點的連續資料根數；不同圖表週期不宜直接比較根數。",
                "**心理提醒**：漂亮的回測成績容易讓人過度自信，虧損紀錄也容易讓人只想等回本；先看較晚期間、交易成本和最深跌幅，避免只挑支持自己想法的結果。",
            ]),
            ("價格、技術與選股指標", [
                "**均線乖離率**：現價相對指定期間平均價高或低多少百分比；用來觀察短期偏離，不是必然反轉訊號。",
                "**量比**：目前成交量與近期平均量相比的倍數；高於 1 代表量比平均活躍，不能判斷買賣方向。",
                "**價量動能估算**：綜合價格漲跌與成交量相對熱度，觀察趨勢參與度；只是價量參考，不是實際籌碼集中度、法人進出或持股變化。",
                "**RSI 相對強弱指標**：依近期上漲與下跌幅度計算的 0–100 動能指標；描述近期漲跌力道，不代表單獨的買賣訊號。",
                "**KD 隨機指標**：比較收盤價在近期高低區間的位置，K、D 線是平滑後的變化；主要描述短期動能。",
                "**MACD**：兩條不同速度的指數移動平均線之差；柱狀圖顯示 MACD 與訊號線的差距，用來觀察趨勢動能變化。",
                "**布林通道**：以移動平均線和近期波動範圍形成上下界；價格靠近通道邊緣表示位置偏向區間一側，不保證反轉。",
                "**ATR 平均真實波幅**：近期價格區間的平均大小，用來估算波動幅度；不表示漲跌方向。",
                "**黃金交叉／死亡交叉**：快均線向上穿越慢均線／向下跌破慢均線；是歷史價格訊號，可能落後且會反覆失效。",
                "**GVI 成長價值指標**：綜合帳面價值、股價與獲利能力估算的研究分數；依賴財報資料與假設，不是合理價或報酬預測。",
                "**ROE 股東權益報酬率**：公司用股東投入的權益創造盈餘的比例；須留意產業差異、財報期間與一次性項目。",
                "**每股淨值／PB 股價淨值比**：每股淨值是帳面資產扣除負債後分攤到每股的金額；PB 顯示股價相對帳面價值的高低，跨產業比較要小心。",
                "**動能／距近 252 根高點比例**：觀察現價離近一年可取得的最高收盤價有多近；越接近高點代表近期趨勢較強，也要留意追價風險。歷史資料少於 50 根時不評分。",
                "**自由現金流收益率**：比較公司創造的自由現金流相對於公司市值的大小；資料期間可能落後股價，也可能受一次性現金流影響。",
                "**PEG 本益成長比**：把公司估值與預期盈餘成長放在一起觀察；成長預估不可靠時，這個數字也會失真。",
                "**策略評分／排名**：依本次股票池和已開啟條件做相對排序，不是通過機率、預期報酬或買進建議。",
                "**心理提醒**：價格靠近高點時容易引發追價，快速下跌時容易引發恐慌；指標只提供觀察線索，不能直接量出所有投資人的情緒。",
            ]),
            ("配息、網格與法人欄位", [
                "**現金殖利率**：用近 12 個月實際發放的現金股利，對照目前股價觀察回饋規模；未來股利可能變動，不能視為固定利息。",
                "**股利發放率**：觀察近期股利相對公司最新可得盈餘的高低；高於 100% 可能表示股利超過當期盈餘，需查明原因。",
                "**配息品質分數**：依配息紀錄、盈餘、現金流、發放率、安全性與估值等目前啟用條件加權整理的 0–100 分；不是配息保證或未來報酬機率。",
                "**利息保障倍數**：營業獲利相對利息支出的倍數；越低通常代表支付利息的緩衝較小，並受資料口徑影響。",
                "**資料覆蓋率**：目前可取得的必要資料占比；缺漏越多，評分越不完整，不等於公司品質分數。",
                "**網格偏離現價**：每個觸發價與目前價格的百分比差距；觸發價是規劃參考，不是券商掛單。",
                "**定期定額累計投入**：模擬期間投入金額加總；和期末資產相比才能看投入後的損益。",
                "**佔股本比**：顯示法人當日買進、賣出或淨買賣超規模相對於公司已發行股數的大小；不是法人持股比例。淨買賣超表示買進與賣出相抵後的方向和數量。",
                "**心理提醒**：高殖利率容易讓人只注意現金入帳、忽略價格與配息能否維持；法人買超也容易引發跟隨心理，兩者都需要搭配公司狀況和多日資料。",
            ]),
            ("投資組合欄位", [
                "**目前配置比例**：顯示各類資產占整個組合的比重；價格與匯率更新狀態會影響結果。",
                "**配置偏離（百分點）**：顯示目前資產比重比目標高或低多少；例如目前 45%、目標 40%，代表高出 5 個百分點。",
                "**調整金額參考**：估算回到目標配置所需的金額差；正數代表相對不足、負數代表相對超出，未扣交易成本或稅費。",
                "**未實現損益估算**：比較目前估值和記錄成本後的帳面差額；不含完整稅費、已領股息或所有外部資金流。",
                "**快照估算回撤**：手動記錄的組合市值與曾記錄高點相比的變化；未調整入金／出金，只能粗略參考。",
                "**最低保留現金**：低於設定金額時顯示提醒；系統不會因此自動買賣或更改配置建議。",
                "**心理提醒**：買入成本常成為心理錨點，虧損時可能不願重新檢視；依事前訂好的配置和風險門檻檢查，能減少臨時追高或恐慌賣出。",
            ]),
        ]
        for heading, definitions in sections:
            st.markdown(f"**{heading}**")
            st.markdown("\n".join(f"- {definition}" for definition in definitions))
