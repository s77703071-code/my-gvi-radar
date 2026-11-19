"""Interactive price charts, overlays and drawing tools."""

from plotly.subplots import make_subplots
import html
import plotly.graph_objects as go
import streamlit as st

from .backtest import (
    compute_kline_indicator_frame,
)
from .chart_tools import (
    render_plotly_delete_hotkey,
)
from .dividend import (
    _finite_number,
)
from .explanations import (
    build_price_volume_market_psychology,
)


def render_chart(auth_user, c_name, change_pct, date_strings, df_chart, personal_ma_configs, selected_chart_indicators, selected_chart_overlays, selected_stock, selected_tf, vol_ratio):
    ma_header_values = []
    for ma in personal_ma_configs:
        p, c = ma["period"], ma["color"]
        ma_series = df_chart['Close'].rolling(window=p).mean().dropna()
        if not ma_series.empty:
            latest_ma_val = ma_series.to_numpy().flatten()[-1]
            ma_header_values.append((int(p), str(c), float(latest_ma_val)))
    chart_top_margin = 72
    st.markdown(f"### 📊 【{c_name}】專業 K 線圖與成交量")
    selected_indicator_text = ", ".join(selected_chart_indicators) if selected_chart_indicators else "未選擇"
    selected_overlay_text = ", ".join(selected_chart_overlays) if selected_chart_overlays else "未選擇"
    st.caption(
        f"目前 K 線時間軸：{selected_tf}；副圖：{selected_indicator_text}；"
        f"圖上疊加：{selected_overlay_text}。各指標週期依目前 K 線級別計算。"
    )
    st.info("🧠 市場行為觀察：" + build_price_volume_market_psychology(change_pct, vol_ratio, selected_tf))

    chart_window_options = {"最近 60 根": 60, "最近 120 根": 120, "最近 240 根": 240, "全部資料": None}
    chart_window_key = f"chart_window_{auth_user['id']}_{selected_stock}_{selected_tf}"
    chart_window_label = st.selectbox(
        "圖表顯示範圍",
        options=list(chart_window_options),
        index=1,
        key=chart_window_key,
        help="只調整畫面顯示範圍，不會刪除歷史資料、均線計算或標線。",
    )
    requested_bars = chart_window_options[chart_window_label]
    visible_bars = min(len(df_chart), requested_bars) if requested_bars else len(df_chart)
    chart_x_range = [len(df_chart) - visible_bars - 0.5, len(df_chart) - 0.5]
    st.caption(
        f"目前顯示最近 {visible_bars} 根 K 棒；均線仍用完整資料計算。"
        "縮放／移動用滑鼠或觸控操作；畫線在圖表右上角，刪除請用圖表下方的標線按鈕。"
    )

    with st.expander("🧭 K 線圖快速操作指南", expanded=False):
        st.markdown(
            "- **縮放與移動：**電腦用滑鼠滾輪縮放、按住拖曳平移；手機用雙指縮放、單指拖曳。\n"
            "- **畫線或標記：**使用圖表右上角工具列；也可以用下方文字標記輸入文字與顏色。\n"
            "- **刪除標記：**圖表下方可刪除線條；文字標記可在文字設定中個別刪除；清除全部會一併移除。"
        )

    with st.expander("🎨 畫線顏色與粗細設定", expanded=False):
        draw_color_col, draw_width_col = st.columns(2)
        with draw_color_col:
            draw_color = st.color_picker("自訂畫線／文字顏色", value="#FF3333", key="draw_line_color_picker")
        with draw_width_col:
            draw_line_width = st.slider("線條粗細", min_value=0.5, max_value=10.0, value=2.5, step=0.5, key=f"draw_line_width_{selected_stock}",
                                        help="設定接下來繪製的趨勢線、矩形與圓形邊框粗細。")
    user_chart_notes_key = f"chart_text_annotations_{auth_user['id']}_{selected_stock}_{selected_tf}"
    chart_notes = st.session_state.get(user_chart_notes_key, [])
    if not isinstance(chart_notes, list):
        chart_notes = []
    st.session_state[user_chart_notes_key] = chart_notes
    with st.expander("📝 輸入文字與顏色標記", expanded=False):
        chart_note_text = st.text_input(
            "輸入要顯示在圖表上的文字",
            key=f"chart_note_text_{auth_user['id']}_{selected_stock}_{selected_tf}",
            max_chars=80,
        )
        chart_note_color = st.color_picker(
            "文字顏色",
            value="#FBBF24",
            key=f"chart_note_color_{auth_user['id']}_{selected_stock}_{selected_tf}",
        )
        if st.button(
            "➕ 加到最新一根 K 棒附近",
            key=f"add_chart_note_{auth_user['id']}_{selected_stock}_{selected_tf}",
            use_container_width=True,
        ):
            if chart_note_text.strip():
                chart_notes.append({"text": chart_note_text.strip(), "color": chart_note_color})
                st.session_state[user_chart_notes_key] = chart_notes
                st.success("文字標記已加入；可在圖上拖曳調整位置。")
            else:
                st.warning("請先輸入要顯示的文字。")
        if chart_notes:
            st.caption("目前文字標記：")
            for note_index, note in enumerate(list(chart_notes)):
                note_col, delete_col = st.columns([0.78, 0.22])
                with note_col:
                    st.caption(f"{note_index + 1}. {note.get('text', '')}｜{note.get('color', '')}")
                with delete_col:
                    if st.button(
                        "刪除",
                        key=f"delete_chart_note_{auth_user['id']}_{selected_stock}_{selected_tf}_{note_index}",
                        use_container_width=True,
                    ):
                        chart_notes.pop(note_index)
                        st.session_state[user_chart_notes_key] = chart_notes
    clear_shapes = st.button(
        "🧹 清除全部畫線與文字標記",
        key=f"clear_shapes_{auth_user['id']}_{selected_stock}_{selected_tf}",
        use_container_width=True,
    )
    if clear_shapes:
        clear_key = f"shape_clear_revision_{auth_user['id']}_{selected_stock}_{selected_tf}"
        st.session_state[clear_key] = int(st.session_state.get(clear_key, 0)) + 1
        chart_notes.clear()
        st.session_state[user_chart_notes_key] = chart_notes

    # 主圖、成交量與使用者選擇的技術指標各自使用獨立面板。
    chart_rows = 2 + len(selected_chart_indicators)
    if selected_chart_indicators:
        row_weight_total = 500 + 145 * len(selected_chart_indicators)
        chart_row_heights = (
            [390 / row_weight_total, 110 / row_weight_total]
            + [145 / row_weight_total] * len(selected_chart_indicators)
        )
        chart_height = 520 + 145 * len(selected_chart_indicators)
    else:
        chart_row_heights = [0.75, 0.25]
        chart_height = 520

    fig = make_subplots(
        rows=chart_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=chart_row_heights,
    )

    # 1️⃣ 上半部：主圖 Candlestick K棒
    fig.add_trace(go.Candlestick(
        x=date_strings,
        open=df_chart['Open'].to_numpy().flatten().tolist(),
        high=df_chart['High'].to_numpy().flatten().tolist(),
        low=df_chart['Low'].to_numpy().flatten().tolist(),
        close=df_chart['Close'].to_numpy().flatten().tolist(),
        name="K線",
        increasing_line_color='#ef5350', # 漲紅
        decreasing_line_color='#26a69a', # 跌綠
        hovertext=[f"日期：{d}" for d in date_strings]
    ), row=1, col=1)

    # 均線繪製
    for ma in personal_ma_configs:
        p, c = ma["period"], ma["color"]
        ma_series = df_chart['Close'].rolling(window=p).mean().dropna()
        if not ma_series.empty:
            ma_list = ma_series.to_numpy().flatten().tolist()
            fig.add_trace(go.Scatter(
                x=date_strings[-len(ma_list):],
                y=ma_list,
                mode='lines',
                name=f'{p}MA',
                line=dict(color=c, width=1.5)
            ), row=1, col=1)

    if "布林通道（20，±2 標準差）" in selected_chart_overlays:
        bb_middle = df_chart["Close"].rolling(window=20, min_periods=20).mean()
        bb_std = df_chart["Close"].rolling(window=20, min_periods=20).std()
        bb_lower = bb_middle - 2 * bb_std
        bb_upper = bb_middle + 2 * bb_std
        fig.add_trace(go.Scatter(
            x=date_strings, y=bb_lower.to_numpy().flatten().tolist(),
            mode="lines", name="布林下軌（20）",
            line=dict(color="rgba(96,165,250,0.75)", width=1, dash="dot"),
            connectgaps=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=date_strings, y=bb_upper.to_numpy().flatten().tolist(),
            mode="lines", name="布林上軌（20）",
            line=dict(color="rgba(96,165,250,0.85)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(96,165,250,0.10)",
            connectgaps=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=date_strings, y=bb_middle.to_numpy().flatten().tolist(),
            mode="lines", name="布林中線（20）",
            line=dict(color="rgba(96,165,250,0.95)", width=1.2),
            connectgaps=False,
        ), row=1, col=1)

    for ema_period, ema_label, ema_color in (
        (12, "EMA 12", "#F59E0B"),
        (26, "EMA 26", "#38BDF8"),
    ):
        if ema_label in selected_chart_overlays:
            ema_series = df_chart["Close"].ewm(span=ema_period, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=date_strings, y=ema_series.to_numpy().flatten().tolist(),
                mode="lines", name=ema_label,
                line=dict(color=ema_color, width=1.7),
            ), row=1, col=1)

    # 2️⃣ 下半部：副圖 成交量 Bar 柱狀圖
    open_arr = df_chart['Open'].to_numpy().flatten()
    close_arr = df_chart['Close'].to_numpy().flatten()
    vol_colors = ['#ef5350' if c >= o else '#26a69a' for o, c in zip(open_arr, close_arr)]

    fig.add_trace(go.Bar(
        x=date_strings,
        y=df_chart['Volume'].to_numpy().flatten().tolist(),
        name="成交量",
        marker_color=vol_colors,
        opacity=0.8
    ), row=2, col=1)

    if "成交量均線（20）" in selected_chart_overlays:
        volume_ma20 = df_chart["Volume"].rolling(window=20, min_periods=20).mean()
        fig.add_trace(go.Scatter(
            x=date_strings, y=volume_ma20.to_numpy().flatten().tolist(),
            mode="lines", name="成交量均線（20）",
            line=dict(color="#A78BFA", width=1.7),
        ), row=2, col=1)

    if selected_chart_indicators:
        indicator_frame = compute_kline_indicator_frame(df_chart)
        for indicator_index, indicator_name in enumerate(selected_chart_indicators):
            indicator_row = indicator_index + 3
            if indicator_name == "KD":
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame['K'].to_numpy().flatten().tolist(),
                    mode='lines', name='K', line=dict(color='#F59E0B', width=1.7),
                ), row=indicator_row, col=1)
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame['D'].to_numpy().flatten().tolist(),
                    mode='lines', name='D', line=dict(color='#38BDF8', width=1.7),
                ), row=indicator_row, col=1)
                for threshold in (20, 80):
                    fig.add_hline(
                        y=threshold, row=indicator_row, col=1,
                        line=dict(color='rgba(148,163,184,0.45)', dash='dot', width=1),
                    )
                fig.update_yaxes(title_text="KD（K／D）", range=[0, 100], row=indicator_row, col=1)
            elif indicator_name == "MACD":
                macd_histogram = indicator_frame['DIF'] - indicator_frame['DEM']
                macd_colors = [
                    '#ef5350' if value >= 0 else '#26a69a'
                    for value in macd_histogram.fillna(0).to_numpy().flatten()
                ]
                fig.add_trace(go.Bar(
                    x=date_strings, y=macd_histogram.to_numpy().flatten().tolist(),
                    name='MACD 柱狀', marker_color=macd_colors, opacity=0.55,
                ), row=indicator_row, col=1)
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame['DIF'].to_numpy().flatten().tolist(),
                    mode='lines', name='DIF', line=dict(color='#F59E0B', width=1.6),
                ), row=indicator_row, col=1)
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame['DEM'].to_numpy().flatten().tolist(),
                    mode='lines', name='訊號線', line=dict(color='#38BDF8', width=1.6),
                ), row=indicator_row, col=1)
                fig.add_hline(
                    y=0, row=indicator_row, col=1,
                    line=dict(color='rgba(148,163,184,0.55)', dash='dot', width=1),
                )
                fig.update_yaxes(title_text="MACD", row=indicator_row, col=1)
            elif indicator_name == "RSI":
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame['RSI'].to_numpy().flatten().tolist(),
                    mode='lines', name='RSI(14)', line=dict(color='#A78BFA', width=1.8),
                ), row=indicator_row, col=1)
                for threshold in (30, 50, 70):
                    fig.add_hline(
                        y=threshold, row=indicator_row, col=1,
                        line=dict(
                            color='rgba(148,163,184,0.45)',
                            dash='dot' if threshold != 50 else 'dash',
                            width=1,
                        ),
                    )
                fig.update_yaxes(title_text="RSI（14）", range=[0, 100], row=indicator_row, col=1)
            elif indicator_name == "BIAS（乖離率）":
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame["BIAS"].to_numpy().flatten().tolist(),
                    mode="lines", name="BIAS（20）", line=dict(color="#34D399", width=1.8),
                ), row=indicator_row, col=1)
                for threshold in (-5, 0, 5):
                    fig.add_hline(
                        y=threshold, row=indicator_row, col=1,
                        line=dict(color="rgba(148,163,184,0.45)", dash="dot", width=1),
                    )
                fig.update_yaxes(title_text="BIAS（20，%）", ticksuffix="%", row=indicator_row, col=1)
            elif indicator_name == "ATR（波動幅度）":
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame["ATR"].to_numpy().flatten().tolist(),
                    mode="lines", name="ATR（14）", line=dict(color="#FB923C", width=1.8),
                ), row=indicator_row, col=1)
                fig.update_yaxes(title_text="ATR（14）", row=indicator_row, col=1)
            elif indicator_name == "OBV（能量潮）":
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame["OBV"].to_numpy().flatten().tolist(),
                    mode="lines", name="OBV", line=dict(color="#22D3EE", width=1.8),
                ), row=indicator_row, col=1)
                fig.update_yaxes(title_text="OBV（累積量）", row=indicator_row, col=1)
            elif indicator_name == "CCI（順勢指標）":
                fig.add_trace(go.Scatter(
                    x=date_strings, y=indicator_frame["CCI"].to_numpy().flatten().tolist(),
                    mode="lines", name="CCI（20）", line=dict(color="#C084FC", width=1.8),
                ), row=indicator_row, col=1)
                for threshold in (-100, 0, 100):
                    fig.add_hline(
                        y=threshold, row=indicator_row, col=1,
                        line=dict(color="rgba(148,163,184,0.45)", dash="dot", width=1),
                    )
                fig.update_yaxes(title_text="CCI（20）", row=indicator_row, col=1)
            elif indicator_name == "ADX（趨勢強度）":
                for column_name, line_name, line_color in (
                    ("ADX", "ADX（趨勢強度）", "#F59E0B"),
                    ("PLUS_DI", "+DI（上升方向）", "#34D399"),
                    ("MINUS_DI", "−DI（下降方向）", "#F87171"),
                ):
                    fig.add_trace(go.Scatter(
                        x=date_strings, y=indicator_frame[column_name].to_numpy().flatten().tolist(),
                        mode="lines", name=line_name, line=dict(color=line_color, width=1.6),
                    ), row=indicator_row, col=1)
                fig.add_hline(
                    y=20, row=indicator_row, col=1,
                    line=dict(color="rgba(148,163,184,0.55)", dash="dot", width=1),
                )
                fig.update_yaxes(title_text="DMI／ADX", range=[0, 100], row=indicator_row, col=1)
            fig.update_yaxes(
                side="right", showgrid=True, gridcolor="rgba(128,128,128,0.2)",
                row=indicator_row, col=1,
            )

    # 🚀 排版設定與畫線/文字預設色彩配置
    chart_x_axes = {}
    bottom_axis_context = selected_chart_indicators[-1] if selected_chart_indicators else "成交量"
    time_axis_title = f"週期：{selected_tf}｜{bottom_axis_context}"
    for axis_index in range(1, chart_rows + 1):
        axis_name = "xaxis" if axis_index == 1 else f"xaxis{axis_index}"
        chart_x_axes[axis_name] = dict(
            type='category',
            range=chart_x_range,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
            nticks=6,
            tickangle=-30,
            tickfont=dict(size=10),
            automargin=True,
            title=dict(text=time_axis_title if axis_index == chart_rows else ""),
        )
    fig.update_layout(
        title=dict(
            text=f"【{html.escape(str(c_name), quote=False)}】K 線圖｜{selected_tf}",
            x=0.01, xanchor="left", y=0.99, yanchor="top",
            font=dict(size=16, color="#173B2C"),
            subtitle=dict(text=""),
        ),
        xaxis_rangeslider_visible=False,
        uirevision=(
            f"chart-{selected_stock}-{selected_tf}-{chart_window_label}-"
            f"{','.join(selected_chart_indicators)}-{','.join(selected_chart_overlays)}"
        ),
        editrevision=int(st.session_state.get(f"shape_clear_revision_{auth_user['id']}_{selected_stock}_{selected_tf}", 0)),
        height=chart_height,
            margin=dict(l=12, r=12, t=chart_top_margin, b=54),
        dragmode='pan',
        showlegend=False,
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", side="right", title="股價（元）"),
        yaxis2=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)", side="right", title="成交量"),
        newshape=dict(
            line=dict(color=draw_color, width=draw_line_width),
            fillcolor=draw_color,
            opacity=0.6
        ),
        annotationdefaults=dict(
            font=dict(color=draw_color, size=14),
            bgcolor="rgba(15,23,42,0.88)",
            bordercolor=draw_color,
            borderwidth=1,
            borderpad=4,
            showarrow=False,
        ),
        **chart_x_axes
    )
    close_for_note = _finite_number(df_chart["Close"].iloc[-1])
    if close_for_note is not None and date_strings:
        note_y_offsets = [0, 20, -20, 40, -40, 60, -60]
        for note_index, note in enumerate(chart_notes):
            note_text = str(note.get("text", "")).strip()
            if not note_text:
                continue
            note_color = str(note.get("color", "#FBBF24"))
            fig.add_annotation(
                x=date_strings[-1], y=close_for_note,
                xref="x", yref="y",
                xanchor="right", yanchor="bottom", xshift=-8,
                yshift=note_y_offsets[note_index % len(note_y_offsets)],
                text=html.escape(note_text, quote=False),
                showarrow=False,
                font=dict(color=note_color, size=14),
                bgcolor="rgba(15,23,42,0.88)",
                bordercolor=note_color, borderwidth=1, borderpad=4,
            )

    chart_key = f"candlestick_chart_{auth_user['id']}_{selected_stock}_{selected_tf}"
    clear_revision = int(st.session_state.get(f"shape_clear_revision_{auth_user['id']}_{selected_stock}_{selected_tf}", 0))
    fig.update_layout(meta={"codex_chart_key": chart_key})
    if ma_header_values:
        ma_header_html = "".join(
            f"<span style=\"color:{html.escape(color, quote=True)};font-weight:700;white-space:nowrap;\">■ MA{period}: {value:,.2f}</span>"
            for period, color, value in ma_header_values
        )
        st.markdown(
            "<div style=\"display:flex;flex-wrap:nowrap;gap:16px;align-items:center;"
            "overflow-x:auto;white-space:nowrap;padding:7px 10px;margin:2px 0 6px;"
            "border:1px solid rgba(148,163,184,.35);border-radius:8px;"
            "background:rgba(15,23,42,.55);font-size:.92rem;\">"
            f"{ma_header_html}</div>",
            unsafe_allow_html=True,
        )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=chart_key,
        config={
            'modeBarButtonsToAdd': [
                'drawline',       # 直線工具
                'drawopenpath',   # 畫筆工具
                'drawrect',       # 矩形
                'drawcircle',     # 圓形
                'drawtext',       # 📝 文字標記工具
                'eraseshape'      # 🧹 橡皮擦擦除工具
            ],
            'displayModeBar': True, # 強制固定顯示右上角 ModeBar 工具列
            'displaylogo': False,
            'editable': True,
            # Keep drawing/shape dragging, but never expose fixed chart
            # labels or annotation source text as editable prompts.
            'edits': {
                'shapePosition': True,
                'annotationText': False,
                'axisTitleText': False,
                'titleText': False,
                'legendText': False,
                'colorbarTitleText': False,
            },
            'scrollZoom': True,
            'responsive': True
        }
    )
    render_plotly_delete_hotkey(st, chart_key, clear_revision)
    st.caption(
        "線條可先點選邊界再按工具列橡皮擦，電腦也可在圖表內按 Delete；"
        "新增文字可在文字設定中個別刪除。清除全部會移除畫線與輸入式文字標記。"
    )
