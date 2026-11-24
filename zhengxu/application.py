"""Streamlit startup, navigation and authenticated routing."""

from pathlib import Path
import os
import re
import streamlit as st

from .account_view import (
    render_account_gate,
)
from .accounts import (
    _confirm_recovery_email,
    _request_account_code,
)
from .administration import (
    _dual_grid_management_context,
    _render_dual_grid_management,
    _require_dual_grid_access,
)
from .dividend import reset_dividend_source_cache
from .module_registry import MODULE_GROUPS
from .panel import (
    render_analysis,
)
from .storage import (
    init_db,
)
from .workbench import (
    _close_workbench_on_module_change,
    _open_workbench_module,
    _render_investment_workbench,
)


def run():
    st.set_page_config(page_title="正序｜投資研究與管理", layout="wide", page_icon="📈")
    reset_dividend_source_cache()
    theme_path = Path(__file__).resolve().parents[1] / "assets" / "zhengxu.css"
    if theme_path.is_file():
        st.markdown(
            f'<style class="zhengxu-theme">{theme_path.read_text(encoding="utf-8")}</style>',
            unsafe_allow_html=True,
        )
    init_db()
    auth_user = render_account_gate()
    navigation = st.sidebar.container()
    with st.sidebar.expander("設定／更新備援 Email"):
        with st.form("recovery_email_request_form"):
            recovery_email = st.text_input("備援 Email", key="recovery_email_input").strip().lower()
            request_email_code = st.form_submit_button("寄送信箱驗證碼", use_container_width=True)
        if request_email_code:
            if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", recovery_email):
                st.error("請輸入有效的 Email 地址。")
            else:
                try:
                    sent = _request_account_code(auth_user["id"], "bind_email", recovery_email)
                    if sent:
                        st.session_state["pending_recovery_email"] = recovery_email
                        st.success("驗證碼已寄出，請查看信箱。")
                    else:
                        st.warning("請稍候一分鐘後再重新寄送驗證碼。")
                except Exception:
                    st.error("目前無法寄送驗證信，請確認系統管理者已設定郵件服務。")
        if st.session_state.get("pending_recovery_email"):
            with st.form("recovery_email_verify_form"):
                email_code = st.text_input("六位數驗證碼", max_chars=6, key="recovery_email_code")
                confirm_email = st.form_submit_button("確認備援 Email", use_container_width=True)
            if confirm_email:
                if _confirm_recovery_email(auth_user["id"], email_code):
                    st.session_state.pop("pending_recovery_email", None)
                    st.success("備援 Email 已驗證並更新。")
                else:
                    st.error("驗證碼無效、已逾時或錯誤次數過多。")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 全球個股即時診斷")
    st.sidebar.caption("💡 提示：上市請加 `.TW`，上櫃請加 `.TWO`（例如：3293.TWO）")
    selected_stock = st.sidebar.text_input(
        "輸入台美股代碼：", value="3293.TWO", key="selected_stock_input"
    ).strip().upper()

    st.sidebar.markdown("---")
    if os.environ.get("GVI_DUAL_GRID_ACCESS") == "1":
        management = _dual_grid_management_context(auth_user)
        if st.session_state.get("module_access_open"):
            if navigation.button("返回策略模組", key="module_access_close"):
                st.session_state.pop("module_access_open", None)
                st.rerun()
            if management is None:
                st.warning("目前無法使用雙策略網格管理。")
            else:
                _render_dual_grid_management(auth_user, management)
            st.stop()
        if management and navigation.button("雙策略網格管理", key="module_access_open_button"):
            st.session_state["module_access_open"] = True
            st.rerun()

    navigation.markdown("### 🧭 核心策略模組導航")
    workbench_open = st.session_state.get("workbench_open", not st.session_state.get("sidebar_strategy_selector"))
    if navigation.button("返回分析面板" if workbench_open else "投資工作台",
                         key="workbench_toggle", width="stretch"):
        st.session_state["workbench_open"] = not workbench_open
        if workbench_open and not st.session_state.get("sidebar_strategy_selector"):
            st.session_state["sidebar_strategy_selector"] = "📊 彩色 K 線圖與成交量"
        st.rerun()
    selected_module = navigation.selectbox(
        "選擇要檢視的分析面板：",
        options=list(MODULE_GROUPS),
        index=None,
        placeholder="選擇分析面板",
        format_func=lambda module: f"{MODULE_GROUPS[module]}｜{module}",
        key="sidebar_strategy_selector",
        on_change=_close_workbench_on_module_change,
    )
    with navigation.container(key="workbench_navigation"):
        for group in dict.fromkeys(MODULE_GROUPS.values()):
            with st.expander(group, expanded=not workbench_open and MODULE_GROUPS.get(selected_module) == group):
                for index, (module, category) in enumerate(MODULE_GROUPS.items()):
                    if category == group:
                        st.button(module, key=f"workbench_menu_{index}", width="stretch",
                                  type="primary" if not workbench_open and module == selected_module else "secondary",
                                  on_click=_open_workbench_module, args=(module,))
    navigation.markdown("### 🗣️ 解說風格與版面")
    explanation_style = navigation.radio(
        "選擇顯示方式",
        options=["一般白話｜專業易讀", "生動童趣｜生活比喻"],
        key="explanation_style_selector",
        help="切換說明用詞、提示卡片和 AI 報告風格；不會改動行情資料、指標公式或策略門檻。",
    )
    navigation.caption("童趣版用比喻幫助理解，仍保留虧損風險、資料限制與非買賣指令提醒。")
    if workbench_open:
        _render_investment_workbench(auth_user, explanation_style)
        st.stop()
    if selected_module == "🧠 智慧型雙策略網格系統":
        _require_dual_grid_access(auth_user)
    render_analysis(auth_user, selected_stock, selected_module, explanation_style)
