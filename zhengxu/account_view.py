"""Public welcome, sign-in and account recovery forms."""

import re
import streamlit as st

from .accounts import (
    _complete_supabase_password_reset,
    _register_account,
    _request_supabase_password_reset,
    _verify_account,
)
from .workbench import (
    _logout_account,
)


def render_account_gate():
    st.sidebar.markdown("### 正序")
    auth_user = st.session_state.get("auth_user")
    if auth_user:
        st.sidebar.success(f"已登入：{auth_user['username']}")
        st.sidebar.button("登出", key="account_logout", on_click=_logout_account)
    else:
        st.sidebar.caption("登入後可使用操盤終端，投資組合資料依帳號分開保存。")
        if not st.session_state.get("account_login_view", False):
            st.markdown("""
                <section class="zhengxu-login-hero">
                  <h1>正序投資研究與管理</h1>
                  <p>登入後可使用市場分析、策略研究與個人投資組合。</p>
                </section>
            """, unsafe_allow_html=True)
            splash_cols = st.columns([1, 1.3, 1])
            with splash_cols[1]:
                if st.button("🔐 登入或建立帳號", type="primary", use_container_width=True, key="open_account_login"):
                    st.session_state["account_login_view"] = True
                    st.rerun()
            st.stop()

        st.markdown("""
            <div style="max-width:820px;margin:1.5rem auto 1rem;text-align:center;">
              <div style="color:#60a5fa;font-size:.75rem;font-weight:800;letter-spacing:.14em;">SECURE INVESTMENT WORKSPACE</div>
              <h2 style="margin:.35rem 0 .25rem;">登入操盤手終端</h2>
              <p style="color:#94a3b8;margin:0;">投資組合資料僅會顯示在登入帳號中。</p>
            </div>
        """, unsafe_allow_html=True)
        auth_columns = st.columns([1, 1.5, 1])
        with auth_columns[1]:
            login_tab, register_tab, reset_tab = st.tabs(["登入", "建立帳號", "忘記密碼"])
            with login_tab:
                with st.form("account_login_form_main"):
                    login_name = st.text_input("註冊 Email", key="login_username_main").strip().lower()
                    login_password = st.text_input("密碼", type="password", key="login_password_main")
                    login_submit = st.form_submit_button("登入", use_container_width=True)
                if login_submit:
                    account = _verify_account(login_name, login_password)
                    if account:
                        st.session_state["auth_user"] = account
                        st.session_state.pop("account_login_view", None)
                        st.rerun()
                    st.error("登入失敗。請確認 Email、密碼，或先完成 Supabase 寄出的信箱驗證。")
            with register_tab:
                st.caption("帳號驗證由 Supabase Auth 管理；登入時請使用註冊 Email。")
                with st.form("account_register_form_main"):
                    register_name = st.text_input("顯示帳號（3–32 位英數或底線）", key="register_username_main").strip()
                    register_email = st.text_input("Email（登入與密碼重設使用）", key="register_email_main").strip().lower()
                    register_password = st.text_input("建立密碼（至少 8 位）", type="password", key="register_password_main")
                    register_confirm = st.text_input("再次輸入密碼", type="password", key="register_password_confirm_main")
                    register_submit = st.form_submit_button("建立帳號", use_container_width=True)
                if register_submit:
                    if not re.fullmatch(r"[A-Za-z0-9_]{3,32}", register_name):
                        st.error("顯示帳號需為 3–32 位英數字或底線。")
                    elif not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", register_email):
                        st.error("請輸入有效的 Email。")
                    elif len(register_password) < 8:
                        st.error("密碼至少需要 8 位。")
                    elif register_password != register_confirm:
                        st.error("兩次輸入的密碼不一致。")
                    else:
                        try:
                            user_id = _register_account(register_name, register_password, register_email)
                            if user_id is None:
                                st.success("註冊申請已送出。請先查看 Email 並完成 Supabase 信箱驗證，再使用 Email 與密碼登入。")
                            else:
                                st.session_state["auth_user"] = {
                                    "id": user_id, "username": register_name, "email": register_email
                                }
                                st.session_state.pop("account_login_view", None)
                                st.rerun()
                        except ValueError as exc:
                            st.error(f"註冊未完成：{exc}")
                        except Exception as exc:
                            st.error(f"註冊未完成：{exc}")
            with reset_tab:
                st.caption("Supabase 將寄送一次性驗證碼；請在信件範本中使用 {{ .Token }}，再於此輸入驗證碼。")
                with st.form("password_reset_request_form_main"):
                    reset_email = st.text_input("註冊 Email", key="reset_request_email_main").strip().lower()
                    request_reset = st.form_submit_button("寄送密碼重設驗證碼", use_container_width=True)
                if request_reset:
                    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", reset_email):
                        st.error("請輸入有效的 Email。")
                    else:
                        try:
                            _request_supabase_password_reset(reset_email)
                            st.session_state["supabase_reset_email"] = reset_email
                            st.success("若此 Email 已註冊，Supabase 將寄送驗證碼。請查看收件匣及垃圾郵件匣。")
                        except Exception as exc:
                            st.error(f"目前無法寄送重設信：{exc}")
                pending_reset_email = st.session_state.get("supabase_reset_email", "")
                if pending_reset_email:
                    with st.form("password_reset_complete_form_main"):
                        st.caption(f"重設帳號：{pending_reset_email}")
                        reset_token = st.text_input("Email 中的驗證碼", max_chars=12, key="supabase_reset_token_main").strip()
                        reset_new_password = st.text_input("新密碼（至少 8 位）", type="password", key="supabase_reset_new_password_main")
                        reset_confirm = st.text_input("再次輸入新密碼", type="password", key="supabase_reset_confirm_password_main")
                        complete_reset = st.form_submit_button("確認重設密碼", use_container_width=True)
                    if complete_reset:
                        if len(reset_new_password) < 8:
                            st.error("新密碼至少需要 8 位。")
                        elif reset_new_password != reset_confirm:
                            st.error("兩次輸入的新密碼不一致。")
                        elif not reset_token:
                            st.error("請輸入 Email 中的驗證碼。")
                        else:
                            try:
                                _complete_supabase_password_reset(
                                    pending_reset_email, reset_token, reset_new_password
                                )
                                st.session_state.pop("supabase_reset_email", None)
                                st.success("密碼已重設，請使用新密碼登入。")
                            except Exception as exc:
                                st.error(f"密碼重設失敗：{exc}")
        back_cols = st.columns([1, 1.5, 1])
        with back_cols[1]:
            if st.button("返回首頁", key="account_login_back", use_container_width=True):
                st.session_state["account_login_view"] = False
                st.rerun()
        st.stop()
    return auth_user
