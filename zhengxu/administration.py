"""Dual-grid administration and scoped access controls."""

from uuid import UUID
import pandas as pd
import streamlit as st

from .foundation import ServiceReadError, failure_message, feature_enabled

from .accounts import (
    _dual_grid_permission_request,
)


def _require_dual_grid_access(account):
    """Stop the module before it can read data or run a strategy."""
    if not feature_enabled("GVI_DUAL_GRID_ACCESS"):
        return

    status, access = _dual_grid_permission_request(account, "read_module_access")
    message = "暫時無法確認雙策略網格權限，請稍後再試。"
    if status == 200 and isinstance(access, dict) and access.get("module") == "dual_grid":
        global_enabled = access.get("global_enabled")
        override = access.get("member_override")
        allowed = access.get("allowed")
        if (type(global_enabled) is bool and type(allowed) is bool
                and "member_override" in access and (override is None or type(override) is bool)
                and allowed == (global_enabled and override is not False)):
            if allowed:
                return
            message = ("雙策略網格目前暫停使用。" if not global_enabled
                       else "這個帳號目前無法使用雙策略網格。")
    elif status in (401, 403):
        message = failure_message(ServiceReadError(status), message)

    for key in ("dual_grid_analysis_results", "dual_grid_analysis_tickers", "dual_grid_backtest_result"):
        st.session_state.pop(key, None)
    st.warning(message)
    st.stop()


def _dual_grid_management_context(account):
    status, context = _dual_grid_permission_request(account, "read_module_management")
    if status != 200 or not isinstance(context, dict) or context.get("module") != "dual_grid":
        return None
    if type(context.get("super_admin")) is not bool or type(context.get("global_enabled")) is not bool:
        return None
    targets = context.get("targets")
    if not isinstance(targets, list) or not (context["super_admin"] or targets):
        return None
    try:
        for row in targets:
            if (not isinstance(row, dict) or "override" not in row
                    or (row["override"] is not None and type(row["override"]) is not bool)
                    or str(UUID(row["member_id"])) != row["member_id"]):
                return None
        if context["super_admin"]:
            scopes = context.get("delegations")
            if not isinstance(scopes, list) or any(
                not isinstance(scope, dict) or str(UUID(scope["admin_id"])) != scope["admin_id"]
                or str(UUID(scope["member_id"])) != scope["member_id"] for scope in scopes
            ):
                return None
    except (KeyError, AttributeError, TypeError, ValueError):
        return None
    return context


def _submit_dual_grid_management(account, route, payload, reason):
    if not reason.strip():
        st.error("請填寫變更原因。")
        return
    status, result = _dual_grid_permission_request(account, route, {**payload, "p_reason": reason})
    if status == 200 and isinstance(result, dict) and type(result.get("changed")) is bool:
        st.session_state["module_access_notice"] = "設定已儲存。" if result["changed"] else "設定已是此狀態。"
        st.rerun()
    if status in (401, 403):
        st.error("權限已變更，這次操作未儲存。請重新開啟管理頁。")
    else:
        st.error("目前無法確認設定是否儲存，請重新載入管理頁確認。")


def _change_dual_grid_management(account, target, enabled, reason):
    _submit_dual_grid_management(account, "change_module_access", {
        "p_target": target, "p_enabled": enabled,
    }, reason)


def _render_dual_grid_delegation(account, scopes):
    st.markdown("#### 管理員委派")
    if scopes:
        st.dataframe(
            [{"管理員識別碼": scope["admin_id"], "會員識別碼": scope["member_id"]} for scope in scopes],
            hide_index=True,
        )
    with st.form("module_access_delegation_form"):
        admin = st.text_input("委派管理員識別碼")
        member = st.text_input("委派會員識別碼")
        active = st.selectbox("委派操作", ("新增委派", "撤銷委派"))
        reason = st.text_input("委派變更原因", max_chars=200)
        if st.form_submit_button("儲存委派"):
            try:
                admin_id, member_id = str(UUID(admin.strip())), str(UUID(member.strip()))
            except ValueError:
                st.error("請輸入有效的管理員與會員識別碼。")
                return
            if admin_id == member_id:
                st.error("管理員與會員必須是不同帳號。")
                return
            _submit_dual_grid_management(account, "change_module_delegation", {
                "p_admin": admin_id, "p_member": member_id, "p_active": active == "新增委派",
            }, reason)


def _render_dual_grid_audit(account):
    st.subheader("雙策略網格權限稽核")
    status, events = _dual_grid_permission_request(account, "read_module_audit")
    if status in (401, 403):
        st.warning("目前沒有查看權限稽核的權限。")
        return
    if status != 200 or not isinstance(events, list):
        st.error("目前無法載入稽核紀錄，請稍後再試。")
        return
    if not events:
        st.info("目前沒有權限變更紀錄。")
        return

    actions = {
        "set_global": ("全域開關", ("關閉", "開啟")),
        "set_member": ("個別設定", ("停用", "允許")),
        "grant_delegation": ("新增委派", ("無委派", "已委派")),
        "revoke_delegation": ("撤銷委派", ("無委派", "已委派")),
    }
    roles = {"super_admin": "Super Admin", "delegated": "委派管理員"}
    rows = []
    try:
        for event in events:
            label, values = actions[event["action"]]
            before, after = event["before_value"], event["after_value"]
            if type(after) is not bool or (before is not None and type(before) is not bool):
                raise ValueError("Unexpected audit value")
            rows.append({
                "時間（台北）": pd.Timestamp(event["recorded_at"]).tz_convert("Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S"),
                "操作": label,
                "操作者識別碼": str(UUID(event["actor_id"])),
                "授權身分": roles[event["authorization_basis"]],
                "會員識別碼": str(UUID(event["target_id"])) if event["target_id"] else "—",
                "委派管理員識別碼": str(UUID(event["delegate_id"])) if event["delegate_id"] else "—",
                "變更前": "沿用預設" if before is None else values[before],
                "變更後": values[after],
                "原因": event["reason"],
            })
    except (AttributeError, KeyError, TypeError, ValueError):
        st.error("目前無法載入稽核紀錄，請稍後再試。")
        return
    st.caption("顯示最近 50 筆，依變更時間由新到舊排列。")
    st.dataframe(rows, hide_index=True)


def _render_dual_grid_management(account, context):
    if context["super_admin"]:
        view = st.radio("管理頁面", ("權限設定", "權限稽核"), horizontal=True)
        if view == "權限稽核":
            _render_dual_grid_audit(account)
            return
    st.subheader("雙策略網格管理")
    st.caption("全域狀態：開啟" if context["global_enabled"] else "全域狀態：關閉")
    if notice := st.session_state.pop("module_access_notice", ""):
        st.success(notice)

    if context["super_admin"]:
        with st.form("module_access_global_form"):
            enabled = st.selectbox("全域設定", ("開啟", "關閉"), index=int(not context["global_enabled"]))
            reason = st.text_input("全域變更原因", max_chars=200)
            if st.form_submit_button("儲存全域設定"):
                _change_dual_grid_management(account, None, enabled == "開啟", reason)

    labels = {None: "沿用預設", True: "允許", False: "停用"}
    with st.form("module_access_member_form"):
        if context["super_admin"]:
            if context["targets"]:
                st.dataframe(
                    [{"會員識別碼": row["member_id"], "個別設定": labels[row["override"]]} for row in context["targets"]],
                    hide_index=True,
                )
            target = st.text_input("會員識別碼")
        else:
            statuses = {row["member_id"]: labels[row["override"]] for row in context["targets"]}
            target = st.selectbox("可管理會員", tuple(statuses), format_func=lambda uid: f"{uid} · {statuses[uid]}")
        allowed = st.selectbox("個別設定", ("允許", "停用"))
        reason = st.text_input("個別變更原因", max_chars=200)
        if st.form_submit_button("儲存個別設定"):
            try:
                target_id = str(UUID(target.strip()))
            except ValueError:
                st.error("請輸入有效的會員識別碼。")
            else:
                _change_dual_grid_management(account, target_id, allowed == "允許", reason)
    if context["super_admin"]:
        _render_dual_grid_delegation(account, context["delegations"])
