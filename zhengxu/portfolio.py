"""Cloud pilot portfolio reads, writes, ledger history and forms."""

from base64 import b64encode
from datetime import date
from math import isfinite
from uuid import UUID
from uuid import uuid4
import html
import json
import os
import pandas as pd
import re
import requests
import streamlit as st
import streamlit.components.v1 as components

from .accounts import (
    DUAL_GRID_SUPABASE_URL,
    _dual_grid_public_api_key,
    _supabase_auth_config,
)
from .portfolio_research import render_portfolio_research
from .storage import (
    _encode_portfolio_export,
)


PILOT_POSITION_COLUMNS = {
    "legacy_id": "id", "symbol": "代碼", "name": "名稱", "asset_type": "資產類型",
    "currency": "幣別", "quantity": "數量", "price": "最新價格", "fx_to_twd": "換算台幣匯率",
    "cost_basis_twd": "台幣成本", "updated_at": "更新時間",
}
PILOT_SNAPSHOT_COLUMNS = {
    "snapshot_date": "記錄時間", "total_value_twd": "組合市值", "stock_value_twd": "股票市值",
    "bond_value_twd": "債券市值", "cash_value_twd": "現金市值",
}


def _validate_pilot_portfolio_rows(data):
    positions, snapshots = data.get("positions"), data.get("snapshots")
    for rows, keys, numeric, nullable in (
        (positions, set(PILOT_POSITION_COLUMNS), {"quantity", "price", "fx_to_twd", "cost_basis_twd"}, set()),
        (snapshots, {*PILOT_SNAPSHOT_COLUMNS, "legacy_id", "note"}, set(PILOT_SNAPSHOT_COLUMNS) - {"snapshot_date"}, {"note"}),
    ):
        if not isinstance(rows, list):
            raise ValueError("Malformed portfolio data")
        ids = set()
        for row in rows:
            if not isinstance(row, dict) or row.keys() != keys:
                raise ValueError("Malformed portfolio data")
            legacy_id = row["legacy_id"]
            if type(legacy_id) is not int or legacy_id < 1 or legacy_id in ids:
                raise ValueError("Malformed portfolio data")
            ids.add(legacy_id)
            for key in keys - numeric - {"legacy_id"}:
                if not isinstance(row[key], str) and not (key in nullable and row[key] is None):
                    raise ValueError("Malformed portfolio data")
            try:
                valid = all(type(row[key]) in (int, float) and isfinite(row[key]) for key in numeric)
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("Malformed portfolio data")
    if data["revision"] == 0 and (positions or snapshots):
        raise ValueError("Malformed portfolio data")


PILOT_SETTING_LABELS = {
    "target_stock_pct": "股票目標（%）", "target_bond_pct": "債券目標（%）",
    "target_cash_pct": "現金目標（%）", "rebalance_threshold_pct": "再平衡門檻（%）",
    "minimum_cash_twd": "最低現金（台幣）", "drawdown_warning_pct": "回撤提醒（%）",
    "drawdown_review_pct": "回撤檢視（%）", "drawdown_limit_pct": "回撤上限（%）",
}


def _render_portfolio_export_link(filename, content, *,
                                  label="下載我的持倉、設定與快照（JSON）", title="投資組合匯出"):
    # Deliver within this signed-in session without publishing a media URL.
    payload = b64encode(content).decode("ascii")
    link = (
        f'<a href="data:application/json;base64,{payload}" download="{filename}">'
        f"{html.escape(label)}</a>"
    )
    if hasattr(st, "iframe"):
        st.iframe(link, height=42, alt=title)
    else:
        components.html(link, height=42)
    st.caption("下載檔包含個人投資資料，請保存在私人位置。")


def _is_portfolio_pilot(account):
    selected = os.environ.get("GVI_PORTFOLIO_PILOT_UID", "").strip()
    if not selected:
        return False
    try:
        return UUID(selected) == UUID(account["supabase_user_id"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def _portfolio_pilot_request(account, route, payload=None):
    token = account.get("access_token")
    if not token:
        raise ValueError("Sign-in required")
    url, key = _supabase_auth_config()
    if url != DUAL_GRID_SUPABASE_URL or not _dual_grid_public_api_key(key):
        raise ValueError("Portfolio pilot configuration")
    method = "GET" if payload is None else "POST"
    profile = "Accept-Profile" if payload is None else "Content-Profile"
    return requests.request(
        method, f"{url}/rest/v1/rpc/{route}",
        headers={"apikey": key, "Authorization": f"Bearer {token}", profile: "gvi_portfolio_pilot_api"},
        json=payload, timeout=(5, 12), allow_redirects=False,
    )


def _read_portfolio_pilot(account):
    response = _portfolio_pilot_request(account, "read_imported_portfolio")
    if response.status_code != 200:
        raise RuntimeError("Portfolio pilot request failed")
    data = response.json()
    if not isinstance(data, dict) or type(data.get("revision")) is not int or data["revision"] < 0:
        raise ValueError("Malformed portfolio data")
    _validate_pilot_portfolio_rows(data)
    settings = data["settings"]
    if data["revision"] == 0:
        if settings is not None:
            raise ValueError("Malformed portfolio settings")
    elif (not isinstance(settings, dict) or settings.keys() != PILOT_SETTING_LABELS.keys()
            or not all(type(value) in (int, float) and isfinite(value) for value in settings.values())):
        raise ValueError("Malformed portfolio settings")
    return data


PILOT_SETTINGS_STATE = "portfolio_pilot_settings_state"
PILOT_SETTINGS_FIELDS = {
    "target_stock_pct": (40.0, 0.0, 100.0),
    "target_bond_pct": (50.0, 0.0, 100.0),
    "target_cash_pct": (10.0, 0.0, 100.0),
    "rebalance_threshold_pct": (5.0, 1.0, 30.0),
    "minimum_cash_twd": (0.0, 0.0, None),
    "drawdown_warning_pct": (8.0, 1.0, 30.0),
    "drawdown_review_pct": (12.0, 1.0, 40.0),
    "drawdown_limit_pct": (15.0, 1.0, 60.0),
}


def _submit_pilot_settings(account, state):
    pending = state["pending"]
    try:
        response = _portfolio_pilot_request(account, "save_settings", pending)
        if response.status_code == 200:
            receipt = response.json()
            if (isinstance(receipt, dict) and type(receipt.get("revision")) is int
                    and receipt["revision"] == pending["p_expected_revision"] + 1
                    and type(receipt.get("replayed")) is bool):
                state.pop("pending")
                state["saved"] = True
                state["reload"] = True
                st.rerun()
        if response.status_code in (400, 401, 403, 409, 423):
            state.pop("pending")
            if response.status_code == 409:
                state["conflict"] = True
                st.rerun()
            if response.status_code == 400:
                st.error("請核對配置合計為 100%，且回撤提醒、檢視與上限依序遞增。")
            else:
                st.error("目前無法儲存配置，請確認登入與試行寫入權限。")
            return
    except (RuntimeError, ValueError, requests.RequestException):
        pass
    st.rerun()


def _render_pilot_settings_form(account, data):
    revision = data["revision"]
    owner = account["supabase_user_id"]
    state = st.session_state.get(PILOT_SETTINGS_STATE)
    if state is None or state["owner"] != owner:
        state = {"owner": owner, "revision": revision, "form_id": uuid4().hex}
        st.session_state[PILOT_SETTINGS_STATE] = state
    if state.pop("saved", False):
        st.success("雲端配置已儲存。")
    if state.pop("reload", False):
        state.update(revision=revision, form_id=uuid4().hex)
    if "pending" in state:
        st.warning("尚未確認儲存結果。請重試同一筆操作。")
        if st.button("重試雲端配置儲存"):
            _submit_pilot_settings(account, state)
        return
    if state["revision"] != revision or state.get("conflict"):
        st.warning("雲端配置版本已更新。請重新讀取後再編輯。")
        if st.button("重新讀取雲端配置"):
            state.pop("conflict", None)
            state["reload"] = True
            st.rerun()
        return

    stored = data["settings"]
    if stored is None:
        st.caption("下方是表單建議值，儲存後才會寫入你的雲端配置。")
    with st.form("pilot_configuration_" + state["form_id"]):
        values = {}
        for fields in (
            ("target_stock_pct", "target_bond_pct", "target_cash_pct"),
            ("rebalance_threshold_pct", "minimum_cash_twd"),
            ("drawdown_warning_pct", "drawdown_review_pct", "drawdown_limit_pct"),
        ):
            for column, name in zip(st.columns(len(fields)), fields):
                default, minimum, maximum = PILOT_SETTINGS_FIELDS[name]
                value = default if stored is None else float(stored[name])
                # Older data can lie outside today's editable bounds.
                values[name] = column.number_input(
                    PILOT_SETTING_LABELS[name], min_value=min(minimum, value),
                    max_value=max(maximum, value) if maximum is not None else None,
                    value=value, key=f"{state['form_id']}_{name}",
                )
        if st.form_submit_button("儲存雲端配置"):
            state["pending"] = {"p_expected_revision": state["revision"],
                                "p_operation_id": str(uuid4()), "p_settings": values}
            _submit_pilot_settings(account, state)


PILOT_POSITION_STATE = "portfolio_pilot_position_state"
PILOT_POSITION_EDIT_STATE = "portfolio_pilot_position_edit_state"
PILOT_POSITION_DELETE_STATE = "portfolio_pilot_position_delete_state"
PILOT_SNAPSHOT_STATE = "portfolio_pilot_snapshot_state"


def _submit_pilot_portfolio_operation(account, state, *, route="add_position"):
    pending = state["pending"]
    action = {
        "add_position": "新增雲端持倉",
        "update_position": "修改雲端持倉",
        "delete_position": "刪除雲端持倉",
        "record_snapshot": "記錄雲端快照",
    }[route]
    try:
        response = _portfolio_pilot_request(account, route, pending)
        if response.status_code == 200:
            receipt = response.json()
            if (isinstance(receipt, dict) and type(receipt.get("revision")) is int
                    and receipt["revision"] == pending["p_expected_revision"] + 1
                    and type(receipt.get("record_id")) is int and receipt["record_id"] > 0
                    and ("p_record_id" not in pending or receipt["record_id"] == pending["p_record_id"])
                    and type(receipt.get("replayed")) is bool):
                state.pop("pending")
                state["saved"] = True
                st.rerun()
        if response.status_code in (400, 401, 403, 404, 409, 423):
            state.pop("pending")
            if response.status_code in (404, 409):
                state["conflict"] = True
                st.rerun()
            if response.status_code == 400:
                if route == "delete_position":
                    st.error("無法辨識要刪除的持倉，請重新選擇。")
                elif route == "record_snapshot":
                    st.error("無法記錄快照，請核對已存持倉的數量、單價、匯率與分類。")
                else:
                    st.error("請核對持倉欄位、數量、價格、匯率及成本。")
            else:
                st.error(f"目前無法{action}，請確認登入與試行權限。")
            return
    except (RuntimeError, ValueError, requests.RequestException):
        pass
    st.rerun()


def _pilot_position_fields(key_prefix, position=None, *, name_column=None):
    stored = position or {}
    asset_types = ("股票", "債券", "現金")
    currencies = ("TWD", "USD", "JPY", "HKD", "EUR")
    asset = stored.get("asset_type", asset_types[0])
    currency = stored.get("currency", currencies[0])
    if asset not in asset_types:
        asset_types += (asset,)
    if currency not in currencies:
        currencies += (currency,)
    name_column = name_column if name_column is not None else st
    name = name_column.text_input("持倉名稱", value=stored.get("name", ""), key=f"{key_prefix}_name")
    asset_column, currency_column = st.columns(2)
    values = {
        "name": name,
        "asset_type": asset_column.selectbox("持倉類型", asset_types, index=asset_types.index(asset), key=f"{key_prefix}_asset"),
        "currency": currency_column.selectbox("持倉幣別", currencies, index=currencies.index(currency), key=f"{key_prefix}_currency"),
    }
    numeric_columns = [*st.columns(2), *st.columns(2)]
    for column, (field, label, minimum) in zip(numeric_columns, (
        ("quantity", "持倉數量", 0.0),
        ("price", "持倉單價", 0.0),
        ("fx_to_twd", "兌台幣匯率", 0.000001),
        ("cost_basis_twd", "持倉總成本（台幣）", 0.0),
    )):
        value = float(stored[field]) if position is not None else None
        help_text = "每一單位持倉幣別可兌換的台幣金額。TWD 請填 1。" if field == "fx_to_twd" else None
        edited = column.number_input(label, min_value=min(minimum, value) if value is not None else minimum,
                                 value=value, format="%.6f", help=help_text, key=f"{key_prefix}_{field}")
        values[field] = stored[field] if position is not None and edited == value else edited
    return values


def _pilot_position_label(position):
    return f"{position['symbol']} · {position['name']}（編號 {position['legacy_id']}）"


def _render_pilot_position_edit(account, data):
    st.markdown("#### 修改雲端持倉")
    owner = account["supabase_user_id"]
    revision = data["revision"]
    state = st.session_state.get(PILOT_POSITION_EDIT_STATE)
    if state is None or state["owner"] != owner or state.pop("reload", False):
        state = {"owner": owner, "revision": revision, "form_id": uuid4().hex}
        st.session_state[PILOT_POSITION_EDIT_STATE] = state
    if state.pop("saved", False):
        st.success("雲端持倉修改已儲存。")
        state.update(revision=revision, form_id=uuid4().hex)
    if "pending" in state:
        st.warning("尚未確認持倉修改結果。請重試同一筆操作。")
        if st.button("重試修改雲端持倉"):
            _submit_pilot_portfolio_operation(account, state, route="update_position")
        return

    positions = {row["legacy_id"]: row for row in data["positions"]}
    if not positions:
        st.info("目前沒有可修改的雲端持倉。")
        return
    if state["revision"] != revision or state.get("conflict"):
        st.warning("雲端投資組合已更新，請重新讀取後再修改持倉。")
        if st.button("重新讀取待修改持倉"):
            state["reload"] = True
            st.rerun()
        return

    record_id = st.selectbox(
        "選擇要修改的持倉", tuple(positions), index=None,
        format_func=lambda row_id: _pilot_position_label(positions[row_id]),
    )
    if record_id is None:
        return
    form_id = f"pilot_position_edit_{state['form_id']}_{record_id}"
    with st.form(form_id):
        values = _pilot_position_fields(form_id, positions[record_id])
        if st.form_submit_button("儲存持倉修改"):
            if any(value is None for value in values.values()) or values["quantity"] <= 0:
                st.error("請填寫正數數量、單價、匯率及總成本。")
                return
            state["pending"] = {"p_expected_revision": state["revision"], "p_operation_id": str(uuid4()),
                                "p_record_id": record_id, "p_values": values}
            _submit_pilot_portfolio_operation(account, state, route="update_position")


def _render_pilot_position_form(account, data):
    st.markdown("#### 新增雲端持倉")
    if data["settings"] is None:
        st.info("請先儲存雲端配置，再新增持倉。")
        return

    revision = data["revision"]
    owner = account["supabase_user_id"]
    state = st.session_state.get(PILOT_POSITION_STATE)
    if state is None or state["owner"] != owner or state.pop("reload", False):
        state = {"owner": owner, "revision": revision, "form_id": uuid4().hex}
        st.session_state[PILOT_POSITION_STATE] = state
    if state.pop("saved", False):
        st.success("雲端持倉已新增。")
        state.update(revision=revision, form_id=uuid4().hex)
    if "pending" in state:
        st.warning("尚未確認持倉儲存結果。請重試同一筆操作。")
        if st.button("重試新增雲端持倉"):
            _submit_pilot_portfolio_operation(account, state)
        return
    if state["revision"] != revision or state.get("conflict"):
        st.warning("雲端投資組合版本已更新。請重新讀取後再新增持倉。")
        if st.button("重新讀取雲端持倉"):
            state["reload"] = True
            st.rerun()
        return

    with st.form("pilot_position_" + state["form_id"]):
        symbol_column, name_column = st.columns((1, 2))
        symbol = symbol_column.text_input("持倉代碼")
        values = _pilot_position_fields(state["form_id"], name_column=name_column)
        if st.form_submit_button("儲存新增持倉"):
            if not symbol.strip() or any(value is None for value in values.values()) or values["quantity"] <= 0:
                st.error("請填寫持倉代碼、正數數量、單價、匯率及總成本。")
                return
            values["symbol"] = symbol.strip()
            values["name"] = values["name"].strip()
            state["pending"] = {"p_expected_revision": state["revision"],
                                "p_operation_id": str(uuid4()), "p_values": values}
            _submit_pilot_portfolio_operation(account, state)


def _render_pilot_position_delete(account, data):
    st.markdown("#### 刪除雲端持倉")
    owner = account["supabase_user_id"]
    revision = data["revision"]
    state = st.session_state.get(PILOT_POSITION_DELETE_STATE)
    if state is None or state["owner"] != owner or state.pop("reload", False):
        state = {"owner": owner, "revision": revision, "form_id": uuid4().hex}
        st.session_state[PILOT_POSITION_DELETE_STATE] = state
    if state.pop("saved", False):
        st.success("雲端持倉已刪除。")
        state.update(revision=revision, form_id=uuid4().hex)
    if "pending" in state:
        st.warning("尚未確認刪除結果。重試會送出同一筆刪除請求。")
        if st.button("重試刪除雲端持倉"):
            _submit_pilot_portfolio_operation(account, state, route="delete_position")
        return

    positions = {row["legacy_id"]: row for row in data["positions"]}
    if state["revision"] != revision or state.get("conflict"):
        st.warning("雲端投資組合已更新，請重新讀取後再選擇要刪除的持倉。")
        if st.button("重新讀取待刪除持倉"):
            state["reload"] = True
            st.rerun()
        return
    if not positions:
        st.info("目前沒有可刪除的雲端持倉。")
        return

    form_id = state["form_id"]
    record_id = st.selectbox(
        "選擇要刪除的持倉", tuple(positions), index=None,
        format_func=lambda row_id: _pilot_position_label(positions[row_id]),
        key=f"pilot_position_delete_{form_id}",
    )
    if record_id is None:
        return
    position = positions[record_id]
    st.warning(f"將刪除 {position['symbol']} · {position['name']}（編號 {record_id}，數量 {position['quantity']}）。"
               "刪除後無法從此頁復原。")
    confirmed = st.checkbox("我確認刪除這筆持倉", key=f"pilot_delete_confirm_{form_id}_{record_id}")
    if st.button("取消刪除持倉"):
        state["reload"] = True
        st.rerun()
    if st.button("確認刪除持倉", disabled=not confirmed):
        state["pending"] = {"p_expected_revision": state["revision"], "p_operation_id": str(uuid4()),
                            "p_record_id": record_id}
        _submit_pilot_portfolio_operation(account, state, route="delete_position")


def _render_pilot_snapshot(account, data):
    owner = account["supabase_user_id"]
    revision = data["revision"]
    state = st.session_state.get(PILOT_SNAPSHOT_STATE)
    if state is None or state["owner"] != owner:
        state = {"owner": owner, "revision": revision}
        st.session_state[PILOT_SNAPSHOT_STATE] = state
    if state.pop("saved", False):
        st.success("雲端快照已記錄。")
        state["revision"] = revision
    if "pending" in state:
        st.warning("尚未確認快照記錄結果。請重試同一筆操作。")
        if st.button("重試記錄雲端快照"):
            _submit_pilot_portfolio_operation(account, state, route="record_snapshot")
        return
    if state["revision"] != revision or state.get("conflict"):
        st.warning("雲端投資組合已更新。請重新讀取持倉後再記錄快照。")
        if st.button("重新讀取快照所用持倉"):
            st.session_state.pop(PILOT_SNAPSHOT_STATE)
            st.rerun()
        return
    if not data["positions"]:
        st.info("目前沒有雲端持倉可供記錄快照。")
        return
    st.caption("以已儲存的持倉數量、單價與匯率計算，未調整入金或出金。")
    if st.button("記錄雲端快照"):
        state["pending"] = {"p_expected_revision": state["revision"], "p_operation_id": str(uuid4())}
        _submit_pilot_portfolio_operation(account, state, route="record_snapshot")


PILOT_INCOME_STATE = "portfolio_pilot_income_state"
PILOT_INCOME_CORRECTION_STATE = "portfolio_pilot_income_correction_state"
PILOT_INCOME_TYPES = {"dividend": "股息", "interest": "利息"}
PILOT_ENTRY_CURRENCIES = ("TWD", "USD", "JPY", "HKD", "EUR")
PILOT_ENTRY_DECIMAL = re.compile(r"[0-9]{1,24}(?:[.][0-9]{1,12})?")


def _submit_pilot_entry(account, state, *, route, label, entry_label, invalid_message, conflict_message=None):
    pending = state["pending"]
    previous = pending.get("p_previous_operation_id")
    try:
        response = _portfolio_pilot_request(account, route, pending)
        if response.status_code == 200:
            result = response.json()
            if (isinstance(result, dict) and result.get("operation_id") == pending["p_operation_id"]
                    and type(result.get("replayed")) is bool
                    and (previous is None or result.get("previous_operation_id") == previous)):
                state.pop("pending")
                state.update(saved=True, form_id=uuid4().hex)
                st.rerun()
        if response.status_code in (400, 401, 403, 404, 409, 422, 423):
            state.pop("pending")
            if response.status_code in (400, 422):
                message = invalid_message
            elif response.status_code == 409:
                message = conflict_message or f"這筆操作已用於不同的{entry_label}。請核對歷史紀錄後再新增。"
            else:
                message = f"目前無法儲存{label}，請確認登入與試行權限。"
            if previous is not None:
                state["error"] = message
                if response.status_code in (400, 401, 403, 422, 423):
                    state["draft"] = pending
                st.rerun()
            st.error(message)
            return
    except (RuntimeError, ValueError, requests.RequestException):
        pass
    st.rerun()


def _submit_pilot_income(account, state):
    _submit_pilot_entry(account, state, route="record_income", label="投資收入", entry_label="收入明細",
                        invalid_message="請核對收入類型、實際入帳日、幣別與實收金額。")


def _validate_pilot_income_rows(rows):
    fields = {"operation_id", "income_type", "booked_on", "currency", "net_amount", "reference", "note", "recorded_at"}
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or row.keys() != fields or row["income_type"] not in PILOT_INCOME_TYPES
        or row["currency"] not in PILOT_ENTRY_CURRENCIES
        or not isinstance(row["net_amount"], str) or not PILOT_ENTRY_DECIMAL.fullmatch(row["net_amount"])
        for row in rows
    ):
        raise ValueError("Invalid income history")


def _pilot_activity_mode(account, kind, correction_enabled):
    options = ("明細", "新增", "更正") if correction_enabled else ("明細", "新增")
    return st.radio("操作", options, horizontal=True,
                    key=f"portfolio_cloud_{account['id']}_{kind}_mode", label_visibility="collapsed")


def _render_pilot_income(account):
    st.markdown("#### 股息與利息")
    correction_enabled = os.environ.get("GVI_PORTFOLIO_PILOT_INCOME_CORRECTION") == "1"
    mode = _pilot_activity_mode(account, "income", correction_enabled)
    if mode == "新增":
        _render_pilot_income_form(account)
        return

    history = None
    if correction_enabled:
        try:
            rows, history = _read_pilot_correction_history(account, "income", _validate_pilot_income_rows)
        except (RuntimeError, ValueError, TypeError, requests.RequestException):
            st.warning("暫時無法讀取收入更正歷史。")
    if mode == "更正":
        if history is not None:
            _render_pilot_correction(account, rows, history, income=True)
        return
    if history is None:
        try:
            response = _portfolio_pilot_request(account, "read_income_entries")
            rows = response.json() if response.status_code == 200 else None
            _validate_pilot_income_rows(rows)
        except (RuntimeError, ValueError, TypeError, requests.RequestException):
            st.error("暫時無法讀取投資收入明細。請稍後再試。")
            return

    if rows:
        for currency in PILOT_ENTRY_CURRENCIES:
            entries = [{"入帳日": row["booked_on"], "類型": PILOT_INCOME_TYPES[row["income_type"]],
                        "實收金額": row["net_amount"], "標的或帳戶": row["reference"], "備註": row["note"]}
                       for row in rows if row["currency"] == currency]
            if entries:
                st.markdown(f"**{currency}**")
                st.dataframe(entries, hide_index=True)
        if st.button("準備匯出投資收入明細"):
            now = pd.Timestamp.now(tz="UTC")
            content = json.dumps({"schema_version": 1, "exported_at_utc": now.isoformat(), "income_entries": rows},
                                 ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            _render_portfolio_export_link(f"gvi-income-{now:%Y%m%dT%H%M%SZ}.json", content,
                                          label="下載股息與利息明細（JSON）", title="投資收入明細匯出")
    else:
        st.info("尚無已儲存的股息或利息明細。")


def _render_pilot_income_form(account):
    owner = account["supabase_user_id"]
    state = st.session_state.get(PILOT_INCOME_STATE)
    if state is None or state["owner"] != owner:
        state = {"owner": owner, "form_id": uuid4().hex}
        st.session_state[PILOT_INCOME_STATE] = state
    if state.pop("saved", False):
        st.success("投資收入已儲存。")
    if "pending" in state:
        st.warning("尚未確認收入儲存結果。請重試同一筆操作。")
        if st.button("重試儲存投資收入"):
            _submit_pilot_income(account, state)
    if "pending" in state:
        return
    with st.form("pilot_income_" + state["form_id"], clear_on_submit=False):
        kind_col, date_col, currency_col = st.columns(3)
        amount_col, reference_col = st.columns(2)
        kind = kind_col.selectbox("收入類型", tuple(PILOT_INCOME_TYPES), format_func=PILOT_INCOME_TYPES.get)
        booked_on = date_col.date_input("實際入帳日", value=None, min_value=pd.Timestamp("1900-01-01").date(),
                                  max_value=pd.Timestamp.now(tz="Asia/Taipei").date())
        currency = currency_col.selectbox("收入幣別", PILOT_ENTRY_CURRENCIES)
        amount = amount_col.text_input("實收金額（原幣）", help="請輸入實際入帳金額，最多可填 12 位小數。").strip()
        reference = reference_col.text_input("標的或帳戶（選填）", max_chars=100).strip()
        note = st.text_area("收入備註（選填）", max_chars=500).strip()
        if st.form_submit_button("儲存投資收入"):
            if booked_on is None or not PILOT_ENTRY_DECIMAL.fullmatch(amount) or not any(c in "123456789" for c in amount):
                st.error("請選擇實際入帳日，並輸入大於零的原幣實收金額。")
                return
            state["pending"] = {"p_operation_id": str(uuid4()), "p_income_type": kind,
                                "p_booked_on": booked_on.isoformat(), "p_currency": currency,
                                "p_net_amount": amount, "p_reference": reference or None, "p_note": note or None}
            _submit_pilot_income(account, state)


PILOT_TRADE_STATE = "portfolio_pilot_trade_state"
PILOT_TRADE_CORRECTION_STATE = "portfolio_pilot_trade_correction_state"
PILOT_TRADE_SIDES = {"buy": "買入", "sell": "賣出"}
PILOT_TRADE_FIELDS = {"operation_id", "executed_on", "side", "symbol", "currency", "quantity",
                     "unit_price", "fee", "tax", "note", "recorded_at"}
PILOT_TRADE_FORM_ERROR = "請填寫成交標的、實際成交日及大於零的數量與單價。費用可留白或填非負數。"


def _submit_pilot_trade(account, state):
    _submit_pilot_entry(account, state, route="record_trade", label="買賣成交", entry_label="成交明細",
                        invalid_message="請核對成交日、標的、幣別、數量、單價及費用。")


def _validate_pilot_trade_rows(rows):
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or row.keys() != PILOT_TRADE_FIELDS or row["side"] not in PILOT_TRADE_SIDES
        or row["currency"] not in PILOT_ENTRY_CURRENCIES
        or any(not isinstance(row[name], str) or not PILOT_ENTRY_DECIMAL.fullmatch(row[name])
               for name in ("quantity", "unit_price"))
        or any(row[name] is not None and (not isinstance(row[name], str)
               or not PILOT_ENTRY_DECIMAL.fullmatch(row[name])) for name in ("fee", "tax"))
        for row in rows
    ):
        raise ValueError("Invalid trade history")


def _read_pilot_trade_entries(account):
    response = _portfolio_pilot_request(account, "read_trade_entries")
    rows = response.json() if response.status_code == 200 else None
    _validate_pilot_trade_rows(rows)
    return rows


def _read_pilot_correction_history(account, kind, validate_rows):
    entries_key = f"{kind}_entries"
    response = _portfolio_pilot_request(account, f"read_{kind}_history")
    history = response.json() if response.status_code == 200 else None
    if not isinstance(history, dict) or history.keys() != {entries_key, "corrections"}:
        raise ValueError("Invalid version history")
    entries, corrections = history[entries_key], history["corrections"]
    validate_rows(entries)
    by_id = {row["operation_id"]: row for row in entries}
    fields = {"operation_id", "previous_operation_id", "reason", "corrected_at"}
    if len(by_id) != len(entries) or not isinstance(corrections, list) or any(
        not isinstance(event, dict) or event.keys() != fields
        or not all(isinstance(value, str) for value in event.values())
        or event["operation_id"] not in by_id or event["previous_operation_id"] not in by_id
        or event["operation_id"] == event["previous_operation_id"]
        for event in corrections
    ):
        raise ValueError("Invalid trade correction history")
    replaced = {event["previous_operation_id"] for event in corrections}
    successors = {event["operation_id"] for event in corrections}
    if len(replaced) != len(corrections) or len(successors) != len(corrections):
        raise ValueError("Conflicting trade version history")
    return [row for row in entries if row["operation_id"] not in replaced], history


def _pilot_trade_display(row):
    return {"成交日": row["executed_on"], "方向": PILOT_TRADE_SIDES[row["side"]], "標的": row["symbol"],
            "原幣": row["currency"], "成交數量": row["quantity"], "成交單價": row["unit_price"],
            "手續費": row["fee"] or "未填", "交易稅": row["tax"] or "未填", "備註": row["note"] or ""}


def _pilot_trade_form_fields(initial=None):
    values = initial or {}
    prefix = "更正後" if initial is not None else ""
    side_col, symbol_col, date_col = st.columns((1, 1, 1.3))
    currency_col, quantity_col, price_col = st.columns(3)
    side = side_col.selectbox(f"{prefix}成交方向", tuple(PILOT_TRADE_SIDES), format_func=PILOT_TRADE_SIDES.get,
                        index=tuple(PILOT_TRADE_SIDES).index(values.get("side", "buy")))
    symbol = symbol_col.text_input(f"{prefix}成交標的", value=values.get("symbol", ""), max_chars=40).strip()
    current_date = pd.Timestamp.now(tz="Asia/Taipei").date()
    initial_date = date.fromisoformat(values["executed_on"]) if values.get("executed_on") else None
    first_date = date(1900, 1, 1)
    executed_on = date_col.date_input(f"{prefix}實際成交日", value=initial_date, min_value=min(first_date, initial_date or first_date),
                                max_value=current_date)
    currency = currency_col.selectbox(f"{prefix}成交幣別", PILOT_ENTRY_CURRENCIES,
                            index=PILOT_ENTRY_CURRENCIES.index(values.get("currency", "TWD")))
    quantity = quantity_col.text_input(f"{prefix}成交數量", value=values.get("quantity", "")).strip()
    price = price_col.text_input(f"{prefix}成交單價（原幣）", value=values.get("unit_price", "")).strip()
    st.caption("數量與金額最多可填 12 位小數。費用使用成交幣別，空白代表未填，無費用請輸入 0。")
    fee_col, tax_col = st.columns(2)
    fee = fee_col.text_input(f"{prefix}手續費（選填）", value=values.get("fee") or "").strip()
    tax = tax_col.text_input(f"{prefix}交易稅（選填）", value=values.get("tax") or "").strip()
    note = st.text_area(f"{prefix}成交備註（選填）", value=values.get("note") or "", max_chars=500).strip()
    return {"side": side, "symbol": symbol, "executed_on": executed_on.isoformat() if executed_on else None,
            "currency": currency, "quantity": quantity, "unit_price": price,
            "fee": fee or None, "tax": tax or None, "note": note or None}


def _valid_pilot_trade_form(values):
    return (
        bool(values["executed_on"] and values["symbol"])
        and all(PILOT_ENTRY_DECIMAL.fullmatch(values[key]) and any(c in "123456789" for c in values[key])
                for key in ("quantity", "unit_price"))
        and all(values[key] is None or PILOT_ENTRY_DECIMAL.fullmatch(values[key]) for key in ("fee", "tax"))
    )


def _pilot_income_display(row):
    return {"入帳日": row["booked_on"], "類型": PILOT_INCOME_TYPES[row["income_type"]], "原幣": row["currency"],
            "實收金額": row["net_amount"], "標的或帳戶": row["reference"] or "", "備註": row["note"] or ""}


def _pilot_income_correction_fields(initial):
    kind_col, date_col, currency_col = st.columns(3)
    amount_col, reference_col = st.columns(2)
    kind = kind_col.selectbox("更正後收入類型", tuple(PILOT_INCOME_TYPES), format_func=PILOT_INCOME_TYPES.get,
                        index=tuple(PILOT_INCOME_TYPES).index(initial["income_type"]))
    original_date = date.fromisoformat(initial["booked_on"])
    booked_on = date_col.date_input("更正後實際入帳日", value=original_date, min_value=min(date(1900, 1, 1), original_date),
                              max_value=max(original_date, pd.Timestamp.now(tz="Asia/Taipei").date()))
    currency = currency_col.selectbox("更正後收入幣別", PILOT_ENTRY_CURRENCIES,
                            index=PILOT_ENTRY_CURRENCIES.index(initial["currency"]))
    amount = amount_col.text_input("更正後實收金額（原幣）", value=initial["net_amount"]).strip()
    reference = reference_col.text_input("更正後標的或帳戶（選填）", value=initial["reference"] or "", max_chars=100).strip()
    note = st.text_area("更正後收入備註（選填）", value=initial["note"] or "", max_chars=500).strip()
    return {"income_type": kind, "booked_on": booked_on.isoformat() if booked_on else None,
            "currency": currency, "net_amount": amount, "reference": reference or None, "note": note or None}


def _valid_pilot_income_correction(values):
    amount = values["net_amount"]
    return bool(values["booked_on"] and PILOT_ENTRY_DECIMAL.fullmatch(amount) and any(c in "123456789" for c in amount))


def _submit_pilot_correction(account, state, kind, category):
    _submit_pilot_entry(
        account, state, route=f"correct_{kind}", label=f"{category}更正", entry_label=f"{category}更正",
        invalid_message=f"請核對更正後的{category}明細及更正理由。",
        conflict_message=f"這筆{category}的版本已變更，或操作內容與先前不符。請重新選擇目前有效的版本。",
    )


def _render_pilot_correction(account, rows, history, *, income=False):
    kind, category, state_key = ("income", "收入", PILOT_INCOME_CORRECTION_STATE) if income else (
        "trade", "成交", PILOT_TRADE_CORRECTION_STATE)
    display_row = _pilot_income_display if income else _pilot_trade_display
    date_key = "booked_on" if income else "executed_on"
    st.markdown(f"##### {category}更正")
    revisions = {row["operation_id"]: row for row in history[f"{kind}_entries"]}
    if history["corrections"]:
        with st.expander(f"查看{category}更正歷史"):
            st.dataframe([
                {"更正時間": event["corrected_at"], "更正理由": event["reason"],
                 "前版識別": event["previous_operation_id"], "更正版識別": event["operation_id"]}
                for event in history["corrections"]
            ], hide_index=True)
            st.dataframe([
                {"版本識別": row["operation_id"], **display_row(row)} for row in revisions.values()
            ], hide_index=True)
    if revisions and st.button(f"準備匯出完整{category}歷史"):
        now = pd.Timestamp.now(tz="UTC")
        content = json.dumps({"schema_version": 2, "exported_at_utc": now.isoformat(), **history},
                             ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        _render_portfolio_export_link(f"gvi-{kind}-history-{now:%Y%m%dT%H%M%SZ}.json", content,
                                      label=f"下載完整{category}歷史（JSON）", title=f"完整{category}歷史匯出")

    owner = account["supabase_user_id"]
    state = st.session_state.get(state_key)
    if state is None or state["owner"] != owner:
        state = {"owner": owner, "form_id": uuid4().hex}
        st.session_state[state_key] = state
    if state.pop("saved", False):
        st.success(f"{category}更正已儲存。")
    error = state.pop("error", None)
    if error:
        st.error(error)
    if "pending" in state:
        st.warning("尚未確認更正儲存結果。請重試同一筆操作。")
        if st.button(f"重試儲存{category}更正"):
            _submit_pilot_correction(account, state, kind, category)
        return

    active = {row["operation_id"]: row for row in rows}
    draft = state.get("draft")
    if state.get("preview") and draft:
        previous = draft["p_previous_operation_id"]
        if previous not in active:
            st.warning(f"原版本已有更新。請重新選擇{category}明細。")
            if st.button(f"重新選擇{category}明細"):
                state.pop("draft")
                state.pop("preview")
                st.rerun()
            return
        before = display_row(active[previous])
        after = display_row({key.removeprefix("p_"): value for key, value in draft.items()})
        st.dataframe([{"欄位": key, "目前版本": before[key], "更正後": after[key]} for key in before],
                     hide_index=True)
        st.text(f"更正理由：{draft['p_reason']}")
        confirmed = st.checkbox("我已核對更正前後內容", key=f"{kind}_confirm_" + draft["p_operation_id"])
        if st.button(f"返回編輯{category}更正"):
            state.pop("preview")
            st.rerun()
        if st.button(f"確認儲存{category}更正", disabled=not confirmed):
            state["pending"] = state.pop("draft")
            state.pop("preview")
            _submit_pilot_correction(account, state, kind, category)
        return

    if not active:
        return
    options = list(active)
    draft_target = draft["p_previous_operation_id"] if draft else None
    def option_label(key):
        row = active[key]
        label = (f"{PILOT_INCOME_TYPES[row['income_type']]} · {row['currency']} {row['net_amount']}" if income else
                 f"{row['symbol']} · {PILOT_TRADE_SIDES[row['side']]}")
        return f"{row[date_key]} · {label} · {key[:8]}"

    selected = st.selectbox(f"選擇要更正的{category}", options,
                            index=options.index(draft_target) if draft_target in active else None,
                            key=f"{kind}_correction_selection", format_func=option_label)
    if selected is None:
        return
    same_draft = draft is not None and draft["p_previous_operation_id"] == selected
    initial = {key.removeprefix("p_"): value for key, value in draft.items()} if same_draft else active[selected]
    try:
        date.fromisoformat(initial[date_key])
    except (TypeError, ValueError):
        st.error(f"無法編輯這筆{category}的日期。請核對歷史紀錄。")
        return
    with st.form(f"pilot_{kind}_correction_{state['form_id']}_{selected}"):
        revised = _pilot_income_correction_fields(initial) if income else _pilot_trade_form_fields(initial)
        reason = st.text_area("更正理由", value=draft["p_reason"] if same_draft else "", max_chars=500,
                              key=f"{kind}_reason_{state['form_id']}_{selected}").strip()
        if st.form_submit_button(f"預覽{category}更正"):
            if not reason:
                st.error("請填寫更正理由。")
            elif not (_valid_pilot_income_correction(revised) if income else _valid_pilot_trade_form(revised)):
                st.error("請核對入帳日與大於零的原幣實收金額。" if income else PILOT_TRADE_FORM_ERROR)
            else:
                state["draft"] = {"p_previous_operation_id": selected, "p_operation_id": str(uuid4()),
                                  "p_reason": reason, **{f"p_{key}": value for key, value in revised.items()}}
                state["preview"] = True
                st.rerun()


def _render_pilot_trade(account):
    st.markdown("#### 買賣成交明細")
    correction_enabled = os.environ.get("GVI_PORTFOLIO_PILOT_TRADE_CORRECTION") == "1"
    mode = _pilot_activity_mode(account, "trade", correction_enabled)
    if mode == "新增":
        _render_pilot_trade_form(account)
        return

    history = None
    if correction_enabled:
        try:
            rows, history = _read_pilot_correction_history(account, "trade", _validate_pilot_trade_rows)
        except (RuntimeError, ValueError, TypeError, requests.RequestException):
            st.warning("暫時無法讀取成交更正歷史。")
    if mode == "更正":
        if history is not None:
            _render_pilot_correction(account, rows, history)
        return
    if history is None:
        try:
            rows = _read_pilot_trade_entries(account)
        except (RuntimeError, ValueError, TypeError, requests.RequestException):
            st.error("暫時無法讀取買賣成交明細。請稍後再試。")
            return

    if rows:
        st.dataframe([_pilot_trade_display(row) for row in rows], hide_index=True)
        if st.button("準備匯出買賣成交明細"):
            now = pd.Timestamp.now(tz="UTC")
            content = json.dumps({"schema_version": 1, "exported_at_utc": now.isoformat(), "trade_entries": rows},
                                 ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            _render_portfolio_export_link(f"gvi-trades-{now:%Y%m%dT%H%M%SZ}.json", content,
                                          label="下載買賣成交明細（JSON）", title="買賣成交明細匯出")
    else:
        st.info("尚無已儲存的買賣成交明細。")


def _render_pilot_trade_form(account):
    owner = account["supabase_user_id"]
    state = st.session_state.get(PILOT_TRADE_STATE)
    if state is None or state["owner"] != owner:
        state = {"owner": owner, "form_id": uuid4().hex}
        st.session_state[PILOT_TRADE_STATE] = state
    if state.pop("saved", False):
        st.success("買賣成交已儲存。")
    if "pending" in state:
        st.warning("尚未確認成交儲存結果。請重試同一筆操作。")
        if st.button("重試儲存買賣成交"):
            _submit_pilot_trade(account, state)
    if "pending" in state:
        return
    st.caption("記錄已成交的買賣供查閱與對帳。目前登記後持倉、現金和快照維持原值。")
    with st.form("pilot_trade_" + state["form_id"], clear_on_submit=False):
        values = _pilot_trade_form_fields()
        if st.form_submit_button("儲存買賣成交"):
            if not _valid_pilot_trade_form(values):
                st.error(PILOT_TRADE_FORM_ERROR)
                return
            state["pending"] = {"p_operation_id": str(uuid4()), **{f"p_{key}": value for key, value in values.items()}}
            _submit_pilot_trade(account, state)


def _render_portfolio_pilot(account, *, open_chart):
    try:
        data = _read_portfolio_pilot(account)
    except (KeyError, ValueError, OverflowError, RuntimeError, requests.RequestException):
        st.error("暫時無法讀取雲端投資組合。請稍後再試，或切回原有投資組合。")
        return

    scope = f"portfolio_cloud_{account['id']}"
    sections = ["持倉", "歷史快照"]
    if os.environ.get("GVI_PORTFOLIO_PILOT_INCOME") == "1":
        sections.append("股息與利息")
    if os.environ.get("GVI_PORTFOLIO_PILOT_TRADE") == "1":
        sections.append("買賣成交")
    sections.append("配置")
    section = st.radio("雲端投資組合功能", sections, horizontal=True, key=f"{scope}_section")

    if section == "持倉":
        _render_pilot_holdings(account, data, scope=scope, open_chart=open_chart)
    elif section == "歷史快照":
        st.markdown("#### 歷史快照")
        if os.environ.get("GVI_PORTFOLIO_PILOT_SNAPSHOT") == "1":
            _render_pilot_snapshot(account, data)
        if data["snapshots"]:
            labels = {**PILOT_SNAPSHOT_COLUMNS, "note": "備註"}
            st.dataframe([{label: row[key] for key, label in labels.items()}
                          for row in data["snapshots"]], hide_index=True)
        else:
            st.info("目前沒有歷史快照。")
    elif section == "股息與利息":
        _render_pilot_income(account)
    elif section == "買賣成交":
        _render_pilot_trade(account)
    else:
        if os.environ.get("GVI_PORTFOLIO_PILOT_SETTINGS_WRITE") == "1":
            _render_pilot_settings_form(account, data)
        elif data["settings"] is not None:
            st.markdown("#### 配置與風險規則")
            st.dataframe([{"設定": label, "數值": data["settings"][key]}
                          for key, label in PILOT_SETTING_LABELS.items()], hide_index=True)
        else:
            st.info("目前沒有雲端配置。")


def _render_pilot_holdings(account, data, *, scope, open_chart):
    actions = {
        label: renderer
        for label, flag, renderer in (
            ("新增持倉", "ADD", _render_pilot_position_form),
            ("修改持倉", "EDIT", _render_pilot_position_edit),
            ("刪除持倉", "DELETE", _render_pilot_position_delete),
        )
        if os.environ.get(f"GVI_PORTFOLIO_PILOT_POSITION_{flag}") == "1"
    }
    if actions:
        mode = st.radio("持倉操作", ("查看持倉", *actions), horizontal=True, key=f"{scope}_holdings_mode")
        if mode in actions:
            actions[mode](account, data)
            return

    positions = data["positions"]
    render_portfolio_research(positions, scope=f"cloud_{account['id']}",
                              source="雲端投資組合", open_chart=open_chart)
    if data["revision"] == 0:
        st.info("目前沒有雲端投資組合資料。")
    if positions:
        st.markdown("#### 持倉")
        labels = {key: label for key, label in PILOT_POSITION_COLUMNS.items() if key != "legacy_id"}
        st.dataframe([{label: row[key] for key, label in labels.items()} for row in positions], hide_index=True)
    if st.button("準備匯出我的投資組合", key="portfolio_prepare_export"):
        filename, content = _encode_portfolio_export(
            data["settings"] or {},
            [{key: value for key, value in row.items() if key != "legacy_id"} for row in positions],
            [{key: value for key, value in row.items() if key != "legacy_id"} for row in data["snapshots"]],
        )
        _render_portfolio_export_link(filename, content)
