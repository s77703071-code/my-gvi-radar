"""Account identity, recovery and authenticated platform requests."""

from base64 import urlsafe_b64decode
import hashlib
import hmac
import json
import os
import pandas as pd
import re
import requests
import secrets
import sqlite3
import streamlit as st
import time

from .research_ai import (
    get_gmail_secret_values,
    send_gmail_report,
)
from .storage import (
    DB_FILE,
)


def _password_digest(password, salt_hex):
    # Retained for the legacy in-app email-binding challenge flow only.
    # Account passwords are verified exclusively by Supabase Auth.
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 310_000
    ).hex()


def _supabase_auth_config():
    """Read Supabase Auth credentials from Streamlit Secrets or environment."""
    try:
        url = str(st.secrets.get("SUPABASE_URL", "") or "").strip().rstrip("/")
        key = str(st.secrets.get("SUPABASE_ANON_KEY", "") or "").strip()
    except Exception:
        url, key = "", ""
    url = url or os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = key or os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        raise RuntimeError(
            "尚未設定 Supabase。請在 Streamlit Cloud → Settings → Secrets 設定 "
            "SUPABASE_URL 與 SUPABASE_ANON_KEY。"
        )
    if not re.match(r"^https://[A-Za-z0-9.-]+$", url):
        raise RuntimeError("SUPABASE_URL 格式不正確，請確認是 Supabase 專案 URL。")
    return url, key


def _supabase_auth_request(path, payload=None, access_token=None, method="POST"):
    url, anon_key = _supabase_auth_config()
    headers = {"apikey": anon_key, "Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    else:
        headers["Authorization"] = f"Bearer {anon_key}"
    response = requests.request(
        method.upper(),
        f"{url}/auth/v1/{path.lstrip('/')}",
        headers=headers,
        json=payload,
        timeout=20,
    )
    try:
        body = response.json()
    except Exception:
        body = {}
    if response.status_code >= 400:
        message = body.get("msg") or body.get("message") or body.get("error_description") or body.get("error")
        if response.status_code in (400, 401, 422):
            raise ValueError(str(message or "帳號或密碼不正確。"))
        raise RuntimeError(str(message or f"Supabase Auth 連線失敗（HTTP {response.status_code}）。"))
    return body


def _stable_local_user_id(auth_user_id):
    """Stable positive integer for legacy SQLite user_id columns; derived from Supabase UUID."""
    digest = hashlib.sha256(str(auth_user_id).encode("utf-8")).hexdigest()
    return int(digest[:15], 16)


def _ensure_local_account(auth_user):
    """Create a local compatibility row; credentials remain exclusively in Supabase Auth."""
    auth_id = str(auth_user.get("id") or "").strip()
    if not auth_id:
        raise RuntimeError("Supabase 沒有回傳有效的使用者 ID。")
    metadata = auth_user.get("user_metadata") or auth_user.get("user", {}).get("user_metadata") or {}
    email = str(auth_user.get("email") or auth_user.get("user", {}).get("email") or "").strip().lower()
    username = str(metadata.get("username") or email.split("@")[0] or "user").strip()
    user_id = _stable_local_user_id(auth_id)
    defaults = {
        "target_stock_pct": 40.0, "target_bond_pct": 50.0,
        "target_cash_pct": 10.0, "rebalance_threshold_pct": 5.0,
        "drawdown_warning_pct": 8.0, "drawdown_review_pct": 12.0,
        "drawdown_limit_pct": 15.0, "minimum_cash_twd": 0.0,
    }
    salt = os.urandom(16).hex()
    # This random digest is never used for authentication. Supabase Auth verifies passwords.
    placeholder_hash = _password_digest(secrets.token_urlsafe(32), salt)
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute("SELECT id FROM app_users WHERE id=?", (user_id,)).fetchone()
        if not existing:
            local_name = username[:32]
            conflict = conn.execute(
                "SELECT 1 FROM app_users WHERE username=? COLLATE NOCASE", (local_name,)
            ).fetchone()
            if conflict:
                local_name = f"{username[:22]}_{str(user_id)[-8:]}"
            conn.execute(
                "INSERT INTO app_users(id,username,email,password_salt,password_hash,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (user_id, local_name, email, salt, placeholder_hash,
                 pd.Timestamp.now().isoformat(timespec="seconds")),
            )
        for key, value in defaults.items():
            conn.execute(
                "INSERT OR IGNORE INTO portfolio_user_settings(user_id,setting_key,setting_value) "
                "VALUES(?,?,?)", (user_id, key, value)
            )
        conn.commit()
    return user_id, username, email


def _register_account(username, password, email):
    """Register through Supabase Auth; never store a reusable password hash locally."""
    result = _supabase_auth_request(
        "signup",
        {"email": email, "password": password, "data": {"username": username}},
    )
    auth_user = result.get("user") or {}
    session = result.get("session")
    # Supabase may require email confirmation before creating a session.
    if not session or not auth_user.get("id"):
        return None
    user_id, _, _ = _ensure_local_account(auth_user)
    return user_id


def _verify_account(email, password):
    """Authenticate with Supabase; login identifier is the registered email address."""
    try:
        result = _supabase_auth_request(
            "token?grant_type=password", {"email": email, "password": password}
        )
    except (ValueError, RuntimeError):
        return None
    auth_user = result.get("user") or {}
    if not auth_user.get("id"):
        return None
    try:
        user_id, username, user_email = _ensure_local_account(auth_user)
    except Exception:
        return None
    return {
        "id": user_id,
        "username": username,
        "email": user_email,
        "supabase_user_id": str(auth_user["id"]),
        "access_token": result.get("access_token", ""),
    }


DUAL_GRID_PROJECT_REF = "aytlgxlbbetzxilyeuho"
DUAL_GRID_SUPABASE_URL = f"https://{DUAL_GRID_PROJECT_REF}.supabase.co"


def _dual_grid_public_api_key(key):
    if re.fullmatch(r"sb_publishable_[A-Za-z0-9_-]+", key):
        return True
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", key):
        return False
    encoded = key.split(".")[1]
    try:
        claims = json.loads(urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except ValueError:
        return False
    return (isinstance(claims, dict) and claims.get("role") == "anon"
            and claims.get("ref") == DUAL_GRID_PROJECT_REF)


def _dual_grid_permission_request(account, route, payload=None):
    token = account.get("access_token")
    if not token:
        return 401, None
    try:
        url, key = _supabase_auth_config()
    except RuntimeError:
        return 0, None
    if url != DUAL_GRID_SUPABASE_URL or not _dual_grid_public_api_key(key):
        return 0, None

    method = "GET" if payload is None else "POST"
    profile = "Accept-Profile" if method == "GET" else "Content-Profile"
    try:
        response = requests.request(
            method, f"{url}/rest/v1/rpc/{route}",
            headers={"apikey": key, "Authorization": f"Bearer {token}", profile: "gvi_access_api"},
            json=payload, timeout=(5, 12), allow_redirects=False,
        )
        return response.status_code, response.json() if response.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return 0, None


def _request_supabase_password_reset(email):
    """Ask Supabase Auth to send its recovery email."""
    _supabase_auth_request("recover", {"email": email})


def _complete_supabase_password_reset(email, token, new_password):
    """Verify Supabase recovery OTP, then set a new password using its short-lived token."""
    verified = _supabase_auth_request(
        "verify", {"type": "recovery", "email": email, "token": token}
    )
    access_token = str(verified.get("access_token") or "")
    if not access_token:
        raise ValueError("驗證碼無效或已逾時，請重新寄送密碼重設信。")
    _supabase_auth_request(
        "user", {"password": new_password}, access_token=access_token, method="PUT"
    )


def _request_account_code(user_id, purpose, email=None):
    """Send a short-lived one-time code for password reset or email binding."""
    user_id = int(user_id)
    now = time.time()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        row = conn.execute("SELECT email FROM app_users WHERE id=?", (user_id,)).fetchone()
        if not row:
            return False
        target_email = str(email or row[0] or "").strip().lower()
        if not target_email:
            return False
        previous = conn.execute(
            "SELECT sent_at FROM account_email_challenges WHERE user_id=? AND purpose=?",
            (user_id, purpose),
        ).fetchone()
        if previous and now - float(previous[0]) < 60:
            return False
    code = f"{secrets.randbelow(1_000_000):06d}"
    salt = os.urandom(16).hex()
    digest = _password_digest(code, salt)
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute(
            "INSERT INTO account_email_challenges(user_id,purpose,pending_email,code_salt,code_hash,expires_at,sent_at,attempts) "
            "VALUES(?,?,?,?,?,?,?,0) ON CONFLICT(user_id,purpose) DO UPDATE SET "
            "pending_email=excluded.pending_email,code_salt=excluded.code_salt,code_hash=excluded.code_hash,"
            "expires_at=excluded.expires_at,sent_at=excluded.sent_at,attempts=0",
            (user_id, purpose, target_email if purpose == "bind_email" else None,
             salt, digest, now + 600, now),
        )
        conn.commit()
    try:
        sender, app_password = get_gmail_secret_values(st.secrets)
        if not sender or not app_password:
            raise RuntimeError("尚未設定 Gmail 寄件資料。請在 Streamlit Secrets 設定 [gmail] sender_email 與 app_password。")
        purpose_text = "密碼重設" if purpose == "password_reset" else "信箱驗證"
        send_gmail_report(
            target_email, f"操盤終端{purpose_text}驗證碼",
            f"您的驗證碼是：{code}\n\n驗證碼 10 分鐘內有效，請勿提供給他人。若非本人操作，請忽略此信。",
            sender, app_password,
        )
    except Exception:
        with sqlite3.connect(DB_FILE) as conn:
            conn.execute("DELETE FROM account_email_challenges WHERE user_id=? AND purpose=?", (user_id, purpose))
            conn.commit()
        raise
    return True


def _check_email_code(user_id, purpose, code):
    now = time.time()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        row = conn.execute(
            "SELECT pending_email,code_salt,code_hash,expires_at,attempts FROM account_email_challenges "
            "WHERE user_id=? AND purpose=?", (int(user_id), purpose),
        ).fetchone()
        if not row:
            return None
        pending_email, salt, expected, expires_at, attempts = row
        if float(expires_at) < now or int(attempts) >= 5:
            conn.execute("DELETE FROM account_email_challenges WHERE user_id=? AND purpose=?", (int(user_id), purpose))
            conn.commit()
            return None
        if not hmac.compare_digest(_password_digest(str(code).strip(), salt), expected):
            conn.execute(
                "UPDATE account_email_challenges SET attempts=attempts+1 WHERE user_id=? AND purpose=?",
                (int(user_id), purpose),
            )
            conn.commit()
            return None
        conn.execute("DELETE FROM account_email_challenges WHERE user_id=? AND purpose=?", (int(user_id), purpose))
        conn.commit()
    return pending_email if purpose == "bind_email" else True


def _reset_account_password(username, code, new_password):
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        row = conn.execute("SELECT id FROM app_users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
    if not row or not _check_email_code(row[0], "password_reset", code):
        return False
    salt = os.urandom(16).hex()
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute(
            "UPDATE app_users SET password_salt=?,password_hash=? WHERE id=?",
            (salt, _password_digest(new_password, salt), int(row[0])),
        )
        conn.commit()
    return True


def _confirm_recovery_email(user_id, code):
    email = _check_email_code(user_id, "bind_email", code)
    if not email:
        return False
    with sqlite3.connect(DB_FILE, timeout=30) as conn:
        conn.execute("UPDATE app_users SET email=? WHERE id=?", (email, int(user_id)))
        conn.commit()
    return True
