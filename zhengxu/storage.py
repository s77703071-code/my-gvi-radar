"""SQLite startup and personal portfolio export."""

from contextlib import closing
from pathlib import Path
import json
import os
import pandas as pd
import sqlite3
import sys


if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = str(Path(__file__).resolve().parents[1])
DB_FILE = os.path.join(BASE_DIR, "market_cache.db")
CACHE_TTL = 86400  # 快取有效期限：24 小時 (86400 秒)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS market_data
                 (ticker TEXT PRIMARY KEY, date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                  gvi REAL, roe REAL, mcap REAL, shares REAL, book_value REAL,
                  roe_source TEXT, bv_source TEXT, updated_at INTEGER)''')
    # Public daily institutional observations are shared market data, not account-specific portfolio data.
    c.execute('''CREATE TABLE IF NOT EXISTS chip_flow_observations (
        symbol TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        market TEXT NOT NULL DEFAULT '',
        foreign_net REAL,
        trust_net REAL,
        dealer_net REAL,
        issued_shares REAL,
        volume_shares REAL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(symbol, trade_date)
    )''')
    # Keep public-market Top 10 snapshots so repeated appearances can be
    # counted from observed report dates without inventing missing sessions.
    c.execute('''CREATE TABLE IF NOT EXISTS institutional_top10_appearances (
        snapshot_date TEXT NOT NULL,
        investor TEXT NOT NULL,
        ranking_type TEXT NOT NULL,
        ticker TEXT NOT NULL,
        stock_name TEXT NOT NULL DEFAULT '',
        market TEXT NOT NULL DEFAULT '',
        data_date TEXT NOT NULL DEFAULT '',
        rank_value REAL,
        captured_at TEXT NOT NULL,
        PRIMARY KEY(snapshot_date, investor, ranking_type, ticker)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS institutional_top10_snapshot_days (
        snapshot_date TEXT NOT NULL,
        investor TEXT NOT NULL,
        ranking_type TEXT NOT NULL,
        captured_at TEXT NOT NULL,
        PRIMARY KEY(snapshot_date, investor, ranking_type)
    )''')
    chip_cols = [row[1] for row in c.execute("PRAGMA table_info(chip_flow_observations)").fetchall()]
    if "volume_shares" not in chip_cols:
        c.execute("ALTER TABLE chip_flow_observations ADD COLUMN volume_shares REAL")

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
    # 投資組合理財：獨立資料表，沿用既有 market_cache.db，不影響原有 market_data。
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symbol TEXT NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        asset_type TEXT NOT NULL,
        currency TEXT NOT NULL DEFAULT 'TWD',
        quantity REAL NOT NULL DEFAULT 0,
        price REAL NOT NULL DEFAULT 0,
        fx_to_twd REAL NOT NULL DEFAULT 1,
        cost_basis_twd REAL NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_settings (
        setting_key TEXT PRIMARY KEY,
        setting_value REAL NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        snapshot_date TEXT NOT NULL,
        total_value_twd REAL NOT NULL,
        stock_value_twd REAL NOT NULL,
        bond_value_twd REAL NOT NULL,
        cash_value_twd REAL NOT NULL,
        note TEXT DEFAULT ''
    )''')
    # Account credentials use salted PBKDF2 hashes. Portfolio rows are scoped
    # by user_id; legacy rows without an owner stay unassigned.
    c.execute('''CREATE TABLE IF NOT EXISTS app_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        email TEXT,
        password_salt TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio_user_settings (
        user_id INTEGER NOT NULL,
        setting_key TEXT NOT NULL,
        setting_value REAL NOT NULL,
        PRIMARY KEY(user_id, setting_key),
        FOREIGN KEY(user_id) REFERENCES app_users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS account_email_challenges (
        user_id INTEGER NOT NULL,
        purpose TEXT NOT NULL,
        pending_email TEXT,
        code_salt TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        expires_at REAL NOT NULL,
        sent_at REAL NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(user_id, purpose),
        FOREIGN KEY(user_id) REFERENCES app_users(id)
    )''')
    user_cols = [row[1] for row in c.execute("PRAGMA table_info(app_users)").fetchall()]
    if "email" not in user_cols:
        c.execute("ALTER TABLE app_users ADD COLUMN email TEXT")
    for table in ("portfolio_positions", "portfolio_snapshots"):
        table_cols = [row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()]
        if "user_id" not in table_cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER")
    defaults = {
        "target_stock_pct": 40.0,
        "target_bond_pct": 50.0,
        "target_cash_pct": 10.0,
        "rebalance_threshold_pct": 5.0,
        "drawdown_warning_pct": 8.0,
        "drawdown_review_pct": 12.0,
        "drawdown_limit_pct": 15.0,
        "minimum_cash_twd": 0.0,
    }
    for key, value in defaults.items():
        c.execute("INSERT OR IGNORE INTO portfolio_settings(setting_key, setting_value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def _export_portfolio(user_id):
    database_uri = Path(DB_FILE).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(database_uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        settings = dict(conn.execute(
            "SELECT setting_key, setting_value FROM portfolio_user_settings "
            "WHERE user_id=? ORDER BY setting_key",
            (user_id,),
        ).fetchall())
        positions = _read_legacy_positions(conn, user_id)
        snapshots = [dict(row) for row in conn.execute(
            "SELECT snapshot_date, total_value_twd, stock_value_twd, bond_value_twd, "
            "cash_value_twd, note FROM portfolio_snapshots WHERE user_id=? ORDER BY id",
            (user_id,),
        )]

    return _encode_portfolio_export(settings, positions, snapshots)


def _read_legacy_positions(conn, user_id):
    return [dict(row) for row in conn.execute(
        "SELECT symbol, name, asset_type, currency, quantity, price, fx_to_twd, "
        "cost_basis_twd, updated_at FROM portfolio_positions WHERE user_id=? ORDER BY id",
        (user_id,),
    )]


def _encode_portfolio_export(settings, positions, snapshots):
    exported_at = pd.Timestamp.now(tz="UTC")
    content = json.dumps({
        "schema_version": 1,
        "exported_at_utc": exported_at.isoformat(),
        "positions": positions,
        "settings": settings,
        "snapshots": snapshots,
    }, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    return f"gvi-portfolio-{exported_at:%Y%m%dT%H%M%SZ}.json", content
