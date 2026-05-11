from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
SCHEMA_MIGRATIONS = (
  "ALTER TABLE events ADD COLUMN rule_type TEXT NOT NULL DEFAULT 'ABOVE'",
  "ALTER TABLE events ADD COLUMN upper_bound REAL",
  "ALTER TABLE events ADD COLUMN event_interval TEXT NOT NULL DEFAULT '1m'",
  "ALTER TABLE events ADD COLUMN settlement_price REAL",
  "ALTER TABLE events ADD COLUMN settlement_quote_time INTEGER",
  "ALTER TABLE events ADD COLUMN settlement_source TEXT",
  "ALTER TABLE events ADD COLUMN strategy_key TEXT NOT NULL DEFAULT 'manual'",
  "ALTER TABLE events ADD COLUMN ai_quality_score REAL",
  "ALTER TABLE events ADD COLUMN ai_quality_passed INTEGER",
  "ALTER TABLE predictions ADD COLUMN strategy_key TEXT NOT NULL DEFAULT 'orderbook_notional_40m'",
  "ALTER TABLE predictions ADD COLUMN trade_quality_score REAL",
  "ALTER TABLE predictions ADD COLUMN trade_quality_passed INTEGER",
  "ALTER TABLE predictions ADD COLUMN trade_quality_gate TEXT",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate TEXT",
  "ALTER TABLE predictions ADD COLUMN high_winrate_rule TEXT",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate_passed INTEGER",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate_value REAL",
  "ALTER TABLE predictions ADD COLUMN high_winrate_gate_min REAL",
  "ALTER TABLE predictions ADD COLUMN entry_price REAL",
  "ALTER TABLE predictions ADD COLUMN exit_price REAL",
  "ALTER TABLE predictions ADD COLUMN actual_return REAL",
  "ALTER TABLE predictions ADD COLUMN prediction_correct INTEGER",
  "ALTER TABLE predictions ADD COLUMN settled_at TEXT",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_gate TEXT",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_passed INTEGER",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_value REAL",
  "ALTER TABLE orders ADD COLUMN external_order_id TEXT",
  "ALTER TABLE orders ADD COLUMN external_status TEXT",
  "ALTER TABLE orders ADD COLUMN external_response TEXT",
  "ALTER TABLE auto_trade_settings ADD COLUMN live_trading_enabled INTEGER NOT NULL DEFAULT 0",
  "ALTER TABLE orderbook_features ADD COLUMN best_bid REAL",
  "ALTER TABLE orderbook_features ADD COLUMN best_ask REAL",
  "ALTER TABLE orderbook_features ADD COLUMN best_bid_qty REAL",
  "ALTER TABLE orderbook_features ADD COLUMN best_ask_qty REAL",
  "ALTER TABLE orderbook_features ADD COLUMN microprice REAL",
  "ALTER TABLE orderbook_features ADD COLUMN microprice_bps REAL",
  "ALTER TABLE orderbook_features ADD COLUMN ofi REAL",
  "ALTER TABLE orderbook_features ADD COLUMN ofi_ratio REAL",
  "ALTER TABLE orderbook_features ADD COLUMN quote_time INTEGER",
)


def get_conn() -> sqlite3.Connection:
  conn = sqlite3.connect(DB_PATH, check_same_thread=False)
  conn.row_factory = sqlite3.Row
  return conn


def init_db() -> None:
  conn = get_conn()
  with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    conn.executescript(f.read())
  _apply_schema_migrations(conn)
  _ensure_auto_trade_settings(conn)
  _ensure_auto_trade_strategies(conn)
  _ensure_ai_event_columns(conn)
  conn.commit()
  conn.close()


def _apply_schema_migrations(conn: sqlite3.Connection) -> None:
  for sql in SCHEMA_MIGRATIONS:
    try:
      conn.execute(sql)
    except sqlite3.OperationalError:
      pass


def _ensure_auto_trade_settings(conn: sqlite3.Connection) -> None:
  conn.execute(
    """
    INSERT OR IGNORE INTO auto_trade_settings(
      id, enabled, live_trading_enabled, symbol, duration, duration_minutes, qty, updated_at
    )
    VALUES(1, 0, 0, 'BTCUSDT', '10m', 10, 5, datetime('now'))
    """
  )


def _ensure_auto_trade_strategies(conn: sqlite3.Connection) -> None:
  _rename_orderbook_notional_strategy(conn)
  _delete_retired_auto_trade_strategies(conn)
  default_exists = conn.execute(
    "SELECT 1 FROM auto_trade_strategies WHERE strategy_key = ?",
    ("orderbook_notional_40m",),
  ).fetchone()
  seed_rows = (
    ("orderbook_notional_40m",),
    ("orderbook_notional_40m_mg",),
    ("orderbook_notional_10m_mg_5102045",),
    ("orderbook_trade_flow_1k",),
    ("orderbook_trade_flow_1k_invert_mg",),
  )
  conn.executemany(
    """
    INSERT OR IGNORE INTO auto_trade_strategies(
      strategy_key, enabled, live_trading_enabled, symbol, duration, duration_minutes, qty, updated_at
    )
    VALUES(?, 0, 0, 'BTCUSDT', '10m', 10, 5, datetime('now'))
    """,
    seed_rows,
  )
  if default_exists is None:
    _copy_legacy_auto_trade_settings(conn)


def _delete_retired_auto_trade_strategies(conn: sqlite3.Connection) -> None:
  conn.execute(
    "DELETE FROM auto_trade_strategies WHERE strategy_key = ?",
    ("complete_day_10m_production",),
  )
  for key in (
      "vegas_fib_resonance",
      "high_winrate_rules",
      "pure_rule_precision",
      "win70_trade_max_rules",
      "daily_trade_floor_tree",
  ):
    conn.execute(
        "DELETE FROM auto_trade_strategies WHERE strategy_key = ?",
        (key,),
    )


def _rename_orderbook_notional_strategy(conn: sqlite3.Connection) -> None:
  exists = conn.execute(
    "SELECT 1 FROM auto_trade_strategies WHERE strategy_key = ?",
    ("orderbook_notional_40m",),
  ).fetchone()
  if exists is not None:
    return
  conn.execute(
    """
    UPDATE auto_trade_strategies
    SET strategy_key = ?
    WHERE strategy_key = ?
    """,
    ("orderbook_notional_40m", "orderbook_notional_50m"),
  )


def _copy_legacy_auto_trade_settings(conn: sqlite3.Connection) -> None:
  conn.execute(
    """
    UPDATE auto_trade_strategies
    SET
      enabled = (SELECT enabled FROM auto_trade_settings WHERE id = 1),
      live_trading_enabled = (SELECT live_trading_enabled FROM auto_trade_settings WHERE id = 1),
      symbol = (SELECT symbol FROM auto_trade_settings WHERE id = 1),
      duration = (SELECT duration FROM auto_trade_settings WHERE id = 1),
      duration_minutes = (SELECT duration_minutes FROM auto_trade_settings WHERE id = 1),
      qty = (SELECT qty FROM auto_trade_settings WHERE id = 1),
      updated_at = datetime('now')
    WHERE strategy_key = 'orderbook_notional_40m'
    """
  )


def _ensure_ai_event_columns(conn: sqlite3.Connection) -> None:
  cols = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
  if "ai_probability_up" in cols:
    return
  conn.execute("ALTER TABLE events ADD COLUMN ai_probability_up REAL")
  conn.execute("ALTER TABLE events ADD COLUMN ai_predicted_direction TEXT")
  conn.execute("ALTER TABLE events ADD COLUMN ai_prediction_correct INTEGER")
  conn.execute("DELETE FROM settlements")
  conn.execute("DELETE FROM orders")
  conn.execute("DELETE FROM events")
