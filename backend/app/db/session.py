from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.db.db_write_retry import DbWriteRetryExhausted, db_write_lock, run_db_write_with_retry
from app.db.schema_migrations import SCHEMA_MIGRATIONS
from app.services.runtime_symbols import configured_runtime_symbols

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data.db"

# 与 auto_trade_service.SUPPORTED_AUTO_DURATIONS、rule_config.DURATION_TO_MINUTES 对齐
_AUTO_TRADE_SLOT_DURATIONS: tuple[str, ...] = ("10m", "30m", "60m", "1d")
_DURATION_MINUTES = {"10m": 10, "30m": 30, "60m": 60, "1d": 1440}
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_conn() -> sqlite3.Connection:
  """timeout：等锁最长时间（秒）。WAL + busy_timeout 减轻多协程/多请求下的 database is locked。"""
  conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
  conn.row_factory = sqlite3.Row
  try:
    conn.execute("PRAGMA journal_mode=WAL")
  except sqlite3.Error as exc:
    conn.close()
    raise RuntimeError(f"failed to enable SQLite WAL mode for {DB_PATH}") from exc
  try:
    conn.execute("PRAGMA busy_timeout=30000")
  except sqlite3.Error as exc:
    conn.close()
    raise RuntimeError(f"failed to configure SQLite busy_timeout for {DB_PATH}") from exc
  return conn


def init_db() -> None:
  conn = get_conn()
  try:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
      conn.executescript(f.read())
    _apply_schema_migrations(conn)
    _ensure_prediction_signal_keys(conn)
    _migrate_auto_trade_strategies_composite_pk(conn)
    _ensure_auto_trade_settings(conn)
    _ensure_auto_trade_strategies(conn)
    _ensure_ai_event_columns(conn)
    conn.commit()
  finally:
    conn.close()


def _apply_schema_migrations(conn: sqlite3.Connection) -> None:
  for sql in SCHEMA_MIGRATIONS:
    try:
      conn.execute(sql)
    except sqlite3.OperationalError as exc:
      message = str(exc).lower()
      if "duplicate column name" in message or "already exists" in message:
        continue
      raise


def _ensure_prediction_signal_keys(conn: sqlite3.Connection) -> None:
  columns = {row["name"] for row in conn.execute("PRAGMA table_info(predictions)")}
  if "signal_key" not in columns:
    conn.execute("ALTER TABLE predictions ADD COLUMN signal_key TEXT")
  pending = conn.execute(
    """
    SELECT 1
    FROM predictions
    WHERE signal_key IS NULL OR signal_key = ''
    LIMIT 1
    """
  ).fetchone()
  if pending is None:
    return
  conn.execute(
    """
    UPDATE predictions
    SET signal_key = strategy_key
    WHERE signal_key IS NULL OR signal_key = ''
    """
  )


def _ensure_auto_trade_settings(conn: sqlite3.Connection) -> None:
  conn.execute(
    """
    INSERT OR IGNORE INTO auto_trade_settings(
      id, enabled, live_trading_enabled, symbol, duration, duration_minutes, qty, updated_at
    )
    VALUES(1, 0, 0, 'BTCUSDT', '10m', 10, 5, datetime('now'))
    """
  )


def _migrate_auto_trade_strategies_composite_pk(conn: sqlite3.Connection) -> None:
  """Rebuild auto trade slots to isolate strategy_key + symbol + duration."""
  if not _table_exists(conn, "auto_trade_strategies"):
    return
  if _auto_trade_strategy_pk(conn) == ("strategy_key", "symbol", "duration"):
    return
  columns = {row["name"] for row in conn.execute("PRAGMA table_info(auto_trade_strategies)")}
  if "symbol" not in columns:
    conn.execute("ALTER TABLE auto_trade_strategies ADD COLUMN symbol TEXT NOT NULL DEFAULT 'BTCUSDT'")
  conn.executescript(
    """
    DROP TABLE IF EXISTS auto_trade_strategies__migration;
    CREATE TABLE IF NOT EXISTS auto_trade_strategies__migration (
      strategy_key TEXT NOT NULL,
      symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
      duration TEXT NOT NULL DEFAULT '10m',
      enabled INTEGER NOT NULL DEFAULT 0,
      live_trading_enabled INTEGER NOT NULL DEFAULT 0,
      duration_minutes INTEGER NOT NULL DEFAULT 10,
      qty REAL NOT NULL DEFAULT 5,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (strategy_key, symbol, duration)
    );
    INSERT OR IGNORE INTO auto_trade_strategies__migration(
      strategy_key, symbol, duration, enabled, live_trading_enabled, duration_minutes, qty, updated_at
    )
    SELECT strategy_key, UPPER(COALESCE(NULLIF(symbol, ''), 'BTCUSDT')), duration,
           enabled, live_trading_enabled, duration_minutes, qty, updated_at
    FROM auto_trade_strategies;
    DROP TABLE auto_trade_strategies;
    ALTER TABLE auto_trade_strategies__migration RENAME TO auto_trade_strategies;
    """
  )


def _ensure_auto_trade_strategies(conn: sqlite3.Connection) -> None:
  _delete_retired_auto_trade_strategies(conn)
  from app.services.strategy_registry import strategy_payloads
  from app.services.auto_trade_default_slots import (
    default_slot_flags,
    disable_simulation_only_live_trading,
    enable_default_simulation_strategy_slots,
  )

  ts = datetime.now(timezone.utc).isoformat()
  payloads = strategy_payloads()
  for symbol in configured_runtime_symbols():
    for payload in payloads:
      key = str(payload["key"])
      for dur in _payload_durations(payload):
        dm = _DURATION_MINUTES[dur]
        enabled, live = default_slot_flags(key)
        conn.execute(
          """
          INSERT OR IGNORE INTO auto_trade_strategies(
            strategy_key, symbol, duration, enabled, live_trading_enabled, duration_minutes, qty, updated_at
          )
          VALUES(?, ?, ?, ?, ?, ?, 5, ?)
          """,
          (key, symbol, dur, enabled, live, dm, ts),
        )
  enable_default_simulation_strategy_slots(conn, _AUTO_TRADE_SLOT_DURATIONS, _DURATION_MINUTES, ts)
  disable_simulation_only_live_trading(conn, _AUTO_TRADE_SLOT_DURATIONS)


def _delete_retired_auto_trade_strategies(conn: sqlite3.Connection) -> None:
  from app.services.retired_strategy_keys import RETIRED_AUTO_TRADE_STRATEGY_KEYS

  for key in sorted(RETIRED_AUTO_TRADE_STRATEGY_KEYS):
    conn.execute(
        "DELETE FROM auto_trade_strategies WHERE strategy_key = ?",
        (key,),
    )


def _payload_durations(payload: dict) -> tuple[str, ...]:
  supported = set(payload.get("supportedDurations") or _AUTO_TRADE_SLOT_DURATIONS)
  return tuple(duration for duration in _AUTO_TRADE_SLOT_DURATIONS if duration in supported)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
  row = conn.execute(
    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
    (table_name,),
  ).fetchone()
  return row is not None


def _auto_trade_strategy_pk(conn: sqlite3.Connection) -> tuple[str, ...]:
  rows = conn.execute("PRAGMA table_info(auto_trade_strategies)").fetchall()
  pk_rows = sorted((int(row["pk"]), str(row["name"])) for row in rows if int(row["pk"]) > 0)
  return tuple(name for _order, name in pk_rows)


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
