from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data.db"

# 与 auto_trade_service.SUPPORTED_AUTO_DURATIONS、rule_config.DURATION_TO_MINUTES 对齐
_AUTO_TRADE_SLOT_DURATIONS: tuple[str, ...] = ("10m", "30m", "60m", "1d")
_DURATION_MINUTES = {"10m": 10, "30m": 30, "60m": 60, "1d": 1440}
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
  "ALTER TABLE predictions ADD COLUMN expected_return REAL",
  "ALTER TABLE predictions ADD COLUMN model_version TEXT",
  "ALTER TABLE predictions ADD COLUMN feature_window INTEGER",
  "ALTER TABLE predictions ADD COLUMN model_duration TEXT",
  "ALTER TABLE predictions ADD COLUMN model_trained_at TEXT",
  "ALTER TABLE predictions ADD COLUMN exit_price REAL",
  "ALTER TABLE predictions ADD COLUMN actual_return REAL",
  "ALTER TABLE predictions ADD COLUMN prediction_correct INTEGER",
  "ALTER TABLE predictions ADD COLUMN settled_at TEXT",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_gate TEXT",
  "ALTER TABLE events ADD COLUMN ai_high_winrate_rule TEXT",
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
  """
  CREATE TABLE IF NOT EXISTS futures_positioning_features (
    symbol TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open_interest REAL,
    open_interest_value REAL,
    long_short_ratio REAL,
    long_account REAL,
    short_account REAL,
    taker_buy_sell_ratio REAL,
    taker_buy_vol REAL,
    taker_sell_vol REAL,
    PRIMARY KEY (symbol, open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS market_sentiment_features (
    source TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    fear_greed_value REAL,
    fear_greed_classification TEXT,
    PRIMARY KEY (source, open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS onchain_features (
    symbol TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    exchange_netflow REAL,
    stablecoin_supply_ratio REAL,
    active_addresses REAL,
    transaction_count REAL,
    PRIMARY KEY (symbol, open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS factor_combo_ranking_cache (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    total INTEGER NOT NULL,
    search_config TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol, duration)
  )
  """,
)


def get_conn() -> sqlite3.Connection:
  """timeout：等锁最长时间（秒）。WAL + busy_timeout 减轻多协程/多请求下的 database is locked。"""
  conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
  conn.row_factory = sqlite3.Row
  try:
    conn.execute("PRAGMA journal_mode=WAL")
  except sqlite3.Error:
    pass
  try:
    conn.execute("PRAGMA busy_timeout=30000")
  except sqlite3.Error:
    pass
  return conn


def init_db() -> None:
  conn = get_conn()
  with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    conn.executescript(f.read())
  _apply_schema_migrations(conn)
  _migrate_auto_trade_strategies_composite_pk(conn)
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


def _migrate_auto_trade_strategies_composite_pk(conn: sqlite3.Connection) -> None:
  """旧库仅有 PRIMARY KEY(strategy_key) 时重建为 (strategy_key, duration)。"""
  row = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name='auto_trade_strategies'"
  ).fetchone()
  sql = (row["sql"] or "") if row else ""
  normalized = sql.replace(" ", "").upper()
  if "PRIMARYKEY(strategy_key,duration)" in normalized or (
      "PRIMARYKEY(strategy_key," in normalized and "duration)" in normalized
  ):
    return
  if not sql:
    return
  conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS auto_trade_strategies__migration (
      strategy_key TEXT NOT NULL,
      duration TEXT NOT NULL DEFAULT '10m',
      enabled INTEGER NOT NULL DEFAULT 0,
      live_trading_enabled INTEGER NOT NULL DEFAULT 0,
      symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
      duration_minutes INTEGER NOT NULL DEFAULT 10,
      qty REAL NOT NULL DEFAULT 5,
      updated_at TEXT NOT NULL,
      PRIMARY KEY (strategy_key, duration)
    );
    INSERT OR IGNORE INTO auto_trade_strategies__migration(
      strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
    )
    SELECT strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
    FROM auto_trade_strategies;
    DROP TABLE auto_trade_strategies;
    ALTER TABLE auto_trade_strategies__migration RENAME TO auto_trade_strategies;
    """
  )


def _ensure_auto_trade_strategies(conn: sqlite3.Connection) -> None:
  _rename_orderbook_notional_strategy(conn)
  _delete_retired_auto_trade_strategies(conn)
  default_exists = conn.execute(
    "SELECT 1 FROM auto_trade_strategies WHERE strategy_key = ? LIMIT 1",
    ("orderbook_notional_40m",),
  ).fetchone()

  from app.services.strategy_registry import strategy_payloads
  from app.services.auto_trade_default_slots import (
    default_slot_flags,
    enable_default_simulation_strategy_slots,
  )

  ts = datetime.now(timezone.utc).isoformat()
  payloads = strategy_payloads()
  for payload in payloads:
    key = str(payload["key"])
    for dur in _payload_durations(payload):
      dm = _DURATION_MINUTES[dur]
      enabled, live = default_slot_flags(key)
      conn.execute(
        """
        INSERT OR IGNORE INTO auto_trade_strategies(
          strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, 'BTCUSDT', ?, 5, ?)
        """,
        (key, dur, enabled, live, dm, ts),
      )
  enable_default_simulation_strategy_slots(conn, _AUTO_TRADE_SLOT_DURATIONS, _DURATION_MINUTES, ts)
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
      duration_minutes = (SELECT duration_minutes FROM auto_trade_settings WHERE id = 1),
      qty = (SELECT qty FROM auto_trade_settings WHERE id = 1),
      updated_at = datetime('now')
    WHERE strategy_key = 'orderbook_notional_40m'
      AND duration = (SELECT duration FROM auto_trade_settings WHERE id = 1)
    """
  )


def _payload_durations(payload: dict) -> tuple[str, ...]:
  supported = set(payload.get("supportedDurations") or _AUTO_TRADE_SLOT_DURATIONS)
  return tuple(duration for duration in _AUTO_TRADE_SLOT_DURATIONS if duration in supported)


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
