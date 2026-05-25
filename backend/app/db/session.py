from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data.db"
_DB_WRITE_LOCK = threading.Lock()

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
  "ALTER TABLE predictions ADD COLUMN strategy_key TEXT NOT NULL DEFAULT 'factor_combo_ranker_v1'",
  "ALTER TABLE predictions ADD COLUMN signal_key TEXT",
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
  "ALTER TABLE futures_positioning_features ADD COLUMN open_interest_chg_1 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN open_interest_value_chg_1 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN open_interest_z_20 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN long_short_ratio_chg_1 REAL",
  "ALTER TABLE futures_positioning_features ADD COLUMN taker_buy_share REAL",
  "ALTER TABLE market_sentiment_features ADD COLUMN fear_greed_chg_1 REAL",
  "ALTER TABLE market_sentiment_features ADD COLUMN fear_greed_z_30 REAL",
  "ALTER TABLE onchain_features ADD COLUMN exchange_netflow_z_20 REAL",
  "ALTER TABLE onchain_features ADD COLUMN active_addresses_chg_1 REAL",
  "ALTER TABLE onchain_features ADD COLUMN transaction_count_chg_1 REAL",
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
    open_interest_chg_1 REAL,
    open_interest_value_chg_1 REAL,
    open_interest_z_20 REAL,
    long_short_ratio_chg_1 REAL,
    taker_buy_share REAL,
    PRIMARY KEY (symbol, open_time)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS market_sentiment_features (
    source TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    fear_greed_value REAL,
    fear_greed_classification TEXT,
    fear_greed_chg_1 REAL,
    fear_greed_z_30 REAL,
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
    exchange_netflow_z_20 REAL,
    active_addresses_chg_1 REAL,
    transaction_count_chg_1 REAL,
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
  """
  CREATE TABLE IF NOT EXISTS factor_combo_signal_cache (
    symbol TEXT NOT NULL PRIMARY KEY,
    updated_at TEXT NOT NULL,
    top_per_duration INTEGER NOT NULL,
    limit_count INTEGER NOT NULL,
    payload TEXT NOT NULL
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS factor_combo_backtest_cache (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    factor_name TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol, duration, factor_name)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS high_winrate_combo_ranking_cache (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    total INTEGER NOT NULL,
    search_config TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS high_winrate_strategy_status (
    strategy_key TEXT NOT NULL,
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    win_rate REAL,
    profit_factor REAL,
    consecutive_losses INTEGER NOT NULL,
    evaluated_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (strategy_key, symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS ensemble_stage_status (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    stage TEXT NOT NULL,
    recommended_stage TEXT NOT NULL,
    recommendation_reason TEXT NOT NULL,
    confirmed_stage TEXT,
    confirmed_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, duration)
  )
  """,
  """
  CREATE TABLE IF NOT EXISTS ensemble_signal_scores (
    symbol TEXT NOT NULL,
    duration TEXT NOT NULL,
    signal_key TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    win_rate REAL NOT NULL,
    avg_return REAL NOT NULL,
    profit_factor REAL NOT NULL,
    consecutive_losses INTEGER NOT NULL,
    stability_score REAL NOT NULL,
    weight_suggestion REAL NOT NULL,
    score REAL NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (symbol, duration, signal_key)
  )
  """,
  "ALTER TABLE events ADD COLUMN prediction_open_time INTEGER",
  "CREATE INDEX IF NOT EXISTS idx_predictions_symbol_duration_open ON predictions(symbol, duration, open_time)",
  "CREATE INDEX IF NOT EXISTS idx_predictions_settled ON predictions(symbol, duration, settled_at)",
  "CREATE INDEX IF NOT EXISTS idx_events_shadow_pairing ON events(symbol, event_interval, status, prediction_open_time)",
  "CREATE INDEX IF NOT EXISTS idx_events_settled_strategy ON events(symbol, event_interval, strategy_key, status)",
  "CREATE INDEX IF NOT EXISTS idx_orders_event_id ON orders(event_id)",
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


@contextmanager
def db_write_lock():
  """Serialize SQLite writes across threads."""
  _DB_WRITE_LOCK.acquire()
  try:
    yield
  finally:
    _DB_WRITE_LOCK.release()


def run_db_write_with_retry(operation, *, attempts: int = 6, base_delay: float = 0.05):
  delay = base_delay
  for attempt in range(attempts):
    try:
      with db_write_lock():
        return operation()
    except sqlite3.OperationalError as exc:
      message = str(exc).lower()
      if "locked" not in message or attempt == attempts - 1:
        raise
      time.sleep(delay)
      delay = min(delay * 2, 1.0)


def init_db() -> None:
  conn = get_conn()
  with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    conn.executescript(f.read())
  _apply_schema_migrations(conn)
  _ensure_prediction_signal_keys(conn)
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
  _delete_retired_auto_trade_strategies(conn)
  from app.services.strategy_registry import strategy_payloads
  from app.services.auto_trade_default_slots import (
    default_slot_flags,
    disable_simulation_only_live_trading,
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
  disable_simulation_only_live_trading(conn, _AUTO_TRADE_SLOT_DURATIONS)


def _delete_retired_auto_trade_strategies(conn: sqlite3.Connection) -> None:
  retired_keys = [
      "complete_day_10m_production",
      "vegas_fib_resonance",
      "high_winrate_rules",
      "pure_rule_precision",
      "win70_trade_max_rules",
      "daily_trade_floor_tree",
      "orderbook_notional_40m",
      "orderbook_notional_40m_mg",
      "orderbook_notional_10m_mg_5102045",
      "orderbook_notional_10m",
      "orderbook_notional_15m",
      "orderbook_notional_15m_mg_51020",
      "orderbook_trade_flow_1k",
      "orderbook_trade_flow_1k_invert_mg",
      "blind_reverse_martingale_v1",
      "three_bar_10m_reverse_martingale_v1",
      "four_bar_10m_reverse_martingale_v1",
      "five_bar_10m_reverse_martingale_v1",
      "high_winrate_factor_combo_v1",
  ]
  for key in retired_keys:
    conn.execute(
        "DELETE FROM auto_trade_strategies WHERE strategy_key = ?",
        (key,),
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
