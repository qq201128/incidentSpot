from __future__ import annotations

import sqlite3

from app.db.session import _ensure_auto_trade_strategies
from app.services.auto_trade_service import AUTO_TRADE_SLOT_DURATIONS
from app.services.rule_config import DURATION_TO_MINUTES
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, ORDERBOOK_NOTIONAL_STRATEGY_KEY


def test_db_seed_enables_existing_factor_combo_sim_slots() -> None:
    conn = _auto_trade_conn()
    for duration in AUTO_TRADE_SLOT_DURATIONS:
        _insert_strategy(conn, FACTOR_COMBO_STRATEGY_KEY, duration, enabled=0, live=0)

    _ensure_auto_trade_strategies(conn)

    rows = conn.execute(
        """
        SELECT duration, enabled, live_trading_enabled
        FROM auto_trade_strategies
        WHERE strategy_key = ?
        """,
        (FACTOR_COMBO_STRATEGY_KEY,),
    ).fetchall()
    by_duration = {row["duration"]: row for row in rows}
    assert set(by_duration) == set(AUTO_TRADE_SLOT_DURATIONS)
    assert all(by_duration[duration]["enabled"] == 1 for duration in AUTO_TRADE_SLOT_DURATIONS)
    assert all(by_duration[duration]["live_trading_enabled"] == 0 for duration in AUTO_TRADE_SLOT_DURATIONS)


def _auto_trade_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE auto_trade_strategies (
          strategy_key TEXT NOT NULL,
          duration TEXT NOT NULL DEFAULT '10m',
          enabled INTEGER NOT NULL DEFAULT 0,
          live_trading_enabled INTEGER NOT NULL DEFAULT 0,
          symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
          duration_minutes INTEGER NOT NULL DEFAULT 10,
          qty REAL NOT NULL DEFAULT 5,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (strategy_key, duration)
        )
        """
    )
    _insert_strategy(conn, ORDERBOOK_NOTIONAL_STRATEGY_KEY, "10m", enabled=0, live=0)
    return conn


def _insert_strategy(
    conn: sqlite3.Connection,
    key: str,
    duration: str,
    *,
    enabled: int,
    live: int,
) -> None:
    conn.execute(
        """
        INSERT INTO auto_trade_strategies(
          strategy_key, duration, enabled, live_trading_enabled, symbol, duration_minutes, qty, updated_at
        )
        VALUES(?, ?, ?, ?, 'BTCUSDT', ?, 5, '2026-05-13T00:00:00+00:00')
        """,
        (key, duration, enabled, live, DURATION_TO_MINUTES[duration]),
    )
