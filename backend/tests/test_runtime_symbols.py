from __future__ import annotations

import sqlite3

from app.db.session import _migrate_auto_trade_strategies_composite_pk


def test_migrates_auto_trade_strategy_pk_to_symbol_duration() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
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
        );
        INSERT INTO auto_trade_strategies
        VALUES('factor_combo_ranker_v1', '10m', 1, 0, 'BTCUSDT', 10, 5, 'now');
        """
    )

    _migrate_auto_trade_strategies_composite_pk(conn)

    pk = _primary_key(conn, "auto_trade_strategies")
    row = conn.execute("SELECT * FROM auto_trade_strategies").fetchone()

    assert pk == ("strategy_key", "symbol", "duration")
    assert row["strategy_key"] == "factor_combo_ranker_v1"
    assert row["symbol"] == "BTCUSDT"
    assert row["duration"] == "10m"


def _primary_key(conn: sqlite3.Connection, table_name: str) -> tuple[str, ...]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    pk_rows = sorted((int(row["pk"]), str(row["name"])) for row in rows if int(row["pk"]) > 0)
    return tuple(name for _order, name in pk_rows)
