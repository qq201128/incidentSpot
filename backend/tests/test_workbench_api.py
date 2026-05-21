from __future__ import annotations

import sqlite3

from app.api import workbench


def test_workbench_summary_returns_recent_symbol_events(monkeypatch) -> None:
    conn = _memory_conn()
    _insert_event(conn, "BTCUSDT", "OPEN")
    _insert_event(conn, "ETHUSDT", "OPEN")
    _insert_event(conn, "BTCUSDT", "SETTLED")
    monkeypatch.setattr(workbench, "get_conn", lambda: conn)

    result = workbench.workbench_summary(symbol="btcusdt", duration="10m", limit=20)

    assert result["symbol"] == "BTCUSDT"
    assert result["dataSource"] == "Binance Index"
    assert result["eventCounts"] == {"OPEN": 1, "SETTLED": 1, "FAILED": 0}
    assert [event["symbol"] for event in result["events"]] == ["BTCUSDT", "BTCUSDT"]


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_key TEXT NOT NULL, symbol TEXT NOT NULL,
          title TEXT NOT NULL, event_interval TEXT NOT NULL, rule_type TEXT NOT NULL,
          strike_value REAL NOT NULL, upper_bound REAL, start_time TEXT NOT NULL,
          end_time TEXT NOT NULL, status TEXT NOT NULL, result TEXT, settlement_price REAL,
          settlement_quote_time INTEGER, settlement_source TEXT, ai_probability_up REAL,
          ai_predicted_direction TEXT, ai_prediction_correct INTEGER, ai_quality_score REAL,
          ai_quality_passed INTEGER, ai_high_winrate_gate TEXT, ai_high_winrate_rule TEXT,
          ai_high_winrate_passed INTEGER, ai_high_winrate_value REAL
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL, side TEXT NOT NULL,
          price REAL NOT NULL, qty REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
          external_order_id TEXT, external_status TEXT, external_response TEXT
        );
        CREATE TABLE settlements (id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, pnl REAL NOT NULL);
        """
    )
    return conn


def _insert_event(conn: sqlite3.Connection, symbol: str, status: str) -> None:
    conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value,
          start_time, end_time, status
        )
        VALUES('manual', ?, 'test', '10m', 'ABOVE', 103215.4, '2026-05-21T14:10:00Z', ?, ?)
        """,
        (symbol, "2026-05-21T14:20:00Z", status),
    )
    conn.commit()
