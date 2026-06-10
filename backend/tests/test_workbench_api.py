from __future__ import annotations

import sqlite3

from app.api import events, workbench
from app.services import workbench_summary_service as summary_service
from app.services.workbench_summary_cache import clear_workbench_summary_cache


def test_workbench_summary_returns_symbol_counts_and_flags(monkeypatch) -> None:
    conn = _memory_conn()
    _insert_event(conn, "BTCUSDT", "OPEN", with_order=True)
    _insert_event(conn, "ETHUSDT", "OPEN", with_order=True)
    _insert_event(conn, "BTCUSDT", "SETTLED", with_order=True)
    monkeypatch.setattr(summary_service, "get_conn", lambda: conn)
    clear_workbench_summary_cache()

    result = workbench.workbench_summary_sync(symbol="btcusdt", duration="10m")

    assert result["symbol"] == "BTCUSDT"
    assert result["dataSource"] == "Binance Index"
    assert result["eventCounts"] == {"OPEN": 1, "SETTLED": 1, "FAILED": 0}
    assert result["eventTotal"] == 2
    assert result["hasOpenPosition"] is True
    assert "overall" in result["aiHistorySuccess"]
    assert "events" not in result


def test_list_events_returns_paginated_symbol_rows(monkeypatch) -> None:
    def make_conn() -> sqlite3.Connection:
        conn = _memory_conn()
        for index in range(12):
            _insert_event(conn, "BTCUSDT", "SETTLED" if index % 2 else "OPEN")
        _insert_event(conn, "ETHUSDT", "OPEN")
        return conn

    monkeypatch.setattr(events, "get_conn", make_conn)

    page_one = events.list_events(symbol="BTCUSDT", page=1, pageSize=8)
    page_two = events.list_events(symbol="BTCUSDT", page=2, pageSize=8)

    assert page_one["total"] == 12
    assert page_one["pageCount"] == 2
    assert len(page_one["items"]) == 8
    assert len(page_two["items"]) == 4
    assert all(item["symbol"] == "BTCUSDT" for item in page_one["items"])


def test_list_events_route_normalizes_optional_query_defaults(monkeypatch) -> None:
    conn = _memory_conn()
    _insert_event(conn, "BTCUSDT", "OPEN", with_order=True)
    monkeypatch.setattr(events, "get_conn", lambda: conn)

    result = events.list_events(symbol="BTCUSDT", page=1, pageSize=8)

    assert result["total"] == 1
    assert result["items"][0]["symbol"] == "BTCUSDT"


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
          ai_high_winrate_passed INTEGER, ai_high_winrate_value REAL,
          market_regime_gate_passed INTEGER
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


def _insert_event(
    conn: sqlite3.Connection,
    symbol: str,
    status: str,
    *,
    with_order: bool = True,
) -> None:
    cursor = conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value,
          start_time, end_time, status, ai_predicted_direction, ai_prediction_correct, result,
          market_regime_gate_passed
        )
        VALUES('manual', ?, 'test', '10m', 'ABOVE', 103215.4, '2026-05-21T14:10:00Z', ?, ?, 'up', 1, 'YES', 1)
        """,
        (symbol, "2026-05-21T14:20:00Z", status),
    )
    if with_order:
        conn.execute(
            """
            INSERT INTO orders(event_id, side, price, qty, status, created_at)
            VALUES(?, 'BUY', 0.8, 5, 'OPEN', '2026-05-21T14:10:00Z')
            """,
            (cursor.lastrowid,),
        )
    conn.commit()
