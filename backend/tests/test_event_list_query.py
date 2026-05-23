from __future__ import annotations

import sqlite3

from app.services.event_list_query import paginated_events


def test_paginated_events_clamps_page_to_last_page() -> None:
    conn = _memory_conn()
    for _ in range(3):
        _insert_event(conn, "BTCUSDT")

    payload = paginated_events(conn, symbol="BTCUSDT", page=99, page_size=2, view="events")

    assert payload["page"] == 2
    assert payload["pageCount"] == 2
    assert payload["total"] == 3
    assert len(payload["items"]) == 1


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_key TEXT NOT NULL, symbol TEXT NOT NULL,
          title TEXT NOT NULL, event_interval TEXT NOT NULL, rule_type TEXT NOT NULL,
          strike_value REAL NOT NULL, upper_bound REAL, start_time TEXT NOT NULL,
          end_time TEXT NOT NULL, status TEXT NOT NULL
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL, side TEXT NOT NULL,
          price REAL NOT NULL, qty REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
          external_order_id TEXT, external_status TEXT, external_response TEXT
        );
        """
    )
    return conn


def _insert_event(conn: sqlite3.Connection, symbol: str) -> None:
    conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value,
          start_time, end_time, status
        )
        VALUES('manual', ?, 'test', '10m', 'ABOVE', 103215.4, '2026-05-21T14:10:00Z', '2026-05-21T14:20:00Z', 'OPEN')
        """,
        (symbol,),
    )
    conn.commit()
