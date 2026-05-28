from __future__ import annotations

import sqlite3

from app.services import event_ai_history
from app.services.ai_history_cache import clear_ai_history_cache


def test_ai_history_success_paginates_by_duration() -> None:
    clear_ai_history_cache()
    conn = _memory_conn()
    _insert_settled_ai_event(conn, "BTCUSDT", "s1", start="2026-05-21T14:10:00Z", end="2026-05-21T14:20:00Z")
    _insert_settled_ai_event(conn, "BTCUSDT", "s2", start="2026-05-21T14:10:00Z", end="2026-05-21T14:20:00Z")
    _insert_settled_ai_event(
        conn, "BTCUSDT", "s3", start="2026-05-21T14:10:00Z", end="2026-05-21T14:50:00Z", event_interval="30m"
    )

    page_one = event_ai_history.query_ai_history_success(
        conn, "BTCUSDT", duration_minutes=10, page=1, page_size=1
    )
    page_two = event_ai_history.query_ai_history_success(
        conn, "BTCUSDT", duration_minutes=10, page=2, page_size=1
    )

    assert page_one["pagination"]["total"] == 2
    assert page_one["pagination"]["pageCount"] == 2
    assert len(page_one["byStrategy"]) == 1
    assert len(page_two["byStrategy"]) == 1
    assert page_one["period"]["factorCount"] == 2
    meta = event_ai_history.query_ai_history_meta(conn, "BTCUSDT")
    assert len(meta["durationSummaries"]) == 2
    assert meta["durationSummaries"][0]["durationMinutes"] == 10
    assert meta["durationSummaries"][1]["durationMinutes"] == 30


def test_ai_history_meta_returns_duration_summaries() -> None:
    clear_ai_history_cache()
    conn = _memory_conn()
    _insert_settled_ai_event(conn, "BTCUSDT", "s1", start="2026-05-21T14:10:00Z", end="2026-05-21T14:20:00Z")
    result = event_ai_history.query_ai_history_meta(conn, "BTCUSDT")
    assert result["symbol"] == "BTCUSDT"
    assert len(result["durationSummaries"]) == 1
    assert result["durationSummaries"][0]["durationMinutes"] == 10


def test_ai_history_success_summary_omits_factor_list() -> None:
    conn = _memory_conn()
    _insert_settled_ai_event(conn, "BTCUSDT", "s1", start="2026-05-21T14:10:00Z", end="2026-05-21T14:20:00Z")
    result = event_ai_history.ai_history_success(conn, "BTCUSDT")
    assert result["overall"]["total"] == 1
    assert "byStrategy" not in result


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_key TEXT NOT NULL, symbol TEXT NOT NULL,
          title TEXT NOT NULL, event_interval TEXT NOT NULL, rule_type TEXT NOT NULL,
          strike_value REAL NOT NULL, upper_bound REAL, start_time TEXT NOT NULL,
          end_time TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
          ai_predicted_direction TEXT, ai_prediction_correct INTEGER,
          ai_high_winrate_rule TEXT
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL, side TEXT NOT NULL,
          price REAL NOT NULL, qty REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE settlements (
          id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL, order_id INTEGER NOT NULL,
          pnl REAL NOT NULL, settled_at TEXT NOT NULL
        );
        """
    )
    return conn


def _insert_settled_ai_event(
    conn: sqlite3.Connection,
    symbol: str,
    strategy_key: str,
    *,
    start: str,
    end: str,
    event_interval: str = "10m",
) -> None:
    cursor = conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value,
          start_time, end_time, status, ai_predicted_direction, ai_prediction_correct, result
        )
        VALUES(?, ?, 'test', ?, 'ABOVE', 103215.4, ?, ?, 'SETTLED', 'up', 1, 'YES')
        """,
        (strategy_key, symbol, event_interval, start, end),
    )
    event_id = cursor.lastrowid
    order_cursor = conn.execute(
        """
        INSERT INTO orders(event_id, side, price, qty, status, created_at)
        VALUES(?, 'BUY', 0.8, 5, 'OPEN', '2026-05-21T14:10:00Z')
        """,
        (event_id,),
    )
    conn.execute(
        "INSERT INTO settlements(event_id, order_id, pnl, settled_at) VALUES(?, ?, ?, ?)",
        (event_id, order_cursor.lastrowid, 4.0, "2026-05-21T14:20:00Z"),
    )
    conn.commit()
