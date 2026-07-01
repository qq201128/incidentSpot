from __future__ import annotations

import sqlite3

from app.services.event_list_query import paginated_events
from app.services.event_search_index import ensure_event_search_index


def test_paginated_events_filters_by_strategy_key() -> None:
    conn = _memory_conn()
    _insert_event(conn, "BTCUSDT", "manual")
    _insert_event(conn, "BTCUSDT", "factor_combo_ranker_v1_combo_abc")

    payload = paginated_events(
        conn,
        symbol="BTCUSDT",
        page=1,
        page_size=10,
        strategy_key="factor_combo_ranker_v1_combo_abc",
    )

    assert payload["total"] == 1
    assert payload["items"][0]["strategy_key"] == "factor_combo_ranker_v1_combo_abc"


def test_paginated_events_filters_by_duration_minutes() -> None:
    conn = _memory_conn()
    _insert_event(conn, "BTCUSDT", "combo_a", event_interval="10m")
    _insert_event(conn, "BTCUSDT", "combo_a", event_interval="30m")
    _insert_event(conn, "BTCUSDT", "combo_a", event_interval="60m")

    payload = paginated_events(
        conn,
        symbol="BTCUSDT",
        page=1,
        page_size=10,
        strategy_key="combo_a",
        duration_minutes=30,
    )

    assert payload["total"] == 1
    assert payload["items"][0]["event_interval"] == "30m"


def test_paginated_events_clamps_page_to_last_page() -> None:
    conn = _memory_conn()
    for _ in range(3):
        _insert_event(conn, "BTCUSDT")

    payload = paginated_events(conn, symbol="BTCUSDT", page=99, page_size=2)

    assert payload["page"] == 2
    assert payload["pageCount"] == 2
    assert payload["total"] == 3
    assert len(payload["items"]) == 1


def test_paginated_events_does_not_join_orders_for_plain_list() -> None:
    conn = _memory_conn()
    event_id = _insert_event(conn, "BTCUSDT", strategy_key="manual", status="OPEN")
    _insert_order(conn, event_id, status="OPEN", external_response="accepted")
    statements: list[str] = []
    conn.set_trace_callback(statements.append)

    payload = paginated_events(conn, symbol="BTCUSDT", page=1, page_size=10)

    selected = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
    assert payload["total"] == 1
    assert not any("FROM orders" in statement for statement in selected)


def test_paginated_events_searches_backend_fields() -> None:
    conn = _memory_conn()
    _insert_event(conn, "BTCUSDT", strategy_key="manual", status="OPEN")
    target_id = _insert_event(conn, "BTCUSDT", strategy_key="combo_search_target", status="SETTLED")
    _insert_order(conn, target_id, status="OPEN", external_response="accepted")

    payload = paginated_events(
        conn,
        symbol="BTCUSDT",
        page=1,
        page_size=10,
        query="search_target",
    )

    assert payload["total"] == 1
    assert payload["unfilteredTotal"] == 2
    assert payload["query"] == "search_target"
    assert payload["items"][0]["id"] == target_id


def test_paginated_events_searches_symbol_and_order_fields() -> None:
    conn = _memory_conn()
    _insert_event(conn, "ETHUSDT", strategy_key="manual", status="OPEN")
    target_id = _insert_event(conn, "BTCUSDT", strategy_key="manual", status="OPEN")
    _insert_order(
        conn,
        target_id,
        status="OPEN",
        side="SELL",
        external_order_id="paper-777",
        external_response="simulated order",
    )

    symbol_payload = paginated_events(conn, symbol=None, page=1, page_size=10, query="btcusdt")
    side_payload = paginated_events(conn, symbol="BTCUSDT", page=1, page_size=10, query="sell")
    external_id_payload = paginated_events(conn, symbol="BTCUSDT", page=1, page_size=10, query="paper-777")

    assert symbol_payload["items"][0]["id"] == target_id
    assert side_payload["items"][0]["id"] == target_id
    assert external_id_payload["items"][0]["id"] == target_id


def test_paginated_events_search_empty_result_is_explicit() -> None:
    conn = _memory_conn()
    _insert_event(conn, "BTCUSDT", strategy_key="manual", status="OPEN")

    payload = paginated_events(conn, symbol="BTCUSDT", page=4, page_size=10, query="missing")

    assert payload["items"] == []
    assert payload["total"] == 0
    assert payload["unfilteredTotal"] == 1
    assert payload["page"] == 1
    assert payload["pageCount"] == 1
    assert payload["query"] == "missing"


def _memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_key TEXT NOT NULL, symbol TEXT NOT NULL,
          title TEXT NOT NULL, event_interval TEXT NOT NULL, rule_type TEXT NOT NULL,
          strike_value REAL NOT NULL, upper_bound REAL, start_time TEXT NOT NULL,
          end_time TEXT NOT NULL, status TEXT NOT NULL, result TEXT, settlement_source TEXT
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER NOT NULL, side TEXT NOT NULL,
          price REAL NOT NULL, qty REAL NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
          external_order_id TEXT, external_status TEXT, external_response TEXT
        );
        """
    )
    ensure_event_search_index(conn)
    return conn


def _insert_event(
    conn: sqlite3.Connection,
    symbol: str,
    strategy_key: str = "manual",
    *,
    event_interval: str = "10m",
    status: str = "OPEN",
    settlement_source: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO events(
          strategy_key, symbol, title, event_interval, rule_type, strike_value,
          start_time, end_time, status, settlement_source
        )
        VALUES(?, ?, 'test', ?, 'ABOVE', 103215.4, '2026-05-21T14:10:00Z', '2026-05-21T14:20:00Z', ?, ?)
        """,
        (strategy_key, symbol, event_interval, status, settlement_source),
    )
    conn.commit()
    ensure_event_search_index(conn)
    return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def _insert_order(
    conn: sqlite3.Connection,
    event_id: int,
    *,
    status: str,
    side: str = "BUY",
    external_order_id: str | None = None,
    external_response: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO orders(event_id, side, price, qty, status, created_at, external_order_id, external_response)
        VALUES(?, ?, 0.5, 1.0, ?, '2026-05-21T14:10:01Z', ?, ?)
        """,
        (event_id, side, status, external_order_id, external_response),
    )
    conn.commit()
    ensure_event_search_index(conn)
