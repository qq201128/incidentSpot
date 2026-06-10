from __future__ import annotations

import re
from typing import Any

EVENT_SEARCH_TABLE = "event_search_fts"
EVENT_SEARCH_META_TABLE = "event_search_meta"
EVENT_SEARCH_INDEX_VERSION = "event_search_v2"


def ensure_event_search_index(conn: Any) -> None:
    _ensure_tables(conn)
    if _index_current(conn):
        return
    _rebuild_index(conn)


def refresh_event_search_row(conn: Any, event_id: int) -> None:
    _ensure_tables(conn)
    conn.execute(f"DELETE FROM {EVENT_SEARCH_TABLE} WHERE event_id = ?", (event_id,))
    conn.execute(
        _event_search_select_sql("WHERE events.id = ?"),
        (event_id,),
    )
    _store_index_meta(conn)


def event_search_match_query(raw: str) -> str:
    tokens = [token for token in re.split(r"\W+", raw) if token]
    return " ".join(f"{token}*" for token in tokens)


def _ensure_tables(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {EVENT_SEARCH_TABLE}
        USING fts5(event_id UNINDEXED, search_text)
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EVENT_SEARCH_META_TABLE} (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        )
        """
    )


def _index_current(conn: Any) -> bool:
    return (
        _stored_version(conn) == EVENT_SEARCH_INDEX_VERSION
        and _event_count(conn) == _index_count(conn)
        and _stored_count(conn, "eventCount") == _event_count(conn)
        and _stored_count(conn, "orderCount") == _order_count(conn)
    )


def _rebuild_index(conn: Any) -> None:
    conn.execute(f"DELETE FROM {EVENT_SEARCH_TABLE}")
    conn.execute(_event_search_select_sql(""))
    _store_index_meta(conn)


def _store_index_meta(conn: Any) -> None:
    values = {
        "version": EVENT_SEARCH_INDEX_VERSION,
        "eventCount": str(_event_count(conn)),
        "orderCount": str(_order_count(conn)),
    }
    for key, value in values.items():
        _store_meta_value(conn, key, value)


def _store_meta_value(conn: Any, key: str, value: str) -> None:
    conn.execute(
        f"""
        INSERT INTO {EVENT_SEARCH_META_TABLE}(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _stored_version(conn: Any) -> str | None:
    row = conn.execute(f"SELECT value FROM {EVENT_SEARCH_META_TABLE} WHERE key = 'version'").fetchone()
    return str(row["value"]) if row else None


def _stored_count(conn: Any, key: str) -> int | None:
    row = conn.execute(f"SELECT value FROM {EVENT_SEARCH_META_TABLE} WHERE key = ?", (key,)).fetchone()
    return int(row["value"]) if row else None


def _event_count(conn: Any) -> int:
    row = conn.execute("SELECT COUNT(*) AS total FROM events").fetchone()
    return int(row["total"] or 0)


def _order_count(conn: Any) -> int:
    row = conn.execute("SELECT COUNT(*) AS total FROM orders").fetchone()
    return int(row["total"] or 0)


def _index_count(conn: Any) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS total FROM {EVENT_SEARCH_TABLE}").fetchone()
    return int(row["total"] or 0)


def _event_search_select_sql(where_sql: str) -> str:
    return f"""
        INSERT INTO {EVENT_SEARCH_TABLE}(event_id, search_text)
        SELECT events.id, {_search_text_sql()}
        FROM events
        LEFT JOIN orders latest_order ON latest_order.id = (
            SELECT id FROM orders WHERE event_id = events.id ORDER BY id DESC LIMIT 1
        )
        {where_sql}
        """


def _search_text_sql() -> str:
    fields = (
        "events.id",
        "events.symbol",
        "events.title",
        "events.status",
        "events.strategy_key",
        "events.event_interval",
        "events.result",
        "events.settlement_source",
        "latest_order.side",
        "latest_order.status",
        "latest_order.external_order_id",
        "latest_order.external_status",
        "latest_order.external_response",
    )
    return " || ' ' || ".join(f"COALESCE({field}, '')" for field in fields)
