from __future__ import annotations

import re
from typing import Any

EVENT_SEARCH_TABLE = "event_search_fts"


def ensure_event_search_index(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS {EVENT_SEARCH_TABLE}
        USING fts5(event_id UNINDEXED, search_text)
        """
    )
    conn.execute(f"DELETE FROM {EVENT_SEARCH_TABLE}")
    conn.execute(
        f"""
        INSERT INTO {EVENT_SEARCH_TABLE}(event_id, search_text)
        SELECT events.id, COALESCE(events.id, '') || ' ' ||
               COALESCE(events.symbol, '') || ' ' ||
               COALESCE(events.title, '') || ' ' ||
               COALESCE(events.status, '') || ' ' ||
               COALESCE(events.strategy_key, '') || ' ' ||
               COALESCE(events.event_interval, '') || ' ' ||
               COALESCE(events.result, '') || ' ' ||
               COALESCE(events.settlement_source, '') || ' ' ||
               COALESCE(latest_order.side, '') || ' ' ||
               COALESCE(latest_order.status, '') || ' ' ||
               COALESCE(latest_order.external_order_id, '') || ' ' ||
               COALESCE(latest_order.external_status, '') || ' ' ||
               COALESCE(latest_order.external_response, '')
        FROM events
        LEFT JOIN orders latest_order ON latest_order.id = (
            SELECT id FROM orders WHERE event_id = events.id ORDER BY id DESC LIMIT 1
        )
        """
    )


def refresh_event_search_row(conn: Any, event_id: int) -> None:
    conn.execute(f"DELETE FROM {EVENT_SEARCH_TABLE} WHERE event_id = ?", (event_id,))
    conn.execute(
        f"""
        INSERT INTO {EVENT_SEARCH_TABLE}(event_id, search_text)
        SELECT events.id, COALESCE(events.id, '') || ' ' ||
               COALESCE(events.symbol, '') || ' ' ||
               COALESCE(events.title, '') || ' ' ||
               COALESCE(events.status, '') || ' ' ||
               COALESCE(events.strategy_key, '') || ' ' ||
               COALESCE(events.event_interval, '') || ' ' ||
               COALESCE(events.result, '') || ' ' ||
               COALESCE(events.settlement_source, '') || ' ' ||
               COALESCE(latest_order.side, '') || ' ' ||
               COALESCE(latest_order.status, '') || ' ' ||
               COALESCE(latest_order.external_order_id, '') || ' ' ||
               COALESCE(latest_order.external_status, '') || ' ' ||
               COALESCE(latest_order.external_response, '')
        FROM events
        LEFT JOIN orders latest_order ON latest_order.id = (
            SELECT id FROM orders WHERE event_id = events.id ORDER BY id DESC LIMIT 1
        )
        WHERE events.id = ?
        """,
        (event_id,),
    )


def event_search_match_query(raw: str) -> str:
    tokens = [token for token in re.split(r"\W+", raw) if token]
    return " ".join(f"{token}*" for token in tokens)
