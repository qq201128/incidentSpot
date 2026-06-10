from __future__ import annotations

import math

from app.services.event_ai_history import event_interval_where
from app.services.event_search_index import event_search_match_query

DEFAULT_PAGE_SIZE = 8
MAX_PAGE_SIZE = 100

_LATEST_ORDER_JOIN = """
LEFT JOIN orders latest_order ON latest_order.id = (
    SELECT id FROM orders WHERE event_id = events.id ORDER BY id DESC LIMIT 1
)
"""


def paginated_events(
    conn,
    *,
    symbol: str | None,
    page: int,
    page_size: int,
    strategy_key: str | None = None,
    duration_minutes: int | None = None,
    query: str | None = None,
) -> dict:
    safe_page = max(1, int(page))
    safe_page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    where_sql, params = _where_clause(
        symbol,
        strategy_key,
        duration_minutes=duration_minutes,
        query=query,
    )
    total = int(
        conn.execute(
            f"SELECT COUNT(*) AS total FROM events {_LATEST_ORDER_JOIN} WHERE {where_sql}",
            params,
        ).fetchone()["total"]
    )
    unfiltered_total = _unfiltered_total(conn, symbol, strategy_key, duration_minutes)
    page_count = max(1, math.ceil(total / safe_page_size)) if total else 1
    safe_page = min(safe_page, page_count)
    offset = (safe_page - 1) * safe_page_size
    rows = conn.execute(
        f"""
        SELECT events.*
        FROM events
        {_LATEST_ORDER_JOIN}
        WHERE {where_sql}
        ORDER BY events.id DESC
        LIMIT ? OFFSET ?
        """,
        (*params, safe_page_size, offset),
    ).fetchall()
    return {
        "items": rows,
        "page": safe_page,
        "pageSize": safe_page_size,
        "total": total,
        "unfilteredTotal": unfiltered_total,
        "pageCount": page_count,
        "query": (query or "").strip(),
    }


def _unfiltered_total(
    conn,
    symbol: str | None,
    strategy_key: str | None,
    duration_minutes: int | None,
) -> int:
    where_sql, params = _where_clause(
        symbol,
        strategy_key,
        duration_minutes=duration_minutes,
        query=None,
    )
    row = conn.execute(
        f"SELECT COUNT(*) AS total FROM events {_LATEST_ORDER_JOIN} WHERE {where_sql}",
        params,
    ).fetchone()
    return int(row["total"])


def _where_clause(
    symbol: str | None,
    strategy_key: str | None = None,
    *,
    duration_minutes: int | None = None,
    query: str | None = None,
) -> tuple[str, list]:
    clauses = ["1 = 1"]
    params: list = []
    if symbol:
        clauses.append("events.symbol = ?")
        params.append(symbol.upper())
    safe_strategy_key = strategy_key.strip() if isinstance(strategy_key, str) and strategy_key.strip() else None
    if safe_strategy_key:
        clauses.append("events.strategy_key = ?")
        params.append(safe_strategy_key)
    interval_sql, interval_params = event_interval_where(duration_minutes, alias="events")
    if interval_sql:
        clauses.append(interval_sql.removeprefix(" AND "))
        params.extend(interval_params)
    search_sql, search_params = _search_clause(query)
    if search_sql:
        clauses.append(search_sql)
        params.extend(search_params)
    return " AND ".join(clauses), params


def _search_clause(query: str | None) -> tuple[str, list]:
    q = (query or "").strip()
    if not q:
        return "", []
    match_query = event_search_match_query(q)
    if not match_query:
        return "", []
    return """
    events.id IN (
        SELECT event_id
        FROM event_search_fts
        WHERE event_search_fts MATCH ?
    )
    """.strip(), [match_query]
