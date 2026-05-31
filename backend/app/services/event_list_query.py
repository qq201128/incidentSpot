from __future__ import annotations

import math

from app.services.event_ai_history import event_interval_where

ALLOWED_VIEWS = frozenset({"events", "orders", "settlements", "failures"})
DEFAULT_PAGE_SIZE = 8
MAX_PAGE_SIZE = 100

_LATEST_ORDER_JOIN = """
LEFT JOIN orders latest_order ON latest_order.id = (
    SELECT id FROM orders WHERE event_id = events.id ORDER BY id DESC LIMIT 1
)
"""


def normalize_view(view: str) -> str:
    normalized = (view or "events").strip().lower()
    if normalized not in ALLOWED_VIEWS:
        allowed = ", ".join(sorted(ALLOWED_VIEWS))
        raise ValueError(f"view must be one of {allowed}")
    return normalized


def paginated_events(
    conn,
    *,
    symbol: str | None,
    page: int,
    page_size: int,
    view: str,
    strategy_key: str | None = None,
    duration_minutes: int | None = None,
    query: str | None = None,
) -> dict:
    safe_view = normalize_view(view)
    safe_page = max(1, int(page))
    safe_page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    where_sql, params = _view_where_clause(
        safe_view,
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
        "pageCount": page_count,
        "view": safe_view,
    }


def _view_where_clause(
    view: str,
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
    if view == "orders":
        clauses.append("latest_order.id IS NOT NULL")
    elif view == "settlements":
        clauses.append("(events.status = 'SETTLED' OR events.settlement_price IS NOT NULL)")
    elif view == "failures":
        clauses.append(_failure_clause())
    return " AND ".join(clauses), params


def _search_clause(query: str | None) -> tuple[str, list]:
    q = (query or "").strip()
    if not q:
        return "", []
    pattern = f"%{q.upper()}%"
    return """
    (
        UPPER(CAST(events.id AS TEXT)) LIKE ?
        OR UPPER(events.symbol) LIKE ?
        OR UPPER(events.title) LIKE ?
        OR UPPER(events.status) LIKE ?
        OR UPPER(events.strategy_key) LIKE ?
        OR UPPER(events.event_interval) LIKE ?
        OR UPPER(COALESCE(events.result, '')) LIKE ?
        OR UPPER(COALESCE(events.settlement_source, '')) LIKE ?
        OR UPPER(COALESCE(latest_order.side, '')) LIKE ?
        OR UPPER(COALESCE(latest_order.status, '')) LIKE ?
        OR UPPER(COALESCE(latest_order.external_order_id, '')) LIKE ?
        OR UPPER(COALESCE(latest_order.external_status, '')) LIKE ?
        OR UPPER(COALESCE(latest_order.external_response, '')) LIKE ?
    )
    """.strip(), [pattern] * 13


def _failure_clause() -> str:
    return """
    (
        events.status = 'FAILED'
        OR UPPER(COALESCE(events.settlement_source, '')) GLOB '*FAIL*'
        OR UPPER(COALESCE(events.settlement_source, '')) GLOB '*ERROR*'
        OR UPPER(COALESCE(events.settlement_source, '')) GLOB '*REJECT*'
        OR latest_order.status = 'FAILED'
        OR UPPER(COALESCE(latest_order.external_status, '')) GLOB '*FAIL*'
        OR UPPER(COALESCE(latest_order.external_status, '')) GLOB '*ERROR*'
        OR UPPER(COALESCE(latest_order.external_status, '')) GLOB '*REJECT*'
        OR UPPER(COALESCE(latest_order.external_response, '')) GLOB '*FAIL*'
        OR UPPER(COALESCE(latest_order.external_response, '')) GLOB '*ERROR*'
        OR UPPER(COALESCE(latest_order.external_response, '')) GLOB '*REJECT*'
    )
    """.strip()
