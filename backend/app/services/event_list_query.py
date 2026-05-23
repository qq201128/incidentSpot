from __future__ import annotations

import math

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


def paginated_events(conn, *, symbol: str | None, page: int, page_size: int, view: str) -> dict:
    safe_view = normalize_view(view)
    safe_page = max(1, int(page))
    safe_page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    where_sql, params = _view_where_clause(safe_view, symbol)
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


def _view_where_clause(view: str, symbol: str | None) -> tuple[str, list]:
    clauses = ["1 = 1"]
    params: list = []
    if symbol:
        clauses.append("events.symbol = ?")
        params.append(symbol.upper())
    if view == "orders":
        clauses.append("latest_order.id IS NOT NULL")
    elif view == "settlements":
        clauses.append("(events.status = 'SETTLED' OR events.settlement_price IS NOT NULL)")
    elif view == "failures":
        clauses.append(_failure_clause())
    return " AND ".join(clauses), params


def _failure_clause() -> str:
    return """
    (
        events.status = 'FAILED'
        OR latest_order.status = 'FAILED'
        OR UPPER(COALESCE(latest_order.external_status, '')) GLOB '*FAIL*'
        OR UPPER(COALESCE(latest_order.external_status, '')) GLOB '*ERROR*'
        OR UPPER(COALESCE(latest_order.external_status, '')) GLOB '*REJECT*'
    )
    """.strip()
