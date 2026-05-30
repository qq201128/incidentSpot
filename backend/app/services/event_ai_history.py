from __future__ import annotations

import math
from datetime import datetime

from app.services.rule_config import DURATION_TO_MINUTES

UNKNOWN_DURATION = -1
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100

MINUTES_TO_INTERVAL: dict[int, str] = {minutes: interval for interval, minutes in DURATION_TO_MINUTES.items()}
KNOWN_INTERVAL_SQL = ", ".join(f"'{key}'" for key in DURATION_TO_MINUTES)

_SETTLEMENT_PNL_JOIN = """
LEFT JOIN (
  SELECT event_id, SUM(pnl) AS pnl_u
  FROM settlements
  GROUP BY event_id
) s ON s.event_id = e.id
"""

_PNL_SQL = "COALESCE(s.pnl_u, 0)"

_SETTLED_AI_WHERE = """
  e.symbol = ?
  AND e.status = 'SETTLED'
  AND e.ai_predicted_direction IS NOT NULL
  AND e.ai_prediction_correct IS NOT NULL
"""


def settled_expected_profit_usdt(*, status: str, order_side: str | None, order_qty, order_price, result) -> float | None:
    qty = _finite_float(order_qty)
    price = _finite_float(order_price)
    if qty is None or qty <= 0 or price is None or price < 0:
        return None
    if status != "SETTLED" or result is None or order_side is None:
        return None
    is_correct = (order_side == "BUY" and result == "YES") or (order_side == "SELL" and result == "NO")
    return qty * price if is_correct else -qty


def ai_history_success(conn, symbol: str) -> dict:
    """Symbol-wide settled AI stats for workbench summary (no factor list)."""
    return {"overall": _fetch_overall_stats(conn, symbol.upper())}


def query_ai_history_meta(conn, symbol: str) -> dict:
    from app.services.ai_history_cache import get_cached_ai_history_meta

    safe_symbol = symbol.upper()

    def _build() -> dict:
        return {
            "symbol": safe_symbol,
            "durationSummaries": _fetch_duration_summaries(conn, safe_symbol),
        }

    return get_cached_ai_history_meta(safe_symbol, build=_build)


def query_ai_history_success(
    conn,
    symbol: str,
    *,
    duration_minutes: int,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    include_summaries: bool = False,
) -> dict:
    from app.services.ai_history_cache import AiHistoryCacheKey, get_cached_ai_history

    safe_symbol = symbol.upper()
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    page = max(1, int(page))

    def _build() -> dict:
        period = _fetch_period_stats(conn, safe_symbol, duration_minutes)
        total = period["factorCount"]
        page_count = max(1, math.ceil(total / page_size)) if total else 1
        safe_page = min(page, page_count)
        by_strategy = _fetch_strategy_page(
            conn,
            safe_symbol,
            duration_minutes,
            page=safe_page,
            page_size=page_size,
        )
        payload = {
            "symbol": safe_symbol,
            "durationMinutes": duration_minutes,
            "period": period,
            "byStrategy": by_strategy,
            "pagination": {
                "page": safe_page,
                "pageSize": page_size,
                "total": total,
                "pageCount": page_count,
            },
        }
        if include_summaries:
            payload["durationSummaries"] = _fetch_duration_summaries(conn, safe_symbol)
        return payload

    return get_cached_ai_history(
        AiHistoryCacheKey(safe_symbol, duration_minutes, page, page_size),
        build=_build,
    )


def event_interval_where(duration_minutes: int | None, *, alias: str = "e") -> tuple[str, tuple]:
    """SQL fragment filtering events by settlement duration (minutes). None = no filter."""
    if duration_minutes is None:
        return "", ()
    column = f"{alias}.event_interval"
    if duration_minutes == UNKNOWN_DURATION:
        return f" AND {column} NOT IN ({KNOWN_INTERVAL_SQL})", ()
    interval = MINUTES_TO_INTERVAL.get(duration_minutes)
    if interval is None:
        return " AND 1 = 0", ()
    return f" AND {column} = ?", (interval,)


def _interval_filter_sql(duration_minutes: int) -> tuple[str, tuple]:
    return event_interval_where(duration_minutes, alias="e")


def _duration_minutes_from_interval(event_interval: str | None) -> int:
    if event_interval in DURATION_TO_MINUTES:
        return DURATION_TO_MINUTES[event_interval]
    return UNKNOWN_DURATION


def _fetch_overall_stats(conn, symbol: str) -> dict:
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS total,
          COALESCE(SUM(e.ai_prediction_correct), 0) AS hits,
          COALESCE(SUM({_PNL_SQL}), 0.0) AS pnl_u
        FROM events e
        {_SETTLEMENT_PNL_JOIN}
        WHERE {_SETTLED_AI_WHERE}
        """,
        (symbol,),
    ).fetchone()
    total = int(row["total"] or 0)
    hits = int(row["hits"] or 0)
    return {
        "total": total,
        "hits": hits,
        "rate": hits / total if total else None,
        "pnlU": float(row["pnl_u"] or 0),
    }


def _fetch_duration_summaries(conn, symbol: str) -> list[dict]:
    rows = conn.execute(
        f"""
        SELECT
          e.event_interval AS event_interval,
          COUNT(DISTINCT e.strategy_key) AS factor_count
        FROM events e
        WHERE {_SETTLED_AI_WHERE}
        GROUP BY e.event_interval
        ORDER BY
          CASE e.event_interval
            WHEN '10m' THEN 10
            WHEN '30m' THEN 30
            WHEN '60m' THEN 60
            WHEN '1d' THEN 1440
            ELSE 999999
          END
        """,
        (symbol,),
    ).fetchall()
    return [
        {
            "durationMinutes": _duration_minutes_from_interval(row["event_interval"]),
            "factorCount": int(row["factor_count"] or 0),
        }
        for row in rows
    ]


def _fetch_period_stats(conn, symbol: str, duration_minutes: int) -> dict:
    interval_sql, interval_params = _interval_filter_sql(duration_minutes)
    row = conn.execute(
        f"""
        SELECT
          COUNT(DISTINCT e.strategy_key) AS factor_count,
          COUNT(*) AS total,
          COALESCE(SUM(e.ai_prediction_correct), 0) AS hits,
          COALESCE(SUM({_PNL_SQL}), 0.0) AS pnl_u
        FROM events e
        {_SETTLEMENT_PNL_JOIN}
        WHERE {_SETTLED_AI_WHERE}
        {interval_sql}
        """,
        (symbol, *interval_params),
    ).fetchone()
    total = int(row["total"] or 0)
    hits = int(row["hits"] or 0)
    return {
        "total": total,
        "hits": hits,
        "pnlU": float(row["pnl_u"] or 0),
        "rate": hits / total if total else None,
        "factorCount": int(row["factor_count"] or 0),
    }


def _fetch_strategy_page(
    conn,
    symbol: str,
    duration_minutes: int,
    *,
    page: int,
    page_size: int,
) -> list[dict]:
    interval_sql, interval_params = _interval_filter_sql(duration_minutes)
    offset = (page - 1) * page_size
    rows = conn.execute(
        f"""
        SELECT
          e.strategy_key AS strategy_key,
          MAX(e.ai_high_winrate_rule) AS factor_name,
          COUNT(*) AS total,
          COALESCE(SUM(e.ai_prediction_correct), 0) AS hits,
          COALESCE(SUM({_PNL_SQL}), 0.0) AS pnl_u
        FROM events e
        {_SETTLEMENT_PNL_JOIN}
        WHERE {_SETTLED_AI_WHERE}
        {interval_sql}
        GROUP BY e.strategy_key
        ORDER BY pnl_u DESC, e.strategy_key ASC
        LIMIT ? OFFSET ?
        """,
        (symbol, *interval_params, page_size, offset),
    ).fetchall()
    return [_row_to_strategy_item(row, duration_minutes) for row in rows]


def _row_to_strategy_item(row, duration_minutes: int) -> dict:
    total = int(row["total"] or 0)
    hits = int(row["hits"] or 0)
    return {
        "strategyKey": row["strategy_key"] or "manual",
        "factorName": row["factor_name"],
        "durationMinutes": duration_minutes,
        "total": total,
        "hits": hits,
        "pnlU": float(row["pnl_u"] or 0),
        "rate": hits / total if total else None,
    }


def _duration_minutes(start_time: str | None, end_time: str | None) -> int:
    start = _parse_iso_ms(start_time)
    end = _parse_iso_ms(end_time)
    if start is None or end is None or end <= start:
        return UNKNOWN_DURATION
    return round((end - start) / 60000)


def _parse_iso_ms(value: str | None) -> float | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp() * 1000
    except ValueError:
        return None


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number
