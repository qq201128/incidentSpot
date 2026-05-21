from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.api.event_response import event_response
from app.db.session import get_conn

router = APIRouter(prefix="/api/workbench", tags=["workbench"])

ALLOWED_DURATIONS = frozenset(("10m", "30m", "60m", "1d"))
MAX_EVENTS_LIMIT = 50


@router.get("/summary")
def workbench_summary(
    symbol: str = Query("BTCUSDT", min_length=6),
    duration: str = Query("10m"),
    limit: int = Query(20, ge=1, le=MAX_EVENTS_LIMIT),
) -> dict:
    safe_symbol = symbol.upper()
    safe_duration = _validate_duration(duration)
    conn = get_conn()
    try:
        rows = _event_rows(conn, safe_symbol, limit)
        events = [event_response(conn, row) for row in rows]
        counts = _event_counts(conn, safe_symbol)
        return {
            "symbol": safe_symbol,
            "duration": safe_duration,
            "dataSource": "Binance Index",
            "serverTime": datetime.now(timezone.utc).isoformat(),
            "eventCounts": counts,
            "events": events,
        }
    finally:
        conn.close()


def _validate_duration(duration: str) -> str:
    if duration not in ALLOWED_DURATIONS:
        allowed = ", ".join(sorted(ALLOWED_DURATIONS))
        raise HTTPException(status_code=400, detail=f"duration must be one of {allowed}")
    return duration


def _event_rows(conn, symbol: str, limit: int):
    return conn.execute(
        """
        SELECT * FROM events
        WHERE symbol = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (symbol, limit),
    ).fetchall()


def _event_counts(conn, symbol: str) -> dict:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS total
        FROM events
        WHERE symbol = ?
        GROUP BY status
        """,
        (symbol,),
    ).fetchall()
    counts = {"OPEN": 0, "SETTLED": 0, "FAILED": 0}
    for row in rows:
        counts[row["status"]] = int(row["total"])
    return counts
