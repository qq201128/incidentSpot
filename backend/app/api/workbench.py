from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.db.session import get_conn
from app.services.event_ai_history import ai_history_success

router = APIRouter(prefix="/api/workbench", tags=["workbench"])

ALLOWED_DURATIONS = frozenset(("10m", "30m", "60m", "1d"))


@router.get("/summary")
def workbench_summary(
    symbol: str = Query("BTCUSDT", min_length=6),
    duration: str = Query("10m"),
) -> dict:
    safe_symbol = symbol.upper()
    safe_duration = _validate_duration(duration)
    conn = get_conn()
    try:
        counts = _event_counts(conn, safe_symbol)
        event_total = sum(counts.values())
        return {
            "symbol": safe_symbol,
            "duration": safe_duration,
            "dataSource": "Binance Index",
            "serverTime": datetime.now(timezone.utc).isoformat(),
            "eventCounts": counts,
            "eventTotal": event_total,
            "hasOpenPosition": _has_open_position(conn, safe_symbol),
            "aiHistorySuccess": ai_history_success(conn, safe_symbol),
        }
    finally:
        conn.close()


def _validate_duration(duration: str) -> str:
    if duration not in ALLOWED_DURATIONS:
        allowed = ", ".join(sorted(ALLOWED_DURATIONS))
        raise HTTPException(status_code=400, detail=f"duration must be one of {allowed}")
    return duration


def _has_open_position(conn, symbol: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM events e
        INNER JOIN orders o ON o.event_id = e.id
        WHERE e.symbol = ? AND e.status = 'OPEN'
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    return row is not None


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
