from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.db.session import get_conn
from app.services.combo_event_governance import combo_event_monitoring
from app.services.event_ai_history import query_ai_history_meta, query_ai_history_success
from app.services.workbench_summary_service import get_workbench_summary

router = APIRouter(prefix="/api/workbench", tags=["workbench"])

ALLOWED_DURATIONS = frozenset(("10m", "30m", "60m", "1d"))


@router.get("/event-governance")
async def workbench_event_governance(
    symbol: str = Query("BTCUSDT", min_length=6),
    duration: str = Query("10m"),
) -> dict:
    safe_symbol = symbol.upper()
    safe_duration = _validate_duration(duration)
    return await asyncio.to_thread(combo_event_monitoring, safe_symbol, safe_duration)


@router.get("/ai-history-success/meta")
def workbench_ai_history_meta(symbol: str = Query("BTCUSDT", min_length=6)) -> dict:
    safe_symbol = symbol.strip().upper()
    conn = get_conn()
    try:
        return query_ai_history_meta(conn, safe_symbol)
    finally:
        conn.close()


@router.get("/ai-history-success")
def workbench_ai_history_success(
    symbol: str = Query("BTCUSDT", min_length=6),
    duration_minutes: int = Query(..., alias="durationMinutes", description="Settlement duration in minutes; -1 for unknown"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100, alias="pageSize"),
) -> dict:
    safe_symbol = symbol.strip().upper()
    conn = get_conn()
    try:
        return query_ai_history_success(
            conn,
            safe_symbol,
            duration_minutes=duration_minutes,
            page=page,
            page_size=page_size,
        )
    finally:
        conn.close()


def workbench_ai_history_success_sync(
    symbol: str = "BTCUSDT",
    duration_minutes: int = 10,
    page: int = 1,
    page_size: int = 10,
) -> dict:
    conn = get_conn()
    try:
        return query_ai_history_success(
            conn,
            symbol.strip().upper(),
            duration_minutes=duration_minutes,
            page=page,
            page_size=page_size,
        )
    finally:
        conn.close()


@router.get("/summary")
async def workbench_summary(
    symbol: str = Query("BTCUSDT", min_length=6),
    duration: str = Query("10m"),
) -> dict:
    safe_symbol = symbol.upper()
    safe_duration = _validate_duration(duration)
    # Fast cached reads; stale-while-revalidate avoids asyncio thread-pool queueing.
    return get_workbench_summary(safe_symbol, safe_duration)


def workbench_summary_sync(symbol: str = "BTCUSDT", duration: str = "10m") -> dict:
    """Sync helper for tests and scripts."""
    return get_workbench_summary(symbol.strip().upper(), _validate_duration(duration))


def _validate_duration(duration: str) -> str:
    if duration not in ALLOWED_DURATIONS:
        allowed = ", ".join(sorted(ALLOWED_DURATIONS))
        raise HTTPException(status_code=400, detail=f"duration must be one of {allowed}")
    return duration
