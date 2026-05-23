from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.services.combo_event_governance import combo_event_monitoring
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


@router.get("/summary")
async def workbench_summary(
    symbol: str = Query("BTCUSDT", min_length=6),
    duration: str = Query("10m"),
) -> dict:
    safe_symbol = symbol.upper()
    safe_duration = _validate_duration(duration)
    return await asyncio.to_thread(get_workbench_summary, safe_symbol, safe_duration)


def workbench_summary_sync(symbol: str = "BTCUSDT", duration: str = "10m") -> dict:
    """Sync helper for tests and scripts."""
    return get_workbench_summary(symbol.strip().upper(), _validate_duration(duration))


def _validate_duration(duration: str) -> str:
    if duration not in ALLOWED_DURATIONS:
        allowed = ", ".join(sorted(ALLOWED_DURATIONS))
        raise HTTPException(status_code=400, detail=f"duration must be one of {allowed}")
    return duration
