from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.event_final_decision_reporting import (
    event_final_decision_summary,
    latest_event_final_decision,
)

router = APIRouter(prefix="/api/event-final-decisions", tags=["event-final-decisions"])


@router.get("/latest")
def read_latest_event_final_decision(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    try:
        sym = symbol.strip().upper()
        latest = latest_event_final_decision(sym, duration)
        return {"symbol": sym, "duration": duration, "latest": latest}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/summary")
def read_event_final_decision_summary(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    try:
        return event_final_decision_summary(symbol, duration)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
