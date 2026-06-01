from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.factor_simulation_trace_service import factor_simulation_trace

router = APIRouter(prefix="/api/factors", tags=["factors"])


@router.get("/simulation-trace")
def simulation_trace(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    factorName: str | None = Query(None),
    strategyKey: str | None = Query(None),
) -> dict:
    try:
        return factor_simulation_trace(
            symbol,
            duration,
            factor_name=factorName,
            strategy_key=strategyKey,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
