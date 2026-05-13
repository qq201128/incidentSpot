from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.factor_learning_service import (
    get_factor_learning_memory,
    run_factor_learning_llm_agent,
    refresh_factor_learning_memory,
)
from app.services.factor_operator_library import factor_operator_payload

router = APIRouter(prefix="/api/factor-learning", tags=["factor-learning"])


@router.get("/memory")
def factor_learning_memory(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    try:
        memory = get_factor_learning_memory(symbol.upper(), duration)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if memory is None:
        raise HTTPException(
            status_code=404,
            detail=f"factor learning memory not found for {symbol.upper()} {duration}",
        )
    return {**memory, "source": {**memory.get("source", {}), "readSource": "memory_file"}}


@router.post("/refresh")
def factor_learning_refresh(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    run_agent: bool = Query(True, alias="runAgent"),
) -> dict:
    try:
        memory = refresh_factor_learning_memory(symbol.upper(), duration, run_llm_agent=run_agent)
        return {**memory, "ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/agent/review")
def factor_learning_agent_review(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    try:
        memory = run_factor_learning_llm_agent(symbol.upper(), duration)
        return {**memory, "ok": True}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/operators")
def factor_learning_operators() -> dict:
    return factor_operator_payload()
