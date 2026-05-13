from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.factor_learning_service import (
    get_factor_learning_memory,
    mark_factor_learning_agent_pending,
    run_factor_learning_llm_agent,
    refresh_factor_learning_memory,
)
from app.services.factor_operator_library import factor_operator_payload

router = APIRouter(prefix="/api/factor-learning", tags=["factor-learning"])
logger = logging.getLogger("uvicorn.error")


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
    background_tasks: BackgroundTasks,
    *,
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
    run_agent: bool = Query(True, alias="runAgent"),
) -> dict:
    sym_u = symbol.upper()
    try:
        if run_agent:
            return _queue_factor_learning_agent(background_tasks, sym_u, duration)
        memory = refresh_factor_learning_memory(sym_u, duration, run_llm_agent=False)
        return {**memory, "ok": True, "agentQueued": False}
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


def _queue_factor_learning_agent(
    background_tasks: BackgroundTasks,
    symbol: str,
    duration: str,
) -> dict:
    memory = refresh_factor_learning_memory(symbol, duration, run_llm_agent=False)
    queued = mark_factor_learning_agent_pending(memory)
    background_tasks.add_task(_background_factor_learning_agent_review, symbol, duration)
    return {
        **queued,
        "ok": True,
        "agentQueued": True,
        "message": "Kimi 因子挖掘已排队，完成后会写回因子学习记忆。",
    }


def _background_factor_learning_agent_review(symbol: str, duration: str) -> None:
    try:
        run_factor_learning_llm_agent(symbol, duration)
    except Exception:
        logger.exception("background factor learning agent failed: %s %s", symbol, duration)


@router.get("/operators")
def factor_learning_operators() -> dict:
    return factor_operator_payload()
