from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.factor_learning_service import (
    get_factor_learning_memory,
    mark_factor_learning_agent_failed,
    mark_factor_learning_agent_pending,
    mark_factor_learning_agent_running,
    run_factor_learning_llm_agent,
    refresh_factor_learning_memory,
)
from app.services.factor_learning_refresh_tasks import (
    mark_factor_learning_refresh_completed,
    mark_factor_learning_refresh_failed,
    mark_factor_learning_refresh_queued,
    mark_factor_learning_refresh_running,
)
from app.services.factor_mined_library import mined_factor_library_summary
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
        return _queue_factor_learning_refresh(background_tasks, sym_u, duration, run_agent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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


def _queue_factor_learning_refresh(
    background_tasks: BackgroundTasks,
    symbol: str,
    duration: str,
    run_agent: bool,
) -> dict:
    queued = mark_factor_learning_refresh_queued(symbol, duration, run_agent=run_agent)
    if run_agent:
        queued = mark_factor_learning_agent_pending(queued)
    background_tasks.add_task(_background_factor_learning_refresh, symbol, duration, run_agent)
    return {
        **queued,
        "ok": True,
        "agentQueued": run_agent,
        "refreshQueued": True,
        "message": _refresh_queue_message(run_agent),
    }


def _background_factor_learning_refresh(symbol: str, duration: str, run_agent: bool) -> None:
    try:
        mark_factor_learning_refresh_running(symbol, duration, run_agent=run_agent)
        memory = refresh_factor_learning_memory(symbol, duration, run_llm_agent=False)
        completed = mark_factor_learning_refresh_completed(memory, run_agent=run_agent)
        if run_agent:
            mark_factor_learning_agent_running(completed)
            run_factor_learning_llm_agent(symbol, duration)
    except Exception as exc:
        mark_factor_learning_refresh_failed(symbol, duration, str(exc), run_agent=run_agent)
        if run_agent:
            mark_factor_learning_agent_failed(symbol, duration, str(exc))
        logger.exception("background factor learning refresh failed: %s %s", symbol, duration)


def _refresh_queue_message(run_agent: bool) -> str:
    if run_agent:
        return "本地复盘与联网 Agent 挖掘已排队（先复盘记忆，再调用 LLM），完成后会写回记忆。"
    return "本地因子学习复盘已排队，完成后会写回记忆。"


@router.get("/operators")
def factor_learning_operators() -> dict:
    return factor_operator_payload()


@router.get("/mined-library")
def factor_learning_mined_library(
    symbol: str = Query(..., min_length=6),
    duration: str = Query("10m"),
) -> dict:
    return mined_factor_library_summary(symbol.upper(), duration)
