from __future__ import annotations

from dataclasses import dataclass
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from app.services.background_loop_status import (
    record_loop_failure,
    record_loop_start,
    record_loop_success,
)
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
BACKGROUND_REFRESH_LOOP = "factor_learning_refresh"


@dataclass(frozen=True)
class FactorLearningRefreshJob:
    symbol: str
    duration: str
    run_agent: bool


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
        job = FactorLearningRefreshJob(sym_u, duration, run_agent)
        return _queue_factor_learning_refresh(background_tasks, job)
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
    job: FactorLearningRefreshJob,
) -> dict:
    queued = mark_factor_learning_refresh_queued(job.symbol, job.duration, run_agent=job.run_agent)
    if job.run_agent:
        queued = mark_factor_learning_agent_pending(queued)
    background_tasks.add_task(_background_factor_learning_refresh, job)
    return {
        **queued,
        "ok": True,
        "agentQueued": job.run_agent,
        "refreshQueued": True,
        "message": _refresh_queue_message(job.run_agent),
    }


def _background_factor_learning_refresh(job: FactorLearningRefreshJob) -> None:
    details = _loop_details(job)
    stage = "refresh_memory"
    record_loop_start(BACKGROUND_REFRESH_LOOP, details)
    try:
        mark_factor_learning_refresh_running(job.symbol, job.duration, run_agent=job.run_agent)
        memory = refresh_factor_learning_memory(job.symbol, job.duration, run_llm_agent=False)
        completed = mark_factor_learning_refresh_completed(memory, run_agent=job.run_agent)
        if job.run_agent:
            stage = "llm_agent"
            mark_factor_learning_agent_running(completed)
            run_factor_learning_llm_agent(job.symbol, job.duration)
        record_loop_success(BACKGROUND_REFRESH_LOOP, {**details, "stage": "completed"})
    except Exception as exc:
        failure_details = {**details, "stage": stage}
        record_loop_failure(BACKGROUND_REFRESH_LOOP, exc, failure_details)
        mark_factor_learning_refresh_failed(job.symbol, job.duration, str(exc), run_agent=job.run_agent)
        if job.run_agent:
            mark_factor_learning_agent_failed(job.symbol, job.duration, str(exc))
        logger.exception("background factor learning refresh failed: %s %s", job.symbol, job.duration)
        raise


def _loop_details(job: FactorLearningRefreshJob) -> dict:
    return {"symbol": job.symbol, "duration": job.duration, "runAgent": job.run_agent}


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
