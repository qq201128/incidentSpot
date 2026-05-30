from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, FastAPI

from app.db.session import init_db
from app.services.background_loop_status import (
    background_loop_statuses,
    record_loop_failure,
    record_loop_start,
    record_loop_success,
)

logger = logging.getLogger(__name__)
BOOTSTRAP_STATUS_STARTING = "starting"
BOOTSTRAP_STATUS_RUNNING = "running"
BOOTSTRAP_STATUS_READY = "ready"
BOOTSTRAP_STATUS_FAILED = "failed"
BOOTSTRAP_LOOP_NAME = "application_bootstrap"


def _configure_asyncio_thread_pool() -> None:
    workers = int(os.getenv("ASYNCIO_THREAD_POOL_WORKERS", "48"))
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=max(8, workers)))

BACKGROUND_SHUTDOWN_TIMEOUT_SECONDS = 5.0
STOP_EVENT_ATTRS = (
    "settlement_stop_event",
    "predict_stop_event",
    "trade_stop_event",
    "factor_ranking_stop_event",
    "market_context_stop_event",
    "factor_combo_daily_stop_event",
    "lstm_candidate_retry_stop_event",
    "lstm_daily_review_stop_event",
    "combo_event_governance_stop_event",
)
BACKGROUND_TASK_ATTRS = (
    "settlement_task",
    "predict_task",
    "trade_task",
    "factor_ranking_task",
    "market_context_task",
    "factor_combo_daily_task",
    "lstm_candidate_retry_task",
    "lstm_daily_review_task",
    "combo_event_governance_task",
)
BACKGROUND_LOOP_STATUS_NAMES = {
    "settlement": "auto_settlement",
    "predict": "auto_predict",
    "trade": "auto_trade",
    "factor_ranking": "factor_ranking",
    "market_context": "market_context",
    "factor_combo_daily": "factor_combo_daily",
    "lstm_candidate_retry": "lstm_candidate_retry",
    "lstm_daily_review": "lstm_daily_review",
    "combo_event_governance": "combo_event_governance",
}


async def bootstrap_application(app: FastAPI) -> None:
    """Fast path: migrate DB, register core APIs off-thread, then load the rest."""
    _set_bootstrap_state(app, BOOTSTRAP_STATUS_STARTING)
    record_loop_start(BOOTSTRAP_LOOP_NAME, {"stage": "core_bootstrap"})
    try:
        _configure_asyncio_thread_pool()
        await asyncio.to_thread(init_db)
        await asyncio.to_thread(_register_core_routers, app)
        _set_bootstrap_state(app, BOOTSTRAP_STATUS_RUNNING)
        record_loop_start(BOOTSTRAP_LOOP_NAME, {"stage": "deferred_bootstrap"})
        app.state.bootstrap_task = asyncio.create_task(_deferred_bootstrap(app))
    except Exception as exc:
        _set_bootstrap_state(app, BOOTSTRAP_STATUS_FAILED, exc)
        record_loop_failure(BOOTSTRAP_LOOP_NAME, exc, _bootstrap_failure_details("core_bootstrap", exc))
        logger.exception("application bootstrap failed")
        raise


def _register_core_routers(app: FastAPI) -> None:
    from app.api.events import router as events_router
    from app.api.market import router as market_router
    from app.api.stream import router as stream_router
    from app.api.workbench import router as workbench_router

    for router in (market_router, events_router, workbench_router, stream_router):
        app.include_router(router)


async def _deferred_bootstrap(app: FastAPI) -> None:
    try:
        routers = await asyncio.to_thread(_load_deferred_routers)
        for router in routers:
            app.include_router(router)
        await asyncio.to_thread(_warm_background_imports)
        await asyncio.to_thread(_warm_ai_history_cache)
        _spawn_background_tasks(app)
        _set_bootstrap_state(app, BOOTSTRAP_STATUS_READY)
        record_loop_success(BOOTSTRAP_LOOP_NAME, {"stage": "deferred_bootstrap"})
        logger.info("deferred application bootstrap complete")
    except Exception as exc:
        _set_bootstrap_state(app, BOOTSTRAP_STATUS_FAILED, exc)
        record_loop_failure(BOOTSTRAP_LOOP_NAME, exc, _bootstrap_failure_details("deferred_bootstrap", exc))
        logger.exception("deferred application bootstrap failed")


def _set_bootstrap_state(app: FastAPI, status: str, exc: Exception | None = None) -> None:
    app.state.bootstrap_status = status
    app.state.bootstrap_error = str(exc) if exc else None
    app.state.bootstrap_exception_type = type(exc).__name__ if exc else None


def _bootstrap_failure_details(stage: str, exc: Exception) -> dict:
    details = {"stage": stage}
    failure_details = getattr(exc, "details", None)
    if failure_details is not None:
        details["failureDetails"] = failure_details
    return details


def _load_deferred_routers() -> list[APIRouter]:
    from app.api.auto_trade import router as auto_trade_router
    from app.api.ensemble import router as ensemble_router
    from app.api.factor_combo_positions import router as factor_combo_positions_router
    from app.api.factor_combinations import router as factor_combinations_router
    from app.api.factor_learning import router as factor_learning_router
    from app.api.factors import router as factors_router
    from app.api.mining import router as mining_router
    from app.api.models import router as models_router
    from app.api.rules import router as rules_router

    return [
        ensemble_router,
        auto_trade_router,
        rules_router,
        factors_router,
        factor_combinations_router,
        factor_combo_positions_router,
        factor_learning_router,
        mining_router,
        models_router,
    ]


def _warm_background_imports() -> None:
    import app.services.auto_predict_service  # noqa: F401
    import app.services.auto_settlement_service  # noqa: F401
    import app.services.auto_trade_service  # noqa: F401
    import app.services.combo_event_governance_background  # noqa: F401
    import app.services.factor_combination_background  # noqa: F401
    import app.services.factor_ranking_background  # noqa: F401
    import app.services.lstm_candidate_retry_background  # noqa: F401
    import app.services.lstm_daily_review_background  # noqa: F401
    import app.services.market_context_background  # noqa: F401


def _spawn_background_tasks(app: FastAPI) -> None:
    from app.services.auto_predict_service import auto_predict_loop
    from app.services.auto_settlement_service import auto_settlement_loop
    from app.services.auto_trade_service import auto_trade_loop
    from app.services.combo_event_governance_background import combo_event_governance_refresh_loop
    from app.services.factor_combination_background import factor_combination_daily_refresh_loop
    from app.services.factor_ranking_background import factor_ranking_refresh_loop
    from app.services.lstm_candidate_retry_background import (
        lstm_candidate_retry_enabled,
        lstm_candidate_retry_loop,
    )
    from app.services.lstm_daily_review_background import (
        lstm_daily_review_enabled,
        lstm_daily_review_loop,
    )
    from app.services.market_context_background import market_context_refresh_loop

    _spawn_loop(app, "settlement", auto_settlement_loop)
    _spawn_loop(app, "predict", auto_predict_loop)
    _spawn_loop(app, "trade", auto_trade_loop)
    _spawn_loop(app, "factor_ranking", factor_ranking_refresh_loop)
    _spawn_loop(app, "market_context", market_context_refresh_loop)
    _spawn_loop(app, "factor_combo_daily", factor_combination_daily_refresh_loop)

    if lstm_daily_review_enabled():
        _spawn_loop(app, "lstm_daily_review", lstm_daily_review_loop)

    if lstm_candidate_retry_enabled():
        _spawn_loop(app, "lstm_candidate_retry", lstm_candidate_retry_loop)

    _spawn_loop(app, "combo_event_governance", combo_event_governance_refresh_loop)


def _spawn_loop(app: FastAPI, name: str, loop_factory) -> None:
    stop_event = asyncio.Event()
    setattr(app.state, f"{name}_stop_event", stop_event)
    task = asyncio.create_task(loop_factory(stop_event))
    task.add_done_callback(lambda done: _record_background_task_result(name, done))
    setattr(app.state, f"{name}_task", task)


def _record_background_task_result(name: str, task: asyncio.Task) -> None:
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is None:
        return
    if not isinstance(exc, Exception):
        logger.error("background task stopped with base exception: %s", exc)
        return
    status_name = BACKGROUND_LOOP_STATUS_NAMES.get(name, name)
    if _already_recorded_task_failure(status_name, exc):
        return
    details = {"stage": "background_task", "taskName": name}
    record_loop_failure(status_name, exc, details)
    logger.error(
        "background task failed: %s",
        name,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def _already_recorded_task_failure(status_name: str, exc: Exception) -> bool:
    status = background_loop_statuses().get(status_name) or {}
    return status.get("status") == "failed" and status.get("lastError") == str(exc)


def _warm_ai_history_cache() -> None:
    from app.db.session import get_conn
    from app.services.ai_history_cache import warm_ai_history_cache

    conn = get_conn()
    try:
        warm_ai_history_cache(conn)
    finally:
        conn.close()


async def shutdown_application(app: FastAPI) -> None:
    bootstrap_task = getattr(app.state, "bootstrap_task", None)
    if bootstrap_task and not bootstrap_task.done():
        bootstrap_task.cancel()
        try:
            await bootstrap_task
        except asyncio.CancelledError:
            logger.debug("deferred application bootstrap cancelled during shutdown")

    for attr in STOP_EVENT_ATTRS:
        ev = getattr(app.state, attr, None)
        if ev:
            ev.set()
    await _cancel_background_tasks(_background_tasks(app))


def _background_tasks(app: FastAPI) -> list[asyncio.Task]:
    tasks = []
    for attr in BACKGROUND_TASK_ATTRS:
        task = getattr(app.state, attr, None)
        if task and not task.done():
            tasks.append(task)
    return tasks


async def _cancel_background_tasks(tasks: list[asyncio.Task]) -> None:
    if not tasks:
        return
    for task in tasks:
        task.cancel()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=BACKGROUND_SHUTDOWN_TIMEOUT_SECONDS,
        )
        _log_background_task_shutdown_results(results)
    except asyncio.TimeoutError:
        logger.warning("background task shutdown timed out after %ss", BACKGROUND_SHUTDOWN_TIMEOUT_SECONDS)


def _log_background_task_shutdown_results(results: list[object]) -> None:
    for result in results:
        if result is None or isinstance(result, asyncio.CancelledError):
            continue
        if isinstance(result, BaseException):
            logger.error(
                "background task failed during shutdown: %s",
                result,
                exc_info=(type(result), result, result.__traceback__),
            )
