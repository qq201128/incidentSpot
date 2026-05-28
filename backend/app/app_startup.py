from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, FastAPI

from app.db.session import init_db

logger = logging.getLogger(__name__)


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


async def bootstrap_application(app: FastAPI) -> None:
    """Fast path: migrate DB, register core APIs off-thread, then load the rest."""
    _configure_asyncio_thread_pool()
    await asyncio.to_thread(init_db)
    await asyncio.to_thread(_register_core_routers, app)
    app.state.bootstrap_task = asyncio.create_task(_deferred_bootstrap(app))


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
        logger.info("deferred application bootstrap complete")
    except Exception:
        logger.exception("deferred application bootstrap failed")


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

    settlement_stop = asyncio.Event()
    app.state.settlement_stop_event = settlement_stop
    app.state.settlement_task = asyncio.create_task(auto_settlement_loop(settlement_stop))

    predict_stop = asyncio.Event()
    app.state.predict_stop_event = predict_stop
    app.state.predict_task = asyncio.create_task(auto_predict_loop(predict_stop))

    trade_stop = asyncio.Event()
    app.state.trade_stop_event = trade_stop
    app.state.trade_task = asyncio.create_task(auto_trade_loop(trade_stop))

    factor_ranking_stop = asyncio.Event()
    app.state.factor_ranking_stop_event = factor_ranking_stop
    app.state.factor_ranking_task = asyncio.create_task(factor_ranking_refresh_loop(factor_ranking_stop))

    market_context_stop = asyncio.Event()
    app.state.market_context_stop_event = market_context_stop
    app.state.market_context_task = asyncio.create_task(market_context_refresh_loop(market_context_stop))

    factor_combo_stop = asyncio.Event()
    app.state.factor_combo_daily_stop_event = factor_combo_stop
    app.state.factor_combo_daily_task = asyncio.create_task(
        factor_combination_daily_refresh_loop(factor_combo_stop)
    )

    if lstm_daily_review_enabled():
        lstm_review_stop = asyncio.Event()
        app.state.lstm_daily_review_stop_event = lstm_review_stop
        app.state.lstm_daily_review_task = asyncio.create_task(lstm_daily_review_loop(lstm_review_stop))

    if lstm_candidate_retry_enabled():
        lstm_retry_stop = asyncio.Event()
        app.state.lstm_candidate_retry_stop_event = lstm_retry_stop
        app.state.lstm_candidate_retry_task = asyncio.create_task(lstm_candidate_retry_loop(lstm_retry_stop))

    governance_stop = asyncio.Event()
    app.state.combo_event_governance_stop_event = governance_stop
    app.state.combo_event_governance_task = asyncio.create_task(
        combo_event_governance_refresh_loop(governance_stop)
    )


def _warm_ai_history_cache() -> None:
    from app.db.session import get_conn
    from app.services.ai_history_cache import warm_ai_history_cache

    conn = get_conn()
    try:
        warm_ai_history_cache(conn)
    except Exception:
        logger.exception("ai history cache warm-up failed")
    finally:
        conn.close()


async def shutdown_application(app: FastAPI) -> None:
    bootstrap_task = getattr(app.state, "bootstrap_task", None)
    if bootstrap_task and not bootstrap_task.done():
        bootstrap_task.cancel()
        try:
            await bootstrap_task
        except asyncio.CancelledError:
            pass

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
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=BACKGROUND_SHUTDOWN_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("background task shutdown timed out after %ss", BACKGROUND_SHUTDOWN_TIMEOUT_SECONDS)
