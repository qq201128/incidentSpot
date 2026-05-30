from __future__ import annotations

import asyncio
import logging

from app.services.background_threads import run_blocking_daemon
from app.services.background_loop_config import float_env
from app.services.background_loop_status import (
    record_loop_failure,
    record_loop_start,
    record_loop_stopped,
    record_loop_success,
)
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.market_context_ingest_service import ingest_market_context_data

logger = logging.getLogger("uvicorn.error")
LOOP_NAME = "market_context"


def _refresh_interval_seconds() -> float:
    return float_env("MARKET_CONTEXT_REFRESH_SECONDS", 300.0, min_value=60.0)


def _initial_delay_seconds() -> float:
    return float_env("MARKET_CONTEXT_INITIAL_DELAY_SECONDS", 2.0, min_value=0.0)


def refresh_all_configured_market_context() -> None:
    symbols = factor_ranking_precomputed_symbols()
    logger.info("market context refresh configured symbols=%s", symbols)
    failed = []
    for symbol in symbols:
        try:
            report = ingest_market_context_data(symbol)
            logger.info("market context updated: %s", report)
        except Exception as exc:
            failed.append(symbol)
            record_loop_failure(LOOP_NAME, exc, {"symbol": symbol})
            logger.exception("market context update failed: %s", symbol)
    details = {"symbolCount": len(symbols), "failedSymbols": failed}
    if failed:
        record_loop_failure(LOOP_NAME, RuntimeError("market context failed for symbols"), details)
        return
    record_loop_success(LOOP_NAME, details)


async def market_context_refresh_loop(stop_event: asyncio.Event) -> None:
    try:
        interval = _refresh_interval_seconds()
        initial = _initial_delay_seconds()
    except Exception as exc:
        record_loop_failure(LOOP_NAME, exc, {"stage": "startup_config"})
        logger.exception("market context background startup config failed")
        raise
    logger.info(
        "market context background: symbols=%s interval=%ss initial_delay=%ss",
        factor_ranking_precomputed_symbols(),
        interval,
        initial,
    )
    record_loop_start(LOOP_NAME, {"intervalSeconds": interval, "initialDelaySeconds": initial})
    if stop_event.is_set():
        record_loop_stopped(LOOP_NAME, "stop_before_first_batch")
        return
    if initial > 0:
        if await _sleep_for(stop_event, initial):
            record_loop_stopped(LOOP_NAME, "stop_during_initial_delay")
            return
    while not stop_event.is_set():
        try:
            await run_blocking_daemon(refresh_all_configured_market_context)
        except Exception as exc:
            record_loop_failure(LOOP_NAME, exc, {"stage": "batch"})
            logger.exception("market context background batch failed")
        if await _sleep_for(stop_event, interval):
            record_loop_stopped(LOOP_NAME, "stop_between_batches")
            return


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False
