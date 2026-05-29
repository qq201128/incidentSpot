from __future__ import annotations

import asyncio
import logging
import os

from app.services.background_threads import run_blocking_daemon
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.market_context_ingest_service import ingest_market_context_data

logger = logging.getLogger("uvicorn.error")


def _refresh_interval_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("MARKET_CONTEXT_REFRESH_SECONDS", "300")))
    except ValueError:
        return 300.0


def _initial_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("MARKET_CONTEXT_INITIAL_DELAY_SECONDS", "2")))
    except ValueError:
        return 2.0


def refresh_all_configured_market_context() -> None:
    symbols = factor_ranking_precomputed_symbols()
    logger.info("market context refresh configured symbols=%s", symbols)
    for symbol in symbols:
        try:
            report = ingest_market_context_data(symbol)
            logger.info("market context updated: %s", report)
        except Exception:
            logger.exception("market context update failed: %s", symbol)


async def market_context_refresh_loop(stop_event: asyncio.Event) -> None:
    interval = _refresh_interval_seconds()
    initial = _initial_delay_seconds()
    logger.info(
        "market context background: symbols=%s interval=%ss initial_delay=%ss",
        factor_ranking_precomputed_symbols(),
        interval,
        initial,
    )
    if initial > 0:
        await _sleep_for(stop_event, initial)
    while not stop_event.is_set():
        await run_blocking_daemon(refresh_all_configured_market_context)
        await _sleep_for(stop_event, interval)


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return
