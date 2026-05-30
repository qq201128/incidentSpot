from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.background_threads import run_blocking_daemon
from app.services.background_loop_config import float_env
from app.services.background_loop_status import (
    record_loop_failure,
    record_loop_start,
    record_loop_stopped,
    record_loop_success,
)
from app.services.factor_backtest_service import run_factor_ranking_report
from app.services.factor_ranking_cache_service import (
    factor_ranking_precomputed_symbols,
    save_cached_ranking,
)
from app.services.market_context_ingest_service import ingest_market_context_data
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

logger = logging.getLogger("uvicorn.error")
LOOP_NAME = "factor_ranking"


def _refresh_interval_seconds() -> float:
    return float_env("FACTOR_RANKING_REFRESH_SECONDS", 1800.0, min_value=60.0)


def _initial_delay_seconds() -> float:
    return float_env("FACTOR_RANKING_INITIAL_DELAY_SECONDS", 8.0, min_value=0.0)


def refresh_ranking_for_symbol_duration(symbol: str, duration: str) -> None:
    """Synchronous: compute full ranking (all categories) and persist."""
    sym = symbol.strip().upper()
    ingest_market_context_data(sym, durations=(duration,))
    report = run_factor_ranking_report(sym, duration, None)
    save_cached_ranking(
        sym,
        duration,
        report["ranking"],
        diagnostics=report["rankingDiagnostics"],
        failures=report["rankingFailures"],
    )


def refresh_symbol_rankings(symbol: str, duration: str | None = None) -> None:
    """Recompute and store cache for one symbol; all rule durations if duration is None."""
    sym = symbol.strip().upper()
    if duration:
        if duration not in SUPPORTED_RULE_DURATIONS:
            raise ValueError(f"unsupported duration: {duration}")
        refresh_ranking_for_symbol_duration(sym, duration)
        return
    ingest_market_context_data(sym)
    for dur in sorted(SUPPORTED_RULE_DURATIONS):
        _save_ranking_report(sym, dur)


def _save_ranking_report(symbol: str, duration: str) -> None:
    report = run_factor_ranking_report(symbol, duration, None)
    save_cached_ranking(
        symbol,
        duration,
        report["ranking"],
        diagnostics=report["rankingDiagnostics"],
        failures=report["rankingFailures"],
    )


def refresh_all_configured_rankings() -> None:
    symbols = factor_ranking_precomputed_symbols()
    logger.info("factor ranking refresh configured symbols=%s", symbols)
    failed = []
    for sym in symbols:
        try:
            refresh_symbol_rankings(sym, None)
            logger.info("factor ranking cache updated: %s (all durations)", sym)
        except Exception as exc:
            failed.append(sym)
            record_loop_failure(LOOP_NAME, exc, {"symbol": sym})
            logger.exception("factor ranking cache failed: %s", sym)
    details = {"symbolCount": len(symbols), "failedSymbols": failed}
    if failed:
        record_loop_failure(LOOP_NAME, RuntimeError("factor ranking failed for symbols"), details)
        return
    record_loop_success(LOOP_NAME, details)


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False


async def factor_ranking_refresh_loop(stop_event: asyncio.Event) -> None:
    """Periodically precompute factor IR rankings for configured symbols (SQLite cache)."""
    try:
        interval = _refresh_interval_seconds()
        initial = _initial_delay_seconds()
    except Exception as exc:
        record_loop_failure(LOOP_NAME, exc, {"stage": "startup_config"})
        logger.exception("factor ranking background startup config failed")
        raise
    logger.info(
        "factor ranking background: symbols=%s interval=%ss initial_delay=%ss",
        factor_ranking_precomputed_symbols(),
        interval,
        initial,
    )
    record_loop_start(LOOP_NAME, {"intervalSeconds": interval, "initialDelaySeconds": initial})
    if initial > 0:
        if await _sleep_for(stop_event, initial):
            record_loop_stopped(LOOP_NAME, "initial_delay_stop")
            logger.info("factor ranking background stopped during initial delay")
            return
    while not stop_event.is_set():
        try:
            await run_blocking_daemon(refresh_all_configured_rankings)
        except Exception as exc:
            record_loop_failure(LOOP_NAME, exc, {"stage": "batch"})
            logger.exception("factor ranking background batch failed")
        if await _sleep_for(stop_event, interval):
            record_loop_stopped(LOOP_NAME, "stop_before_next_refresh")
            logger.info("factor ranking background stopped before next refresh")
            return
