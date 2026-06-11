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
from app.services.combo_event_governance import compute_combo_event_monitoring
from app.services.combo_event_governance_cache import (
    store_governance,
    store_monitoring,
    store_shadow_report,
)
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols

logger = logging.getLogger("uvicorn.error")
GOVERNANCE_DURATIONS: tuple[str, ...] = ("10m", "30m", "60m", "1d")
LOOP_NAME = "combo_event_governance"


def _refresh_interval_seconds() -> float:
    return float_env("COMBO_EVENT_GOVERNANCE_REFRESH_SECONDS", 180.0, min_value=60.0)


def _initial_delay_seconds() -> float:
    return float_env("COMBO_EVENT_GOVERNANCE_INITIAL_DELAY_SECONDS", 15.0, min_value=0.0)


def refresh_combo_event_governance_all() -> None:
    symbols = factor_ranking_precomputed_symbols()
    failed = []
    for symbol in symbols:
        for duration in GOVERNANCE_DURATIONS:
            try:
                monitoring = compute_combo_event_monitoring(symbol, duration)
                governance = {
                    "symbol": monitoring["symbol"],
                    "duration": monitoring["duration"],
                    "batchComboDemotion": monitoring["batchComboDemotion"],
                    "factorCandidateDemotion": monitoring["factorCandidateDemotion"],
                }
                store_monitoring(symbol, duration, monitoring)
                store_governance(symbol, duration, governance)
                store_shadow_report(symbol, duration, monitoring["shadowEventDeviation"])
                from app.services.workbench_summary_cache import store_workbench_summary
                from app.services.workbench_summary_service import build_workbench_summary

                store_workbench_summary(symbol, duration, build_workbench_summary(symbol, duration))
                # logger.info(
                #     "combo event governance refreshed: %s %s paired=%s batch_watch=%s single_watch=%s",
                #     symbol,
                #     duration,
                #     monitoring.get("shadowEventDeviation", {}).get("summary", {}).get("pairedCount"),
                #     monitoring.get("batchComboDemotion", {}).get("watchlistCount"),
                #     monitoring.get("factorCandidateDemotion", {}).get("watchlistCount"),
                # )
            except Exception as exc:
                failed.append({"symbol": symbol, "duration": duration})
                record_loop_failure(LOOP_NAME, exc, {"symbol": symbol, "duration": duration})
                logger.exception("combo event governance refresh failed: %s %s", symbol, duration)
    details = {"symbolCount": len(symbols), "failedItems": failed}
    if failed:
        record_loop_failure(LOOP_NAME, RuntimeError("combo event governance failed for items"), details)
        return
    record_loop_success(LOOP_NAME, details)


async def combo_event_governance_refresh_loop(stop_event: asyncio.Event) -> None:
    try:
        interval = _refresh_interval_seconds()
        initial = _initial_delay_seconds()
    except Exception as exc:
        record_loop_failure(LOOP_NAME, exc, {"stage": "startup_config"})
        logger.exception("combo event governance background startup config failed")
        raise
    logger.info(
        "combo event governance background: symbols=%s interval=%ss initial_delay=%ss",
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
            await run_blocking_daemon(refresh_combo_event_governance_all)
        except Exception as exc:
            record_loop_failure(LOOP_NAME, exc, {"stage": "batch"})
            logger.exception("combo event governance background batch failed")
        if await _sleep_for(stop_event, interval):
            record_loop_stopped(LOOP_NAME, "stop_between_batches")
            return


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False
