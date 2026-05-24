from __future__ import annotations

import asyncio
import logging
import os

from app.services.background_threads import run_blocking_daemon
from app.services.combo_event_governance import compute_combo_event_monitoring
from app.services.combo_event_governance_cache import (
    store_governance,
    store_monitoring,
    store_shadow_report,
)
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols

logger = logging.getLogger("uvicorn.error")
GOVERNANCE_DURATIONS: tuple[str, ...] = ("10m", "30m", "60m", "1d")


def _refresh_interval_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("COMBO_EVENT_GOVERNANCE_REFRESH_SECONDS", "180")))
    except ValueError:
        return 180.0


def _initial_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("COMBO_EVENT_GOVERNANCE_INITIAL_DELAY_SECONDS", "15")))
    except ValueError:
        return 15.0


def refresh_combo_event_governance_all() -> None:
    symbols = factor_ranking_precomputed_symbols()
    for symbol in symbols:
        for duration in GOVERNANCE_DURATIONS:
            try:
                monitoring = compute_combo_event_monitoring(symbol, duration)
                governance = {
                    "symbol": monitoring["symbol"],
                    "duration": monitoring["duration"],
                    "batchComboDemotion": monitoring["batchComboDemotion"],
                }
                store_monitoring(symbol, duration, monitoring)
                store_governance(symbol, duration, governance)
                store_shadow_report(symbol, duration, monitoring["shadowEventDeviation"])
                from app.services.workbench_summary_cache import store_workbench_summary
                from app.services.workbench_summary_service import build_workbench_summary

                store_workbench_summary(symbol, duration, build_workbench_summary(symbol, duration))
                logger.info(
                    "combo event governance refreshed: %s %s paired=%s watchlist=%s",
                    symbol,
                    duration,
                    monitoring.get("shadowEventDeviation", {}).get("summary", {}).get("pairedCount"),
                    monitoring.get("batchComboDemotion", {}).get("watchlistCount"),
                )
            except Exception:
                logger.exception("combo event governance refresh failed: %s %s", symbol, duration)


async def combo_event_governance_refresh_loop(stop_event: asyncio.Event) -> None:
    interval = _refresh_interval_seconds()
    initial = _initial_delay_seconds()
    logger.info(
        "combo event governance background: symbols=%s interval=%ss initial_delay=%ss",
        factor_ranking_precomputed_symbols(),
        interval,
        initial,
    )
    if initial > 0:
        await _sleep_for(stop_event, initial)
    while not stop_event.is_set():
        await run_blocking_daemon(refresh_combo_event_governance_all)
        await _sleep_for(stop_event, interval)


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return
