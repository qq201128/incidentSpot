from __future__ import annotations

import asyncio
import logging
import os

from app.services.background_threads import run_blocking_daemon
from app.services.combo_event_governance import (
    compute_combo_event_monitoring,
    compute_combo_event_governance,
)
from app.services.combo_event_governance_cache import (
    store_governance,
    store_monitoring,
    store_shadow_report,
)
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.high_winrate_strategy_demotion import run_pending_high_winrate_goal_refresh

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


def _goal_refresh_interval_seconds() -> float:
    try:
        return max(300.0, float(os.getenv("HIGH_WINRATE_GOAL_REFRESH_SECONDS", "3600")))
    except ValueError:
        return 3600.0


def refresh_combo_event_governance_all(*, run_goal_refresh: bool = False) -> None:
    symbols = factor_ranking_precomputed_symbols()
    for symbol in symbols:
        for duration in GOVERNANCE_DURATIONS:
            try:
                monitoring = compute_combo_event_monitoring(symbol, duration)
                governance = {
                    "symbol": monitoring["symbol"],
                    "duration": monitoring["duration"],
                    "highWinrateStatus": monitoring["highWinrateStatus"],
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
        if run_goal_refresh:
            for duration in GOVERNANCE_DURATIONS:
                try:
                    result = run_pending_high_winrate_goal_refresh(symbol, duration)
                    if result is not None:
                        logger.info(
                            "high winrate goal refresh completed: %s %s status=%s reason=%s",
                            symbol,
                            duration,
                            result.get("status"),
                            result.get("reason"),
                        )
                except Exception:
                    logger.exception("high winrate goal refresh failed: %s %s", symbol, duration)


async def combo_event_governance_refresh_loop(stop_event: asyncio.Event) -> None:
    interval = _refresh_interval_seconds()
    initial = _initial_delay_seconds()
    goal_interval = _goal_refresh_interval_seconds()
    logger.info(
        "combo event governance background: symbols=%s interval=%ss initial_delay=%ss goal_refresh=%ss",
        factor_ranking_precomputed_symbols(),
        interval,
        initial,
        goal_interval,
    )
    if initial > 0:
        await _sleep_for(stop_event, initial)
    last_goal_refresh = 0.0
    while not stop_event.is_set():
        run_goal = False
        now = asyncio.get_running_loop().time()
        if now - last_goal_refresh >= goal_interval:
            run_goal = True
            last_goal_refresh = now
        await run_blocking_daemon(refresh_combo_event_governance_all, run_goal_refresh=run_goal)
        await _sleep_for(stop_event, interval)


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return
