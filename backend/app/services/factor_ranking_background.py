from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from app.services.factor_backtest_service import run_factor_ranking
from app.services.factor_ranking_cache_service import (
    factor_ranking_precomputed_symbols,
    save_cached_ranking,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

logger = logging.getLogger("uvicorn.error")


def _refresh_interval_seconds() -> float:
    try:
        return max(60.0, float(os.getenv("FACTOR_RANKING_REFRESH_SECONDS", "1800")))
    except ValueError:
        return 1800.0


def _initial_delay_seconds() -> float:
    try:
        return max(0.0, float(os.getenv("FACTOR_RANKING_INITIAL_DELAY_SECONDS", "8")))
    except ValueError:
        return 8.0


def refresh_ranking_for_symbol_duration(symbol: str, duration: str) -> None:
    """Synchronous: compute full ranking (all categories) and persist."""
    sym = symbol.strip().upper()
    ranking = run_factor_ranking(sym, duration, None)
    save_cached_ranking(sym, duration, ranking)


def refresh_symbol_rankings(symbol: str, duration: str | None = None) -> None:
    """Recompute and store cache for one symbol; all rule durations if duration is None."""
    sym = symbol.strip().upper()
    if duration:
        if duration not in SUPPORTED_RULE_DURATIONS:
            raise ValueError(f"unsupported duration: {duration}")
        refresh_ranking_for_symbol_duration(sym, duration)
        return
    for dur in sorted(SUPPORTED_RULE_DURATIONS):
        refresh_ranking_for_symbol_duration(sym, dur)


def refresh_all_configured_rankings() -> None:
    symbols = factor_ranking_precomputed_symbols()
    for sym in symbols:
        try:
            refresh_symbol_rankings(sym, None)
            logger.info("factor ranking cache updated: %s (all durations)", sym)
        except Exception:
            logger.exception("factor ranking cache failed: %s", sym)


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def factor_ranking_refresh_loop(stop_event: asyncio.Event) -> None:
    """Periodically precompute factor IR rankings for configured symbols (SQLite cache)."""
    interval = _refresh_interval_seconds()
    initial = _initial_delay_seconds()
    logger.info(
        "factor ranking background: symbols=%s interval=%ss initial_delay=%ss",
        factor_ranking_precomputed_symbols(),
        interval,
        initial,
    )
    if initial > 0:
        await _sleep_for(stop_event, initial)
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(refresh_all_configured_rankings)
        except Exception:
            logger.exception("factor ranking background batch failed")
        await _sleep_for(stop_event, interval)
