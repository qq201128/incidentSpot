from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.services.factor_backtest_batch_service import BACKTEST_DURATION_ORDER
from app.db.session import get_conn
from app.services.auto_trade_default_slots import enable_default_simulation_strategy_slots
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import (
    CombinationSearchConfig,
    run_factor_combination_ranking,
)
from app.services.factor_learning_service import refresh_factor_learning_memory
from app.services.factor_mined_library import upsert_good_combinations
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.lstm_combo_sync_service import sync_lstm_model_to_combo_ranking
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS

logger = logging.getLogger("uvicorn.error")
DAILY_REFRESH_TZ = ZoneInfo("Asia/Shanghai")
DAILY_REFRESH_AT = time(hour=0, minute=30)


def refresh_combination_ranking_for_symbol_duration(
    symbol: str,
    duration: str,
    config: CombinationSearchConfig | None = None,
) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    report = run_factor_combination_ranking(symbol.strip().upper(), duration, config)
    save_cached_combination_ranking(report)
    promotion = upsert_good_combinations(report)
    logger.info(
        "mined factor library updated: %s %s promoted=%s total=%s",
        promotion["symbol"],
        promotion["duration"],
        promotion["promoted"],
        promotion["libraryTotal"],
    )
    _sync_lstm_shadow_model(symbol.strip().upper(), duration, report)
    refresh_factor_learning_memory(
        symbol.strip().upper(),
        duration,
        ranking_report=report,
        run_llm_agent=True,
    )


def _sync_lstm_shadow_model(symbol: str, duration: str, report: dict) -> None:
    sync = sync_lstm_model_to_combo_ranking(symbol, duration, ranking_report=report)
    logger.info(
        "lstm combo sync: %s %s status=%s model=%s",
        symbol,
        duration,
        sync["status"],
        sync.get("modelVersion"),
    )


def refresh_symbol_combination_rankings(
    symbol: str,
    duration: str | None = None,
    config: CombinationSearchConfig | None = None,
) -> None:
    if duration is not None:
        refresh_combination_ranking_for_symbol_duration(symbol, duration, config)
        return
    for dur in _refresh_durations():
        refresh_combination_ranking_for_symbol_duration(symbol, dur, config)


def refresh_all_configured_combination_rankings(
    config: CombinationSearchConfig | None = None,
) -> None:
    _sync_default_simulation_slots()
    for symbol in factor_ranking_precomputed_symbols():
        try:
            refresh_symbol_combination_rankings(symbol, None, config)
            logger.info("factor combo ranking cache updated: %s (all durations)", symbol)
        except Exception:
            logger.exception("factor combo ranking cache failed: %s", symbol)
    _sync_default_simulation_slots()


def _sync_default_simulation_slots() -> None:
    conn = get_conn()
    try:
        enable_default_simulation_strategy_slots(
            conn,
            _refresh_durations(),
            DURATION_TO_MINUTES,
            datetime.now(DAILY_REFRESH_TZ).isoformat(),
        )
        conn.commit()
    finally:
        conn.close()


def seconds_until_next_daily_refresh(now: datetime | None = None) -> float:
    current = now or datetime.now(DAILY_REFRESH_TZ)
    current = _localized(current)
    target = datetime.combine(current.date(), DAILY_REFRESH_AT, DAILY_REFRESH_TZ)
    if target <= current:
        target += timedelta(days=1)
    return (target - current).total_seconds()


async def factor_combination_daily_refresh_loop(stop_event: asyncio.Event) -> None:
    logger.info(
        "factor combo daily review: symbols=%s at=00:30 timezone=Asia/Shanghai",
        factor_ranking_precomputed_symbols(),
    )
    while not stop_event.is_set():
        await _sleep_for(stop_event, seconds_until_next_daily_refresh())
        if stop_event.is_set():
            return
        try:
            await asyncio.to_thread(refresh_all_configured_combination_rankings)
        except Exception:
            logger.exception("factor combo daily refresh batch failed")


def _refresh_durations() -> tuple[str, ...]:
    return tuple(dur for dur in BACKTEST_DURATION_ORDER if dur in SUPPORTED_RULE_DURATIONS)


def _localized(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=DAILY_REFRESH_TZ)
    return value.astimezone(DAILY_REFRESH_TZ)


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass
