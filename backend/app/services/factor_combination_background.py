from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from app.services.background_threads import run_blocking_daemon
from app.services.background_loop_status import (
    record_loop_failure,
    record_loop_start,
    record_loop_stopped,
    record_loop_success,
)
from app.services.binance_service import fetch_klines
from app.services.factor_backtest_batch_service import BACKTEST_DURATION_ORDER
from app.db.session import get_conn
from app.services.auto_trade_default_slots import enable_default_simulation_strategy_slots
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import (
    CombinationSearchConfig,
    run_factor_combination_ranking,
)
from app.services.factor_combination_stale_refresh import failed_stale_item, ranking_cache_needs_refresh, stale_refresh_details
from app.services.factor_combination_live_service import rebuild_combination_signal_watchlist
from app.services.factor_combination_signal_cache_service import save_cached_combination_signals
from app.services.factor_learning_service import refresh_factor_learning_memory
from app.services.factor_mined_library import upsert_good_combinations
from app.services.factor_ranking_cache_service import factor_ranking_precomputed_symbols
from app.services.kline_backfill import count_klines, oldest_open_time, upsert_klines_rows
from app.services.lstm_combo_sync_service import sync_lstm_model_to_combo_ranking
from app.services.market_context_ingest_service import ingest_market_context_data
from app.services.paper_live_daily_loop_status import daily_loop_failure_details
from app.services.qualified_factor_simulation_slots import sync_qualified_simulation_slots
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS

logger = logging.getLogger("uvicorn.error")
DAILY_REFRESH_TZ = ZoneInfo("Asia/Shanghai")
DAILY_REFRESH_AT = time(hour=0, minute=30)
REFRESH_KLINE_LIMIT = 1000
MIN_DURATION_KLINE_ROWS = 540
MAX_DURATION_BACKFILL_ROUNDS = 10
LOOP_NAME = "factor_combo_daily"


@dataclass(frozen=True)
class BackfillCursor:
    symbol: str
    duration: str
    end_time: int | None


def refresh_combination_ranking_for_symbol_duration(
    symbol: str,
    duration: str,
    config: CombinationSearchConfig | None = None,
    *,
    refresh_signal_cache: bool = True,
    run_learning_agent: bool = True,
) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    sym = symbol.strip().upper()
    ingest_market_context_data(sym, durations=(duration,))
    _refresh_duration_klines(sym, duration)
    report = run_factor_combination_ranking(sym, duration, config)
    save_cached_combination_ranking(report)
    promotion = upsert_good_combinations(report)
    logger.info(
        "mined factor library updated: %s %s promoted=%s total=%s",
        promotion["symbol"],
        promotion["duration"],
        promotion["promoted"],
        promotion["libraryTotal"],
    )
    _sync_lstm_shadow_model(sym, duration, report)
    refresh_factor_learning_memory(
        sym,
        duration,
        ranking_report=report,
        run_llm_agent=run_learning_agent,
    )
    if refresh_signal_cache:
        _refresh_signal_watchlist_cache(sym)
    slot_report = sync_qualified_simulation_slots(sym, duration)
    logger.info(
        "qualified simulation slots synced: %s %s single=%s combo=%s enabled=%s",
        sym,
        duration,
        slot_report["singleFactorSlots"],
        slot_report["comboFactorSlots"],
        slot_report["enabledSlots"],
    )


def _refresh_duration_klines(symbol: str, duration: str) -> None:
    rows = fetch_klines(symbol, duration, limit=REFRESH_KLINE_LIMIT)
    if not rows:
        raise ValueError(f"no latest {duration} klines returned for {symbol}")
    upsert_klines_rows(symbol, duration, rows)
    _backfill_duration_klines(symbol, duration)


def _backfill_duration_klines(symbol: str, duration: str) -> None:
    cursor = BackfillCursor(symbol=symbol, duration=duration, end_time=_initial_backfill_end_time(symbol, duration))
    rounds = 0
    while count_klines(symbol, duration) < MIN_DURATION_KLINE_ROWS:
        if rounds >= MAX_DURATION_BACKFILL_ROUNDS:
            break
        rounds += 1
        rows = fetch_klines(symbol, duration, limit=REFRESH_KLINE_LIMIT, end_time=cursor.end_time)
        if not rows:
            raise ValueError(f"no historical {duration} klines returned for {symbol} before {cursor.end_time}")
        upsert_klines_rows(symbol, duration, rows)
        cursor = _next_backfill_cursor(rows, cursor)


def _initial_backfill_end_time(symbol: str, duration: str) -> int | None:
    oldest = oldest_open_time(symbol, duration)
    return int(oldest) - 1 if oldest is not None else None


def _next_backfill_cursor(rows: list[dict], cursor: BackfillCursor) -> BackfillCursor:
    new_oldest = min(int(row["openTime"]) for row in rows)
    if cursor.end_time is not None and new_oldest >= cursor.end_time:
        raise ValueError(f"historical {cursor.duration} kline backfill did not move earlier for {cursor.symbol}")
    return BackfillCursor(cursor.symbol, cursor.duration, new_oldest - 1)


def _sync_lstm_shadow_model(symbol: str, duration: str, report: dict) -> None:
    sync = sync_lstm_model_to_combo_ranking(symbol, duration, ranking_report=report)
    logger.info(
        "lstm combo sync: %s %s status=%s model=%s",
        symbol,
        duration,
        sync["status"],
        sync.get("modelVersion"),
    )


def _refresh_signal_watchlist_cache(symbol: str) -> None:
    save_cached_combination_signals(rebuild_combination_signal_watchlist(symbol))


def refresh_symbol_combination_rankings(
    symbol: str,
    duration: str | None = None,
    config: CombinationSearchConfig | None = None,
    *,
    run_learning_agent: bool = True,
) -> None:
    if duration is not None:
        refresh_combination_ranking_for_symbol_duration(
            symbol,
            duration,
            config,
            run_learning_agent=run_learning_agent,
        )
        return
    sym = symbol.strip().upper()
    for dur in _refresh_durations():
        refresh_combination_ranking_for_symbol_duration(
            sym,
            dur,
            config,
            refresh_signal_cache=False,
            run_learning_agent=run_learning_agent,
        )
    _refresh_signal_watchlist_cache(sym)


def refresh_all_configured_combination_rankings(
    config: CombinationSearchConfig | None = None,
) -> None:
    _sync_default_simulation_slots()
    failed = []
    for symbol in factor_ranking_precomputed_symbols():
        try:
            refresh_symbol_combination_rankings(symbol, None, config)
            logger.info("factor combo ranking cache updated: %s (all durations)", symbol)
        except Exception as exc:
            failed.append(symbol)
            record_loop_failure(LOOP_NAME, exc, {"symbol": symbol, "stage": "all_configured"})
            logger.exception("factor combo ranking cache failed: %s", symbol)
    _sync_default_simulation_slots()
    details = {"stage": "all_configured", "failedSymbols": failed}
    if failed:
        record_loop_failure(LOOP_NAME, RuntimeError("factor combo ranking failed for symbols"), details)
        return
    record_loop_success(LOOP_NAME, details)


def refresh_stale_configured_combination_rankings(
    config: CombinationSearchConfig | None = None,
) -> None:
    _sync_default_simulation_slots()
    refreshed = []
    failed = []
    skipped = []
    for symbol in factor_ranking_precomputed_symbols():
        for duration in _refresh_durations():
            if not _ranking_cache_needs_refresh(symbol, duration):
                skipped.append({"symbol": symbol, "duration": duration})
                continue
            try:
                refresh_combination_ranking_for_symbol_duration(
                    symbol,
                    duration,
                    config,
                    run_learning_agent=False,
                )
                refreshed.append({"symbol": symbol, "duration": duration})
                logger.info("stale factor combo ranking cache updated: %s %s", symbol, duration)
            except Exception as exc:
                failed_item = failed_stale_item(symbol, duration, exc)
                failed.append(failed_item)
                record_loop_failure(LOOP_NAME, exc, {"stage": "stale", **failed_item})
                logger.exception("stale factor combo ranking cache failed: %s %s", symbol, duration)
    _sync_default_simulation_slots()
    details = stale_refresh_details(refreshed, failed, skipped)
    if failed:
        record_loop_failure(LOOP_NAME, RuntimeError("stale factor combo ranking failed for items"), details)
        return
    record_loop_success(LOOP_NAME, details)


def _ranking_cache_needs_refresh(symbol: str, duration: str) -> bool:
    return ranking_cache_needs_refresh(symbol, duration)


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
    record_loop_start(LOOP_NAME, {"schedule": "00:30 Asia/Shanghai"})
    while not stop_event.is_set():
        if await _sleep_for(stop_event, seconds_until_next_daily_refresh()):
            record_loop_stopped(LOOP_NAME, "stop_before_scheduled_run")
            logger.info("factor combo daily refresh loop stopped before next scheduled run")
            return
        if stop_event.is_set():
            return
        try:
            from app.services.paper_live_daily_loop_service import run_paper_live_daily_closed_loop

            report = await run_blocking_daemon(run_paper_live_daily_closed_loop)
            details = {"stage": "paper_live_daily_loop", "status": report.get("status")}
            if report.get("status") == "passed":
                record_loop_success(LOOP_NAME, details)
            else:
                details.update(daily_loop_failure_details(report))
                record_loop_failure(LOOP_NAME, RuntimeError("paper-live daily loop failed"), details)
        except Exception as exc:
            record_loop_failure(LOOP_NAME, exc, {"stage": "paper_live_daily_loop"})
            logger.exception("factor combo daily refresh batch failed")


def _refresh_durations() -> tuple[str, ...]:
    return tuple(dur for dur in BACKTEST_DURATION_ORDER if dur in SUPPORTED_RULE_DURATIONS)


def _localized(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=DAILY_REFRESH_TZ)
    return value.astimezone(DAILY_REFRESH_TZ)


async def _sleep_for(stop_event: asyncio.Event, seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        return False
