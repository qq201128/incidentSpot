from __future__ import annotations

import asyncio
import logging

from app.db.session import get_conn
from app.services.auto_trade_service import get_auto_trade_settings, list_auto_trade_settings
from app.services.auto_trade_types import AutoTradeSettings
from app.services.binance_service import fetch_klines
from app.services.forward_validation_service import settle_due_predictions
from app.services.kline_timing import (
    MS_PER_MINUTE,
    current_rule_entry_open_time_for_duration,
    is_within_entry_grace,
    seconds_until_next_rule_entry_for_duration,
    utc_now_ms,
)
from app.services.prediction_cache_service import (
    prediction_exists,
    prediction_passed_exists,
    prediction_response,
    save_prediction,
)
from app.services.rule_signal_service import predict_rule_direction
from app.services.strategy_registry import (
    DEFAULT_STRATEGY_KEY,
    is_continuous_orderbook_strategy,
    strategy_entry_grace_ms,
    strategy_supports_duration,
)

logger = logging.getLogger("uvicorn.error")
_SUBSCRIBERS: dict[tuple[str, str, str], set] = {}

DEFAULT_PREDICT_SECONDS = 1
REFRESH_KLINE_LIMIT = 5
INITIAL_KLINE_LIMIT = 1000
DEFAULT_DURATION = "10m"


async def auto_predict_loop(stop_event: asyncio.Event, poll_seconds: int = DEFAULT_PREDICT_SECONDS) -> None:
    logger.info("predict loop: running during each enabled strategy's kline entry window")
    while not stop_event.is_set():
        try:
            targets = await asyncio.to_thread(_prediction_targets)
            await _predict_due_entries(targets)
            wait_seconds = _next_predict_wait(targets, poll_seconds)
        except Exception:
            logger.exception("auto prediction failed")
            wait_seconds = float(poll_seconds)
        await _sleep_for(stop_event, wait_seconds)


async def _predict_due_entries(targets: list[AutoTradeSettings]) -> None:
    due_targets = await asyncio.to_thread(_due_prediction_targets, targets)
    if not due_targets:
        return
    await _prepare_prediction_inputs(due_targets)
    await _run_prediction_batch(due_targets)


async def _prepare_prediction_inputs(settings_list: list[AutoTradeSettings]) -> None:
    # 每次预测前拉取最新 1m K 写入库，保证任意周期（10m/30m/60m/1d）结算与规则输入与行情对齐。
    by_symbol: dict[str, list[int]] = {}
    for settings in settings_list:
        sym = settings.symbol.upper()
        by_symbol.setdefault(sym, []).append(current_rule_entry_open_time_for_duration(settings.duration))
    if by_symbol:
        await asyncio.gather(
            *(
                asyncio.to_thread(_refresh_prediction_input, symbol, max(buckets))
                for symbol, buckets in by_symbol.items()
                if buckets
            )
        )
    await asyncio.gather(
        *(
            asyncio.to_thread(settle_due_predictions, symbol, duration)
            for symbol, duration in _unique_symbol_durations(settings_list)
        )
    )


async def _run_prediction_batch(settings_list: list[AutoTradeSettings]) -> None:
    write_lock = asyncio.Lock()
    results = await asyncio.gather(
        *(_run_prediction(settings, write_lock=write_lock) for settings in settings_list),
        return_exceptions=True,
    )
    _raise_prediction_failures(settings_list, results)


async def _run_prediction(
    settings: AutoTradeSettings,
    *,
    write_lock: asyncio.Lock,
) -> None:
    entry_open_time = current_rule_entry_open_time_for_duration(settings.duration)
    result = await asyncio.to_thread(
        predict_rule_direction,
        settings.symbol,
        settings.duration,
        entry_open_time=entry_open_time,
        strategy_key=settings.strategy_key,
    )
    allow_existing = is_continuous_orderbook_strategy(settings.strategy_key)
    if not await _save_prediction(result, write_lock, allow_existing=allow_existing):
        return
    await _broadcast(prediction_response(result))
    logger.info(
        "predict: %s %s entry=%s -> %s (conf=%.4f quality=%.4f qualityPassed=%s)",
        settings.symbol,
        settings.duration,
        entry_open_time,
        result["direction"],
        result["confidence"],
        result["trade_quality_score"],
        result["trade_quality_passed"],
    )


async def _save_prediction(
    result: dict,
    write_lock: asyncio.Lock,
    *,
    allow_existing: bool = False,
) -> bool:
    async with write_lock:
        return await asyncio.to_thread(save_prediction, result, allow_existing=allow_existing)


def _raise_prediction_failures(settings_list: list[AutoTradeSettings], results: list[object]) -> None:
    failures = _prediction_failures(settings_list, results)
    for settings, exc in failures:
        logger.error(
            "predict strategy failed strategy=%s",
            settings.strategy_key,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    if failures:
        failed_keys = ", ".join(settings.strategy_key for settings, _exc in failures)
        raise RuntimeError(f"auto prediction failed for strategies: {failed_keys}") from failures[0][1]


def _prediction_failures(
    settings_list: list[AutoTradeSettings],
    results: list[object],
) -> list[tuple[AutoTradeSettings, Exception]]:
    return [
        (settings, result)
        for settings, result in zip(settings_list, results)
        if isinstance(result, Exception)
    ]


def _prediction_targets() -> list[AutoTradeSettings]:
    settings = list_auto_trade_settings()
    enabled = [
        item for item in settings
        if item.enabled and strategy_supports_duration(item.strategy_key, item.duration)
    ]
    if enabled:
        return enabled
    return [get_auto_trade_settings(DEFAULT_STRATEGY_KEY)]


def _due_prediction_targets(targets: list[AutoTradeSettings]) -> list[AutoTradeSettings]:
    return [settings for settings in targets if _should_predict_entry(settings)]


def _should_predict_entry(settings: AutoTradeSettings) -> bool:
    bucket = current_rule_entry_open_time_for_duration(settings.duration)
    if not is_within_entry_grace(
        bucket,
        grace_ms=strategy_entry_grace_ms(settings.strategy_key),
    ):
        return False
    if is_continuous_orderbook_strategy(settings.strategy_key):
        return not prediction_passed_exists(
            strategy_key=settings.strategy_key,
            symbol=settings.symbol,
            duration=settings.duration,
            open_time=bucket,
        )
    return (
        not prediction_exists(
            strategy_key=settings.strategy_key,
            symbol=settings.symbol,
            duration=settings.duration,
            open_time=bucket,
        )
    )


def _unique_symbol_durations(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str]]:
    return sorted({(settings.symbol.upper(), settings.duration) for settings in settings_list})


def _refresh_prediction_input(symbol: str, entry_open_time: int) -> None:
    latest_open_time = _latest_1m_open_time(symbol)
    limit = INITIAL_KLINE_LIMIT if latest_open_time is None else REFRESH_KLINE_LIMIT
    rows = fetch_klines(symbol, "1m", limit=limit)
    if not rows:
        raise ValueError(f"no latest 1m klines returned for {symbol.upper()}")
    _upsert_1m_klines(symbol, rows)
    _assert_entry_input_ready(symbol, entry_open_time)


def _assert_entry_input_ready(symbol: str, entry_open_time: int) -> None:
    required_open_time = int(entry_open_time) - MS_PER_MINUTE
    latest_open_time = _latest_1m_open_time(symbol)
    if latest_open_time is None or latest_open_time < required_open_time:
        raise ValueError(
            f"missing completed 1m kline before entry {entry_open_time} for {symbol.upper()}"
        )


def _latest_1m_open_time(symbol: str) -> int | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT MAX(open_time) AS max_open_time FROM klines WHERE symbol = ? AND interval = ?",
        (symbol.upper(), "1m"),
    ).fetchone()
    conn.close()
    if row is None or row["max_open_time"] is None:
        return None
    return int(row["max_open_time"])


def _upsert_1m_klines(symbol: str, rows: list[dict]) -> None:
    conn = get_conn()
    for item in rows:
        conn.execute(
            """
            INSERT INTO klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              volume=excluded.volume,
              close_time=excluded.close_time
            """,
            (
                symbol.upper(),
                "1m",
                item["openTime"],
                item["open"],
                item["high"],
                item["low"],
                item["close"],
                item["volume"],
                item["closeTime"],
            ),
        )
    conn.commit()
    conn.close()


def _next_predict_wait(targets: list[AutoTradeSettings], poll_seconds: int) -> float:
    if not targets:
        return float(poll_seconds)
    now_ms = utc_now_ms()
    min_wait = float("inf")
    for settings in targets:
        bucket = current_rule_entry_open_time_for_duration(settings.duration, now_ms)
        grace_ms = strategy_entry_grace_ms(settings.strategy_key)
        if is_within_entry_grace(bucket, now_ms, grace_ms=grace_ms):
            min_wait = min(min_wait, float(poll_seconds))
        else:
            min_wait = min(min_wait, seconds_until_next_rule_entry_for_duration(settings.duration, now_ms))
    return float(min_wait) if min_wait != float("inf") else float(poll_seconds)


async def _sleep_for(stop_event: asyncio.Event, wait_seconds: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
    except TimeoutError:
        return


async def _broadcast(result: dict) -> None:
    key = (
        result["symbol"].upper(),
        result["duration"],
        result.get("strategyKey") or DEFAULT_STRATEGY_KEY,
    )
    websockets = _SUBSCRIBERS.get(key, set())
    dead = set()
    for ws in websockets:
        try:
            await ws.send_json(result)
        except Exception:
            dead.add(ws)
    if dead:
        websockets -= dead

def subscribe(
    ws,
    symbol: str,
    duration: str,
    strategy_key: str = DEFAULT_STRATEGY_KEY,
) -> None:
    _SUBSCRIBERS.setdefault((symbol.upper(), duration, strategy_key), set()).add(ws)


def unsubscribe(
    ws,
    symbol: str,
    duration: str,
    strategy_key: str = DEFAULT_STRATEGY_KEY,
) -> None:
    s = _SUBSCRIBERS.get((symbol.upper(), duration, strategy_key))
    if s:
        s.discard(ws)
