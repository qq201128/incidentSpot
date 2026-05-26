from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from app.services.auto_trade_types import AutoTradeSettings


async def prepare_prediction_inputs(settings_list: list[AutoTradeSettings], deps: dict[str, Any]) -> None:
    by_symbol: dict[str, list[int]] = {}
    by_symbol_duration: dict[tuple[str, str], list[int]] = {}
    for settings in settings_list:
        bucket = deps["current_entry"](settings.duration)
        sym = settings.symbol.upper()
        by_symbol.setdefault(sym, []).append(bucket)
        by_symbol_duration.setdefault((sym, settings.duration), []).append(bucket)
    await refresh_inputs(by_symbol, by_symbol_duration, deps)
    await asyncio.to_thread(deps["db_side_effects"], settings_list)


async def refresh_inputs(by_symbol: dict[str, list[int]], by_symbol_duration: dict[tuple[str, str], list[int]], deps: dict[str, Any]) -> None:
    if by_symbol:
        await asyncio.gather(*(asyncio.to_thread(deps["refresh_1m"], symbol, max(buckets)) for symbol, buckets in by_symbol.items() if buckets))
    if by_symbol_duration:
        await asyncio.gather(
            *(
                asyncio.to_thread(deps["refresh_duration"], symbol, duration, max(buckets))
                for (symbol, duration), buckets in by_symbol_duration.items()
                if buckets
            )
        )


async def run_prediction_batch(settings_list: list[AutoTradeSettings], run_prediction: Callable[..., Awaitable[None]]) -> None:
    write_lock = asyncio.Lock()
    results = await asyncio.gather(*(run_prediction(settings, write_lock=write_lock) for settings in settings_list), return_exceptions=True)
    failures = prediction_failures(settings_list, results)
    if failures:
        failed_keys = ", ".join(settings.strategy_key for settings, _exc in failures)
        raise RuntimeError(f"auto prediction failed for strategies: {failed_keys}") from failures[0][1]


async def run_candidate_collection_batch(settings_list: list[AutoTradeSettings], save_collection: Callable[..., Awaitable[None]], logger: logging.Logger) -> None:
    write_lock = asyncio.Lock()
    results = await asyncio.gather(*(save_collection(settings, write_lock=write_lock) for settings in settings_list), return_exceptions=True)
    failures = prediction_failures(settings_list, results)
    for settings, exc in failures:
        logger.error(
            "candidate collection failed strategy=%s symbol=%s duration=%s",
            settings.strategy_key,
            settings.symbol,
            settings.duration,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    if failures:
        failed = ", ".join(f"{item.symbol}:{item.duration}" for item, _exc in failures)
        raise RuntimeError(f"candidate collection failed for: {failed}") from failures[0][1]


def prediction_failures(settings_list: list[AutoTradeSettings], results: list[object]) -> list[tuple[AutoTradeSettings, Exception]]:
    return [(settings, result) for settings, result in zip(settings_list, results) if isinstance(result, Exception)]


async def broadcast(result: dict, subscribers: dict[tuple[str, str, str], set], default_strategy_key: str) -> None:
    key = (result["symbol"].upper(), result["duration"], result.get("strategyKey") or default_strategy_key)
    websockets = subscribers.get(key, set())
    dead = set()
    for ws in websockets:
        try:
            await ws.send_json(result)
        except Exception:
            dead.add(ws)
    if dead:
        websockets -= dead
