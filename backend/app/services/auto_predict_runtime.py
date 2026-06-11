from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.services.auto_trade_types import AutoTradeSettings


@dataclass(frozen=True)
class PredictionFailure:
    settings: AutoTradeSettings
    exception: BaseException


class PredictionBatchError(RuntimeError):
    def __init__(self, failures: list[PredictionFailure]) -> None:
        self.failures = tuple(failures)
        self.details = [_failure_detail(failure) for failure in failures]
        failed_keys = ", ".join(failure.settings.strategy_key for failure in failures)
        super().__init__(f"auto prediction failed for strategies: {failed_keys}")


class CandidateCollectionBatchError(RuntimeError):
    def __init__(self, failures: list[PredictionFailure]) -> None:
        self.failures = tuple(failures)
        self.details = [_failure_detail(failure) for failure in failures]
        failed = ", ".join(f"{failure.settings.symbol}:{failure.settings.duration}" for failure in failures)
        super().__init__(f"candidate collection failed for: {failed}")


class BroadcastDeliveryError(RuntimeError):
    def __init__(self, failures: list[BaseException]) -> None:
        self.details = [_exception_detail(exc) for exc in failures]
        super().__init__(f"prediction broadcast failed for {len(failures)} subscriber(s)")


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
        raise PredictionBatchError(failures) from failures[0].exception


async def run_candidate_collection_batch(settings_list: list[AutoTradeSettings], save_collection: Callable[..., Awaitable[None]], logger: logging.Logger) -> None:
    write_lock = asyncio.Lock()
    failures = []
    for settings in settings_list:
        try:
            await save_collection(settings, write_lock=write_lock)
        except BaseException as exc:
            failures.append(PredictionFailure(settings, exc))
    for failure in failures:
        settings = failure.settings
        exc = failure.exception
        logger.error(
            "candidate collection failed strategy=%s symbol=%s duration=%s",
            settings.strategy_key,
            settings.symbol,
            settings.duration,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    if failures:
        raise CandidateCollectionBatchError(failures) from failures[0].exception


def prediction_failures(settings_list: list[AutoTradeSettings], results: list[object]) -> list[PredictionFailure]:
    return [
        PredictionFailure(settings, result)
        for settings, result in zip(settings_list, results)
        if isinstance(result, BaseException)
    ]


def _failure_detail(failure: PredictionFailure) -> dict[str, Any]:
    settings = failure.settings
    exc = failure.exception
    detail = {
        "strategyKey": settings.strategy_key,
        "symbol": settings.symbol,
        "duration": settings.duration,
        "error": str(exc),
        "exceptionType": type(exc).__name__,
    }
    nested = getattr(exc, "details", None)
    if nested is not None:
        detail["details"] = nested
    return detail


def _exception_detail(exc: BaseException) -> dict[str, str]:
    return {"error": str(exc), "exceptionType": type(exc).__name__}


async def broadcast(result: dict, subscribers: dict[tuple[str, str, str], set], default_strategy_key: str) -> None:
    key = (result["symbol"].upper(), result["duration"], result.get("strategyKey") or default_strategy_key)
    websockets = subscribers.get(key, set())
    dead = set()
    failures = []
    for ws in websockets:
        try:
            await ws.send_json(result)
        except Exception as exc:
            dead.add(ws)
            failures.append(exc)
    if dead:
        websockets -= dead
    if failures:
        raise BroadcastDeliveryError(failures) from failures[0]


async def sleep_for(stop_event: asyncio.Event, wait_seconds: float) -> bool:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
        return True
    except TimeoutError:
        return False
