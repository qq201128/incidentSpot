from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

PAGE_CACHE_TTL_SECONDS = 8.0
META_CACHE_TTL_SECONDS = 30.0

_lock = threading.Lock()
_page_cache: dict[tuple, tuple[float, dict[str, Any]]] = {}
_meta_cache: dict[str, tuple[float, dict[str, Any]]] = {}


@dataclass(frozen=True)
class AiHistoryWarmFailure:
    symbol: str
    error: str
    exception_type: str


@dataclass(frozen=True)
class AiHistoryCacheKey:
    symbol: str
    duration_minutes: int
    page: int
    page_size: int

    def normalized(self) -> tuple[str, int, int, int]:
        return (self.symbol.strip().upper(), int(self.duration_minutes), int(self.page), int(self.page_size))


class AiHistoryWarmupError(RuntimeError):
    def __init__(self, failures: list[AiHistoryWarmFailure]) -> None:
        self.failures = failures
        self.details = {"failures": [_failure_payload(item) for item in failures]}
        symbols = ", ".join(item.symbol for item in failures)
        super().__init__(f"AI history cache warm-up failed for: {symbols}")


def get_cached_ai_history(
    cache_key: AiHistoryCacheKey,
    *,
    build: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    key = cache_key.normalized()
    now = time.monotonic()
    with _lock:
        entry = _page_cache.get(key)
        if entry is not None and now - entry[0] <= PAGE_CACHE_TTL_SECONDS:
            return entry[1]

    payload = build()
    with _lock:
        _page_cache[key] = (time.monotonic(), payload)
    return payload


def get_cached_ai_history_meta(symbol: str, *, build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    symbol = symbol.strip().upper()
    now = time.monotonic()
    with _lock:
        entry = _meta_cache.get(symbol)
        if entry is not None and now - entry[0] <= META_CACHE_TTL_SECONDS:
            return entry[1]

    payload = build()
    with _lock:
        _meta_cache[symbol] = (time.monotonic(), payload)
    return payload


def warm_ai_history_cache(conn, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> None:
    from app.services.event_ai_history import query_ai_history_meta, query_ai_history_success

    failures: list[AiHistoryWarmFailure] = []
    for symbol in symbols:
        safe = symbol.strip().upper()
        try:
            query_ai_history_meta(conn, safe)
            for minutes in (10, 30, 60, 1440):
                query_ai_history_success(conn, safe, duration_minutes=minutes, page=1, page_size=10)
        except Exception as exc:
            failures.append(AiHistoryWarmFailure(safe, str(exc), type(exc).__name__))
    if failures:
        raise AiHistoryWarmupError(failures)


def clear_ai_history_cache() -> None:
    with _lock:
        _page_cache.clear()
        _meta_cache.clear()


def _failure_payload(failure: AiHistoryWarmFailure) -> dict[str, str]:
    return {"symbol": failure.symbol, "error": failure.error, "exceptionType": failure.exception_type}
