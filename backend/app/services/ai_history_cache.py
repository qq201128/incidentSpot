from __future__ import annotations

import threading
import time
from typing import Any, Callable

PAGE_CACHE_TTL_SECONDS = 8.0
META_CACHE_TTL_SECONDS = 30.0

_lock = threading.Lock()
_page_cache: dict[tuple, tuple[float, dict[str, Any]]] = {}
_meta_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def get_cached_ai_history(
    symbol: str,
    duration_minutes: int,
    page: int,
    page_size: int,
    *,
    build: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    symbol = symbol.strip().upper()
    key = (symbol, int(duration_minutes), int(page), int(page_size))
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

    for symbol in symbols:
        safe = symbol.strip().upper()
        try:
            query_ai_history_meta(conn, safe)
            for minutes in (10, 30, 60, 1440):
                query_ai_history_success(conn, safe, duration_minutes=minutes, page=1, page_size=10)
        except Exception:
            continue


def clear_ai_history_cache() -> None:
    with _lock:
        _page_cache.clear()
        _meta_cache.clear()
