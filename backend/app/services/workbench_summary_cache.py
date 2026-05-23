from __future__ import annotations

import threading
import time
from typing import Any, Callable

SUMMARY_CACHE_TTL_SECONDS = 5.0

_lock = threading.Lock()
_summary_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def get_cached_workbench_summary(
    symbol: str,
    duration: str,
    *,
    build: Callable[[str, str], dict[str, Any]],
    max_age_seconds: float = SUMMARY_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    key = (symbol.strip().upper(), duration)
    now = time.monotonic()
    with _lock:
        entry = _summary_cache.get(key)
        if entry is not None and (now - entry[0]) <= max_age_seconds:
            return {**entry[1], "cache": {"cached": True, "ageSeconds": round(now - entry[0], 2)}}
    payload = build(key[0], key[1])
    with _lock:
        _summary_cache[key] = (time.monotonic(), payload)
    return {**payload, "cache": {"cached": False, "ageSeconds": 0.0}}


def store_workbench_summary(symbol: str, duration: str, payload: dict[str, Any]) -> None:
    key = (symbol.strip().upper(), duration)
    with _lock:
        _summary_cache[key] = (time.monotonic(), payload)


def clear_workbench_summary_cache() -> None:
    with _lock:
        _summary_cache.clear()
