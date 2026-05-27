from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

DEFAULT_TTL_SECONDS = 45.0

_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def cache_ttl_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("MINING_OVERVIEW_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def get_cached_mining_overview(
    symbol: str,
    duration: str,
    *,
    build: Callable[[str, str], dict[str, Any]],
) -> dict[str, Any]:
    key = (symbol.strip().upper(), duration)
    ttl = cache_ttl_seconds()
    now = time.monotonic()
    with _lock:
        entry = _cache.get(key)
        if entry is not None and now - entry[0] <= ttl:
            return {**entry[1], "cache": {"hit": True, "ageSeconds": round(now - entry[0], 2)}}

    payload = build(key[0], key[1])
    with _lock:
        _cache[key] = (time.monotonic(), payload)
    return {**payload, "cache": {"hit": False, "ageSeconds": 0.0}}


def clear_mining_overview_cache() -> None:
    with _lock:
        _cache.clear()
