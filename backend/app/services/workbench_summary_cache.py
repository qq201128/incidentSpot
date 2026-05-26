from __future__ import annotations

import threading
import time
from typing import Any, Callable

# Background governance refreshes about every 180s; avoid rebuilding on every poll.
SUMMARY_CACHE_TTL_SECONDS = 30.0
SUMMARY_STALE_SERVE_SECONDS = 300.0

_lock = threading.Lock()
_refresh_lock = threading.Lock()
_summary_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_refresh_in_flight: set[tuple[str, str]] = set()


def get_cached_workbench_summary(
    symbol: str,
    duration: str,
    *,
    build: Callable[[str, str], dict[str, Any]],
    max_age_seconds: float | None = None,
) -> dict[str, Any]:
    fresh_ttl = SUMMARY_CACHE_TTL_SECONDS if max_age_seconds is None else max_age_seconds
    stale_ttl = SUMMARY_STALE_SERVE_SECONDS
    key = (symbol.strip().upper(), duration)
    now = time.monotonic()
    with _lock:
        entry = _summary_cache.get(key)
        if entry is not None:
            age = now - entry[0]
            if age <= fresh_ttl:
                return _with_cache_meta(entry[1], cached=True, stale=False, age_seconds=age)
            if age <= stale_ttl:
                _schedule_background_refresh(key, build)
                return _with_cache_meta(entry[1], cached=True, stale=True, age_seconds=age)

    payload = build(key[0], key[1])
    with _lock:
        _summary_cache[key] = (time.monotonic(), payload)
    return _with_cache_meta(payload, cached=False, stale=False, age_seconds=0.0)


def store_workbench_summary(symbol: str, duration: str, payload: dict[str, Any]) -> None:
    key = (symbol.strip().upper(), duration)
    with _lock:
        _summary_cache[key] = (time.monotonic(), payload)


def clear_workbench_summary_cache() -> None:
    with _lock:
        _summary_cache.clear()
    with _refresh_lock:
        _refresh_in_flight.clear()


def _schedule_background_refresh(
    key: tuple[str, str],
    build: Callable[[str, str], dict[str, Any]],
) -> None:
    with _refresh_lock:
        if key in _refresh_in_flight:
            return
        _refresh_in_flight.add(key)

    def _run() -> None:
        try:
            payload = build(key[0], key[1])
            store_workbench_summary(key[0], key[1], payload)
        finally:
            with _refresh_lock:
                _refresh_in_flight.discard(key)

    threading.Thread(target=_run, name=f"workbench-summary-refresh-{key[0]}-{key[1]}", daemon=True).start()


def _with_cache_meta(
    payload: dict[str, Any],
    *,
    cached: bool,
    stale: bool,
    age_seconds: float,
) -> dict[str, Any]:
    return {
        **payload,
        "cache": {
            "cached": cached,
            "stale": stale,
            "ageSeconds": round(max(0.0, age_seconds), 2),
        },
    }
