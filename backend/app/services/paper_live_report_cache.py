from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

DEFAULT_TTL_SECONDS = 30.0

_lock = threading.Lock()
_refresh_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_refreshing: set[tuple[str, str]] = set()


def cache_ttl_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("PAPER_LIVE_REPORT_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def get_cached_paper_live_report(
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
        if entry is not None:
            age = now - entry[0]
            if age <= ttl:
                return _with_cache_meta(entry[1], hit=True, stale=False, warming=False, age_seconds=age)
            _schedule_refresh(key, build)
            return _with_cache_meta(entry[1], hit=True, stale=True, warming=True, age_seconds=age)

    payload = build(key[0], key[1])
    with _lock:
        _cache[key] = (time.monotonic(), payload)
    return _with_cache_meta(payload, hit=False, stale=False, warming=False, age_seconds=0.0)


def clear_paper_live_report_cache() -> None:
    with _lock:
        _cache.clear()
    with _refresh_lock:
        _refreshing.clear()


def store_paper_live_report_cache(symbol: str, duration: str, payload: dict[str, Any]) -> None:
    key = (symbol.strip().upper(), duration)
    stored = dict(payload)
    stored.pop("cache", None)
    with _lock:
        _cache[key] = (time.monotonic(), stored)


def _schedule_refresh(
    key: tuple[str, str],
    build: Callable[[str, str], dict[str, Any]],
) -> None:
    with _refresh_lock:
        if key in _refreshing:
            return
        _refreshing.add(key)
    thread = threading.Thread(target=_refresh, args=(key, build), daemon=True)
    thread.start()


def _refresh(
    key: tuple[str, str],
    build: Callable[[str, str], dict[str, Any]],
) -> None:
    try:
        payload = build(key[0], key[1])
        with _lock:
            _cache[key] = (time.monotonic(), payload)
    finally:
        with _refresh_lock:
            _refreshing.discard(key)


def _with_cache_meta(
    payload: dict[str, Any],
    *,
    hit: bool,
    stale: bool,
    warming: bool,
    age_seconds: float,
) -> dict[str, Any]:
    return {
        **payload,
        "cache": {
            "hit": hit,
            "stale": stale,
            "warming": warming,
            "ageSeconds": round(max(0.0, age_seconds), 2),
        },
    }
