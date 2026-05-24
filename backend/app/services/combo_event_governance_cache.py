from __future__ import annotations

import threading
import time
from typing import Any, Callable

_lock = threading.Lock()
_shadow_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_governance_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_monitoring_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def get_cached_shadow_report(
    symbol: str,
    duration: str,
    *,
    compute: Callable[[str, str], dict[str, Any]],
    max_age_seconds: float = 120.0,
    force_refresh: bool = False,
) -> dict[str, Any]:
    return _get_cached(
        _shadow_cache,
        symbol,
        duration,
        compute=compute,
        max_age_seconds=max_age_seconds,
        force_refresh=force_refresh,
    )


def get_cached_governance(
    symbol: str,
    duration: str,
    *,
    compute: Callable[[str, str], dict[str, Any]],
    max_age_seconds: float = 120.0,
    force_refresh: bool = False,
) -> dict[str, Any]:
    return _get_cached(
        _governance_cache,
        symbol,
        duration,
        compute=compute,
        max_age_seconds=max_age_seconds,
        force_refresh=force_refresh,
    )


def get_warm_monitoring_snapshot(symbol: str, duration: str) -> dict[str, Any]:
    key = (symbol.strip().upper(), duration)
    now = time.monotonic()
    with _lock:
        entry = _monitoring_cache.get(key)
        if entry is not None:
            return _with_cache_meta(entry[1], age_seconds=now - entry[0], cached=True)
    return _warming_monitoring_payload(key[0], key[1])


def _warming_monitoring_payload(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "warming": True,
        "shadowEventDeviation": {
            "symbol": symbol,
            "duration": duration,
            "summary": {"pairedCount": 0},
            "issues": [],
            "byStrategy": [],
            "pairs": [],
        },
        "highWinrateStatus": None,
        "batchComboDemotion": {
            "symbol": symbol,
            "duration": duration,
            "evaluations": [],
            "watchlist": [],
            "observeOnly": True,
        },
        "factorCandidateDemotion": {
            "symbol": symbol,
            "duration": duration,
            "evaluations": [],
            "watchlist": [],
            "observeOnly": True,
        },
        "simulationObservation": {
            "evaluatedCount": 0,
            "watchlistCount": 0,
            "batchComboWatchlistCount": 0,
            "factorCandidateWatchlistCount": 0,
        },
        "cache": {"cached": False, "warming": True, "ageSeconds": 0.0},
    }


def store_shadow_report(symbol: str, duration: str, payload: dict[str, Any]) -> None:
    _store(_shadow_cache, symbol, duration, payload)


def store_governance(symbol: str, duration: str, payload: dict[str, Any]) -> None:
    _store(_governance_cache, symbol, duration, payload)


def store_monitoring(symbol: str, duration: str, payload: dict[str, Any]) -> None:
    _store(_monitoring_cache, symbol, duration, payload)


def clear_combo_event_governance_cache() -> None:
    with _lock:
        _shadow_cache.clear()
        _governance_cache.clear()
        _monitoring_cache.clear()


def _get_cached(
    cache: dict[tuple[str, str], tuple[float, dict[str, Any]]],
    symbol: str,
    duration: str,
    *,
    compute: Callable[[str, str], dict[str, Any]],
    max_age_seconds: float,
    force_refresh: bool,
) -> dict[str, Any]:
    key = (symbol.strip().upper(), duration)
    now = time.monotonic()
    if not force_refresh:
        with _lock:
            entry = cache.get(key)
            if entry is not None and now - entry[0] <= max_age_seconds:
                return _with_cache_meta(entry[1], age_seconds=now - entry[0], cached=True)
    payload = compute(key[0], key[1])
    _store(cache, key[0], key[1], payload)
    return _with_cache_meta(payload, age_seconds=0.0, cached=False)


def _store(cache: dict[tuple[str, str], tuple[float, dict[str, Any]]], symbol: str, duration: str, payload: dict[str, Any]) -> None:
    key = (symbol.strip().upper(), duration)
    with _lock:
        cache[key] = (time.monotonic(), payload)


def _with_cache_meta(payload: dict[str, Any], *, age_seconds: float, cached: bool) -> dict[str, Any]:
    return {
        **payload,
        "cache": {
            "cached": cached,
            "ageSeconds": round(max(0.0, age_seconds), 2),
        },
    }
