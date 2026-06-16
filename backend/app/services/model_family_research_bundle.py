from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time
from typing import Any

from app.services.lstm_combo_snapshot import current_combo_snapshot
from app.services.model_family_config import MODEL_FAMILIES
from app.services.model_family_status_service import model_family_research_status

DEFAULT_TTL_SECONDS = 30.0

_cache_lock = threading.Lock()
_refresh_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
_refreshing: set[tuple[str, str]] = set()


def model_family_research_bundle(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    key = (sym, duration)
    ttl = _cache_ttl_seconds()
    now = time.monotonic()
    with _cache_lock:
        entry = _cache.get(key)
        if entry is not None:
            age = now - entry[0]
            if age <= ttl:
                return _with_cache_meta(entry[1], hit=True, stale=False, warming=False, age_seconds=age)
            _schedule_refresh(key)
            return _with_cache_meta(entry[1], hit=True, stale=True, warming=True, age_seconds=age)

    payload = _build_model_family_research_bundle(sym, duration)
    with _cache_lock:
        _cache[key] = (time.monotonic(), payload)
    return _with_cache_meta(payload, hit=False, stale=False, warming=False, age_seconds=0.0)


def clear_model_family_research_bundle_cache() -> None:
    with _cache_lock:
        _cache.clear()
    with _refresh_lock:
        _refreshing.clear()


def _cache_ttl_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("MODEL_RESEARCH_BUNDLE_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def _build_model_family_research_bundle(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    shared_combo = current_combo_snapshot(sym, duration)

    def load(family: str) -> dict[str, Any]:
        try:
            status = model_family_research_status(
                family,
                sym,
                duration,
                current_combo_snapshot=shared_combo,
            )
            return _slim_research_row(status)
        except Exception as exc:
            return _failed_row(family, sym, duration, str(exc))

    workers = min(6, len(MODEL_FAMILIES))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        models = list(pool.map(load, MODEL_FAMILIES))
    return {"symbol": sym, "duration": duration, "models": models}


def _schedule_refresh(key: tuple[str, str]) -> None:
    with _refresh_lock:
        if key in _refreshing:
            return
        _refreshing.add(key)
    thread = threading.Thread(target=_refresh, args=(key,), daemon=True)
    thread.start()


def _refresh(key: tuple[str, str]) -> None:
    try:
        payload = _build_model_family_research_bundle(key[0], key[1])
        with _cache_lock:
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


def _slim_research_row(status: dict[str, Any]) -> dict[str, Any]:
    admission = status.get("paperLiveAdmission") or {}
    sample_counts = status.get("sampleCounts") or {}
    return {
        "modelFamily": status.get("modelFamily"),
        "strategyKey": status.get("strategyKey"),
        "modelVersion": status.get("modelVersion"),
        "featureWindow": status.get("featureWindow"),
        "validationWinRate": admission.get("validationWinRate") or status.get("validationWinRate"),
        "validationSampleCount": sample_counts.get("validation"),
        "testWinRate": status.get("testWinRate"),
        "paperLiveAdmission": admission,
        "paperLiveStatus": status.get("paperLiveStatus") or admission.get("status"),
        "cleanEventFeatures": status.get("cleanEventFeatures"),
        "regimeValidation": status.get("regimeValidation"),
        "shadowPredictionBlockedReason": status.get("shadowPredictionBlockedReason"),
        "validationFailureReason": status.get("validationFailureReason"),
    }


def _failed_row(family: str, symbol: str, duration: str, reason: str) -> dict[str, Any]:
    return {
        "modelFamily": family,
        "strategyKey": None,
        "modelVersion": None,
        "paperLiveStatus": "model_status_failed",
        "shadowPredictionBlockedReason": reason,
        "validationFailureReason": reason,
    }
