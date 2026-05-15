from __future__ import annotations

from typing import Any

from app.services.factor_backtest_batch_service import BACKTEST_DURATION_ORDER
from app.services.factor_cache_metadata import (
    cache_is_usable_for_live_signal,
    live_signal_cache_reason,
)
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_signal_cache_service import (
    get_cached_combination_signals,
    save_cached_combination_signals,
)
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.factor_combo_simulation_keys import factor_combo_simulation_strategy_key
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


DEFAULT_TOP_PER_DURATION = 3


def build_combination_signal_watchlist(
    symbol: str,
    limit: int = 12,
    *,
    top_per_duration: int = DEFAULT_TOP_PER_DURATION,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if top_per_duration <= 0:
        raise ValueError("top_per_duration must be > 0")
    cached = get_cached_combination_signals(symbol)
    if _signal_cache_matches(cached, symbol, limit, top_per_duration):
        return cached
    payload = rebuild_combination_signal_watchlist(
        symbol,
        limit=limit,
        top_per_duration=top_per_duration,
    )
    save_cached_combination_signals(payload)
    return payload


def rebuild_combination_signal_watchlist(
    symbol: str,
    limit: int = 12,
    *,
    top_per_duration: int = DEFAULT_TOP_PER_DURATION,
) -> dict[str, Any]:
    frame_by_duration: dict[str, Any] = {}
    signals = []
    missing = []
    failures = []
    cache_issues = []
    for duration in BACKTEST_DURATION_ORDER:
        if duration not in SUPPORTED_RULE_DURATIONS:
            continue
        cached = get_cached_combination_ranking(symbol, duration)
        if cached is None:
            missing.append(duration)
            continue
        if not cache_is_usable_for_live_signal(cached):
            cache_issues.append(_cache_issue(duration, cached))
            continue
        frame = frame_by_duration.setdefault(duration, load_factor_frame(symbol, duration))
        mined = materialize_mined_factor_frame(frame, symbol=symbol, duration=duration)
        failures.extend(mined.failures)
        for rank, row in enumerate(_top_ranking_rows(cached, top_per_duration), start=1):
            try:
                ranked_row = {**row, "comboRank": rank}
                signal = build_live_signal_from_ranking(
                    mined.frame,
                    ranked_row,
                    symbol=symbol,
                    duration=duration,
                )
                signals.append(_simulation_signal(signal, rank))
            except Exception as exc:
                failures.append(_signal_failure(row, duration, exc))
    return {
        "symbol": symbol.upper(),
        "signals": signals[:limit],
        "total": min(len(signals), limit),
        "limit": limit,
        "missingDurations": missing,
        "topPerDuration": top_per_duration,
        "signalFailures": failures,
        "cacheIssues": cache_issues,
        "durationCacheReasons": _duration_cache_reasons(symbol),
    }


def _top_ranking_rows(cached: dict[str, Any], top_per_duration: int) -> list[dict[str, Any]]:
    ranking = cached.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return []
    return [dict(row) for row in ranking[:top_per_duration]]


def _simulation_signal(signal: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        **signal,
        "comboRank": rank,
        "simulationMode": "paper_live",
        "simulationStrategyKey": factor_combo_simulation_strategy_key(rank),
    }


def _signal_failure(row: dict[str, Any], duration: str, exc: Exception) -> dict[str, Any]:
    return {
        "duration": duration,
        "factorName": row.get("factorName"),
        "stage": "build_live_signal",
        "error": str(exc),
    }


def _cache_issue(duration: str, cached: dict[str, Any]) -> dict[str, Any]:
    return {
        "duration": duration,
        "stage": "load_cached_combination_ranking",
        "error": "stale_combination_ranking_cache",
        "cacheStatus": cached.get("cacheStatus"),
    }


def _signal_cache_matches(
    payload: dict[str, Any] | None,
    symbol: str,
    limit: int,
    top_per_duration: int,
) -> bool:
    if payload is None:
        return False
    if str(payload.get("symbol") or "").upper() != symbol.strip().upper():
        return False
    if int(payload.get("topPerDuration") or 0) != int(top_per_duration):
        return False
    if int(payload.get("limit") or 0) != int(limit):
        return False
    cached_reasons = payload.get("durationCacheReasons")
    return cached_reasons == _duration_cache_reasons(symbol)


def _duration_cache_reasons(symbol: str) -> dict[str, str]:
    reasons = {}
    for duration in BACKTEST_DURATION_ORDER:
        if duration not in SUPPORTED_RULE_DURATIONS:
            continue
        cached = get_cached_combination_ranking(symbol, duration)
        reasons[duration] = _duration_cache_stamp(cached)
    return reasons


def _duration_cache_stamp(cached: dict[str, Any] | None) -> str:
    if cached is None:
        return "stale::"
    status = cached.get("cacheStatus")
    current = status.get("currentMarketData") if isinstance(status, dict) else {}
    row_count = current.get("rowCount") if isinstance(current, dict) else ""
    max_open_time = current.get("maxOpenTime") if isinstance(current, dict) else ""
    updated_at = str(cached.get("updatedAt") or "")
    return f"{live_signal_cache_reason(cached)}:{updated_at}:{row_count}:{max_open_time}"
