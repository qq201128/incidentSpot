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
)
from app.services.factor_combination_signal_service import SignalBuildContext, build_live_signal_from_ranking
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_learning_memory_store import load_factor_learning_memory
from app.services.factor_mined_candidates import materialize_mined_factor_frame_for_rows
from app.services.factor_mined_library import mined_factor_rows_for_duration
from app.services.factor_combo_simulation_keys import factor_combo_simulation_strategy_key
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


DEFAULT_TOP_PER_DURATION = 3


def build_combination_signal_watchlist(
    symbol: str,
    limit: int = 12,
    *,
    top_per_duration: int = DEFAULT_TOP_PER_DURATION,
) -> dict[str, Any]:
    _validate_watchlist_request(limit, top_per_duration)
    cached = get_cached_combination_signals(symbol)
    if cached is None:
        return _empty_signal_watchlist(symbol, limit, top_per_duration)
    return _cached_signal_watchlist(cached, symbol, limit, top_per_duration)


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
        rows = _top_ranking_rows(cached, top_per_duration)
        source_rows = mined_factor_rows_for_duration(symbol, duration)
        frame = frame_by_duration.setdefault(duration, load_factor_frame(symbol, duration))
        mined = materialize_mined_factor_frame_for_rows(
            frame,
            symbol=symbol,
            duration=duration,
            target_rows=rows,
            source_rows=source_rows,
        )
        context = SignalBuildContext(
            load_factor_learning_memory(symbol, duration),
            {},
            _rows_by_name(source_rows),
        )
        failures.extend(mined.failures)
        for rank, row in enumerate(rows, start=1):
            try:
                ranked_row = {**row, "comboRank": rank}
                signal = build_live_signal_from_ranking(
                    mined.frame,
                    ranked_row,
                    symbol=symbol,
                    duration=duration,
                    context=context,
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


def _rows_by_name(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("factorName")): row for row in rows}


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


def _validate_watchlist_request(limit: int, top_per_duration: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if top_per_duration <= 0:
        raise ValueError("top_per_duration must be > 0")


def _empty_signal_watchlist(symbol: str, limit: int, top_per_duration: int) -> dict[str, Any]:
    return {
        "symbol": symbol.strip().upper(),
        "signals": [],
        "total": 0,
        "limit": limit,
        "missingDurations": [],
        "topPerDuration": top_per_duration,
        "signalFailures": [],
        "cacheIssues": [],
        "durationCacheReasons": {},
        "source": "none",
        "signalCacheStatus": {
            "usable": False,
            "reason": "signal_cache_missing",
            "message": "周期信号数据层缓存不存在；请通过组合刷新任务写入缓存。",
        },
    }


def _cached_signal_watchlist(
    cached: dict[str, Any],
    symbol: str,
    limit: int,
    top_per_duration: int,
) -> dict[str, Any]:
    status = _cached_signal_status(cached, symbol, limit, top_per_duration)
    signals = cached.get("signals") if isinstance(cached.get("signals"), list) else []
    return {
        **cached,
        "signals": signals[:limit],
        "total": min(len(signals), limit),
        "limit": limit,
        "topPerDuration": top_per_duration,
        "signalCacheStatus": status,
    }


def _cached_signal_status(
    cached: dict[str, Any],
    symbol: str,
    limit: int,
    top_per_duration: int,
) -> dict[str, Any]:
    config_matches = _signal_cache_config_matches(cached, symbol, limit, top_per_duration)
    data_matches = bool(config_matches and cached.get("durationCacheReasons") == _duration_cache_reasons(symbol))
    reason = "usable" if data_matches else _signal_cache_mismatch_reason(config_matches)
    return {
        "usable": data_matches,
        "reason": reason,
        "message": _signal_cache_message(reason),
    }


def _signal_cache_config_matches(
    payload: dict[str, Any],
    symbol: str,
    limit: int,
    top_per_duration: int,
) -> bool:
    if str(payload.get("symbol") or "").upper() != symbol.strip().upper():
        return False
    if int(payload.get("topPerDuration") or 0) != int(top_per_duration):
        return False
    return int(payload.get("limit") or 0) >= int(limit)


def _signal_cache_mismatch_reason(config_matches: bool) -> str:
    return "signal_cache_stale" if config_matches else "signal_cache_config_mismatch"


def _signal_cache_message(reason: str) -> str:
    if reason == "usable":
        return "周期信号来自数据层缓存。"
    if reason == "signal_cache_stale":
        return "周期信号来自数据层缓存，但底层组合/行情指纹已变化；请通过刷新任务重建缓存。"
    return "周期信号缓存参数与本次请求不一致；请通过刷新任务按当前参数重建缓存。"


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
