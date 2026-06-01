from __future__ import annotations

import sqlite3
from typing import Any

from app.services.factor_backtest_gate import backtest_gate_thresholds
from app.services.factor_cache_metadata import cache_is_usable_for_live_signal
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_learning_common import finite
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking


def simulation_candidate_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = [
        *_single_factor_candidates(symbol, duration),
        *_combo_factor_candidates(symbol, duration),
    ]
    return _dedupe_candidates(rows)


def _single_factor_candidates(symbol: str, duration: str) -> list[dict[str, Any]]:
    cache, cache_error = _cache_payload(lambda: get_cached_ranking(symbol, duration))
    rows = _single_cache_rows(cache, cache_error, symbol, duration)
    rows.extend(_agent_factor_rows(symbol, duration))
    return rows or [_cache_missing_row(symbol, duration, "single_factor", "factor_ranking_cache")]


def _single_cache_rows(
    cache: dict[str, Any] | None,
    cache_error: str | None,
    symbol: str,
    duration: str,
) -> list[dict[str, Any]]:
    if cache_error:
        return [_cache_missing_row(symbol, duration, "single_factor", "factor_ranking_cache", cache_error)]
    if cache is None:
        return []
    if not cache_is_usable_for_live_signal(cache):
        return [_cache_unusable_row(symbol, duration, "single_factor", "factor_ranking_cache", cache)]
    return [
        _candidate_payload(row, "single_factor", "factor_ranking_cache", symbol, duration)
        for row in cache.get("ranking") or []
        if isinstance(row, dict)
    ]


def _agent_factor_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    from app.services.agent_mined_factor_library import agent_factor_rows_for_duration

    return [
        _candidate_payload(_agent_metrics_row(row), "single_factor", "agent_mined_factor_library", symbol, duration)
        for row in agent_factor_rows_for_duration(symbol, duration)
    ]


def _combo_factor_candidates(symbol: str, duration: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, loader in _combo_cache_loaders():
        cache, cache_error = _cache_payload(lambda loader=loader: loader(symbol, duration))
        rows.extend(_combo_cache_rows(cache, cache_error, source, symbol, duration))
    return rows or [_cache_missing_row(symbol, duration, "factor_combo", "factor_combo_ranking_cache")]


def _combo_cache_rows(
    cache: dict[str, Any] | None,
    cache_error: str | None,
    source: str,
    symbol: str,
    duration: str,
) -> list[dict[str, Any]]:
    if cache_error:
        return [_cache_missing_row(symbol, duration, "factor_combo", source, cache_error)]
    if cache is None:
        return []
    if not cache_is_usable_for_live_signal(cache):
        return [_cache_unusable_row(symbol, duration, "factor_combo", source, cache)]
    ranking = cache.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return [_cache_missing_row(symbol, duration, "factor_combo", source, "offline_ranking_empty")]
    return [_candidate_payload(dict(row), "factor_combo", source, symbol, duration) for row in ranking if isinstance(row, dict)]


def _candidate_payload(
    row: dict[str, Any],
    candidate_type: str,
    source: str,
    symbol: str,
    duration: str,
) -> dict[str, Any]:
    metrics = _metrics(row)
    reason = _gate_rejection_reason(metrics)
    factor_name = str(row.get("factorName") or "").strip()
    return {
        "strategyKey": _strategy_key(candidate_type, factor_name),
        "candidateType": candidate_type,
        "factorName": factor_name,
        "symbol": symbol,
        "duration": duration,
        "source": source,
        "gatePassed": reason is None,
        "gateStatus": "not_enabled" if reason is None else "rejected",
        "rejectionReason": reason,
        "metrics": metrics,
    }


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "winRate": row.get("winRate") if row.get("winRate") is not None else row.get("backtestWinRate"),
        "profitFactor": row.get("profitFactor"),
        "totalPeriods": row.get("totalPeriods") or row.get("trades"),
    }


def _gate_rejection_reason(metrics: dict[str, Any]) -> str | None:
    thresholds = backtest_gate_thresholds()
    win_rate = finite(metrics.get("winRate"))
    profit_factor = finite(metrics.get("profitFactor"))
    total_periods = int(metrics.get("totalPeriods") or 0)
    if win_rate is None:
        return "win_rate_missing"
    if win_rate < float(thresholds["minWinRate"]):
        return "win_rate_below_min"
    if profit_factor is None:
        return "profit_factor_missing"
    if profit_factor < float(thresholds["minProfitFactor"]):
        return "profit_factor_below_min"
    if total_periods < int(thresholds["minTotalPeriods"]):
        return "sample_count_below_min"
    return None


def _cache_payload(loader: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return loader(), None
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return None, "cache_table_missing"
        raise


def _cache_missing_row(
    symbol: str,
    duration: str,
    candidate_type: str,
    source: str,
    reason: str = "cache_unavailable",
) -> dict[str, Any]:
    return _rejected_system_row(symbol, duration, candidate_type, source, reason)


def _cache_unusable_row(symbol: str, duration: str, candidate_type: str, source: str, cache: dict[str, Any]) -> dict[str, Any]:
    reason = (cache.get("cacheStatus") or {}).get("reason") or "cache_unavailable"
    return _rejected_system_row(symbol, duration, candidate_type, source, f"cache_unavailable:{reason}")


def _rejected_system_row(symbol: str, duration: str, candidate_type: str, source: str, reason: str) -> dict[str, Any]:
    return {
        "strategyKey": None,
        "candidateType": candidate_type,
        "factorName": None,
        "symbol": symbol,
        "duration": duration,
        "source": source,
        "gatePassed": False,
        "gateStatus": "rejected",
        "rejectionReason": reason,
        "metrics": {},
        "slot": None,
        "latestEvent": None,
        "latestFailure": None,
    }


def _agent_metrics_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return {**row, "winRate": metrics.get("winRate"), "profitFactor": metrics.get("profitFactor"), "totalPeriods": metrics.get("totalPeriods")}


def _strategy_key(candidate_type: str, factor_name: str) -> str | None:
    if not factor_name:
        return None
    if candidate_type == "single_factor":
        return factor_candidate_signal_key(factor_name)
    return simulation_strategy_key_for_factor_name(factor_name)


def _combo_cache_loaders() -> tuple[tuple[str, Any], ...]:
    return (
        ("factor_combo_ranking_cache", get_cached_combination_ranking),
        ("high_winrate_combo_ranking_cache", get_cached_high_winrate_combo_ranking),
    )


def _dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        fallback = f"{row['candidateType']}:{row['source']}:{row.get('rejectionReason')}"
        key = str(row.get("strategyKey") or fallback)
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result
