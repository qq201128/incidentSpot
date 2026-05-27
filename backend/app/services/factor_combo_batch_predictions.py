from __future__ import annotations

from typing import Any

from app.services.factor_backtest_gate import meets_backtest_gate
from app.services.factor_cache_metadata import assert_cache_usable_for_live_signal
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combo_strategy import predict_factor_combo_row_direction
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.paper_live_candidate_service import OBSERVATION_POOL_LIMIT
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


def eligible_factor_combo_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    sym = symbol.strip().upper()
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    return offline_candidate_screening_report(sym, duration)["focusedCandidates"]


def backtest_qualified_factor_combo_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    """All ranked combos from live caches that pass the shared backtest gate (simulation order pool)."""
    sym = symbol.strip().upper()
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cache in _usable_caches(sym, duration):
        ranking = cache.get("ranking")
        if not isinstance(ranking, list):
            continue
        for rank, row in enumerate(ranking, start=1):
            ranked = {**dict(row), "comboRank": rank}
            factor_name = str(ranked.get("factorName") or "").strip()
            if not factor_name or factor_name in seen:
                continue
            if not meets_backtest_gate(_row_for_backtest_gate(ranked)):
                continue
            seen.add(factor_name)
            rows.append(ranked)
    return rows


def offline_candidate_screening_report(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    rows, rejected = _screen_offline_candidates(sym, duration)
    rows.sort(key=_offline_candidate_rank_key, reverse=True)
    overflow = [_rejected_row(row, "outside_observation_pool_limit") for row in rows[OBSERVATION_POOL_LIMIT:]]
    rejected.extend(overflow)
    focused = rows[:OBSERVATION_POOL_LIMIT]
    return {
        "policy": "offline_oos_walk_forward_recent_rolling_prefilter_only",
        "observationPoolLimit": OBSERVATION_POOL_LIMIT,
        "focusedCount": len(focused),
        "candidateCount": len(rows),
        "rejectedCount": len(rejected),
        "focusedCandidates": focused,
        "rejectedReasons": rejected[:50],
        "reasonCounts": _reason_counts(rejected),
    }


def predict_eligible_factor_combo_rows(
    symbol: str,
    duration: str,
    *,
    entry_open_time: int,
    entry_grace_ms: int,
) -> list[dict[str, Any]]:
    return [
        _prediction_for_backtest_simulation(
            symbol,
            duration,
            row,
            entry_open_time=entry_open_time,
            entry_grace_ms=entry_grace_ms,
        )
        for row in eligible_factor_combo_rows(symbol, duration)
    ]


def _prediction_for_backtest_simulation(
    symbol: str,
    duration: str,
    row: dict[str, Any],
    *,
    entry_open_time: int,
    entry_grace_ms: int,
) -> dict[str, Any]:
    prediction = predict_factor_combo_row_direction(
        symbol,
        duration,
        row,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
    )
    if meets_backtest_gate(_row_for_backtest_gate(row)):
        prediction["trade_quality_passed"] = True
    return prediction


def _row_for_backtest_gate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "winRate": row.get("winRate") if row.get("winRate") is not None else row.get("backtestWinRate"),
        "totalPeriods": row.get("totalPeriods") or row.get("trades"),
    }


def _usable_caches(symbol: str, duration: str) -> list[dict[str, Any]]:
    caches = []
    for cache in (
        get_cached_combination_ranking(symbol, duration),
        get_cached_high_winrate_combo_ranking(symbol, duration),
    ):
        if cache is None:
            continue
        assert_cache_usable_for_live_signal(cache, f"factor combination ranking {symbol} {duration}")
        caches.append(cache)
    return caches


def _screen_offline_candidates(symbol: str, duration: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    caches = _usable_caches(symbol, duration)
    if not caches:
        rejected.append({"symbol": symbol, "duration": duration, "reason": "no_usable_offline_candidate_cache"})
        return rows, rejected
    frame = materialize_mined_factor_frame(load_factor_frame(symbol, duration), symbol=symbol, duration=duration).frame
    for cache in caches:
        accepted, cache_rejected = _screen_cache_rows(frame, symbol, duration, cache)
        rows.extend(accepted)
        rejected.extend(cache_rejected)
    return rows, rejected


def _screen_cache_rows(
    frame: Any,
    symbol: str,
    duration: str,
    cache: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranking = cache.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return [], [{"symbol": symbol, "duration": duration, "reason": "offline_ranking_empty"}]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rank, row in enumerate(ranking, start=1):
        ranked = {**dict(row), "comboRank": rank}
        signal = build_live_signal_from_ranking(frame, ranked, symbol=symbol, duration=duration, apply_quality_gate=False)
        if signal.get("qualityPassed"):
            accepted.append(ranked)
        else:
            rejected.append(_rejected_row(ranked, str(signal.get("qualityGateReason") or "quality_gate_failed")))
    return accepted, rejected


def _offline_candidate_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    walk_forward = row.get("walkForward") if isinstance(row.get("walkForward"), dict) else {}
    return (
        _num(walk_forward.get("stabilityScore")),
        1.0 if row.get("walkForwardPassed") else 0.0,
        _num(walk_forward.get("oosWinRate")),
        _num(row.get("factorScore")),
        _num(row.get("profitFactor")),
    )


def _num(value: Any) -> float:
    return float(value) if value is not None else float("-inf")


def _rejected_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "factorName": row.get("factorName"),
        "comboRank": row.get("comboRank"),
        "reason": reason,
        "backtestWinRate": row.get("backtestWinRate") or row.get("winRate"),
        "oosWinRate": _walk_forward(row).get("oosWinRate"),
        "walkForwardResult": row.get("walkForward"),
        "recentRollingResult": row.get("recentRollingResult"),
        "paperLiveStatus": "rejected_offline_prefilter",
    }


def _walk_forward(row: dict[str, Any]) -> dict[str, Any]:
    walk_forward = row.get("walkForward")
    return walk_forward if isinstance(walk_forward, dict) else {}


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts
