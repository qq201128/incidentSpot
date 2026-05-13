from __future__ import annotations

from typing import Any

from app.services.factor_backtest_batch_service import BACKTEST_DURATION_ORDER
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_candidates import materialize_mined_factor_frame
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
    frame = load_factor_frame(symbol)
    signals = []
    missing = []
    failures = []
    for duration in BACKTEST_DURATION_ORDER:
        if duration not in SUPPORTED_RULE_DURATIONS:
            continue
        cached = get_cached_combination_ranking(symbol, duration)
        if cached is None:
            missing.append(duration)
            continue
        mined = materialize_mined_factor_frame(frame, symbol=symbol, duration=duration)
        failures.extend(mined.failures)
        for rank, row in enumerate(_top_ranking_rows(cached, top_per_duration), start=1):
            try:
                signal = build_live_signal_from_ranking(mined.frame, row, symbol=symbol, duration=duration)
                signals.append({**signal, "comboRank": rank})
            except Exception as exc:
                failures.append(_signal_failure(row, duration, exc))
    return {
        "symbol": symbol.upper(),
        "signals": signals[:limit],
        "total": min(len(signals), limit),
        "missingDurations": missing,
        "topPerDuration": top_per_duration,
        "signalFailures": failures,
    }


def _top_ranking_rows(cached: dict[str, Any], top_per_duration: int) -> list[dict[str, Any]]:
    ranking = cached.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return []
    return [dict(row) for row in ranking[:top_per_duration]]


def _signal_failure(row: dict[str, Any], duration: str, exc: Exception) -> dict[str, Any]:
    return {
        "duration": duration,
        "factorName": row.get("factorName"),
        "stage": "build_live_signal",
        "error": str(exc),
    }
