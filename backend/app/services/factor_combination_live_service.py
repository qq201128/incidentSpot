from __future__ import annotations

from typing import Any

from app.services.factor_backtest_batch_service import BACKTEST_DURATION_ORDER
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_frame_service import load_factor_frame
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


def build_combination_signal_watchlist(symbol: str, limit: int = 4) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    frame = load_factor_frame(symbol)
    signals = []
    missing = []
    for duration in BACKTEST_DURATION_ORDER:
        if duration not in SUPPORTED_RULE_DURATIONS:
            continue
        cached = get_cached_combination_ranking(symbol, duration)
        if cached is None:
            missing.append(duration)
            continue
        top_row = _top_ranking_row(cached)
        if top_row is not None:
            signals.append(build_live_signal_from_ranking(frame, top_row, symbol=symbol, duration=duration))
    return {
        "symbol": symbol.upper(),
        "signals": signals[:limit],
        "total": min(len(signals), limit),
        "missingDurations": missing,
    }


def _top_ranking_row(cached: dict[str, Any]) -> dict[str, Any] | None:
    ranking = cached.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return None
    return dict(ranking[0])
