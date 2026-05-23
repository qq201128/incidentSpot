from __future__ import annotations

from typing import Any

from app.services.factor_cache_metadata import assert_cache_usable_for_live_signal
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combo_strategy import predict_factor_combo_row_direction
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


def eligible_factor_combo_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    sym = symbol.strip().upper()
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    rows = []
    for cache in _usable_caches(sym, duration):
        rows.extend(_eligible_rows_for_cache(sym, duration, cache))
    return rows


def predict_eligible_factor_combo_rows(
    symbol: str,
    duration: str,
    *,
    entry_open_time: int,
    entry_grace_ms: int,
) -> list[dict[str, Any]]:
    return [
        predict_factor_combo_row_direction(
            symbol,
            duration,
            row,
            entry_open_time=entry_open_time,
            entry_grace_ms=entry_grace_ms,
        )
        for row in eligible_factor_combo_rows(symbol, duration)
        if int(row.get("comboRank") or 0) > 1
    ]


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


def _eligible_rows_for_cache(symbol: str, duration: str, cache: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = cache.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return []
    frame = materialize_mined_factor_frame(
        load_factor_frame(symbol, duration),
        symbol=symbol,
        duration=duration,
    ).frame
    rows = []
    for rank, row in enumerate(ranking, start=1):
        ranked = {**dict(row), "comboRank": rank}
        signal = build_live_signal_from_ranking(
            frame,
            ranked,
            symbol=symbol,
            duration=duration,
            apply_quality_gate=False,
        )
        if signal.get("qualityPassed"):
            rows.append(ranked)
    return rows
