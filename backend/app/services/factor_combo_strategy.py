from __future__ import annotations

from typing import Any

from app.services.factor_cache_metadata import (
    assert_cache_usable_for_live_signal,
    live_signal_cache_reason,
)
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_duration_alignment import duration_entry_source_open_time
from app.services.factor_frame_service import load_factor_frame, lookback_days_for_bars
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.kline_timing import MS_PER_MINUTE, current_rule_entry_open_time_for_duration
from app.services.factor_combo_simulation_keys import is_high_winrate_combo_name
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
from app.services.factor_combo_frame_materialization import materialize_factor_combo_frame_for_row
from app.services.factor_combo_rule_reasons import factor_combo_rule_reasons
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.high_winrate_strategy_demotion import high_winrate_active_rank
from app.services.rule_config import RULE_DURATION, SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    FACTOR_COMBO_RULE_NAME,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_RULE_NAME,
)

LIVE_SIGNAL_LOOKBACK_BARS = 720


def predict_factor_combo_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
) -> dict[str, Any]:
    return predict_factor_combo_rank_direction(
        symbol,
        duration,
        combo_rank=1,
        result_strategy_key=FACTOR_COMBO_STRATEGY_KEY,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
    )


def predict_high_winrate_factor_combo_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
) -> dict[str, Any]:
    combo_rank = _available_high_winrate_rank(symbol, duration, high_winrate_active_rank(symbol, duration))
    return predict_factor_combo_rank_direction(
        symbol,
        duration,
        combo_rank=combo_rank,
        result_strategy_key=HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
        require_high_winrate_goal=True,
    )


def predict_factor_combo_rank_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    combo_rank: int,
    result_strategy_key: str,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
    require_high_winrate_goal: bool = False,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        supported = sorted(SUPPORTED_RULE_DURATIONS)
        raise ValueError(f"factor combo strategy supports only {supported}, got {duration}")
    cached = _ranking_cache(symbol, duration, require_high_winrate_goal)
    if cached is None:
        raise ValueError(f"no cached combination ranking for {symbol.upper()} {duration}")
    assert_cache_usable_for_live_signal(cached, f"factor combination ranking {symbol.upper()} {duration}")
    top = _ranked_combo(cached, combo_rank)
    if require_high_winrate_goal:
        _assert_high_winrate_combo(top, combo_rank, symbol=symbol, duration=duration)
    _refresh_factor_combo_source_klines(symbol, duration, entry_open_time)
    frame = materialize_factor_combo_frame_for_row(
        _load_live_factor_frame(symbol, duration),
        symbol=symbol,
        duration=duration,
        row=top,
    )
    signal = build_live_signal_from_ranking(
        frame,
        top,
        symbol=symbol,
        duration=duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
        apply_quality_gate=False,
    )
    return _prediction_payload(
        signal,
        entry_open_time,
        result_strategy_key,
        cache_reason=live_signal_cache_reason(cached),
    )


def predict_factor_combo_row_direction(
    symbol: str,
    duration: str,
    row: dict[str, Any],
    *,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        supported = sorted(SUPPORTED_RULE_DURATIONS)
        raise ValueError(f"factor combo strategy supports only {supported}, got {duration}")
    cached = _ranking_cache(symbol, duration, is_high_winrate_combo_name(str(row.get("factorName") or "")))
    if cached is None:
        raise ValueError(f"no cached combination ranking for {symbol.upper()} {duration}")
    assert_cache_usable_for_live_signal(cached, f"factor combination ranking {symbol.upper()} {duration}")
    _refresh_factor_combo_source_klines(symbol, duration, entry_open_time)
    frame = materialize_factor_combo_frame_for_row(
        _load_live_factor_frame(symbol, duration),
        symbol=symbol,
        duration=duration,
        row=row,
    )
    signal = build_live_signal_from_ranking(
        frame,
        row,
        symbol=symbol,
        duration=duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
        apply_quality_gate=False,
    )
    return _prediction_payload(
        signal,
        entry_open_time,
        simulation_strategy_key_for_factor_name(str(row["factorName"])),
        cache_reason=live_signal_cache_reason(cached),
    )


def _ranked_combo(cached: dict[str, Any], combo_rank: int) -> dict[str, Any]:
    ranking = cached.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("cached combination ranking is empty")
    if combo_rank <= 0 or combo_rank > len(ranking):
        raise ValueError(f"cached combination ranking has no Top{combo_rank}")
    return {**dict(ranking[combo_rank - 1]), "comboRank": combo_rank}


def _ranking_cache(symbol: str, duration: str, high_winrate_goal: bool) -> dict[str, Any] | None:
    if high_winrate_goal:
        return get_cached_high_winrate_combo_ranking(symbol, duration)
    return get_cached_combination_ranking(symbol, duration)


def _available_high_winrate_rank(symbol: str, duration: str, preferred_rank: int) -> int:
    cached = get_cached_high_winrate_combo_ranking(symbol, duration)
    ranking = None if cached is None else cached.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return preferred_rank
    if 0 < preferred_rank <= len(ranking):
        return preferred_rank
    return 1


def _load_live_factor_frame(symbol: str, duration: str):
    return load_factor_frame(
        symbol,
        duration,
        lookback_days=lookback_days_for_bars(duration, LIVE_SIGNAL_LOOKBACK_BARS),
    )


def _assert_high_winrate_combo(row: dict[str, Any], rank: int, *, symbol: str, duration: str) -> None:
    factor_name = str(row.get("factorName") or "")
    if is_high_winrate_combo_name(factor_name):
        return
    raise ValueError(
        f"Top{rank} for {symbol.upper()} {duration} is not a high-winrate goal combo: {factor_name}"
    )


def _prediction_payload(
    signal: dict[str, Any],
    entry_open_time: int | None,
    strategy_key: str,
    *,
    cache_reason: str,
) -> dict[str, Any]:
    open_time = int(entry_open_time if entry_open_time is not None else signal["sourceOpenTime"])
    gate_name = _gate_name(strategy_key)
    return {
        "symbol": signal["symbol"],
        "signal_key": strategy_key,
        "strategy_key": strategy_key,
        "duration": signal["duration"],
        "open_time": open_time,
        "entry_price": float(signal["entryPrice"]),
        "direction": signal["direction"],
        "probability_up": signal["probabilityUp"],
        "confidence": signal["confidence"],
        "certainty_label": "FACTOR_COMBO_TRADE" if signal["qualityPassed"] else "FACTOR_COMBO_WAIT",
        "trade_quality_score": _trade_quality_score(signal),
        "trade_quality_passed": signal["qualityPassed"],
        "trade_quality_gate": gate_name,
        "high_winrate_gate": gate_name,
        "high_winrate_rule": str(signal["factorName"]),
        "high_winrate_gate_passed": signal["qualityPassed"],
        "high_winrate_gate_value": signal["historicalWinRate"],
        "high_winrate_gate_min": signal["qualityMinWinRate"],
        "model_version": str(signal["factorName"]),
        "model_family": "factor_combo",
        "model_duration": signal["duration"],
        "oos_win_rate": signal.get("oosWinRate"),
        "walk_forward_result": signal.get("walkForwardResult"),
        "recent_rolling_result": signal.get("recentRollingResult"),
        "data_freshness_status": cache_reason,
        "missing_feature_status": "complete",
        "quality_gate_reason": signal["qualityGateReason"],
        "signal_source": signal["source"],
        "rule_score": signal["score"],
        "rule_reasons": factor_combo_rule_reasons(signal, cache_reason, _signal_rule_name(signal)),
        "orderbook": None,
        "timeframe_votes": [],
    }


def _gate_name(strategy_key: str) -> str:
    if strategy_key.startswith(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY):
        return HIGH_WINRATE_FACTOR_COMBO_RULE_NAME
    return FACTOR_COMBO_RULE_NAME


def _signal_rule_name(signal: dict[str, Any]) -> str:
    if is_high_winrate_combo_name(str(signal.get("factorName") or "")):
        return HIGH_WINRATE_FACTOR_COMBO_RULE_NAME
    return FACTOR_COMBO_RULE_NAME


def _trade_quality_score(signal: dict[str, Any]) -> float:
    learning = signal.get("factorLearning")
    if isinstance(learning, dict) and learning.get("qualityScore") is not None:
        return float(learning["qualityScore"])
    return float(signal["confidence"])


def _refresh_factor_combo_source_klines(
    symbol: str,
    duration: str,
    entry_open_time: int | None,
) -> None:
    entry = (
        int(entry_open_time)
        if entry_open_time is not None
        else current_rule_entry_open_time_for_duration(duration)
    )
    sym = symbol.strip().upper()
    source_open_time = duration_entry_source_open_time(entry, duration)
    refresh_prediction_klines(sym, "1m", entry - MS_PER_MINUTE)
    refresh_prediction_klines(sym, duration, source_open_time)
