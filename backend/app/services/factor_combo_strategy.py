from __future__ import annotations

from typing import Any

from app.services.factor_cache_metadata import (
    assert_cache_usable_for_live_signal,
    live_signal_cache_reason,
)
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.rule_config import RULE_DURATION, SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, FACTOR_COMBO_RULE_NAME


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


def predict_factor_combo_rank_direction(
    symbol: str,
    duration: str = RULE_DURATION,
    *,
    combo_rank: int,
    result_strategy_key: str,
    entry_open_time: int | None = None,
    entry_grace_ms: int | None = None,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        supported = sorted(SUPPORTED_RULE_DURATIONS)
        raise ValueError(f"factor combo strategy supports only {supported}, got {duration}")
    cached = get_cached_combination_ranking(symbol, duration)
    if cached is None:
        raise ValueError(f"no cached combination ranking for {symbol.upper()} {duration}")
    assert_cache_usable_for_live_signal(cached, f"factor combination ranking {symbol.upper()} {duration}")
    top = _ranked_combo(cached, combo_rank)
    frame = materialize_mined_factor_frame(
        load_factor_frame(symbol, duration),
        symbol=symbol,
        duration=duration,
    ).frame
    signal = build_live_signal_from_ranking(
        frame,
        top,
        symbol=symbol,
        duration=duration,
        entry_open_time=entry_open_time,
        entry_grace_ms=entry_grace_ms,
    )
    return _prediction_payload(
        signal,
        entry_open_time,
        result_strategy_key,
        cache_reason=live_signal_cache_reason(cached),
    )


def _ranked_combo(cached: dict[str, Any], combo_rank: int) -> dict[str, Any]:
    ranking = cached.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("cached combination ranking is empty")
    if combo_rank <= 0 or combo_rank > len(ranking):
        raise ValueError(f"cached combination ranking has no Top{combo_rank}")
    return {**dict(ranking[combo_rank - 1]), "comboRank": combo_rank}


def _prediction_payload(
    signal: dict[str, Any],
    entry_open_time: int | None,
    strategy_key: str,
    *,
    cache_reason: str,
) -> dict[str, Any]:
    open_time = int(entry_open_time if entry_open_time is not None else signal["sourceOpenTime"])
    return {
        "symbol": signal["symbol"],
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
        "trade_quality_gate": FACTOR_COMBO_RULE_NAME,
        "high_winrate_gate": None,
        "high_winrate_rule": str(signal["factorName"]),
        "high_winrate_gate_passed": None,
        "high_winrate_gate_value": signal["historicalWinRate"],
        "high_winrate_gate_min": signal["qualityMinWinRate"],
        "quality_gate_reason": signal["qualityGateReason"],
        "signal_source": signal["source"],
        "rule_score": signal["score"],
        "rule_reasons": _rule_reasons(signal, cache_reason),
        "orderbook": None,
        "timeframe_votes": [],
    }


def _rule_reasons(signal: dict[str, Any], cache_reason: str) -> list[str]:
    member_names = ",".join(str(member["name"]) for member in signal["members"])
    reasons = [
        f"rule={FACTOR_COMBO_RULE_NAME}",
        f"combo={signal['factorName']}",
        f"combo_rank={signal.get('comboRank') or 1}",
        f"combo_cache={cache_reason}",
        f"members={member_names}",
        f"method={signal['method']}",
        f"historical_win_rate={signal['historicalWinRate']}",
        f"historical_profit_factor={signal['historicalProfitFactor']}",
        f"score={signal['score']}",
        f"quality_gate={signal['qualityGateReason']}",
        f"factor_timing={signal.get('factorTimingReason')}",
        f"factor_timing_passed={signal.get('factorTimingPassed')}",
        f"factor_timing_blocked={','.join(signal.get('factorTimingBlockedMembers') or [])}",
        f"quality_min_win_rate={signal['qualityMinWinRate']}",
        f"quality_min_profit_factor={signal['qualityMinProfitFactor']}",
    ]
    learning = signal.get("factorLearning")
    if isinstance(learning, dict):
        reasons.extend(_factor_learning_reasons(learning))
    return reasons


def _trade_quality_score(signal: dict[str, Any]) -> float:
    learning = signal.get("factorLearning")
    if isinstance(learning, dict) and learning.get("qualityScore") is not None:
        return float(learning["qualityScore"])
    return float(signal["confidence"])


def _factor_learning_reasons(learning: dict[str, Any]) -> list[str]:
    matches = learning.get("lossPatternMatches") or []
    return [
        f"factor_learning={learning.get('state')}",
        f"factor_learning_filter_passed={learning.get('filterPassed')}",
        f"factor_learning_confirmations={learning.get('confirmationCount')}",
        f"factor_learning_loss_matches={len(matches) if isinstance(matches, list) else 0}",
    ]
