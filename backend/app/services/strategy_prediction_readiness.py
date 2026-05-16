from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_cache_metadata import cache_is_usable_for_live_signal, live_signal_cache_reason
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY


@dataclass(frozen=True)
class PredictionReadiness:
    ready: bool
    reason: str


def strategy_prediction_readiness(strategy_key: str, symbol: str, duration: str) -> PredictionReadiness:
    if strategy_key == FACTOR_COMBO_STRATEGY_KEY:
        return _ranking_cache_readiness(get_cached_combination_ranking(symbol, duration))
    if strategy_key == HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY:
        return _ranking_cache_readiness(get_cached_high_winrate_combo_ranking(symbol, duration))
    return PredictionReadiness(True, "ready")


def _ranking_cache_readiness(cache: dict[str, Any] | None) -> PredictionReadiness:
    if cache is None:
        return PredictionReadiness(False, "ranking_cache_missing")
    if not cache_is_usable_for_live_signal(cache):
        return PredictionReadiness(False, f"ranking_cache_{live_signal_cache_reason(cache)}")
    ranking = cache.get("ranking")
    if not isinstance(ranking, list):
        return PredictionReadiness(False, "ranking_cache_malformed")
    if not ranking:
        return PredictionReadiness(False, "ranking_cache_empty")
    return PredictionReadiness(True, "ready")
