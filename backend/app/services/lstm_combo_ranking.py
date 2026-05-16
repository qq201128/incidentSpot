from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.services.factor_cache_metadata import cache_is_usable, cache_is_usable_for_live_signal
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking

LSTM_COMBO_SOURCE_PRIMARY = "factor_combination_ranking_cache"
LSTM_COMBO_SOURCE_HIGH_WINRATE = "high_winrate_combo_ranking_cache"

RankingLoader = Callable[[str, str], dict[str, Any] | None]


def resolve_lstm_combo_ranking(
    symbol: str,
    duration: str,
    *,
    primary_loader: RankingLoader | None = None,
    high_winrate_loader: RankingLoader | None = None,
) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    primary = _load(primary_loader or get_cached_combination_ranking, sym, duration)
    if _usable_primary_ranking(primary):
        return _source_payload(primary, LSTM_COMBO_SOURCE_PRIMARY, "primary_ready")

    high_winrate = _load(high_winrate_loader or get_cached_high_winrate_combo_ranking, sym, duration)
    if _usable_high_winrate_ranking(high_winrate):
        reason = f"primary_{_ranking_state(primary)};high_winrate_{_ranking_state(high_winrate)}"
        return _source_payload(high_winrate, LSTM_COMBO_SOURCE_HIGH_WINRATE, reason)

    if primary is not None:
        return _source_payload(primary, LSTM_COMBO_SOURCE_PRIMARY, _ranking_state(primary))
    return None


def lstm_combo_ranking_source(ranking: dict[str, Any] | None) -> str | None:
    if ranking is None:
        return None
    source = ranking.get("lstmComboRankingSource")
    return str(source) if source else None


def _load(loader: RankingLoader, symbol: str, duration: str) -> dict[str, Any] | None:
    ranking = loader(symbol, duration)
    return dict(ranking) if isinstance(ranking, dict) else None


def _usable_primary_ranking(ranking: dict[str, Any] | None) -> bool:
    rows = None if ranking is None else ranking.get("ranking")
    return bool(cache_is_usable(ranking) and isinstance(rows, list) and rows)


def _usable_high_winrate_ranking(ranking: dict[str, Any] | None) -> bool:
    rows = None if ranking is None else ranking.get("ranking")
    return bool(cache_is_usable_for_live_signal(ranking) and isinstance(rows, list) and rows)


def _ranking_state(ranking: dict[str, Any] | None) -> str:
    if ranking is None:
        return "missing"
    if not cache_is_usable_for_live_signal(ranking):
        status = ranking.get("cacheStatus") or {}
        reason = status.get("reason") if isinstance(status, dict) else None
        return f"stale_{reason or 'unknown'}"
    rows = ranking.get("ranking")
    if not isinstance(rows, list):
        return "invalid"
    if not rows:
        return "empty"
    return "ready"


def _source_payload(ranking: dict[str, Any], source: str, reason: str) -> dict[str, Any]:
    return {
        **ranking,
        "lstmComboRankingSource": source,
        "lstmComboRankingReason": reason,
    }
