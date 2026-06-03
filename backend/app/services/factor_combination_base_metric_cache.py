from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.services.factor_cache_metadata import bar_aligned_features_match, cache_is_usable
from app.services.factor_ranking_cache_service import get_cached_ranking


def cached_factor_metrics_by_name(symbol: str, duration: str) -> dict[str, dict[str, Any]]:
    cached = get_cached_ranking(symbol, duration)
    if not cache_is_usable(cached):
        return {}
    if not _feature_dependencies_match(cached, symbol, duration):
        return {}
    ranking = cached.get("ranking") if isinstance(cached, dict) else None
    if not isinstance(ranking, list):
        return {}
    return {
        name: deepcopy(row)
        for row in ranking
        if isinstance(row, dict) and (name := str(row.get("factorName") or ""))
    }


def _feature_dependencies_match(cached: dict[str, Any] | None, symbol: str, duration: str) -> bool:
    if not isinstance(cached, dict):
        return False
    return bar_aligned_features_match(cached.get("cacheMeta"), symbol, duration)
