from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_combination_service import run_factor_combination_ranking_on_frame
from app.services.factor_learning_common import finite
from app.services.factor_learning_patterns import factor_rows

COMBO_FACTOR_PREFIXES = ("combo__", "goal_combo__")
LEARNING_METRIC_KEYS = ("winRate", "profitFactor", "sharpe", "ir")


def current_ranking_report(
    symbol: str,
    duration: str,
    frame: pd.DataFrame,
    *,
    use_cache: bool = True,
) -> dict[str, Any]:
    if use_cache:
        cached = get_cached_combination_ranking(symbol, duration)
        fresh = _fresh_cached_ranking(cached)
        if fresh is not None:
            return {**fresh, "learningRefreshSource": "cache"}
        stale = _stale_cached_ranking_for_learning(cached)
        if stale is not None:
            return {**stale, "learningRefreshSource": "stale_cache"}
    report = run_factor_combination_ranking_on_frame(
        frame,
        symbol=symbol,
        duration=duration,
        config=CombinationSearchConfig(),
    )
    if use_cache:
        save_cached_combination_ranking(report)
        return {**report, "learningRefreshSource": "rebuilt_cache"}
    return {**report, "learningRefreshSource": "recent_window_uncached"}


def _fresh_cached_ranking(cached: dict[str, Any] | None) -> dict[str, Any] | None:
    if cached is None:
        return None
    if not cache_is_usable(cached):
        return None
    if not _has_learning_metric_rows(cached):
        return None
    return cached


def _stale_cached_ranking_for_learning(cached: dict[str, Any] | None) -> dict[str, Any] | None:
    if cached is None:
        return None
    if cache_is_usable(cached):
        return None
    if not _has_learning_metric_rows(cached):
        return None
    return cached


def _has_learning_metric_rows(report: dict[str, Any]) -> bool:
    for row in factor_rows(report):
        name = str(row.get("name") or "")
        if name.startswith(COMBO_FACTOR_PREFIXES):
            continue
        if any(finite(row.get(key)) is not None for key in LEARNING_METRIC_KEYS):
            return True
    return False
