from __future__ import annotations

import logging
from typing import Any

from app.services.background_loop_status import record_loop_failure, record_loop_success
from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_ranking_background import refresh_symbol_rankings
from app.services.factor_ranking_cache_service import (
    factor_ranking_precomputed_symbols,
    get_cached_ranking,
)
from app.services.factor_ranking_page import build_ranking_page

BACKGROUND_REFRESH_LOOP = "factor_ranking"
logger = logging.getLogger("uvicorn.error")


def factor_ranking_payload(
    *,
    symbol: str,
    duration: str,
    category: str | None,
    query: str | None,
    page: int,
    page_size: int,
) -> dict[str, Any]:
    precomputed = factor_ranking_precomputed_symbols()
    cached = get_cached_ranking(symbol, duration)
    if cached is None:
        return _empty_payload(symbol, duration, category, query, page, page_size, precomputed)
    if not cache_is_usable(cached):
        return _stale_payload(symbol, duration, category, query, page, page_size, cached, precomputed)
    return _cache_payload(symbol, duration, category, query, page, page_size, cached, precomputed)


def background_refresh_rankings(symbol: str, duration: str | None) -> None:
    try:
        refresh_symbol_rankings(symbol, duration)
        record_loop_success(
            BACKGROUND_REFRESH_LOOP,
            {"stage": "manual_api_refresh", "symbol": symbol, "duration": duration},
        )
    except Exception as exc:
        record_loop_failure(
            BACKGROUND_REFRESH_LOOP,
            exc,
            {"stage": "manual_api_refresh", "symbol": symbol, "duration": duration},
        )
        logger.exception("background factor ranking refresh failed: %s %s", symbol, duration)
        raise


def _empty_payload(
    symbol: str,
    duration: str,
    category: str | None,
    query: str | None,
    page: int,
    page_size: int,
    precomputed: list[str],
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "category": category,
        "updatedAt": None,
        "source": "none",
        "precomputedSymbols": precomputed,
        "hint": "排名由后台定时写入缓存；当前交易对/周期尚无数据。可将该交易对加入 FACTOR_RANKING_SYMBOLS 或使用 POST /ranking/refresh 排队重算。",
        **build_ranking_page([], query, page, page_size),
    }


def _stale_payload(
    symbol: str,
    duration: str,
    category: str | None,
    query: str | None,
    page: int,
    page_size: int,
    cached: dict[str, Any],
    precomputed: list[str],
) -> dict[str, Any]:
    rows = _filter_ranking_by_category(list(cached.get("ranking") or []), category)
    return {
        "symbol": symbol,
        "duration": duration,
        "category": category,
        "updatedAt": cached.get("updatedAt"),
        "source": "stale_cache",
        "staleRankingTotal": cached.get("total"),
        "cacheStatus": cached.get("cacheStatus"),
        "precomputedSymbols": precomputed,
        "hint": "因子排名缓存对应的历史数据已变化或缺少数据指纹；请刷新重算后再使用。",
        **build_ranking_page(rows, query, page, page_size),
    }


def _cache_payload(
    symbol: str,
    duration: str,
    category: str | None,
    query: str | None,
    page: int,
    page_size: int,
    cached: dict[str, Any],
    precomputed: list[str],
) -> dict[str, Any]:
    rows = _filter_ranking_by_category(list(cached["ranking"]), category)
    rows.sort(key=_ranking_sort_key, reverse=True)
    page_payload = build_ranking_page(rows, query, page, page_size)
    return {
        "symbol": symbol,
        "duration": duration,
        "category": category,
        "ranking": page_payload["ranking"],
        "updatedAt": cached["updatedAt"],
        "source": "cache",
        "precomputedSymbols": precomputed,
        "rankingDiagnostics": cached.get("rankingDiagnostics") or {},
        "rankingFailures": cached.get("rankingFailures") or [],
        **page_payload,
    }


def _filter_ranking_by_category(ranking: list[dict[str, Any]], category: str | None) -> list[dict[str, Any]]:
    if not category:
        return ranking
    return [row for row in ranking if row.get("category") == category]


def _ranking_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row.get("factorScore") or 0.0), abs(float(row.get("ir") or 0.0)))
