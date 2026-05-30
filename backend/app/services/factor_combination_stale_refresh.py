from __future__ import annotations

from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_combination_cache_service import get_cached_combination_ranking


def ranking_cache_needs_refresh(symbol: str, duration: str) -> bool:
    cached = get_cached_combination_ranking(symbol, duration)
    return cached is None or not cache_is_usable(cached)


def failed_stale_item(symbol: str, duration: str, exc: Exception) -> dict:
    return {
        "symbol": symbol,
        "duration": duration,
        "error": str(exc),
        "exceptionType": type(exc).__name__,
    }


def stale_refresh_details(refreshed: list[dict], failed: list[dict], skipped: list[dict]) -> dict:
    return {
        "stage": "stale_configured",
        "refreshedItems": refreshed,
        "failedItems": failed,
        "skippedItems": skipped,
    }
