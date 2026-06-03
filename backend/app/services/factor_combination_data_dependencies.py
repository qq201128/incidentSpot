from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.services.market_context_ingest_service import ingest_market_context_data
from app.services.market_data_backfill_service import (
    backfill_duration_klines,
    backfill_funding_features,
    backfill_orderbook_features,
)

RefreshDurationKlines = Callable[[str, str], Any]


def refresh_factor_combination_data_dependencies(
    symbol: str,
    duration: str,
    *,
    refresh_duration_klines: RefreshDurationKlines | None = None,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    refresh = refresh_duration_klines or backfill_duration_klines
    context = ingest_market_context_data(sym, durations=(duration,))
    duration_klines = refresh(sym, duration)
    feature_fill = backfill_bar_aligned_factor_dependencies(sym, duration)
    return {
        "symbol": sym,
        "duration": duration,
        "marketContext": context,
        "durationKlines": duration_klines,
        "featureFill": feature_fill,
    }


def backfill_bar_aligned_factor_dependencies(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    return {
        "funding": backfill_funding_features(sym, duration),
        "orderbook": backfill_orderbook_features(sym, duration),
    }
