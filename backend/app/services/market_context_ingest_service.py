from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.services.binance_context_data import (
    fetch_fear_greed_index,
    fetch_global_long_short_ratio,
    fetch_open_interest_statistics,
    fetch_taker_buy_sell_volume,
)
from app.services.external_factor_data import upsert_positioning_rows, upsert_sentiment_rows

DEFAULT_CONTEXT_PERIOD = "5m"
DEFAULT_CONTEXT_LIMIT = 500


def ingest_market_context_data(
    symbol: str,
    *,
    period: str = DEFAULT_CONTEXT_PERIOD,
    limit: int = DEFAULT_CONTEXT_LIMIT,
) -> dict[str, Any]:
    sym = symbol.upper()
    positioning_rows = _merged_positioning_rows(sym, period, limit)
    sentiment_rows = fetch_fear_greed_index(limit=60)
    upsert_positioning_rows(sym, positioning_rows)
    upsert_sentiment_rows(sentiment_rows)
    return {
        "symbol": sym,
        "period": period,
        "positioningRows": len(positioning_rows),
        "sentimentRows": len(sentiment_rows),
    }


def _merged_positioning_rows(symbol: str, period: str, limit: int) -> list[dict[str, Any]]:
    rows_by_time: dict[int, dict[str, Any]] = defaultdict(dict)
    for row in fetch_open_interest_statistics(symbol, period, limit=limit):
        rows_by_time[int(row["open_time"])].update(row)
    for row in fetch_global_long_short_ratio(symbol, period, limit=limit):
        rows_by_time[int(row["open_time"])].update(row)
    for row in fetch_taker_buy_sell_volume(symbol, period, limit=limit):
        rows_by_time[int(row["open_time"])].update(row)
    return [
        {"open_time": open_time, **payload}
        for open_time, payload in sorted(rows_by_time.items())
    ]
