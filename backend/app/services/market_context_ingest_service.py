from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.services.binance_service import fetch_funding_rate, fetch_orderbook
from app.services.binance_context_data import (
    fetch_fear_greed_index,
    fetch_global_long_short_ratio,
    fetch_open_interest_statistics,
    fetch_taker_buy_sell_volume,
)
from app.services.external_factor_data import (
    upsert_funding_rows,
    upsert_positioning_rows,
    upsert_sentiment_rows,
)
from app.services.kline_timing import current_rule_entry_open_time_for_duration
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.rule_orderbook_service import orderbook_rule_score, persist_orderbook_features

DEFAULT_CONTEXT_PERIOD = "5m"
DEFAULT_CONTEXT_LIMIT = 500
DEFAULT_REALTIME_ORDERBOOK_LIMIT = 500


def ingest_market_context_data(
    symbol: str,
    *,
    period: str = DEFAULT_CONTEXT_PERIOD,
    limit: int = DEFAULT_CONTEXT_LIMIT,
    durations: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    sym = symbol.upper()
    positioning_rows = _merged_positioning_rows(sym, period, limit)
    sentiment_rows = fetch_fear_greed_index(limit=60)
    upsert_positioning_rows(sym, positioning_rows)
    upsert_sentiment_rows(sentiment_rows)
    realtime = _persist_realtime_market_rows(sym, _context_durations(durations))
    return {
        "symbol": sym,
        "period": period,
        "positioningRows": len(positioning_rows),
        "sentimentRows": len(sentiment_rows),
        **realtime,
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


def _persist_realtime_market_rows(symbol: str, durations: tuple[str, ...]) -> dict[str, int]:
    orderbook = orderbook_rule_score(fetch_orderbook(symbol, limit=DEFAULT_REALTIME_ORDERBOOK_LIMIT))
    funding_rate = fetch_funding_rate(symbol)
    if funding_rate is None:
        raise ValueError(f"funding rate unavailable for {symbol}")
    open_times = _duration_open_times(durations)
    for open_time in open_times:
        persist_orderbook_features(symbol, open_time, orderbook)
    upsert_funding_rows(
        symbol,
        [{"open_time": open_time, "funding_rate": funding_rate} for open_time in open_times],
    )
    return {"orderbookRows": len(open_times), "fundingRows": len(open_times)}


def _context_durations(durations: tuple[str, ...] | None) -> tuple[str, ...]:
    selected = durations or tuple(sorted(SUPPORTED_RULE_DURATIONS))
    unsupported = sorted(set(selected) - set(SUPPORTED_RULE_DURATIONS))
    if unsupported:
        raise ValueError(f"unsupported market context durations: {unsupported}")
    return tuple(dict.fromkeys(selected))


def _duration_open_times(durations: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(current_rule_entry_open_time_for_duration(duration) for duration in durations)
    )
