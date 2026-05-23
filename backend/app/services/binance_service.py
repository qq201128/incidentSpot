from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.services.binance_http import DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT, FAPI_BASE_URL, retry_get as _retry_get
from app.services.binance_market_data import (
    LAST_FUNDING_RATE,
    LAST_ORDERBOOK,
    LAST_TICKER,
    fetch_24h_ticker,
    fetch_agg_trades_display,
    fetch_funding_rate,
    fetch_orderbook,
    fetch_orderbook_depth_display,
    fetch_premium_index,
    get_premium_index_display,
    get_cached_funding_rate,
    get_cached_orderbook,
    get_cached_ticker,
)
from app.services.kline_aggregation import (
    ONE_MINUTE_MS,
    aggregate_1m_klines as _aggregate_1m_klines,
    raw_klines_from_response as _raw_klines_from_response,
    trim_incomplete_edge_aggregates as _trim_incomplete_edge_aggregates,
    trim_leading_aggregate_if_first_bucket_incomplete as _trim_leading_aggregate_if_first_bucket_incomplete,
    trim_trailing_aggregate_if_last_bucket_incomplete as _trim_trailing_aggregate_if_last_bucket_incomplete,
)

TEN_MINUTE_ROWS = 10
TEN_MINUTE_MS = TEN_MINUTE_ROWS * ONE_MINUTE_MS
SYNTHETIC_KLINE_LIMIT_MAX = 1500
SYNTHETIC_KLINE_MIN_BARS = 50
SYNTHETIC_KLINE_PADDING_ROWS = 20


@dataclass(frozen=True)
class _SyntheticKlineRequest:
    endpoint: str
    symbol_param: str
    symbol_value: str
    limit: int
    start_time: int | None
    end_time: int | None
    request_options: dict[str, Any]
    include_forming: bool = False


def fetch_klines(
    symbol: str,
    interval: str,
    *,
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict]:
    sym = symbol.upper()
    if interval == "10m":
        return _fetch_synthetic_10m_klines(
            _SyntheticKlineRequest(
                endpoint="/fapi/v1/klines",
                symbol_param="symbol",
                symbol_value=sym,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
                request_options={},
            )
        )

    params = {"symbol": sym, "interval": parse_interval(interval), "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    rows = _retry_get(f"{FAPI_BASE_URL}/fapi/v1/klines", params)
    return _raw_klines_from_response(rows)


def fetch_index_price_klines(
    pair: str,
    interval: str,
    *,
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
    request_options: dict[str, Any] | None = None,
    include_forming: bool = False,
) -> list[dict]:
    """
    Index price OHLCV from GET /fapi/v1/indexPriceKlines (pair + interval).

    Used as the underlying index series for USD-M futures; aligns with mark/index
    context for event-style settlement references. Binance has no native 10m:
    we aggregate from 1m index klines.
    """
    pair_u = pair.upper()
    options = request_options or {}
    max_attempts = int(options.get("max_attempts", DEFAULT_MAX_ATTEMPTS))
    timeout = options.get("timeout", DEFAULT_TIMEOUT)
    if interval == "10m":
        return _fetch_synthetic_10m_klines(
            _SyntheticKlineRequest(
                endpoint="/fapi/v1/indexPriceKlines",
                symbol_param="pair",
                symbol_value=pair_u,
                limit=limit,
                start_time=start_time,
                end_time=end_time,
                request_options=options,
                include_forming=include_forming,
            )
        )

    params = {"pair": pair_u, "interval": parse_interval(interval), "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    rows = _retry_get(
        f"{FAPI_BASE_URL}/fapi/v1/indexPriceKlines",
        params,
        max_attempts=max_attempts,
        timeout=timeout,
    )
    return _raw_klines_from_response(rows)


def _fetch_synthetic_10m_klines(request: _SyntheticKlineRequest) -> list[dict]:
    params: dict[str, Any] = {
        request.symbol_param: request.symbol_value,
        "interval": "1m",
        "limit": _synthetic_kline_fetch_limit(request.limit),
    }
    _add_time_bounds(params, request.start_time, request.end_time)
    rows = _retry_get(
        f"{FAPI_BASE_URL}{request.endpoint}",
        params,
        max_attempts=int(request.request_options.get("max_attempts", DEFAULT_MAX_ATTEMPTS)),
        timeout=request.request_options.get("timeout", DEFAULT_TIMEOUT),
    )
    raw_rows = _raw_klines_from_response(rows)
    aggregated = _aggregate_1m_klines(raw_rows, TEN_MINUTE_MS)
    if request.include_forming:
        trimmed = _trim_leading_aggregate_if_first_bucket_incomplete(
            raw_rows, aggregated, TEN_MINUTE_MS
        )
        trimmed = _mark_forming_tail(trimmed, TEN_MINUTE_MS)
    else:
        trimmed = _trim_incomplete_edge_aggregates(raw_rows, aggregated, TEN_MINUTE_MS)
    return _tail_limit(trimmed, request.limit)


def _mark_forming_tail(rows: list[dict], bar_ms: int) -> list[dict]:
    """Mark the trailing in-progress aggregate so chart clients can overlay live price."""
    if not rows:
        return rows
    bucket = (int(time.time() * 1000) // bar_ms) * bar_ms
    if int(rows[-1]["openTime"]) != bucket:
        return rows
    last = dict(rows[-1])
    last["isClosed"] = False
    return [*rows[:-1], last]


def _synthetic_kline_fetch_limit(limit: int) -> int:
    requested_bars = max(limit, SYNTHETIC_KLINE_MIN_BARS)
    return min(SYNTHETIC_KLINE_LIMIT_MAX, requested_bars * TEN_MINUTE_ROWS + SYNTHETIC_KLINE_PADDING_ROWS)


def _add_time_bounds(params: dict[str, Any], start_time: int | None, end_time: int | None) -> None:
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time


def _tail_limit(rows: list[dict], limit: int) -> list[dict]:
    return rows[-limit:] if len(rows) > limit else rows


def parse_interval(interval: str) -> str:
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "60m": "1h",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
    }
    if interval not in mapping:
        raise ValueError(f"unsupported interval: {interval}")
    return mapping[interval]


def kline_ws_stream_name(interval: str) -> str:
    """Stream suffix for fstream URL, e.g. btcusdt@kline_30m. 10m uses underlying 1m."""
    if interval == "10m":
        return "kline_1m"
    return f"kline_{parse_interval(interval)}"
