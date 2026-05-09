from __future__ import annotations

import json
from typing import Any

_MS_PER_SECOND = 1000
_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60
_HOURS_PER_DAY = 24
_TEN_MINUTES = 10
_THIRTY_MINUTES = 30
_FOUR_HOURS = 4
_TEN_M_LAST_MINUTE_INDEX = 9
_ONE_M_MS = _SECONDS_PER_MINUTE * _MS_PER_SECOND
_TEN_MS = _TEN_MINUTES * _ONE_M_MS
_INTERVAL_MS = {
    "10m": _TEN_MINUTES * _ONE_M_MS,
    "30m": _THIRTY_MINUTES * _ONE_M_MS,
    "60m": _MINUTES_PER_HOUR * _ONE_M_MS,
    "1h": _MINUTES_PER_HOUR * _ONE_M_MS,
    "4h": _FOUR_HOURS * _MINUTES_PER_HOUR * _ONE_M_MS,
    "1d": _HOURS_PER_DAY * _MINUTES_PER_HOUR * _ONE_M_MS,
}


def unwrap_fstream_ws_message(raw: str | bytes) -> dict[str, Any]:
    """Combined streams wrap the inner event under ``data``."""
    if isinstance(raw, bytes):
        raw = raw.decode()
    outer: Any = json.loads(raw)
    if not isinstance(outer, dict):
        return {}
    inner = outer.get("data")
    if isinstance(inner, dict):
        return inner
    return outer


def candle_from_k_obj(k: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return {
            "openTime": int(k["t"]),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k.get("v") or 0),
            "closeTime": int(k["T"]),
            "isClosed": _binance_kline_closed(k.get("x")),
        }
    except (KeyError, TypeError, ValueError):
        return None


def candle_from_index_price_event(
    event: dict[str, Any],
    interval: str,
    state: dict | None,
) -> tuple[dict | None, dict | None]:
    try:
        event_time = int(event["E"])
        price = float(event["i"])
    except (KeyError, TypeError, ValueError):
        return None, state
    bucket = (event_time // _interval_ms(interval)) * _interval_ms(interval)
    next_state = _merge_price_tick_into_state(state, bucket, price, event_time)
    return _synthetic_from_state(bucket, next_state, False), next_state


def candle_for_interval(c1: dict, interval: str, ten_m_state: dict | None) -> tuple[dict, dict | None]:
    if interval != "10m":
        return {
            "openTime": c1["openTime"],
            "open": c1["open"],
            "high": c1["high"],
            "low": c1["low"],
            "close": c1["close"],
            "volume": c1["volume"],
            "closeTime": c1["closeTime"],
            "isClosed": c1["isClosed"],
        }, ten_m_state
    bucket = (c1["openTime"] // _TEN_MS) * _TEN_MS
    next_state = _merge_1m_into_10m_state(ten_m_state, bucket, c1)
    minutes_in = (c1["openTime"] - bucket) // _ONE_M_MS
    ten_closed = c1["isClosed"] and minutes_in == _TEN_M_LAST_MINUTE_INDEX
    return _synthetic_from_state(bucket, next_state, ten_closed), next_state


def _binance_kline_closed(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1", "yes")
    return bool(raw)


def _interval_ms(interval: str) -> int:
    if interval not in _INTERVAL_MS:
        raise ValueError(f"unsupported synthetic index interval: {interval}")
    return _INTERVAL_MS[interval]


def _merge_price_tick_into_state(
    state: dict | None,
    bucket: int,
    price: float,
    event_time: int,
) -> dict:
    if state is None or state.get("bucket") != bucket:
        return {
            "bucket": bucket,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 0.0,
            "closeTime": event_time,
        }
    return {
        "bucket": bucket,
        "open": state["open"],
        "high": max(state["high"], price),
        "low": min(state["low"], price),
        "close": price,
        "volume": state["volume"],
        "closeTime": event_time,
    }


def _merge_1m_into_10m_state(state: dict | None, bucket: int, c1: dict) -> dict:
    if state is None or state.get("bucket") != bucket:
        return {
            "bucket": bucket,
            "open": c1["open"],
            "high": c1["high"],
            "low": c1["low"],
            "close": c1["close"],
            "volume": c1["volume"],
            "closeTime": c1["closeTime"],
        }
    return {
        "bucket": bucket,
        "open": state["open"],
        "high": max(state["high"], c1["high"]),
        "low": min(state["low"], c1["low"]),
        "close": c1["close"],
        "volume": state["volume"] + c1["volume"],
        "closeTime": c1["closeTime"],
    }


def _synthetic_from_state(bucket: int, state: dict, is_closed: bool) -> dict:
    return {
        "openTime": bucket,
        "open": float(state["open"]),
        "high": float(state["high"]),
        "low": float(state["low"]),
        "close": float(state["close"]),
        "volume": float(state["volume"]),
        "closeTime": int(state["closeTime"]),
        "isClosed": is_closed,
    }
