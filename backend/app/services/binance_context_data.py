from __future__ import annotations

from typing import Any

from app.services.binance_service import FAPI_BASE_URL, _retry_get

ALTERNATIVE_ME_BASE_URL = "https://api.alternative.me"


def fetch_open_interest_statistics(
    symbol: str,
    period: str = "5m",
    *,
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict[str, Any]]:
    params = _futures_data_params(symbol, period, limit, start_time, end_time)
    rows = _retry_get(f"{FAPI_BASE_URL}/futures/data/openInterestHist", params)
    return [
        {
            "open_time": int(row["timestamp"]),
            "open_interest": float(row.get("sumOpenInterest", 0) or 0),
            "open_interest_value": float(row.get("sumOpenInterestValue", 0) or 0),
        }
        for row in rows
    ]


def fetch_global_long_short_ratio(
    symbol: str,
    period: str = "5m",
    *,
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict[str, Any]]:
    params = _futures_data_params(symbol, period, limit, start_time, end_time)
    rows = _retry_get(f"{FAPI_BASE_URL}/futures/data/globalLongShortAccountRatio", params)
    return [
        {
            "open_time": int(row["timestamp"]),
            "long_short_ratio": float(row.get("longShortRatio", 0) or 0),
            "long_account": float(row.get("longAccount", 0) or 0),
            "short_account": float(row.get("shortAccount", 0) or 0),
        }
        for row in rows
    ]


def fetch_taker_buy_sell_volume(
    symbol: str,
    period: str = "5m",
    *,
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[dict[str, Any]]:
    params = _futures_data_params(symbol, period, limit, start_time, end_time)
    rows = _retry_get(f"{FAPI_BASE_URL}/futures/data/takerlongshortRatio", params)
    return [
        {
            "open_time": int(row["timestamp"]),
            "taker_buy_sell_ratio": float(row.get("buySellRatio", 0) or 0),
            "taker_buy_vol": float(row.get("buyVol", 0) or 0),
            "taker_sell_vol": float(row.get("sellVol", 0) or 0),
        }
        for row in rows
    ]


def fetch_fear_greed_index(limit: int = 30) -> list[dict[str, Any]]:
    params = {"limit": max(1, min(int(limit), 365)), "format": "json"}
    data = _retry_get(f"{ALTERNATIVE_ME_BASE_URL}/fng/", params, timeout=(10, 20))
    rows = data.get("data", []) if isinstance(data, dict) else []
    return [
        {
            "open_time": int(row["timestamp"]) * 1000,
            "fear_greed_value": float(row.get("value", 0) or 0),
            "fear_greed_classification": str(row.get("value_classification", "")),
        }
        for row in rows
    ]


def _futures_data_params(
    symbol: str,
    period: str,
    limit: int,
    start_time: int | None,
    end_time: int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "symbol": symbol.upper(),
        "period": period,
        "limit": max(1, min(int(limit), 500)),
    }
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)
    return params
