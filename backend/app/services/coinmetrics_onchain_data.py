from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.binance_http import SHORT_TIMEOUT, retry_get

COINMETRICS_COMMUNITY_BASE = "https://community-api.coinmetrics.io/v4"
DEFAULT_ONCHAIN_PAGE_SIZE = 1000
DEFAULT_ONCHAIN_LOOKBACK_DAYS = 500

SYMBOL_TO_ASSET: dict[str, str] = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
}

FLOW_METRICS = ("AdrActCnt", "TxCnt", "FlowInExNtv", "FlowOutExNtv")


def fetch_onchain_feature_rows(
    symbol: str,
    *,
    lookback_days: int = DEFAULT_ONCHAIN_LOOKBACK_DAYS,
    page_size: int = DEFAULT_ONCHAIN_PAGE_SIZE,
) -> list[dict[str, Any]]:
    sym = symbol.strip().upper()
    asset = SYMBOL_TO_ASSET.get(sym)
    if asset is None:
        return []
    start_time, end_time = _lookback_window(lookback_days)
    flow_rows = _fetch_asset_metric_pages(
        assets=asset,
        metrics=",".join(FLOW_METRICS),
        start_time=start_time,
        end_time=end_time,
        page_size=page_size,
    )
    stablecoin_rows = _fetch_asset_metric_pages(
        assets=f"{asset},usdt,usdc",
        metrics="SplyCur,CapMrktCurUSD",
        start_time=start_time,
        end_time=end_time,
        page_size=page_size,
    )
    stablecoin_by_time = _stablecoin_supply_ratio_by_time(stablecoin_rows, asset)
    return _merge_onchain_rows(flow_rows, stablecoin_by_time)


def _lookback_window(lookback_days: int) -> tuple[str, str]:
    days = max(30, int(lookback_days))
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def _fetch_asset_metric_pages(
    *,
    assets: str,
    metrics: str,
    start_time: str,
    end_time: str,
    page_size: int,
) -> list[dict[str, Any]]:
    url = f"{COINMETRICS_COMMUNITY_BASE}/timeseries/asset-metrics"
    params: dict[str, Any] = {
        "assets": assets,
        "metrics": metrics,
        "frequency": "1d",
        "page_size": max(1, min(int(page_size), 10000)),
        "start_time": start_time,
        "end_time": end_time,
    }
    rows: list[dict[str, Any]] = []
    while True:
        payload = retry_get(url, params, timeout=SHORT_TIMEOUT)
        if not isinstance(payload, dict):
            break
        rows.extend(payload.get("data", []))
        token = payload.get("next_page_token")
        if not token:
            break
        params = {
            "assets": assets,
            "metrics": metrics,
            "frequency": "1d",
            "page_size": params["page_size"],
            "next_page_token": token,
        }
    return rows


def _merge_onchain_rows(
    flow_rows: list[dict[str, Any]],
    stablecoin_by_time: dict[int, float | None],
) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in flow_rows:
        open_time = _iso_time_to_open_time_ms(str(row["time"]))
        payload = merged.setdefault(open_time, {"open_time": open_time})
        inflow = _metric_value(row, "FlowInExNtv")
        outflow = _metric_value(row, "FlowOutExNtv")
        if inflow is not None and outflow is not None:
            payload["exchange_netflow"] = inflow - outflow
        active_addresses = _metric_value(row, "AdrActCnt")
        if active_addresses is not None:
            payload["active_addresses"] = active_addresses
        transaction_count = _metric_value(row, "TxCnt")
        if transaction_count is not None:
            payload["transaction_count"] = transaction_count
        ratio = stablecoin_by_time.get(open_time)
        if ratio is not None:
            payload["stablecoin_supply_ratio"] = ratio
    return [merged[key] for key in sorted(merged)]


def _stablecoin_supply_ratio_by_time(
    rows: list[dict[str, Any]],
    cap_asset: str,
) -> dict[int, float | None]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        open_time = _iso_time_to_open_time_ms(str(row["time"]))
        grouped[open_time][str(row["asset"])] = row
    ratios: dict[int, float | None] = {}
    for open_time, assets in grouped.items():
        cap_row = assets.get(cap_asset)
        usdt_row = assets.get("usdt")
        usdc_row = assets.get("usdc")
        if cap_row is None or usdt_row is None or usdc_row is None:
            continue
        market_cap = _metric_value(cap_row, "CapMrktCurUSD")
        usdt_supply = _metric_value(usdt_row, "SplyCur")
        usdc_supply = _metric_value(usdc_row, "SplyCur")
        if market_cap is None or market_cap <= 0 or usdt_supply is None or usdc_supply is None:
            continue
        ratios[open_time] = (usdt_supply + usdc_supply) / market_cap
    return ratios


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    raw = row.get(metric)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # NaN guard


def _iso_time_to_open_time_ms(iso_time: str) -> int:
    normalized = iso_time.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
