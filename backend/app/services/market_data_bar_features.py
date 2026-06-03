from __future__ import annotations

from typing import Any

import pandas as pd

from app.db.session import get_conn, run_db_write_with_retry
from app.services.binance_http import FAPI_BASE_URL, retry_get
from app.services.binance_service import fetch_orderbook
from app.services.external_factor_data import upsert_funding_rows
from app.services.rule_orderbook_service import _UPSERT_ORDERBOOK_FEATURE_SQL, _feature_values, orderbook_rule_score

FUNDING_HISTORY_LIMIT = 1000
FEATURE_FILL_BATCH = 500
FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000
FUNDING_PAGE_STEP_MS = 1


def backfill_funding_features(symbol: str, duration: str) -> dict[str, int]:
    sym = symbol.upper()
    kline_times = _kline_open_times(sym, duration)
    if not kline_times:
        return {"inserted": 0, "klineBars": 0}

    history = fetch_funding_rate_history(
        sym,
        start_time=min(kline_times) - FUNDING_INTERVAL_MS,
        end_time=max(kline_times),
        limit=FUNDING_HISTORY_LIMIT,
    )
    if not history:
        return {"inserted": 0, "klineBars": len(kline_times)}

    rows = _aligned_funding_rows(kline_times, history)
    if rows:
        upsert_funding_rows(sym, rows)
    return {"inserted": len(rows), "klineBars": len(kline_times)}


def backfill_orderbook_features(symbol: str, duration: str) -> dict[str, int]:
    sym = symbol.upper()
    missing = _missing_feature_open_times(sym, duration, "orderbook_features")
    if not missing:
        return {"inserted": 0, "missing": 0}

    orderbook = orderbook_rule_score(fetch_orderbook(sym, limit=500))
    inserted = 0
    for index in range(0, len(missing), FEATURE_FILL_BATCH):
        batch = missing[index : index + FEATURE_FILL_BATCH]
        _upsert_orderbook_feature_batch(sym, batch, orderbook)
        inserted += len(batch)
    return {"inserted": inserted, "missing": len(missing)}


def fetch_funding_rate_history(
    symbol: str,
    *,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int = FUNDING_HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    sym = symbol.upper()
    page_limit = min(1000, max(1, int(limit)))
    cursor = start_time
    rows: list[dict[str, Any]] = []
    while True:
        page = _fetch_funding_rate_page(
            sym,
            start_time=cursor,
            end_time=end_time,
            limit=page_limit,
        )
        if not page:
            break
        rows.extend(page)
        if len(page) < page_limit or end_time is None:
            break
        next_cursor = int(page[-1]["open_time"]) + FUNDING_PAGE_STEP_MS
        if cursor is not None and next_cursor <= cursor:
            raise ValueError(f"funding history pagination did not advance for {sym}")
        if next_cursor > int(end_time):
            break
        cursor = next_cursor
    return sorted(_dedupe_funding_rows(rows), key=lambda row: row["open_time"])


def _upsert_orderbook_feature_batch(symbol: str, open_times: list[int], orderbook: dict[str, Any]) -> None:
    values = [_feature_values(symbol, int(open_time), orderbook) for open_time in open_times]

    def _upsert() -> None:
        conn = get_conn()
        try:
            conn.executemany(_UPSERT_ORDERBOOK_FEATURE_SQL, values)
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_upsert)


def _fetch_funding_rate_page(
    symbol: str,
    *,
    start_time: int | None,
    end_time: int | None,
    limit: int,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"symbol": symbol, "limit": limit}
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)
    data = retry_get(f"{FAPI_BASE_URL}/fapi/v1/fundingRate", params)
    if not isinstance(data, list):
        raise ValueError(f"funding history response is not a list for {symbol}")
    return _funding_rate_rows(data)


def _funding_rate_rows(items: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        row = _funding_rate_row(item)
        if row is not None:
            rows.append(row)
    return rows


def _funding_rate_row(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    funding_time = item.get("fundingTime")
    rate = item.get("fundingRate")
    if funding_time is None or rate is None:
        return None
    return {"open_time": int(funding_time), "funding_rate": float(rate)}


def _dedupe_funding_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_time = {int(row["open_time"]): row for row in rows}
    return list(by_time.values())


def _aligned_funding_rows(kline_times: list[int], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    funding_df = pd.DataFrame(history).sort_values("open_time")
    target_df = pd.DataFrame({"open_time": kline_times})
    aligned = pd.merge_asof(target_df, funding_df, on="open_time", direction="backward")
    return [
        {"open_time": int(row.open_time), "funding_rate": float(row.funding_rate)}
        for row in aligned.itertuples(index=False)
        if pd.notna(row.funding_rate)
    ]


def _kline_open_times(symbol: str, duration: str) -> list[int]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time FROM klines
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time ASC
            """,
            (symbol.upper(), duration),
        ).fetchall()
    finally:
        conn.close()
    return [int(row["open_time"]) for row in rows]


def _missing_feature_open_times(symbol: str, duration: str, table: str) -> list[int]:
    conn = get_conn()
    try:
        rows = conn.execute(
            f"""
            SELECT k.open_time
            FROM klines k
            LEFT JOIN {table} f
              ON f.symbol = k.symbol AND f.open_time = k.open_time
            WHERE k.symbol = ? AND k.interval = ? AND f.open_time IS NULL
            ORDER BY k.open_time ASC
            """,
            (symbol.upper(), duration),
        ).fetchall()
    finally:
        conn.close()
    return [int(row["open_time"]) for row in rows]
