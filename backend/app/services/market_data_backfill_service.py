from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from app.db.session import get_conn, run_db_write_with_retry
from app.services.binance_http import FAPI_BASE_URL, retry_get
from app.services.binance_service import fetch_klines, fetch_orderbook
from app.services.external_factor_data import upsert_funding_rows
from app.services.kline_backfill import backfill_1m_history, count_klines, upsert_klines_rows
from app.services.kline_prediction_refresh import KlineRefreshRequest, refresh_required_klines
from app.services.kline_timing import (
    current_rule_entry_open_time_for_duration,
    rule_interval_ms_for_duration,
)
from app.services.market_context_ingest_service import ingest_market_context_data
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.rule_orderbook_service import _UPSERT_ORDERBOOK_FEATURE_SQL, _feature_values, orderbook_rule_score

logger = logging.getLogger("uvicorn.error")

DEFAULT_1M_TARGET_ROWS = 50_000
KLINES_MULTI_INTERVALS = ("5m", "15m", "1h")
DEFAULT_MARKET_CONTEXT_LIMIT = 1500
FUNDING_HISTORY_LIMIT = 1000
FEATURE_FILL_BATCH = 500


def backfill_symbol_market_data(
    symbol: str,
    *,
    durations: tuple[str, ...] | None = None,
    target_1m_rows: int | None = None,
    market_context_limit: int | None = None,
    sync_multi: bool = True,
    fill_bar_features: bool = True,
) -> dict[str, Any]:
    """Backfill klines, multi-timeframe cache, market context, and bar-aligned features."""
    sym = symbol.strip().upper()
    selected = durations or tuple(sorted(SUPPORTED_RULE_DURATIONS))
    unsupported = sorted(set(selected) - set(SUPPORTED_RULE_DURATIONS))
    if unsupported:
        raise ValueError(f"unsupported durations: {unsupported}")

    target_rows = target_1m_rows if target_1m_rows is not None else _target_1m_rows()
    context_limit = market_context_limit if market_context_limit is not None else DEFAULT_MARKET_CONTEXT_LIMIT

    report: dict[str, Any] = {
        "symbol": sym,
        "durations": list(selected),
        "klines1m": _backfill_1m(sym, target_rows),
        "klinesByDuration": {},
        "klinesMulti": {},
        "marketContext": {},
        "featureFill": {},
    }

    for duration in selected:
        report["klinesByDuration"][duration] = _backfill_duration_klines(sym, duration)

    if sync_multi:
        report["klinesMulti"] = sync_klines_multi(sym)

    report["marketContext"] = ingest_market_context_data(
        sym,
        limit=context_limit,
        durations=selected,
    )

    if fill_bar_features:
        for duration in selected:
            report["featureFill"][duration] = {
                "funding": backfill_funding_features(sym, duration),
                "orderbook": backfill_orderbook_features(sym, duration),
            }

    return report


def _backfill_1m(symbol: str, target_rows: int) -> dict[str, int]:
    before = count_klines(symbol, "1m")
    after = backfill_1m_history(symbol, target_rows=target_rows)
    logger.info("1m kline backfill %s: %s -> %s rows", symbol, before, after)
    return {"before": before, "after": after, "target": target_rows}


def _backfill_duration_klines(symbol: str, duration: str) -> dict[str, Any]:
    before = count_klines(symbol, duration)
    required_open_time = _last_completed_entry_open_time(duration)
    refresh_required_klines(
        KlineRefreshRequest(symbol, duration, required_open_time),
    )
    after = count_klines(symbol, duration)
    logger.info("%s kline backfill %s: %s -> %s rows", duration, symbol, before, after)
    return {"before": before, "after": after, "requiredOpenTime": required_open_time}


def sync_klines_multi(
    symbol: str,
    intervals: tuple[str, ...] = KLINES_MULTI_INTERVALS,
    *,
    limit: int = 1500,
) -> dict[str, int]:
    sym = symbol.upper()
    counts: dict[str, int] = {}
    for interval in intervals:
        rows = fetch_klines(sym, interval, limit=min(1500, limit))
        if rows:
            _upsert_klines_multi(sym, interval, rows)
        counts[interval] = len(rows)
        logger.info("klines_multi %s %s: %s rows", sym, interval, len(rows))
    return counts


def backfill_funding_features(symbol: str, duration: str) -> dict[str, int]:
    sym = symbol.upper()
    kline_times = _kline_open_times(sym, duration)
    if not kline_times:
        return {"inserted": 0, "klineBars": 0}

    history = fetch_funding_rate_history(sym, limit=FUNDING_HISTORY_LIMIT)
    if not history:
        return {"inserted": 0, "klineBars": len(kline_times)}

    funding_df = pd.DataFrame(history).sort_values("open_time")
    target_df = pd.DataFrame({"open_time": kline_times})
    aligned = pd.merge_asof(
        target_df,
        funding_df,
        on="open_time",
        direction="backward",
    )
    rows = [
        {"open_time": int(row.open_time), "funding_rate": float(row.funding_rate)}
        for row in aligned.itertuples(index=False)
        if pd.notna(row.funding_rate)
    ]
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


def fetch_funding_rate_history(symbol: str, *, limit: int = FUNDING_HISTORY_LIMIT) -> list[dict[str, Any]]:
    sym = symbol.upper()
    data = retry_get(
        f"{FAPI_BASE_URL}/fapi/v1/fundingRate",
        {"symbol": sym, "limit": min(1000, max(1, int(limit)))},
    )
    if not isinstance(data, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        funding_time = item.get("fundingTime")
        rate = item.get("fundingRate")
        if funding_time is None or rate is None:
            continue
        rows.append({"open_time": int(funding_time), "funding_rate": float(rate)})
    rows.sort(key=lambda row: row["open_time"])
    return rows


def _upsert_klines_multi(symbol: str, interval: str, rows: list[dict]) -> None:
    def _upsert() -> None:
        conn = get_conn()
        try:
            for item in rows:
                conn.execute(
                    """
                    INSERT INTO klines_multi(symbol, interval, open_time, open, high, low, close, volume, close_time)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
                      open=excluded.open,
                      high=excluded.high,
                      low=excluded.low,
                      close=excluded.close,
                      volume=excluded.volume,
                      close_time=excluded.close_time
                    """,
                    (
                        symbol.upper(),
                        interval,
                        item["openTime"],
                        item["open"],
                        item["high"],
                        item["low"],
                        item["close"],
                        item["volume"],
                        item["closeTime"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_upsert)


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


def _last_completed_entry_open_time(duration: str) -> int:
    current = current_rule_entry_open_time_for_duration(duration)
    return current - rule_interval_ms_for_duration(duration)


def _target_1m_rows() -> int:
    raw = os.getenv("MARKET_DATA_1M_TARGET_ROWS", str(DEFAULT_1M_TARGET_ROWS)).strip()
    try:
        return max(10_000, int(raw))
    except ValueError:
        return DEFAULT_1M_TARGET_ROWS
