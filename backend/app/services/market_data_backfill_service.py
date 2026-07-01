from __future__ import annotations

import logging
import os
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.binance_service import TEN_MINUTE_MS, fetch_klines
from app.services.kline_aggregation import aggregate_1m_klines, trim_incomplete_edge_aggregates
from app.services.kline_backfill import (
    backfill_1m_history,
    backfill_interval_history,
    count_klines,
    upsert_klines_rows,
)
from app.services.kline_prediction_refresh import KlineRefreshRequest, refresh_required_klines
from app.services.kline_timing import (
    current_rule_entry_open_time_for_duration,
    rule_interval_ms_for_duration,
)
from app.services.market_data_bar_features import backfill_funding_features, backfill_orderbook_features
from app.services.market_context_ingest_service import ingest_market_context_data
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

logger = logging.getLogger("uvicorn.error")

DEFAULT_1M_TARGET_ROWS = 50_000
KLINES_MULTI_INTERVALS = ("5m", "10m", "15m", "30m", "60m", "1h", "1d")
DEFAULT_MARKET_CONTEXT_LIMIT = 1500


def backfill_symbol_market_data(
    symbol: str,
    *,
    durations: tuple[str, ...] | None = None,
    target_1m_rows: int | None = None,
    market_context_limit: int | None = None,
    sync_multi: bool = True,
    fill_bar_features: bool = True,
    full_history: bool = False,
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
        "klines1m": _backfill_1m(sym, target_rows, full_history=full_history),
        "klinesByDuration": {},
        "klinesMulti": {},
        "marketContext": {},
        "featureFill": {},
    }

    for duration in selected:
        report["klinesByDuration"][duration] = _backfill_duration(sym, duration, full_history)

    if sync_multi:
        report["klinesMulti"] = sync_full_klines_multi(sym) if full_history else sync_klines_multi(sym)

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

    return {**report, "fullHistory": full_history}


def _backfill_1m(symbol: str, target_rows: int, *, full_history: bool) -> dict[str, Any]:
    before = count_klines(symbol, "1m")
    after = _backfill_full_interval(symbol, "1m") if full_history else backfill_1m_history(symbol, target_rows=target_rows)
    logger.info("1m kline backfill %s: %s -> %s rows", symbol, before, after)
    return {"before": before, "after": after, "target": None if full_history else target_rows}


def _backfill_duration(symbol: str, duration: str, full_history: bool) -> dict[str, Any]:
    if not full_history:
        return backfill_duration_klines(symbol, duration)
    if duration == "10m":
        return aggregate_full_10m_from_1m(symbol)
    return backfill_full_duration_klines(symbol, duration)


def backfill_full_duration_klines(symbol: str, duration: str) -> dict[str, Any]:
    before = count_klines(symbol, duration)
    after = _backfill_full_interval(symbol, duration)
    logger.info("full %s kline backfill %s: %s -> %s rows", duration, symbol, before, after)
    return {"before": before, "after": after, "fullHistory": True}


def _backfill_full_interval(symbol: str, interval: str) -> int:
    return backfill_interval_history(symbol, interval, target_rows=None, max_rounds=None)


def aggregate_full_10m_from_1m(symbol: str) -> dict[str, Any]:
    before = count_klines(symbol, "10m")
    rows = _local_1m_rows(symbol)
    aggregated = trim_incomplete_edge_aggregates(rows, aggregate_1m_klines(rows, TEN_MINUTE_MS), TEN_MINUTE_MS)
    if aggregated:
        upsert_klines_rows(symbol, "10m", aggregated)
    after = count_klines(symbol, "10m")
    return {"before": before, "after": after, "source1mRows": len(rows), "fullHistory": True}


def backfill_duration_klines(symbol: str, duration: str) -> dict[str, Any]:
    before = count_klines(symbol, duration)
    required_open_time = _last_completed_entry_open_time(duration)
    refresh_required_klines(
        KlineRefreshRequest(symbol.upper(), duration, required_open_time),
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


def sync_full_klines_multi(
    symbol: str,
    intervals: tuple[str, ...] = KLINES_MULTI_INTERVALS,
    *,
    chunk: int = 1000,
) -> dict[str, dict[str, Any]]:
    sym = symbol.upper()
    return {interval: _backfill_full_multi_interval(sym, interval, chunk) for interval in intervals}


def _backfill_full_multi_interval(symbol: str, interval: str, chunk: int) -> dict[str, Any]:
    before = _count_klines_multi(symbol, interval)
    state = _multi_backfill_state(symbol, interval)
    while True:
        rows = fetch_klines(symbol, interval, limit=min(1000, chunk), end_time=state["end_time"])
        if not rows:
            if state["current"] == 0:
                raise ValueError(f"no historical klines_multi {interval} rows returned for {symbol}")
            break
        _upsert_klines_multi(symbol, interval, rows)
        oldest = min(int(row["openTime"]) for row in rows)
        if state["end_time"] is not None and oldest >= state["end_time"]:
            raise ValueError(f"historical klines_multi {interval} backfill did not move earlier for {symbol}")
        state = {"current": _count_klines_multi(symbol, interval), "end_time": oldest - 1}
    return {"before": before, "after": state["current"], "fullHistory": True}


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


def _count_klines_multi(symbol: str, interval: str) -> int:
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM klines_multi WHERE symbol = ? AND interval = ?",
            (symbol.upper(), interval),
        ).fetchone()
    finally:
        conn.close()
    return int(row["c"])


def _multi_backfill_state(symbol: str, interval: str) -> dict[str, int | None]:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c, MIN(open_time) AS min_open_time
            FROM klines_multi
            WHERE symbol = ? AND interval = ?
            """,
            (symbol.upper(), interval),
        ).fetchone()
    finally:
        conn.close()
    oldest = row["min_open_time"]
    return {"current": int(row["c"]), "end_time": int(oldest) - 1 if oldest is not None else None}


def _local_1m_rows(symbol: str) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time, open, high, low, close, volume, close_time
            FROM klines
            WHERE symbol = ? AND interval = '1m'
            ORDER BY open_time ASC
            """,
            (symbol.upper(),),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "openTime": int(row["open_time"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "closeTime": int(row["close_time"]),
        }
        for row in rows
    ]


def _last_completed_entry_open_time(duration: str) -> int:
    current = current_rule_entry_open_time_for_duration(duration)
    return current - rule_interval_ms_for_duration(duration)


def _target_1m_rows() -> int:
    raw = os.getenv("MARKET_DATA_1M_TARGET_ROWS", str(DEFAULT_1M_TARGET_ROWS)).strip()
    try:
        return max(10_000, int(raw))
    except ValueError:
        return DEFAULT_1M_TARGET_ROWS
