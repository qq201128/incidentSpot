from __future__ import annotations

import logging
import os
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.binance_service import fetch_klines
from app.services.kline_backfill import backfill_1m_history, count_klines, upsert_klines_rows
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
KLINES_MULTI_INTERVALS = ("5m", "15m", "1h")
DEFAULT_MARKET_CONTEXT_LIMIT = 1500


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
        report["klinesByDuration"][duration] = backfill_duration_klines(sym, duration)

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


def _last_completed_entry_open_time(duration: str) -> int:
    current = current_rule_entry_open_time_for_duration(duration)
    return current - rule_interval_ms_for_duration(duration)


def _target_1m_rows() -> int:
    raw = os.getenv("MARKET_DATA_1M_TARGET_ROWS", str(DEFAULT_1M_TARGET_ROWS)).strip()
    try:
        return max(10_000, int(raw))
    except ValueError:
        return DEFAULT_1M_TARGET_ROWS
