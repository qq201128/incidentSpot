from __future__ import annotations

from dataclasses import dataclass
from sqlite3 import Connection
from typing import Callable

from app.db.session import get_conn
from app.services.binance_service import fetch_klines
from app.services.kline_backfill import upsert_klines_rows
from app.services.kline_timing import MS_PER_MINUTE
from app.services.rule_config import DURATION_TO_MINUTES

INITIAL_KLINE_LIMIT = 1000
MAX_FETCH_LIMIT = 1000
VERIFY_LOOKBACK_BARS = 1000


@dataclass(frozen=True)
class KlineRefreshRequest:
    symbol: str
    interval: str
    required_open_time: int


@dataclass(frozen=True)
class KlineRefreshDeps:
    connect: Callable[[], Connection] = get_conn
    fetch: Callable[..., list[dict]] = fetch_klines
    upsert: Callable[[str, str, list[dict]], None] = upsert_klines_rows


@dataclass(frozen=True)
class MissingKlineRange:
    start_open_time: int
    end_open_time: int


def refresh_prediction_klines(
    symbol: str,
    interval: str,
    required_open_time: int,
) -> None:
    request = KlineRefreshRequest(symbol.upper(), interval, int(required_open_time))
    refresh_required_klines(request)


def refresh_required_klines(
    request: KlineRefreshRequest,
    deps: KlineRefreshDeps | None = None,
) -> None:
    active_deps = deps or KlineRefreshDeps()
    step_ms = _interval_ms(request.interval)
    latest_open_time = _latest_open_time(request, active_deps)
    if latest_open_time is None:
        _fetch_initial_snapshot(request, active_deps)
        latest_open_time = _latest_open_time(request, active_deps)
    if latest_open_time is not None and latest_open_time < request.required_open_time:
        _fetch_range(request, latest_open_time + step_ms, request.required_open_time, active_deps)
    _fill_missing_ranges(request, active_deps)
    _assert_required_window_ready(request, active_deps)


def _fetch_initial_snapshot(request: KlineRefreshRequest, deps: KlineRefreshDeps) -> None:
    rows = deps.fetch(request.symbol, request.interval, limit=INITIAL_KLINE_LIMIT)
    if not rows:
        raise ValueError(f"no latest {request.interval} klines returned for {request.symbol}")
    deps.upsert(request.symbol, request.interval, rows)


def _fill_missing_ranges(request: KlineRefreshRequest, deps: KlineRefreshDeps) -> None:
    for missing_range in _missing_ranges(request, deps):
        _fetch_range(
            request,
            missing_range.start_open_time,
            missing_range.end_open_time,
            deps,
        )


def _fetch_range(
    request: KlineRefreshRequest,
    start_open_time: int,
    end_open_time: int,
    deps: KlineRefreshDeps,
) -> None:
    step_ms = _interval_ms(request.interval)
    current = int(start_open_time)
    target = int(end_open_time)
    while current <= target:
        limit = _fetch_limit(current, target, step_ms)
        chunk_end = current + (limit - 1) * step_ms
        rows = deps.fetch(
            request.symbol,
            request.interval,
            limit=limit,
            start_time=current,
            end_time=chunk_end + step_ms - 1,
        )
        if not rows:
            raise ValueError(_empty_range_message(request, current, chunk_end))
        deps.upsert(request.symbol, request.interval, rows)
        current = chunk_end + step_ms


def _assert_required_window_ready(
    request: KlineRefreshRequest,
    deps: KlineRefreshDeps,
) -> None:
    if not _open_time_exists(request, deps):
        raise ValueError(
            f"missing completed {request.interval} kline at "
            f"{request.required_open_time} for {request.symbol}"
        )
    missing = _missing_ranges(request, deps)
    if missing:
        first = missing[0]
        raise ValueError(
            f"missing {request.interval} kline range "
            f"{first.start_open_time}-{first.end_open_time} for {request.symbol}"
        )


def _missing_ranges(
    request: KlineRefreshRequest,
    deps: KlineRefreshDeps,
) -> list[MissingKlineRange]:
    step_ms = _interval_ms(request.interval)
    oldest = _oldest_open_time(request, deps)
    if oldest is None:
        return [MissingKlineRange(request.required_open_time, request.required_open_time)]
    start_open_time = _verify_start_open_time(request.required_open_time, step_ms, oldest)
    existing = _existing_open_times(request, start_open_time, deps)
    missing: list[MissingKlineRange] = []
    range_start: int | None = None
    current = start_open_time
    while current <= request.required_open_time:
        if current not in existing and range_start is None:
            range_start = current
        if current in existing and range_start is not None:
            missing.append(MissingKlineRange(range_start, current - step_ms))
            range_start = None
        current += step_ms
    if range_start is not None:
        missing.append(MissingKlineRange(range_start, request.required_open_time))
    return missing


def _latest_open_time(
    request: KlineRefreshRequest,
    deps: KlineRefreshDeps,
) -> int | None:
    conn = deps.connect()
    try:
        row = conn.execute(
            "SELECT MAX(open_time) AS max_open_time FROM klines WHERE symbol = ? AND interval = ?",
            (request.symbol, request.interval),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["max_open_time"] is None:
        return None
    return int(row["max_open_time"])


def _oldest_open_time(
    request: KlineRefreshRequest,
    deps: KlineRefreshDeps,
) -> int | None:
    conn = deps.connect()
    try:
        row = conn.execute(
            "SELECT MIN(open_time) AS min_open_time FROM klines WHERE symbol = ? AND interval = ?",
            (request.symbol, request.interval),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["min_open_time"] is None:
        return None
    return int(row["min_open_time"])


def _existing_open_times(
    request: KlineRefreshRequest,
    start_open_time: int,
    deps: KlineRefreshDeps,
) -> set[int]:
    conn = deps.connect()
    try:
        rows = conn.execute(
            """
            SELECT open_time FROM klines
            WHERE symbol = ? AND interval = ? AND open_time BETWEEN ? AND ?
            ORDER BY open_time ASC
            """,
            (request.symbol, request.interval, int(start_open_time), request.required_open_time),
        ).fetchall()
    finally:
        conn.close()
    return {int(row["open_time"]) for row in rows}


def _open_time_exists(request: KlineRefreshRequest, deps: KlineRefreshDeps) -> bool:
    conn = deps.connect()
    try:
        row = conn.execute(
            """
            SELECT 1 FROM klines
            WHERE symbol = ? AND interval = ? AND open_time = ?
            """,
            (request.symbol, request.interval, request.required_open_time),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _fetch_limit(start_open_time: int, end_open_time: int, step_ms: int) -> int:
    expected_rows = ((int(end_open_time) - int(start_open_time)) // int(step_ms)) + 1
    return min(MAX_FETCH_LIMIT, expected_rows)


def _verify_start_open_time(required_open_time: int, step_ms: int, oldest_open_time: int) -> int:
    lookback_start = int(required_open_time) - ((VERIFY_LOOKBACK_BARS - 1) * int(step_ms))
    return max(lookback_start, int(oldest_open_time))


def _interval_ms(interval: str) -> int:
    if interval == "1m":
        return MS_PER_MINUTE
    return int(DURATION_TO_MINUTES[interval]) * MS_PER_MINUTE


def _empty_range_message(
    request: KlineRefreshRequest,
    start_open_time: int,
    end_open_time: int,
) -> str:
    return (
        f"no {request.interval} klines returned for {request.symbol} "
        f"range {start_open_time}-{end_open_time}"
    )
