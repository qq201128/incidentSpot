from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from app.db.session import get_conn, run_db_write_with_retry
from app.services.binance_service import fetch_klines
from app.services.kline_backfill import upsert_klines_rows

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "10m": 600_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "60m": 3_600_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}
MAX_REFETCH_LIMIT = 1000
PRICE_JUMP_LIMIT = 0.20
REPAIR_SOURCE = "binance_refetch"


class MarketDataRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketDataIssue:
    issue_type: str
    start_open_time: int
    end_open_time: int
    reason: str
    details: dict[str, Any]


def repair_market_klines(
    symbol: str,
    interval: str,
    *,
    fetcher: Callable[..., list[dict]] = fetch_klines,
    upsert: Callable[[str, str, list[dict]], None] = upsert_klines_rows,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    issues = scan_market_klines(sym, interval)
    repaired = 0
    for issue in issues:
        try:
            rows = _refetch_issue(sym, interval, issue, fetcher)
            upsert(sym, interval, rows)
            _record_issue(sym, interval, issue, "repaired", None)
            repaired += 1
        except Exception as exc:
            _record_issue(sym, interval, issue, "failed", str(exc))
            raise MarketDataRepairError(_repair_failure(sym, interval, issue, exc)) from exc
    return {"symbol": sym, "interval": interval, "issues": len(issues), "repaired": repaired}


def scan_market_klines(symbol: str, interval: str) -> list[MarketDataIssue]:
    step = _interval_ms(interval)
    rows = _kline_rows(symbol, interval)
    issues = []
    issues.extend(_gap_issues(rows, step))
    issues.extend(_invalid_ohlc_issues(rows))
    issues.extend(_price_jump_issues(rows))
    return issues


def _kline_rows(symbol: str, interval: str) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time, open, high, low, close, volume
            FROM klines
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time ASC
            """,
            (symbol.strip().upper(), interval),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _gap_issues(rows: list[dict[str, Any]], step: int) -> list[MarketDataIssue]:
    issues = []
    for prev, cur in zip(rows, rows[1:]):
        expected = int(prev["open_time"]) + step
        current = int(cur["open_time"])
        if current > expected:
            issues.append(_issue("gap", expected, current - step, "missing_kline_range"))
        if current == int(prev["open_time"]):
            issues.append(_issue("duplicate", current, current, "duplicate_open_time"))
    return issues


def _invalid_ohlc_issues(rows: list[dict[str, Any]]) -> list[MarketDataIssue]:
    return [
        _issue("invalid_ohlc", int(row["open_time"]), int(row["open_time"]), "invalid_ohlc_values", row)
        for row in rows
        if not _valid_ohlc(row)
    ]


def _price_jump_issues(rows: list[dict[str, Any]]) -> list[MarketDataIssue]:
    issues = []
    for prev, cur in zip(rows, rows[1:]):
        prev_close = float(prev["close"] or 0)
        cur_close = float(cur["close"] or 0)
        if prev_close <= 0 or cur_close <= 0:
            continue
        jump = abs(cur_close / prev_close - 1.0)
        if jump > PRICE_JUMP_LIMIT:
            issues.append(_issue("price_jump", int(prev["open_time"]), int(cur["open_time"]), "price_jump_exceeds_limit", {"jump": jump}))
    return issues


def _valid_ohlc(row: dict[str, Any]) -> bool:
    try:
        open_, high, low, close = (float(row[key]) for key in ("open", "high", "low", "close"))
    except (TypeError, ValueError):
        return False
    if min(open_, high, low, close) <= 0:
        return False
    return low <= min(open_, close) and high >= max(open_, close) and low <= high


def _refetch_issue(symbol: str, interval: str, issue: MarketDataIssue, fetcher) -> list[dict]:
    step = _interval_ms(interval)
    limit = min(((issue.end_open_time - issue.start_open_time) // step) + 1, MAX_REFETCH_LIMIT)
    rows = fetcher(
        symbol,
        interval,
        limit=limit,
        start_time=issue.start_open_time,
        end_time=issue.end_open_time + step - 1,
    )
    if not rows:
        raise MarketDataRepairError("binance returned no rows for repair range")
    return rows


def _record_issue(
    symbol: str,
    interval: str,
    issue: MarketDataIssue,
    status: str,
    error: str | None,
) -> None:
    def _write() -> None:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO market_data_quality_reports(
                  symbol, interval, issue_type, start_open_time, end_open_time,
                  status, reason, repair_source, details_json, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol, interval, issue.issue_type, issue.start_open_time,
                    issue.end_open_time, status, error or issue.reason, REPAIR_SOURCE,
                    json.dumps(issue.details, ensure_ascii=True, sort_keys=True), _utc_now(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_write)


def _issue(
    issue_type: str,
    start: int,
    end: int,
    reason: str,
    details: dict[str, Any] | None = None,
) -> MarketDataIssue:
    return MarketDataIssue(issue_type, int(start), int(end), reason, details or {})


def _repair_failure(symbol: str, interval: str, issue: MarketDataIssue, exc: Exception) -> str:
    return (
        f"failed to repair {symbol} {interval} {issue.issue_type} "
        f"{issue.start_open_time}-{issue.end_open_time}: {exc}"
    )


def _interval_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f"unsupported kline interval for repair: {interval}")
    return INTERVAL_MS[interval]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
