from __future__ import annotations

import time

MS_PER_SECOND = 1000
SECONDS_PER_MINUTE = 60
ONE_MINUTE_MS = SECONDS_PER_MINUTE * MS_PER_SECOND


def raw_klines_from_response(rows: list) -> list[dict]:
    klines: list[dict] = []
    for item in rows:
        klines.append(
            {
                "openTime": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "closeTime": int(item[6]),
            }
        )
    return klines


def aggregate_1m_klines(rows_1m: list[dict], bar_ms: int) -> list[dict]:
    if not rows_1m:
        return []
    rows = sorted(rows_1m, key=lambda row: row["openTime"])
    out: list[dict] = []
    cur: dict | None = None
    bucket: int | None = None
    for row in rows:
        open_time = int(row["openTime"])
        bucket_open = (open_time // bar_ms) * bar_ms
        if bucket is None or bucket_open != bucket:
            cur = _append_started_bucket(out, cur, row, bucket_open=bucket_open)
            bucket = bucket_open
            continue
        _merge_into_bucket(cur, row)
    if cur is not None:
        out.append(cur)
    return out


def trim_incomplete_edge_aggregates(
    raw_rows_asc: list[dict],
    aggregated: list[dict],
    bar_ms: int,
    *,
    now_ms: int | None = None,
) -> list[dict]:
    trimmed = trim_leading_aggregate_if_first_bucket_incomplete(raw_rows_asc, aggregated, bar_ms)
    return trim_trailing_aggregate_if_last_bucket_incomplete(
        raw_rows_asc,
        trimmed,
        bar_ms,
        now_ms=now_ms,
    )


def trim_leading_aggregate_if_first_bucket_incomplete(
    raw_rows_asc: list[dict],
    aggregated: list[dict],
    bar_ms: int,
) -> list[dict]:
    if not aggregated or not raw_rows_asc:
        return aggregated
    first_open = int(raw_rows_asc[0]["openTime"])
    bucket_start = (first_open // bar_ms) * bar_ms
    if first_open > bucket_start:
        return aggregated[1:]
    return aggregated


def trim_trailing_aggregate_if_last_bucket_incomplete(
    raw_rows_asc: list[dict],
    aggregated: list[dict],
    bar_ms: int,
    *,
    now_ms: int | None = None,
) -> list[dict]:
    if not aggregated or not raw_rows_asc:
        return aggregated
    rows = sorted(raw_rows_asc, key=lambda row: row["openTime"])
    if not _last_bucket_has_expected_rows(rows, bar_ms):
        return aggregated[:-1]
    if not _last_one_minute_closed(rows[-1], now_ms):
        return aggregated[:-1]
    return aggregated


def _append_started_bucket(
    out: list[dict],
    cur: dict | None,
    row: dict,
    *,
    bucket_open: int,
) -> dict:
    if cur is not None:
        out.append(cur)
    return {
        "openTime": bucket_open,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
        "closeTime": int(row["closeTime"]),
    }


def _merge_into_bucket(cur: dict | None, row: dict) -> None:
    if cur is None:
        raise ValueError("cannot merge 1m kline without an active aggregate bucket")
    cur["high"] = max(cur["high"], float(row["high"]))
    cur["low"] = min(cur["low"], float(row["low"]))
    cur["close"] = float(row["close"])
    cur["volume"] += float(row["volume"])
    cur["closeTime"] = int(row["closeTime"])


def _last_bucket_has_expected_rows(rows_asc: list[dict], bar_ms: int) -> bool:
    expected_rows = bar_ms // ONE_MINUTE_MS
    last_open = int(rows_asc[-1]["openTime"])
    bucket_start = (last_open // bar_ms) * bar_ms
    bucket_rows = [row for row in rows_asc if int(row["openTime"]) >= bucket_start]
    return len(bucket_rows) == expected_rows and int(bucket_rows[0]["openTime"]) == bucket_start


def _last_one_minute_closed(last_row: dict, now_ms: int | None) -> bool:
    current_ms = now_ms if now_ms is not None else int(time.time() * MS_PER_SECOND)
    return int(last_row["closeTime"]) < current_ms
