from __future__ import annotations

from app.services.binance_service import (
    _aggregate_1m_klines,
    _mark_forming_tail,
    _trim_leading_aggregate_if_first_bucket_incomplete,
    _trim_trailing_aggregate_if_last_bucket_incomplete,
)

TEN_MS = 10 * 60 * 1000
ONE_M_MS = 60 * 1000


def _aligned_10m_open(ts: int) -> int:
    return (int(ts) // TEN_MS) * TEN_MS


def _bar(open_time: int, o: float, h: float, l: float, c: float) -> dict:
    return {
        "openTime": open_time,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": 0.0,
        "closeTime": open_time + ONE_M_MS - 1,
    }


def test_aggregate_ten_one_minute_rows_full_bucket() -> None:
    base = _aligned_10m_open(1_720_000_000_000)
    rows = [_bar(base + i * ONE_M_MS, 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10)]
    agg = _aggregate_1m_klines(rows, TEN_MS)
    assert len(agg) == 1
    assert agg[0]["openTime"] == base
    assert agg[0]["open"] == 100.0
    assert agg[0]["close"] == 109.5


def test_trim_when_first_one_m_starts_mid_bucket() -> None:
    """Oldest row at minute +3 of bucket: first merged bar's open is meaningless vs exchange."""
    base = _aligned_10m_open(1_720_000_000_000)
    partial = [_bar(base + i * ONE_M_MS, 100 + i, 101 + i, 99 + i, 100 + i) for i in range(3, 10)]
    full_next = [_bar(base + TEN_MS + i * ONE_M_MS, 200 + i, 201 + i, 199 + i, 200 + i) for i in range(10)]
    rows = partial + full_next
    agg = _aggregate_1m_klines(rows, TEN_MS)
    trimmed = _trim_leading_aggregate_if_first_bucket_incomplete(rows, agg, TEN_MS)
    assert len(trimmed) == 1
    assert trimmed[0]["openTime"] == base + TEN_MS
    assert trimmed[0]["open"] == 200.0


def test_no_trim_when_series_starts_at_bucket_open() -> None:
    base = _aligned_10m_open(1_720_000_000_000)
    rows = [_bar(base + i * ONE_M_MS, 50, 51, 49, 50.5) for i in range(10)]
    agg = _aggregate_1m_klines(rows, TEN_MS)
    trimmed = _trim_leading_aggregate_if_first_bucket_incomplete(rows, agg, TEN_MS)
    assert trimmed == agg
    assert len(trimmed) == 1


def test_trim_when_last_bucket_is_missing_closing_minutes() -> None:
    base = _aligned_10m_open(1_720_000_000_000)
    full = [_bar(base + i * ONE_M_MS, 100, 101, 99, 100.5) for i in range(10)]
    partial = [_bar(base + TEN_MS + i * ONE_M_MS, 200, 201, 199, 200.5) for i in range(3)]
    rows = full + partial
    agg = _aggregate_1m_klines(rows, TEN_MS)
    trimmed = _trim_trailing_aggregate_if_last_bucket_incomplete(
        rows,
        agg,
        TEN_MS,
        now_ms=base + TEN_MS * 2,
    )
    assert len(trimmed) == 1
    assert trimmed[0]["openTime"] == base


def test_trim_when_last_one_minute_is_not_closed_yet() -> None:
    base = _aligned_10m_open(1_720_000_000_000)
    rows = [_bar(base + i * ONE_M_MS, 100, 101, 99, 100.5) for i in range(10)]
    agg = _aggregate_1m_klines(rows, TEN_MS)
    trimmed = _trim_trailing_aggregate_if_last_bucket_incomplete(
        rows,
        agg,
        TEN_MS,
        now_ms=rows[-1]["closeTime"],
    )
    assert trimmed == []


def test_no_trim_when_last_bucket_is_complete_and_closed() -> None:
    base = _aligned_10m_open(1_720_000_000_000)
    rows = [_bar(base + i * ONE_M_MS, 100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(20)]
    agg = _aggregate_1m_klines(rows, TEN_MS)
    trimmed = _trim_trailing_aggregate_if_last_bucket_incomplete(
        rows,
        agg,
        TEN_MS,
        now_ms=rows[-1]["closeTime"] + ONE_M_MS,
    )
    assert trimmed == agg
    assert len(trimmed) == 2


def test_mark_forming_tail_flags_current_bucket() -> None:
    import time

    bar_ms = TEN_MS
    now_ms = int(time.time() * 1000)
    bucket = (now_ms // bar_ms) * bar_ms
    rows = [
        {
            "openTime": bucket - bar_ms,
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "volume": 0.0,
            "closeTime": bucket - 1,
        },
        {
            "openTime": bucket,
            "open": 2.0,
            "high": 3.0,
            "low": 1.5,
            "close": 2.5,
            "volume": 0.0,
            "closeTime": now_ms,
        },
    ]
    marked = _mark_forming_tail(rows, bar_ms)
    assert marked[-1]["isClosed"] is False
    assert marked[0].get("isClosed") is None
