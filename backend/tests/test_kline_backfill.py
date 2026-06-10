from __future__ import annotations

import pytest

from app.services import kline_backfill
from app.services.kline_backfill import KlineBackfillError


def test_backfill_1m_history_raises_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kline_backfill, "count_klines", lambda *_args: 10)
    monkeypatch.setattr(kline_backfill, "oldest_open_time", lambda *_args: 1000)

    def fail_fetch(*_args, **_kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(kline_backfill, "fetch_klines", fail_fetch)

    with pytest.raises(KlineBackfillError, match="network unavailable"):
        kline_backfill.backfill_1m_history("btcusdt", target_rows=20)


def test_backfill_1m_history_raises_when_target_not_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = iter([10, 11])
    monkeypatch.setattr(kline_backfill, "count_klines", lambda *_args: next(counts))
    monkeypatch.setattr(kline_backfill, "oldest_open_time", lambda *_args: 1000)
    monkeypatch.setattr(kline_backfill, "fetch_klines", lambda *_args, **_kwargs: [_row(900)])
    monkeypatch.setattr(kline_backfill, "upsert_klines_rows", lambda *_args: None)

    with pytest.raises(KlineBackfillError, match="11/20 rows"):
        kline_backfill.backfill_1m_history("btcusdt", target_rows=20, max_rounds=1)


def test_backfill_interval_history_full_runs_until_exchange_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = iter([0, 2])
    fetches = []
    monkeypatch.setattr(kline_backfill, "count_klines", lambda *_args: next(counts))
    monkeypatch.setattr(kline_backfill, "oldest_open_time", lambda *_args: None)

    def fetch(symbol: str, interval: str, **kwargs):
        fetches.append((symbol, interval, kwargs))
        return [_row(200), _row(100)] if len(fetches) == 1 else []

    monkeypatch.setattr(kline_backfill, "fetch_klines", fetch)
    monkeypatch.setattr(kline_backfill, "upsert_klines_rows", lambda *_args: None)

    result = kline_backfill.backfill_interval_history("btcusdt", "30m", target_rows=None, max_rounds=None)

    assert result == 2
    assert fetches == [
        ("BTCUSDT", "30m", {"limit": 1000, "end_time": None}),
        ("BTCUSDT", "30m", {"limit": 1000, "end_time": 99}),
    ]


def test_backfill_interval_history_full_raises_when_first_fetch_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kline_backfill, "count_klines", lambda *_args: 0)
    monkeypatch.setattr(kline_backfill, "oldest_open_time", lambda *_args: None)
    monkeypatch.setattr(kline_backfill, "fetch_klines", lambda *_args, **_kwargs: [])

    with pytest.raises(KlineBackfillError, match="no historical 30m klines"):
        kline_backfill.backfill_interval_history("btcusdt", "30m", target_rows=None, max_rounds=None)


def _row(open_time: int) -> dict:
    return {
        "openTime": open_time,
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 1.0,
        "closeTime": open_time + 1,
    }
