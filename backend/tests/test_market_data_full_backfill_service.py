from __future__ import annotations

from app.services import market_data_backfill_service as service


def test_full_history_backfill_aggregates_10m_and_backfills_native_durations(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(service, "_backfill_1m", lambda *args, **kwargs: {"after": 12})
    monkeypatch.setattr(service, "sync_full_klines_multi", lambda symbol: {"5m": {"after": 12}})
    monkeypatch.setattr(service, "ingest_market_context_data", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(service, "backfill_funding_features", lambda *args: {"inserted": 1})
    monkeypatch.setattr(service, "backfill_orderbook_features", lambda *args: {"inserted": 1})

    def aggregate_10m(symbol: str) -> dict:
        calls.append(("aggregate_10m", symbol))
        return {"after": 10}

    def full_native(symbol: str, duration: str) -> dict:
        calls.append(("full_native", symbol, duration))
        return {"after": 20}

    monkeypatch.setattr(service, "aggregate_full_10m_from_1m", aggregate_10m)
    monkeypatch.setattr(service, "backfill_full_duration_klines", full_native)

    report = service.backfill_symbol_market_data(
        "btcusdt",
        durations=("10m", "30m"),
        full_history=True,
    )

    assert report["fullHistory"] is True
    assert report["klinesByDuration"]["10m"] == {"after": 10}
    assert report["klinesByDuration"]["30m"] == {"after": 20}
    assert calls == [("aggregate_10m", "BTCUSDT"), ("full_native", "BTCUSDT", "30m")]


def test_full_multi_interval_runs_until_empty(monkeypatch) -> None:
    states = iter([
        {"current": 0, "end_time": None},
        {"current": 2, "end_time": 99},
    ])
    fetches = []
    monkeypatch.setattr(service, "_multi_backfill_state", lambda *_args: next(states))
    monkeypatch.setattr(service, "_count_klines_multi", lambda *_args: 2)
    monkeypatch.setattr(service, "_upsert_klines_multi", lambda *_args: None)

    def fetch(symbol: str, interval: str, **kwargs):
        fetches.append((symbol, interval, kwargs))
        return [_row(200), _row(100)] if len(fetches) == 1 else []

    monkeypatch.setattr(service, "fetch_klines", fetch)

    report = service._backfill_full_multi_interval("BTCUSDT", "5m", 1000)

    assert report == {"before": 2, "after": 2, "fullHistory": True}
    assert fetches == [
        ("BTCUSDT", "5m", {"limit": 1000, "end_time": None}),
        ("BTCUSDT", "5m", {"limit": 1000, "end_time": 99}),
    ]


def _row(open_time: int) -> dict:
    return {"openTime": open_time}
