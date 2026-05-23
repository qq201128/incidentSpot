from __future__ import annotations

import pytest

from app.services import market_context_ingest_service as service


def test_ingest_market_context_persists_realtime_rows(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(service, "fetch_open_interest_statistics", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "fetch_global_long_short_ratio", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "fetch_taker_buy_sell_volume", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "fetch_fear_greed_index", lambda **_kwargs: [])
    monkeypatch.setattr(service, "fetch_onchain_feature_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(service, "upsert_positioning_rows", lambda *_args: None)
    monkeypatch.setattr(service, "upsert_sentiment_rows", lambda *_args: None)
    monkeypatch.setattr(service, "upsert_onchain_rows", lambda *_args: None)
    monkeypatch.setattr(service, "fetch_orderbook", lambda *_args, **_kwargs: _orderbook_snapshot())
    monkeypatch.setattr(service, "fetch_funding_rate", lambda _symbol: 0.0001)
    monkeypatch.setattr(service, "orderbook_rule_score", lambda orderbook: orderbook)
    monkeypatch.setattr(service, "current_rule_entry_open_time_for_duration", _open_time_for_duration)
    monkeypatch.setattr(service, "persist_orderbook_features", _capture_orderbook(captured))
    monkeypatch.setattr(service, "upsert_funding_rows", _capture_funding(captured))

    report = service.ingest_market_context_data("btcusdt", durations=("10m", "30m"))

    assert report["orderbookRows"] == 2
    assert report["fundingRows"] == 2
    assert captured["orderbookOpenTimes"] == [10, 30]
    assert captured["fundingRows"] == [
        {"open_time": 10, "funding_rate": 0.0001},
        {"open_time": 30, "funding_rate": 0.0001},
    ]


def test_ingest_market_context_exposes_missing_funding(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_orderbook", lambda *_args, **_kwargs: _orderbook_snapshot())
    monkeypatch.setattr(service, "fetch_funding_rate", lambda _symbol: None)
    monkeypatch.setattr(service, "orderbook_rule_score", lambda orderbook: orderbook)
    monkeypatch.setattr(service, "current_rule_entry_open_time_for_duration", _open_time_for_duration)

    with pytest.raises(ValueError, match="funding rate unavailable"):
        service._persist_realtime_market_rows("BTCUSDT", ("10m",))


def _orderbook_snapshot() -> dict:
    return {"score": 1.0, "quoteTime": 1}


def _open_time_for_duration(duration: str) -> int:
    return {"10m": 10, "30m": 30}[duration]


def _capture_orderbook(captured: dict):
    def persist(_symbol: str, open_time: int, _orderbook: dict) -> None:
        rows = captured.setdefault("orderbookOpenTimes", [])
        rows.append(open_time)

    return persist


def _capture_funding(captured: dict):
    def persist(_symbol: str, rows: list[dict]) -> None:
        captured["fundingRows"] = rows

    return persist
