from __future__ import annotations

import pytest

from app.services import settlement_service as service

EVENT_END_MS = 1778122220848
EVENT_END_TIME = "2026-05-07T02:50:20.848944+00:00"
LIVE_QUOTE_TIME = EVENT_END_MS + 400
NEAR_QUOTE_TIME = EVENT_END_MS - 900
FALLBACK_QUOTE_TIME = EVENT_END_MS - 7_844
INDEX_PRICE = 81055.30021739


def test_fetch_settlement_quote_prefers_live_premium_index(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", lambda _symbol: _premium_index(LIVE_QUOTE_TIME))
    monkeypatch.setattr(service, "nearest_index_price_tick", _unexpected_stored_tick)
    monkeypatch.setattr(service, "nearest_available_index_price_tick", _unexpected_fallback)

    quote = service._fetch_settlement_quote(_event())

    assert quote.price == INDEX_PRICE
    assert quote.quote_time_ms == LIVE_QUOTE_TIME
    assert quote.source == "premiumIndex.rest.current;driftMs=400"


def test_fetch_settlement_quote_uses_precise_tick_when_live_fetch_fails(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", _failed_live_quote)
    monkeypatch.setattr(service, "nearest_index_price_tick", lambda *_args: _tick(NEAR_QUOTE_TIME))
    monkeypatch.setattr(service, "nearest_available_index_price_tick", _unexpected_fallback)

    quote = service._fetch_settlement_quote(_event())

    assert quote.price == INDEX_PRICE
    assert quote.quote_time_ms == NEAR_QUOTE_TIME
    assert quote.source == "premiumIndex.tick.nearest_endTime;driftMs=900"


def test_fetch_settlement_quote_falls_back_to_nearest_available_tick(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", _failed_live_quote)
    monkeypatch.setattr(service, "nearest_index_price_tick", lambda *_args: None)
    monkeypatch.setattr(service, "nearest_available_index_price_tick", lambda *_args: _tick(FALLBACK_QUOTE_TIME))

    quote = service._fetch_settlement_quote(_event())

    assert quote.price == INDEX_PRICE
    assert quote.quote_time_ms == FALLBACK_QUOTE_TIME
    assert quote.source == "premiumIndex.tick.nearest_available.after_live_failure;driftMs=7844"


def test_fetch_settlement_quote_still_exposes_missing_price_data(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", _failed_live_quote)
    monkeypatch.setattr(service, "nearest_index_price_tick", lambda *_args: None)
    monkeypatch.setattr(service, "nearest_available_index_price_tick", lambda *_args: None)

    with pytest.raises(ValueError, match="no stored index price tick"):
        service._fetch_settlement_quote(_event())


def _event() -> dict:
    return {"symbol": "BTCUSDT", "end_time": EVENT_END_TIME}


def _tick(quote_time: int) -> dict:
    return {"quote_time": quote_time, "index_price": INDEX_PRICE}


def _premium_index(quote_time: int) -> dict:
    return {"time": quote_time, "indexPrice": INDEX_PRICE}


def _failed_live_quote(_symbol: str) -> None:
    raise RuntimeError("live quote unavailable")


def _unexpected_stored_tick(*_args) -> None:
    raise AssertionError("stored tick should not be called when live quote succeeds")


def _unexpected_fallback(*_args) -> None:
    raise AssertionError("fallback should not be called when precise tick exists")
