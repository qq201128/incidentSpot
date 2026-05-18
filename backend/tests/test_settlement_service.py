from __future__ import annotations

import pytest

from app.services import settlement_service as service

EVENT_END_MS = 1778122220848
EVENT_END_TIME = "2026-05-07T02:50:20.848944+00:00"
LIVE_QUOTE_TIME = EVENT_END_MS + 400
STALE_LIVE_QUOTE_TIME = EVENT_END_MS + 3_000
INDEX_PRICE = 81055.30021739


def test_fetch_settlement_quote_uses_live_current_premium_index(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", lambda _symbol: _premium_index(LIVE_QUOTE_TIME))

    quote = service._fetch_settlement_quote(_event())

    assert quote.price == INDEX_PRICE
    assert quote.quote_time_ms == LIVE_QUOTE_TIME
    assert quote.source == "premiumIndex.rest.current;driftMs=400"


def test_fetch_settlement_quote_accepts_current_price_without_drift_block(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", lambda _symbol: _premium_index(STALE_LIVE_QUOTE_TIME))

    quote = service._fetch_settlement_quote(_event())

    assert quote.price == INDEX_PRICE
    assert quote.quote_time_ms == STALE_LIVE_QUOTE_TIME
    assert quote.source == "premiumIndex.rest.current;driftMs=3000"


def test_fetch_settlement_quote_exposes_live_fetch_failure(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", _failed_live_quote)

    with pytest.raises(RuntimeError, match="live quote unavailable"):
        service._fetch_settlement_quote(_event())


def test_fetch_settlement_quote_exposes_invalid_live_price(monkeypatch) -> None:
    monkeypatch.setattr(service, "fetch_premium_index", lambda _symbol: {"time": LIVE_QUOTE_TIME, "indexPrice": 0})

    with pytest.raises(ValueError, match="premium index settlement response is invalid"):
        service._fetch_settlement_quote(_event())


def _event() -> dict:
    return {"symbol": "BTCUSDT", "end_time": EVENT_END_TIME}


def _premium_index(quote_time: int) -> dict:
    return {"time": quote_time, "indexPrice": INDEX_PRICE}


def _failed_live_quote(_symbol: str) -> None:
    raise RuntimeError("live quote unavailable")
