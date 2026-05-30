from __future__ import annotations

import time

from app.services import binance_market_data as market_data
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses


def test_premium_index_display_uses_cache(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = {"count": 0}

    def fake_retry_get(url: str, params: dict, *, max_attempts: int, timeout):
        calls["count"] += 1
        return {
            "symbol": params["symbol"],
            "markPrice": "100.0",
            "indexPrice": "101.0",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1,
            "time": 1_700_000_000_000,
        }

    monkeypatch.setattr(market_data, "retry_get", fake_retry_get)
    monkeypatch.setattr(market_data, "_maybe_persist_premium_index", lambda *_args: None)
    market_data.LAST_PREMIUM_INDEX.clear()

    first = market_data.get_premium_index_display("BTCUSDT")
    second = market_data.get_premium_index_display("BTCUSDT")

    assert first["indexPrice"] == 101.0
    assert second["indexPrice"] == 101.0
    assert calls["count"] == 1
    status = background_loop_statuses()["premium_index_fetch"]
    assert status["status"] == "passed"
    assert status["lastSuccessDetails"] == {"stage": "fetch_premium_index", "symbol": "BTCUSDT"}


def test_premium_index_display_returns_stale_on_fetch_error(monkeypatch) -> None:
    reset_background_loop_statuses()
    market_data.LAST_PREMIUM_INDEX.clear()
    market_data.LAST_PREMIUM_INDEX["BTCUSDT"] = (
        time.monotonic() - 5.0,
        {
            "symbol": "BTCUSDT",
            "markPrice": 100.0,
            "indexPrice": 99.5,
            "lastFundingRate": 0.0,
            "nextFundingTime": 0,
            "time": 1,
        },
    )

    def fail_retry_get(*_args, **_kwargs):
        raise TimeoutError("network timeout")

    monkeypatch.setattr(market_data, "retry_get", fail_retry_get)

    row = market_data.get_premium_index_display("BTCUSDT")

    assert row["indexPrice"] == 99.5
    assert row["stale"] is True
    assert row["staleReason"] == "network timeout"
    assert row["staleExceptionType"] == "TimeoutError"
    status = background_loop_statuses()["premium_index_fetch"]
    assert status["status"] == "failed"
    assert status["lastError"] == "network timeout"
    assert status["lastFailureDetails"] == {"stage": "fetch_premium_index", "symbol": "BTCUSDT"}


def test_premium_index_persist_failure_is_visible(monkeypatch) -> None:
    reset_background_loop_statuses()
    market_data._LAST_PREMIUM_PERSIST_AT.clear()

    def fail_persist(_result: dict) -> None:
        raise RuntimeError("readonly database")

    monkeypatch.setattr(market_data, "persist_index_price_tick", fail_persist)

    market_data._maybe_persist_premium_index("BTCUSDT", 100.0, {"symbol": "BTCUSDT"})

    status = background_loop_statuses()["premium_index_persist"]
    assert status["status"] == "failed"
    assert status["lastError"] == "readonly database"
    assert status["lastFailureDetails"] == {"stage": "persist_index_price_tick", "symbol": "BTCUSDT"}


def test_premium_index_persist_success_is_visible(monkeypatch) -> None:
    reset_background_loop_statuses()
    market_data._LAST_PREMIUM_PERSIST_AT.clear()
    calls = []

    monkeypatch.setattr(market_data, "persist_index_price_tick", lambda result: calls.append(result))

    market_data._maybe_persist_premium_index("BTCUSDT", 100.0, {"symbol": "BTCUSDT"})

    assert calls == [{"symbol": "BTCUSDT"}]
    status = background_loop_statuses()["premium_index_persist"]
    assert status["status"] == "passed"
    assert status["lastSuccessDetails"] == {"stage": "persist_index_price_tick", "symbol": "BTCUSDT"}
