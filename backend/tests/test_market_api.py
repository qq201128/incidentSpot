from __future__ import annotations

from fastapi import BackgroundTasks

from app.api import market
from app.services import market_data_backfill_service
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses


def test_market_backfill_queues_visible_background_task(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = []

    def fake_backfill(symbol: str, *, durations: tuple[str, ...] | None) -> dict:
        calls.append((symbol, durations))
        return {"durations": list(durations or ("10m", "30m"))}

    monkeypatch.setattr(market, "backfill_symbol_market_data", fake_backfill)

    tasks = BackgroundTasks()
    response = market.backfill_market_data(tasks, symbol="btcusdt", duration="10m")

    assert response["ok"] is True
    assert response["symbol"] == "BTCUSDT"
    assert response["duration"] == "10m"
    assert len(tasks.tasks) == 1

    tasks.tasks[0].func(*tasks.tasks[0].args, **tasks.tasks[0].kwargs)

    assert calls == [("BTCUSDT", ("10m",))]
    status = background_loop_statuses()["market_backfill"]
    assert status["status"] == "passed"
    assert status["lastSuccessDetails"] == {
        "stage": "manual_api_backfill",
        "symbol": "BTCUSDT",
        "durations": ["10m"],
        "durationCount": 1,
    }


def test_market_backfill_background_failure_is_visible(monkeypatch) -> None:
    reset_background_loop_statuses()

    def fake_backfill(symbol: str, *, durations: tuple[str, ...] | None) -> dict:
        raise RuntimeError(f"backfill failed for {symbol} {durations}")

    monkeypatch.setattr(market, "backfill_symbol_market_data", fake_backfill)

    try:
        market._background_market_backfill("BTCUSDT", ("10m",))
    except RuntimeError as exc:
        assert "backfill failed for BTCUSDT" in str(exc)
    else:
        raise AssertionError("market backfill failure was swallowed")

    status = background_loop_statuses()["market_backfill"]
    assert status["status"] == "failed"
    assert status["lastExceptionType"] == "RuntimeError"
    assert status["lastFailureDetails"] == {
        "stage": "manual_api_backfill",
        "symbol": "BTCUSDT",
        "durations": ["10m"],
    }


def test_market_backfill_records_1m_fetch_failure(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(market_data_backfill_service, "count_klines", lambda *_args: 10)
    monkeypatch.setattr(market_data_backfill_service, "backfill_1m_history", _fail_1m_backfill)

    try:
        market._background_market_backfill("BTCUSDT", ("10m",))
    except RuntimeError as exc:
        assert str(exc) == "failed to fetch 1m klines"
    else:
        raise AssertionError("1m backfill fetch failure was swallowed")

    status = background_loop_statuses()["market_backfill"]
    assert status["status"] == "failed"
    assert status["lastError"] == "failed to fetch 1m klines"
    assert status["lastFailureDetails"]["stage"] == "manual_api_backfill"


def _fail_1m_backfill(*_args, **_kwargs) -> int:
    raise RuntimeError("failed to fetch 1m klines")
