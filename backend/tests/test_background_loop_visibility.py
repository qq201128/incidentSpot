from __future__ import annotations

from app.services import (
    auto_settlement_service,
    factor_ranking_background,
    market_context_background,
    settlement_service,
)
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses


def test_factor_ranking_symbol_failure_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(factor_ranking_background, "factor_ranking_precomputed_symbols", lambda: ["BTCUSDT"])
    monkeypatch.setattr(
        factor_ranking_background,
        "refresh_symbol_rankings",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("ranking failed")),
    )

    factor_ranking_background.refresh_all_configured_rankings()

    status = background_loop_statuses()["factor_ranking"]
    assert status["status"] == "failed"
    assert status["lastError"] == "factor ranking failed for symbols"
    assert status["lastFailureDetails"]["failedSymbols"] == ["BTCUSDT"]


def test_factor_ranking_refresh_backfills_dependencies_before_ranking(monkeypatch) -> None:
    calls = []

    def fake_dependencies(symbol: str, duration: str) -> None:
        calls.append(("dependencies", symbol, duration))

    def fake_report(symbol: str, duration: str, category: str | None) -> dict:
        calls.append(("rank", symbol, duration, category))
        return {
            "ranking": [{"factorName": "factor_a"}],
            "rankingDiagnostics": {"rankedFactorCount": 1},
            "rankingFailures": [],
        }

    def fake_save(symbol: str, duration: str, ranking: list[dict], **kwargs) -> None:
        calls.append(("save", symbol, duration, ranking, kwargs))

    monkeypatch.setattr(factor_ranking_background, "refresh_factor_combination_data_dependencies", fake_dependencies)
    monkeypatch.setattr(factor_ranking_background, "run_factor_ranking_report", fake_report)
    monkeypatch.setattr(factor_ranking_background, "save_cached_ranking", fake_save)

    factor_ranking_background.refresh_ranking_for_symbol_duration("ethusdt", "10m")

    assert calls == [
        ("dependencies", "ETHUSDT", "10m"),
        ("rank", "ETHUSDT", "10m", None),
        (
            "save",
            "ETHUSDT",
            "10m",
            [{"factorName": "factor_a"}],
            {"diagnostics": {"rankedFactorCount": 1}, "failures": []},
        ),
    ]


def test_auto_settlement_event_failure_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(
        auto_settlement_service,
        "scan_due_open_events",
        lambda: settlement_service.DueOpenEventScan(due_ids=[123], invalid_events=[]),
    )
    monkeypatch.setattr(
        auto_settlement_service,
        "settle_event",
        lambda _event_id: (_ for _ in ()).throw(RuntimeError("settle failed")),
    )

    async def run_once() -> None:
        import asyncio

        stop = asyncio.Event()

        async def stop_after_first_wait(awaitable, **_kwargs):
            awaitable.close()
            stop.set()

        monkeypatch.setattr(auto_settlement_service.asyncio, "wait_for", stop_after_first_wait)
        await auto_settlement_service.auto_settlement_loop(stop, poll_seconds=1)

    import asyncio

    asyncio.run(run_once())

    status = background_loop_statuses()["auto_settlement"]
    assert status["status"] == "failed"
    assert status["lastError"] == "auto settlement failed for events"
    assert status["lastFailureDetails"]["failedEventIds"] == [123]


def test_auto_settlement_invalid_event_time_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    invalid = {
        "eventId": 456,
        "endTime": "bad-time",
        "error": "Invalid isoformat string: 'bad-time'",
        "exceptionType": "ValueError",
    }
    monkeypatch.setattr(
        auto_settlement_service,
        "scan_due_open_events",
        lambda: settlement_service.DueOpenEventScan(due_ids=[], invalid_events=[invalid]),
    )

    async def run_once() -> None:
        import asyncio

        stop = asyncio.Event()

        async def stop_after_first_wait(awaitable, **_kwargs):
            awaitable.close()
            stop.set()

        monkeypatch.setattr(auto_settlement_service.asyncio, "wait_for", stop_after_first_wait)
        await auto_settlement_service.auto_settlement_loop(stop, poll_seconds=1)

    import asyncio

    asyncio.run(run_once())

    status = background_loop_statuses()["auto_settlement"]
    assert status["status"] == "failed"
    assert status["lastError"] == "auto settlement failed for events"
    assert status["lastFailureDetails"]["invalidEvents"] == [invalid]


def test_factor_ranking_invalid_interval_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setenv("FACTOR_RANKING_REFRESH_SECONDS", "bad")

    async def run_loop() -> None:
        import asyncio

        await factor_ranking_background.factor_ranking_refresh_loop(asyncio.Event())

    import asyncio

    try:
        asyncio.run(run_loop())
    except ValueError as exc:
        assert "FACTOR_RANKING_REFRESH_SECONDS must be numeric" in str(exc)
    else:
        raise AssertionError("invalid factor ranking interval was not exposed")

    status = background_loop_statuses()["factor_ranking"]
    assert status["status"] == "failed"
    assert status["lastFailureDetails"]["stage"] == "startup_config"


def test_market_context_invalid_initial_delay_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setenv("MARKET_CONTEXT_INITIAL_DELAY_SECONDS", "bad")

    async def run_loop() -> None:
        import asyncio

        await market_context_background.market_context_refresh_loop(asyncio.Event())

    import asyncio

    try:
        asyncio.run(run_loop())
    except ValueError as exc:
        assert "MARKET_CONTEXT_INITIAL_DELAY_SECONDS must be numeric" in str(exc)
    else:
        raise AssertionError("invalid market context initial delay was not exposed")

    status = background_loop_statuses()["market_context"]
    assert status["status"] == "failed"
    assert status["lastFailureDetails"]["stage"] == "startup_config"


def test_market_context_batch_failure_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(market_context_background, "_refresh_interval_seconds", lambda: 60.0)
    monkeypatch.setattr(market_context_background, "_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(
        market_context_background,
        "run_blocking_daemon",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("market batch failed")),
    )

    async def stop_after_batch(stop_event, _seconds):
        stop_event.set()
        return True

    monkeypatch.setattr(market_context_background, "_sleep_for", stop_after_batch)

    async def run_loop() -> None:
        import asyncio

        await market_context_background.market_context_refresh_loop(asyncio.Event())

    import asyncio

    asyncio.run(run_loop())

    status = background_loop_statuses()["market_context"]
    assert status["status"] == "failed"
    assert status["lastError"] == "market batch failed"
    assert status["lastFailureDetails"]["stage"] == "batch"


def test_market_context_initial_stop_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(market_context_background, "_refresh_interval_seconds", lambda: 60.0)
    monkeypatch.setattr(market_context_background, "_initial_delay_seconds", lambda: 5.0)
    monkeypatch.setattr(market_context_background, "run_blocking_daemon", _unexpected_background_run)

    async def stop_during_delay(stop_event, _seconds):
        stop_event.set()
        return True

    monkeypatch.setattr(market_context_background, "_sleep_for", stop_during_delay)

    async def run_loop() -> None:
        import asyncio

        await market_context_background.market_context_refresh_loop(asyncio.Event())

    import asyncio

    asyncio.run(run_loop())

    status = background_loop_statuses()["market_context"]
    assert status["status"] == "stopped"
    assert status["stopReason"] == "stop_during_initial_delay"


async def _unexpected_background_run(*_args):
    raise AssertionError("background batch should not run")
