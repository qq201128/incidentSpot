from __future__ import annotations

import asyncio

from app.services import auto_settlement_service, auto_trade_service, settlement_service
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses


def test_auto_settlement_stop_between_scans_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(
        auto_settlement_service,
        "scan_due_open_events",
        lambda: settlement_service.DueOpenEventScan(due_ids=[], invalid_events=[]),
    )

    async def stop_after_scan(stop_event, _poll_seconds):
        stop_event.set()
        return True

    monkeypatch.setattr(auto_settlement_service, "_wait_for_next_poll", stop_after_scan)

    asyncio.run(auto_settlement_service.auto_settlement_loop(asyncio.Event(), poll_seconds=1))

    status = background_loop_statuses()["auto_settlement"]
    assert status["status"] == "stopped"
    assert status["stopReason"] == "stop_between_scans"


def test_auto_settlement_scan_runs_in_background_daemon(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = []

    async def fake_run_blocking_daemon(func):
        calls.append(func.__name__)
        return func()

    monkeypatch.setattr(auto_settlement_service, "run_blocking_daemon", fake_run_blocking_daemon)
    monkeypatch.setattr(
        auto_settlement_service,
        "scan_due_open_events",
        lambda: settlement_service.DueOpenEventScan(due_ids=[], invalid_events=[]),
    )

    async def stop_after_scan(stop_event, _poll_seconds):
        stop_event.set()
        return True

    monkeypatch.setattr(auto_settlement_service, "_wait_for_next_poll", stop_after_scan)

    asyncio.run(auto_settlement_service.auto_settlement_loop(asyncio.Event(), poll_seconds=1))

    assert calls == ["_run_settlement_scan_once"]


def test_auto_trade_stop_between_ticks_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(auto_trade_service, "run_auto_trade_once", lambda: [])

    async def stop_after_tick(stop_event, _started, _poll_seconds):
        stop_event.set()
        return True

    monkeypatch.setattr(auto_trade_service, "_sleep_until_next_tick", stop_after_tick)

    asyncio.run(auto_trade_service.auto_trade_loop(asyncio.Event(), poll_seconds=1))

    status = background_loop_statuses()["auto_trade"]
    assert status["status"] == "stopped"
    assert status["stopReason"] == "stop_between_ticks"
