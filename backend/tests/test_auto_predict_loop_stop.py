from __future__ import annotations

import asyncio

from app.services import auto_predict_service as service
from app.services.auto_predict_loop_status import auto_predict_loop_status
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses


def test_auto_predict_initial_stop_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 5.0)
    monkeypatch.setattr(service, "_prediction_targets", _unexpected_prediction_targets)

    async def stop_during_delay(stop_event, _seconds):
        stop_event.set()
        return True

    monkeypatch.setattr(service, "_sleep_for", stop_during_delay)

    asyncio.run(service.auto_predict_loop(asyncio.Event(), poll_seconds=2))

    status = auto_predict_loop_status()
    background = background_loop_statuses()["auto_predict"]
    assert status["status"] == "stopped"
    assert status["stopReason"] == "stop_during_initial_delay"
    assert status["failureDetails"] is None
    assert background["status"] == "stopped"
    assert background["stopReason"] == "stop_during_initial_delay"


def test_auto_predict_stop_after_failure_keeps_failed_status(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(service, "_predict_initial_delay_seconds", lambda: 0.0)
    monkeypatch.setattr(service, "_prediction_targets", _failed_prediction_targets)

    async def stop_after_failure(_stop_event, _seconds):
        return True

    monkeypatch.setattr(service, "_sleep_for", stop_after_failure)

    asyncio.run(service.auto_predict_loop(asyncio.Event(), poll_seconds=3))

    status = auto_predict_loop_status()
    background = background_loop_statuses()["auto_predict"]
    assert status["status"] == "failed"
    assert status["error"] == "target failed"
    assert status["stopReason"] == "stop_between_cycles"
    assert background["status"] == "failed"
    assert background["lastError"] == "target failed"
    assert background["stopReason"] == "stop_between_cycles"


def _unexpected_prediction_targets() -> None:
    raise AssertionError("prediction targets should not be loaded")


def _failed_prediction_targets() -> None:
    raise RuntimeError("target failed")
