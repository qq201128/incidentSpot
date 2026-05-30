from __future__ import annotations

import asyncio

from app.services import background_threads
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses


def test_blocking_daemon_notify_runtime_error_is_logged(monkeypatch) -> None:
    reset_background_loop_statuses()
    logged = []
    event_loop = asyncio.new_event_loop()
    future = event_loop.create_future()

    class Logger:
        def exception(self, message: str, *, exc_info=None) -> None:
            logged.append((message, exc_info))

    class ClosedLoop:
        def call_soon_threadsafe(self, *_args, **_kwargs) -> None:
            raise RuntimeError("loop closed")

    monkeypatch.setattr(background_threads, "logger", Logger())

    try:
        background_threads._notify(ClosedLoop(), future, "value", is_error=False)
    finally:
        event_loop.close()

    assert logged[0][0] == "blocking daemon result delivery failed"
    assert logged[0][1][0] is RuntimeError
    status = background_loop_statuses()["blocking_daemon_delivery"]
    assert status["status"] == "failed"
    assert status["lastError"] == "loop closed"
    assert status["lastExceptionType"] == "RuntimeError"
    assert status["lastFailureDetails"] == {"isErrorResult": False, "valueType": "str"}


def test_blocking_daemon_notify_preserves_worker_exception_details(monkeypatch) -> None:
    reset_background_loop_statuses()
    event_loop = asyncio.new_event_loop()
    future = event_loop.create_future()

    class Logger:
        def exception(self, _message: str, *, exc_info=None) -> None:
            return None

    class ClosedLoop:
        def call_soon_threadsafe(self, *_args, **_kwargs) -> None:
            raise RuntimeError("loop closed")

    monkeypatch.setattr(background_threads, "logger", Logger())

    try:
        background_threads._notify(ClosedLoop(), future, ValueError("worker failed"), is_error=True)
    finally:
        event_loop.close()

    status = background_loop_statuses()["blocking_daemon_delivery"]
    assert status["status"] == "failed"
    assert status["lastError"] == "loop closed"
    assert status["lastFailureDetails"] == {
        "isErrorResult": True,
        "valueType": "ValueError",
        "originalError": "worker failed",
        "originalExceptionType": "ValueError",
    }
