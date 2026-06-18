from __future__ import annotations

import asyncio

from fastapi import FastAPI

from app import app_startup
from app.main import app, health
from app.services.background_loop_status import (
    background_loop_statuses,
    record_loop_failure,
    reset_background_loop_statuses,
)


def test_deferred_bootstrap_failure_is_exposed(monkeypatch) -> None:
    reset_background_loop_statuses()
    test_app = FastAPI()

    monkeypatch.setattr(
        app_startup,
        "_load_deferred_routers",
        lambda: (_ for _ in ()).throw(RuntimeError("router failed")),
    )

    asyncio.run(app_startup._deferred_bootstrap(test_app))

    assert test_app.state.bootstrap_status == "failed"
    assert test_app.state.bootstrap_error == "router failed"
    assert test_app.state.bootstrap_exception_type == "RuntimeError"
    status = background_loop_statuses()["application_bootstrap"]
    assert status["status"] == "failed"
    assert status["lastError"] == "router failed"
    assert status["lastFailureDetails"] == {"stage": "deferred_bootstrap"}


def test_health_surfaces_bootstrap_failure() -> None:
    previous = (
        app.state.bootstrap_status,
        app.state.bootstrap_error,
        app.state.bootstrap_exception_type,
    )
    app.state.bootstrap_status = "failed"
    app.state.bootstrap_error = "deferred failed"
    app.state.bootstrap_exception_type = "RuntimeError"
    try:
        payload = asyncio.run(health())
    finally:
        app.state.bootstrap_status = previous[0]
        app.state.bootstrap_error = previous[1]
        app.state.bootstrap_exception_type = previous[2]

    assert payload["ready"] is False
    assert payload["bootstrap"] == {
        "status": "failed",
        "error": "deferred failed",
        "exceptionType": "RuntimeError",
    }


def test_sync_bootstrap_failure_is_exposed(monkeypatch) -> None:
    reset_background_loop_statuses()
    test_app = FastAPI()
    logged = []

    class Logger:
        def exception(self, message: str) -> None:
            logged.append(message)

    def fail_thread_pool() -> None:
        raise ValueError("bad thread pool")

    monkeypatch.setattr(app_startup, "logger", Logger())
    monkeypatch.setattr(app_startup, "_configure_asyncio_thread_pool", fail_thread_pool)

    try:
        asyncio.run(app_startup.bootstrap_application(test_app))
    except ValueError as exc:
        assert str(exc) == "bad thread pool"
    else:
        raise AssertionError("sync bootstrap failure was not exposed")

    assert test_app.state.bootstrap_status == "failed"
    assert test_app.state.bootstrap_error == "bad thread pool"
    assert test_app.state.bootstrap_exception_type == "ValueError"
    assert logged == ["application bootstrap failed"]
    status = background_loop_statuses()["application_bootstrap"]
    assert status["status"] == "failed"
    assert status["lastError"] == "bad thread pool"
    assert status["lastFailureDetails"] == {"stage": "core_bootstrap"}


def test_deferred_bootstrap_success_is_recorded(monkeypatch) -> None:
    reset_background_loop_statuses()
    test_app = FastAPI()
    monkeypatch.setattr(app_startup, "_load_deferred_routers", lambda: [])
    monkeypatch.setattr(app_startup, "_warm_background_imports", lambda: None)
    monkeypatch.setattr(app_startup, "_spawn_background_tasks", lambda _app: None)

    asyncio.run(app_startup._deferred_bootstrap(test_app))

    assert test_app.state.bootstrap_status == "ready"
    status = background_loop_statuses()["application_bootstrap"]
    assert status["status"] == "passed"
    assert status["lastSuccessDetails"] == {"stage": "deferred_bootstrap"}


def test_spawned_background_task_failure_is_visible() -> None:
    reset_background_loop_statuses()
    test_app = FastAPI()

    async def failing_loop(_stop_event: asyncio.Event) -> None:
        raise RuntimeError("predict crashed")

    async def run_task() -> None:
        app_startup._spawn_loop(test_app, "predict", failing_loop)
        task = test_app.state.predict_task
        try:
            await task
        except RuntimeError:
            return

    asyncio.run(run_task())

    status = background_loop_statuses()["auto_predict"]
    assert status["status"] == "failed"
    assert status["lastError"] == "predict crashed"
    assert status["lastFailureDetails"] == {"stage": "background_task", "taskName": "predict"}


def test_spawned_background_task_base_exception_is_visible() -> None:
    reset_background_loop_statuses()

    class BaseFailure(BaseException):
        pass

    async def run_callback() -> None:
        future = asyncio.get_running_loop().create_future()
        future.set_exception(BaseFailure("base crashed"))
        app_startup._record_background_task_result("predict", future)

    asyncio.run(run_callback())

    status = background_loop_statuses()["auto_predict"]
    assert status["status"] == "failed"
    assert status["lastError"] == "base crashed"
    assert status["lastExceptionType"] == "BaseFailure"
    assert status["lastFailureDetails"] == {"stage": "background_task", "taskName": "predict"}


def test_spawned_background_task_keeps_specific_recorded_failure() -> None:
    reset_background_loop_statuses()
    test_app = FastAPI()

    async def failing_loop(_stop_event: asyncio.Event) -> None:
        exc = RuntimeError("config crashed")
        record_loop_failure("auto_predict", exc, {"stage": "startup_config"})
        raise exc

    async def run_task() -> None:
        app_startup._spawn_loop(test_app, "predict", failing_loop)
        task = test_app.state.predict_task
        try:
            await task
        except RuntimeError:
            return

    asyncio.run(run_task())

    status = background_loop_statuses()["auto_predict"]
    assert status["status"] == "failed"
    assert status["lastError"] == "config crashed"
    assert status["lastFailureDetails"] == {"stage": "startup_config"}


def test_shutdown_logs_background_task_exceptions(monkeypatch) -> None:
    reset_background_loop_statuses()
    logged = []

    class Logger:
        def error(self, message: str, *args, exc_info=None) -> None:
            logged.append((message, args, exc_info))

    async def failed_task() -> None:
        raise RuntimeError("shutdown failed")

    async def run_shutdown() -> None:
        task = asyncio.create_task(failed_task())
        await asyncio.sleep(0)
        await app_startup._cancel_background_tasks([("predict", task)])

    monkeypatch.setattr(app_startup, "logger", Logger())

    asyncio.run(run_shutdown())

    assert logged[0][0] == "background task failed during shutdown: %s"
    assert str(logged[0][1][0]) == "shutdown failed"
    assert logged[0][2][0] is RuntimeError
    status = background_loop_statuses()["auto_predict"]
    assert status["status"] == "failed"
    assert status["lastError"] == "shutdown failed"
    assert status["lastFailureDetails"] == {"stage": "shutdown", "taskName": "predict"}


def test_shutdown_records_background_base_exceptions(monkeypatch) -> None:
    reset_background_loop_statuses()
    logged = []

    class BaseFailure(BaseException):
        pass

    class Logger:
        def error(self, message: str, *args, exc_info=None) -> None:
            logged.append((message, args, exc_info))

    monkeypatch.setattr(app_startup, "logger", Logger())

    app_startup._log_background_task_shutdown_results(
        [("predict", object())],
        [BaseFailure("shutdown base")],
    )

    status = background_loop_statuses()["auto_predict"]
    assert status["status"] == "failed"
    assert status["lastError"] == "shutdown base"
    assert status["lastExceptionType"] == "BaseFailure"
    assert status["lastFailureDetails"] == {"stage": "shutdown", "taskName": "predict"}
    assert logged[0][0] == "background task failed during shutdown: %s"
