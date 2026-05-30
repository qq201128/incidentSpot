from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import TypeVar

from app.services.background_loop_status import record_loop_failure

T = TypeVar("T")
logger = logging.getLogger(__name__)
LOOP_NAME = "blocking_daemon_delivery"


async def run_blocking_daemon(func: Callable[[], T]) -> T:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[T] = loop.create_future()
    thread = threading.Thread(target=_runner, args=(loop, future, func), daemon=True)
    thread.start()
    return await future


def _runner(loop: asyncio.AbstractEventLoop, future: asyncio.Future[T], func: Callable[[], T]) -> None:
    try:
        result = func()
    except Exception as exc:
        _notify(loop, future, exc, is_error=True)
        return
    _notify(loop, future, result, is_error=False)


def _notify(loop: asyncio.AbstractEventLoop, future: asyncio.Future[T], value: object, *, is_error: bool) -> None:
    try:
        loop.call_soon_threadsafe(_finish_future, future, value, is_error)
    except RuntimeError as exc:
        record_loop_failure(LOOP_NAME, exc, _delivery_failure_details(value, is_error=is_error))
        logger.exception("blocking daemon result delivery failed", exc_info=(type(exc), exc, exc.__traceback__))


def _finish_future(future: asyncio.Future[T], value: object, is_error: bool) -> None:
    if future.cancelled() or future.done():
        return
    if is_error:
        future.set_exception(value)  # type: ignore[arg-type]
        return
    future.set_result(value)  # type: ignore[arg-type]


def _delivery_failure_details(value: object, *, is_error: bool) -> dict[str, object]:
    details: dict[str, object] = {"isErrorResult": is_error, "valueType": type(value).__name__}
    if is_error and isinstance(value, BaseException):
        details["originalError"] = str(value)
        details["originalExceptionType"] = type(value).__name__
    return details
