from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


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
    except RuntimeError:
        return


def _finish_future(future: asyncio.Future[T], value: object, is_error: bool) -> None:
    if future.cancelled() or future.done():
        return
    if is_error:
        future.set_exception(value)  # type: ignore[arg-type]
        return
    future.set_result(value)  # type: ignore[arg-type]

