from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def record_timing(timings: dict[str, Any] | None, key: str, started: float) -> None:
    if timings is not None:
        timings[key] = elapsed_seconds(started)


def elapsed_seconds(started: float) -> float:
    return round(max(time.perf_counter() - started, 0.0), 6)


def timed_call(label: str, timings: dict[str, Any] | None, fn: Callable[[], T]) -> T:
    started = time.perf_counter()
    try:
        return fn()
    finally:
        record_timing(timings, label, started)
