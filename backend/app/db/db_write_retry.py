from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager

_DB_WRITE_LOCK = threading.Lock()


class DbWriteRetryExhausted(sqlite3.OperationalError):
    def __init__(
        self,
        *,
        attempts: int,
        base_delay: float,
        slept_delays: list[float],
        cause: sqlite3.OperationalError,
    ) -> None:
        self.details = {
            "attempts": int(attempts),
            "baseDelaySeconds": float(base_delay),
            "sleptDelaysSeconds": [float(value) for value in slept_delays],
            "causeError": str(cause),
            "causeExceptionType": type(cause).__name__,
        }
        super().__init__(f"SQLite write retry exhausted after {attempts} attempts: {cause}")


@contextmanager
def db_write_lock():
    """Serialize SQLite writes across threads."""
    _DB_WRITE_LOCK.acquire()
    try:
        yield
    finally:
        _DB_WRITE_LOCK.release()


def run_db_write_with_retry(operation, *, attempts: int = 8, base_delay: float = 0.05):
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    delay = base_delay
    slept_delays: list[float] = []
    for attempt in range(attempts):
        try:
            with db_write_lock():
                return operation()
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if "locked" not in message and "busy" not in message:
                raise
            if attempt == attempts - 1:
                raise DbWriteRetryExhausted(
                    attempts=attempts,
                    base_delay=base_delay,
                    slept_delays=slept_delays,
                    cause=exc,
                ) from exc
            time.sleep(delay)
            slept_delays.append(delay)
            delay = min(delay * 2, 2.0)
