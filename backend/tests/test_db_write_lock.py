from __future__ import annotations

import sqlite3
import time

from app.db.session import run_db_write_with_retry


def test_run_db_write_with_retry_recovers_from_locked(monkeypatch) -> None:
    attempts = {"count": 0}

    def flaky_write() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    assert run_db_write_with_retry(flaky_write) == "ok"
    assert attempts["count"] == 2
