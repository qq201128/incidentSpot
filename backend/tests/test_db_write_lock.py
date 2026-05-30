from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from app.db import session
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


def test_run_db_write_with_retry_recovers_from_database_busy(monkeypatch) -> None:
    attempts = {"count": 0}

    def flaky_write() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise sqlite3.OperationalError("database is busy")
        return "ok"

    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    assert run_db_write_with_retry(flaky_write) == "ok"
    assert attempts["count"] == 2


def test_run_db_write_with_retry_exposes_exhaustion_details(monkeypatch) -> None:
    attempts = {"count": 0}
    slept = []

    def locked_write() -> None:
        attempts["count"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(time, "sleep", lambda seconds: slept.append(seconds))

    try:
        run_db_write_with_retry(locked_write, attempts=3, base_delay=0.1)
    except session.DbWriteRetryExhausted as exc:
        assert isinstance(exc, sqlite3.OperationalError)
        assert str(exc) == "SQLite write retry exhausted after 3 attempts: database is locked"
        assert exc.details == {
            "attempts": 3,
            "baseDelaySeconds": 0.1,
            "sleptDelaysSeconds": [0.1, 0.2],
            "causeError": "database is locked",
            "causeExceptionType": "OperationalError",
        }
    else:
        raise AssertionError("write retry exhaustion details were not exposed")

    assert attempts["count"] == 3
    assert slept == [0.1, 0.2]


def test_run_db_write_with_retry_rejects_non_positive_attempts() -> None:
    try:
        run_db_write_with_retry(lambda: None, attempts=0)
    except ValueError as exc:
        assert str(exc) == "attempts must be positive"
    else:
        raise AssertionError("non-positive attempts were not rejected")


def test_get_conn_exposes_wal_configuration_failure(monkeypatch) -> None:
    class BadConn:
        closed = False
        row_factory = None

        def execute(self, _sql: str):
            raise sqlite3.OperationalError("readonly database")

        def close(self) -> None:
            self.closed = True

    conn = BadConn()
    monkeypatch.setattr(session.sqlite3, "connect", lambda *args, **kwargs: conn)

    try:
        session.get_conn()
    except RuntimeError as exc:
        assert "failed to enable SQLite WAL mode" in str(exc)
    else:
        raise AssertionError("WAL configuration failure was not exposed")

    assert conn.closed is True


def test_schema_migration_reraises_unexpected_operational_error() -> None:
    class BadMigrationConn:
        def execute(self, _sql: str):
            raise sqlite3.OperationalError("syntax error near broken")

    try:
        session._apply_schema_migrations(BadMigrationConn())
    except sqlite3.OperationalError as exc:
        assert "syntax error" in str(exc)
    else:
        raise AssertionError("unexpected schema migration error was swallowed")


def test_schema_creates_candidate_settled_index(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(session.SCHEMA_PATH.read_text(encoding="utf-8"))
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(predictions)").fetchall()}
    finally:
        conn.close()

    assert "idx_predictions_candidate_settled" in indexes
