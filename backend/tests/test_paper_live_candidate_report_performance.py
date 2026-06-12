from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from tempfile import gettempdir

import pytest

from app.services import paper_live_candidate_service as service


def test_candidate_report_batches_settled_prediction_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-batched-report") / "candidates.db"
    _create_db(db_path)
    _insert_candidate_predictions(db_path, candidate_count=3, settled_per_candidate=2)
    holder: dict[str, _CountingConnection] = {}

    def connect() -> _CountingConnection:
        conn = _CountingConnection(_connect(db_path))
        holder["conn"] = conn
        return conn

    monkeypatch.setattr(service, "get_conn", connect)

    report = service.paper_live_candidate_report("BTCUSDT", "10m")

    assert report["allCandidateCount"] == 3
    assert {row["paperLiveSampleCount"] for row in report["allCandidates"]} == {2}
    assert holder["conn"].prediction_select_count == 2
    assert holder["conn"].ddl_count == 0


def test_candidate_report_keeps_full_metrics_with_bounded_recent_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-bounded-report") / "candidates.db"
    _create_db(db_path)
    _insert_candidate_predictions(db_path, candidate_count=1, settled_per_candidate=150)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    report = service.paper_live_candidate_report("BTCUSDT", "10m")

    candidate = report["allCandidates"][0]
    assert candidate["paperLiveSampleCount"] == 150
    assert candidate["metrics"]["sampleCount"] == 150
    assert candidate["metrics"]["paperLiveWindows"]["recent100"]["sampleCount"] == 100


class _CountingConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.prediction_select_count = 0
        self.ddl_count = 0

    def execute(self, sql: str, parameters: tuple = ()):
        if _is_prediction_select(sql):
            self.prediction_select_count += 1
        if _is_ddl(sql):
            self.ddl_count += 1
        return self._conn.execute(sql, parameters)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _is_prediction_select(sql: str) -> bool:
    normalized = " ".join(sql.upper().split())
    return normalized.startswith(("SELECT", "WITH")) and " FROM PREDICTIONS" in normalized


def _is_ddl(sql: str) -> bool:
    normalized = " ".join(sql.upper().split())
    return normalized.startswith(("ALTER TABLE ", "CREATE TABLE ", "CREATE INDEX ", "DROP TABLE "))


def _create_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          signal_key TEXT,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          open_time INTEGER NOT NULL,
          direction TEXT NOT NULL,
          high_winrate_rule TEXT,
          high_winrate_gate_value REAL,
          high_winrate_gate_min REAL,
          model_family TEXT,
          model_version TEXT,
          validation_win_rate REAL,
          feature_window INTEGER,
          model_duration TEXT,
          model_trained_at TEXT,
          oos_win_rate REAL,
          walk_forward_result TEXT,
          recent_rolling_result TEXT,
          data_freshness_status TEXT,
          missing_feature_status TEXT,
          entry_price REAL,
          exit_price REAL,
          actual_return REAL,
          prediction_correct INTEGER,
          settled_at TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE paper_live_prediction_failures (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          candidate_key TEXT NOT NULL,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          stage TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE paper_live_prediction_stage_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          signal_key TEXT NOT NULL,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          open_time INTEGER,
          stage TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT,
          details_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE paper_live_candidate_status_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          candidate_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          old_status TEXT,
          new_status TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          changed_at TEXT NOT NULL
        );
        CREATE TABLE auto_trade_strategies (
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 0,
          live_trading_enabled INTEGER NOT NULL DEFAULT 0,
          duration_minutes INTEGER NOT NULL DEFAULT 10,
          qty REAL NOT NULL DEFAULT 5.0,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(strategy_key, symbol, duration)
        );
        """
    )
    conn.close()


def _insert_candidate_predictions(path: Path, *, candidate_count: int, settled_per_candidate: int) -> None:
    conn = _connect(path)
    rows = [
        _prediction_row(candidate, sample)
        for candidate in range(candidate_count)
        for sample in range(settled_per_candidate)
    ]
    conn.executemany(
        """
        INSERT INTO predictions(
          signal_key, strategy_key, symbol, duration, open_time, direction,
          high_winrate_rule, high_winrate_gate_value, data_freshness_status,
          missing_feature_status, actual_return, prediction_correct, settled_at, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()


def _prediction_row(candidate: int, sample: int) -> tuple:
    key = f"factor_{candidate}"
    open_time = candidate * 100 + sample
    return (
        key,
        key,
        "BTCUSDT",
        "10m",
        open_time,
        "up",
        key,
        0.7,
        "fresh",
        "complete",
        0.01,
        1,
        "2026-05-26T00:00:00+00:00",
        "2026-05-26T00:00:00+00:00",
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _runtime_path(name: str) -> Path:
    path = Path(gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
