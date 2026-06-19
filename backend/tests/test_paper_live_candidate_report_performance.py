from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from tempfile import gettempdir

import pytest

from app.services import paper_live_candidate_service as service


def test_candidate_report_batches_settled_event_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert holder["conn"].prediction_select_count == 1
    assert holder["conn"].event_select_count == 1
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


def test_candidate_report_trims_public_all_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-slim-report") / "candidates.db"
    _create_db(db_path)
    _insert_candidate_predictions(db_path, candidate_count=220, settled_per_candidate=1)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    report = service.paper_live_candidate_report("BTCUSDT", "10m")
    full_report = service.refresh_paper_live_candidate_states("BTCUSDT", "10m")

    assert report["payloadMode"] == "dashboard_slim"
    assert report["allCandidateCount"] == 220
    assert len(report["allCandidates"]) < report["allCandidateCount"]
    assert len(full_report["allCandidates"]) == 220


def test_candidate_report_paginates_dashboard_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-paged-report") / "candidates.db"
    _create_db(db_path)
    _insert_candidate_predictions(db_path, candidate_count=45, settled_per_candidate=1)
    monkeypatch.setattr(service, "get_conn", lambda: _connect(db_path))

    report = service.paper_live_candidate_report("BTCUSDT", "10m")
    first_page = service.paginate_paper_live_candidate_report(report, page=1, page_size=18)
    second_page = service.paginate_paper_live_candidate_report(report, page=2, page_size=18)

    assert first_page["pagination"] == {
        "page": 1,
        "pageSize": 18,
        "totalRows": 45,
        "totalPages": 3,
        "returnedRows": 18,
        "hasPrevious": False,
        "hasNext": True,
        "allCandidateCount": 45,
    }
    assert first_page["summary"]["sampleCount"] == 45
    assert first_page["summary"]["settledCandidateCount"] == 45
    assert first_page["summary"]["settledCoverage"] == 1
    assert first_page["summary"]["weightedWinRate"] == 1
    assert len(first_page["allCandidates"]) == 18
    assert len(second_page["allCandidates"]) == 18
    assert {row["candidateKey"] for row in first_page["allCandidates"]}.isdisjoint(
        {row["candidateKey"] for row in second_page["allCandidates"]}
    )


def test_dashboard_pagination_uses_settled_sample_rows() -> None:
    report = {
        "payloadMode": "dashboard_slim",
        "allCandidateCount": 6,
        "allCandidates": [
            {"candidateKey": "settled_a", "status": "paper_collecting", "paperLiveSampleCount": 2},
            {"candidateKey": "empty_a", "status": "paper_collecting", "paperLiveSampleCount": 0},
            {"candidateKey": "settled_b", "status": "paper_collecting", "paperLiveSampleCount": 1},
            {"candidateKey": "empty_b", "status": "paper_collecting", "paperLiveSampleCount": 0},
            {"candidateKey": "settled_c", "status": "paper_failed", "paperLiveSampleCount": 3},
            {"candidateKey": "empty_c", "status": "paper_collecting", "paperLiveSampleCount": None},
        ],
    }

    first_page = service.paginate_paper_live_candidate_report(report, page=1, page_size=2)
    second_page = service.paginate_paper_live_candidate_report(report, page=2, page_size=2)

    assert first_page["pagination"]["totalRows"] == 3
    assert first_page["pagination"]["totalPages"] == 2
    assert first_page["pagination"]["allCandidateCount"] == 6
    assert [row["candidateKey"] for row in first_page["allCandidates"]] == ["settled_a", "settled_b"]
    assert [row["candidateKey"] for row in second_page["allCandidates"]] == ["settled_c"]


def test_candidate_report_prefers_status_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-status-snapshot") / "candidates.db"
    _create_db(db_path)
    _insert_status_snapshot(db_path)
    holder: dict[str, _CountingConnection] = {}

    def connect() -> _CountingConnection:
        conn = _CountingConnection(_connect(db_path))
        holder["conn"] = conn
        return conn

    monkeypatch.setattr(service, "get_conn", connect)

    report = service.paper_live_candidate_report("BTCUSDT", "10m")

    assert report["payloadSource"] == "candidate_status_snapshot"
    assert report["allCandidateCount"] == 1
    assert report["allCandidates"][0]["candidateKey"] == "factor_snapshot"
    assert holder["conn"].prediction_select_count == 0


class _CountingConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.prediction_select_count = 0
        self.event_select_count = 0
        self.ddl_count = 0

    def execute(self, sql: str, parameters: tuple = ()):
        if _is_prediction_select(sql):
            self.prediction_select_count += 1
        if _is_event_select(sql):
            self.event_select_count += 1
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


def _is_event_select(sql: str) -> bool:
    normalized = " ".join(sql.upper().split())
    return normalized.startswith(("SELECT", "WITH")) and " FROM EVENTS" in normalized


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
        CREATE TABLE paper_live_candidate_status (
          candidate_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(candidate_key, symbol, duration)
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
        CREATE TABLE events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          event_interval TEXT NOT NULL,
          start_time TEXT NOT NULL,
          end_time TEXT NOT NULL,
          status TEXT NOT NULL,
          result TEXT,
          ai_predicted_direction TEXT,
          ai_prediction_correct INTEGER,
          ai_high_winrate_rule TEXT,
          prediction_open_time INTEGER,
          prediction_id INTEGER,
          market_regime_gate_passed INTEGER
        );
        CREATE TABLE orders (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          event_id INTEGER NOT NULL,
          side TEXT NOT NULL,
          qty REAL NOT NULL,
          price REAL NOT NULL
        );
        """
    )
    conn.close()


def _insert_status_snapshot(path: Path) -> None:
    conn = _connect(path)
    candidate = {
        "candidateKey": "factor_snapshot",
        "strategyKey": "factor_snapshot",
        "candidateType": "factor",
        "factorName": "snapshot",
        "paperLiveWinRate": 0.7,
        "paperLiveSampleCount": 30,
        "paperLiveStatus": "paper_stable",
        "status": "paper_stable",
        "reason": "stable_paper_live_target_met",
        "metrics": {
            "sampleCount": 30,
            "winRate": 0.7,
            "profitFactor": 2.0,
            "avgReturn": 0.01,
            "paperStability": {
                "rollingWindows": [{"sampleCount": 10, "winRate": 0.7}],
            },
            "paperLiveWindows": {
                "recent30": {"sampleCount": 30, "winRate": 0.7},
                "recent60": {"sampleCount": 30, "winRate": 0.7},
                "recent100": {"sampleCount": 30, "winRate": 0.7},
            },
        },
    }
    conn.execute(
        """
        INSERT INTO paper_live_candidate_status(
          candidate_key, symbol, duration, status, reason, details_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "factor_snapshot",
            "BTCUSDT",
            "10m",
            "paper_stable",
            "stable_paper_live_target_met",
            json.dumps(candidate),
            "2026-06-12T00:00:00+00:00",
        ),
    )
    conn.commit()
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
    _insert_event_outcomes(conn, candidate_count=candidate_count, settled_per_candidate=settled_per_candidate)
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


def _insert_event_outcomes(conn: sqlite3.Connection, *, candidate_count: int, settled_per_candidate: int) -> None:
    for candidate in range(candidate_count):
        key = f"factor_{candidate}"
        for sample in range(settled_per_candidate):
            prediction_id = candidate * settled_per_candidate + sample + 1
            cursor = conn.execute(
                """
                INSERT INTO events(
                  strategy_key, symbol, event_interval, start_time, end_time, status,
                  result, ai_predicted_direction, ai_prediction_correct,
                  ai_high_winrate_rule, prediction_open_time, prediction_id, market_regime_gate_passed
                )
                VALUES(?, 'BTCUSDT', '10m', ?, ?, 'SETTLED', 'YES', 'up', 1, ?, ?, ?, 1)
                """,
                (
                    key,
                    f"2026-05-26T00:{sample % 60:02d}:00+00:00",
                    f"2026-05-26T00:{sample % 60:02d}:30+00:00",
                    key,
                    candidate * 100 + sample,
                    prediction_id,
                ),
            )
            conn.execute(
                "INSERT INTO orders(event_id, side, qty, price) VALUES(?, 'BUY', 5.0, 0.8)",
                (cursor.lastrowid,),
            )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _runtime_path(name: str) -> Path:
    path = Path(gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
