from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from tempfile import gettempdir

import pytest

from app.services import paper_live_candidate_live_control as control
from app.services import paper_live_candidate_service as report_service
from app.services import paper_live_report_cache as report_cache
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.high_winrate_strategy_metrics import high_winrate_metrics


@pytest.fixture(autouse=True)
def _clear_report_cache() -> None:
    report_cache.clear_paper_live_report_cache()
    yield
    report_cache.clear_paper_live_report_cache()


def test_stable_candidate_live_trading_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-control") / "candidates.db"
    candidate_key = factor_candidate_signal_key("alpha")
    _create_db(db_path)
    _insert_predictions(db_path, candidate_key, correct_count=30)
    _insert_event_outcomes(db_path, candidate_key, [True] * 30, start_prediction_id=1)
    _patch_db(monkeypatch, db_path)
    report_cache.store_paper_live_report_cache(
        "BTCUSDT",
        "10m",
        {"stable": [{"candidateKey": candidate_key, "liveTradingEnabled": False}]},
    )

    result = control.set_candidate_live_trading(
        "BTCUSDT",
        "10m",
        candidate_key=candidate_key,
        live_trading_enabled=True,
    )

    slot = _slot_row(db_path, candidate_key)
    candidate = result["report"]["stable"][0]
    cached = report_cache.get_cached_paper_live_report(
        "BTCUSDT",
        "10m",
        build=lambda _symbol, _duration: pytest.fail("expected live-control report cache hit"),
    )
    assert result["liveTradingEnabled"] is True
    assert slot["enabled"] == 1
    assert slot["live_trading_enabled"] == 1
    assert candidate["liveTradingEnabled"] is True
    assert result["report"]["realTradingEnabled"] is True
    assert cached["cache"]["hit"] is True
    assert cached["stable"][0]["liveTradingEnabled"] is True


def test_live_trading_uses_status_snapshot_without_full_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-control-snapshot") / "candidates.db"
    candidate_key = factor_candidate_signal_key("snapshot")
    _create_db(db_path)
    _insert_status_snapshot(db_path, candidate_key)
    _patch_db(monkeypatch, db_path)
    monkeypatch.setattr(
        control,
        "refresh_paper_live_candidate_states",
        lambda *_args, **_kwargs: pytest.fail("expected status snapshot, not full refresh"),
    )

    result = control.set_candidate_live_trading(
        "BTCUSDT",
        "10m",
        candidate_key=candidate_key,
        live_trading_enabled=True,
    )

    slot = _slot_row(db_path, candidate_key)
    candidate = result["report"]["stable"][0]
    assert slot["enabled"] == 1
    assert slot["live_trading_enabled"] == 1
    assert candidate["candidateKey"] == candidate_key
    assert candidate["liveTradingEnabled"] is True
    assert result["report"]["payloadSource"] == "candidate_status_snapshot"


def test_non_stable_candidate_live_trading_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-control-blocked") / "candidates.db"
    candidate_key = factor_candidate_signal_key("beta")
    _create_db(db_path)
    _insert_predictions(db_path, candidate_key, correct_count=1)
    _patch_db(monkeypatch, db_path)

    with pytest.raises(ValueError, match="candidate is not stable"):
        control.set_candidate_live_trading(
            "BTCUSDT",
            "10m",
            candidate_key=candidate_key,
            live_trading_enabled=True,
        )

    assert _slot_row(db_path, candidate_key) is None


def test_non_stable_candidate_live_trading_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("paper-live-control-disable-failed") / "candidates.db"
    candidate_key = factor_candidate_signal_key("gamma")
    _create_db(db_path)
    _insert_predictions(db_path, candidate_key, correct_count=1)
    _insert_slot(db_path, candidate_key, live_trading_enabled=True)
    _patch_db(monkeypatch, db_path)

    result = control.set_candidate_live_trading(
        "BTCUSDT",
        "10m",
        candidate_key=candidate_key,
        live_trading_enabled=False,
    )

    slot = _slot_row(db_path, candidate_key)
    assert result["liveTradingEnabled"] is False
    assert slot["enabled"] == 1
    assert slot["live_trading_enabled"] == 0


def _patch_db(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setattr(control, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(report_service, "get_conn", lambda: _connect(db_path))


def _create_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          signal_key TEXT NOT NULL,
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


def _insert_slot(path: Path, strategy_key: str, *, live_trading_enabled: bool) -> None:
    conn = _connect(path)
    conn.execute(
        """
        INSERT INTO auto_trade_strategies(
          strategy_key, symbol, duration, enabled, live_trading_enabled,
          duration_minutes, qty, updated_at
        )
        VALUES(?, 'BTCUSDT', '10m', 1, ?, 10, 5.0, '2026-05-26T00:00:00+00:00')
        """,
        (strategy_key, int(live_trading_enabled)),
    )
    conn.commit()
    conn.close()


def _insert_status_snapshot(path: Path, candidate_key: str) -> None:
    conn = _connect(path)
    candidate = {
        "candidateKey": candidate_key,
        "strategyKey": candidate_key,
        "candidateType": "factor",
        "factorName": "snapshot",
        "paperLiveWinRate": 0.6,
        "paperLiveSampleCount": 30,
        "paperLiveStatus": "paper_stable",
        "status": "paper_stable",
        "reason": "stable_paper_live_target_met",
        "metrics": _stable_metrics(),
        "liveTradingEnabled": False,
    }
    conn.execute(
        """
        INSERT INTO paper_live_candidate_status(
          candidate_key, symbol, duration, status, reason, details_json, updated_at
        )
        VALUES(?, 'BTCUSDT', '10m', 'paper_stable', 'stable_paper_live_target_met', ?, ?)
        """,
        (
            candidate_key,
            json.dumps(candidate),
            "2026-05-26T00:00:00+00:00",
        ),
    )
    conn.commit()
    conn.close()


def _insert_predictions(path: Path, signal_key: str, *, correct_count: int) -> None:
    conn = _connect(path)
    rows = [_prediction_row(signal_key, index, index < correct_count) for index in range(30)]
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


def _prediction_row(signal_key: str, index: int, correct: bool) -> tuple:
    actual_return = 0.01 if correct else -0.02
    return (
        signal_key,
        signal_key,
        "BTCUSDT",
        "10m",
        index,
        "up",
        signal_key,
        0.7,
        "fresh",
        "complete",
        actual_return,
        int(correct),
        "2026-05-26T00:00:00+00:00",
        "2026-05-26T00:00:00+00:00",
    )


def _insert_event_outcomes(path: Path, signal_key: str, outcomes: list[bool], *, start_prediction_id: int) -> None:
    conn = _connect(path)
    for index, correct in enumerate(outcomes):
        cursor = conn.execute(
            """
            INSERT INTO events(
              strategy_key, symbol, event_interval, start_time, end_time, status,
              result, ai_predicted_direction, ai_prediction_correct,
              ai_high_winrate_rule, prediction_open_time, prediction_id, market_regime_gate_passed
            )
            VALUES(?, 'BTCUSDT', '10m', ?, ?, 'SETTLED', ?, 'up', ?, ?, ?, ?, 1)
            """,
            (
                signal_key,
                f"2026-05-26T00:{index:02d}:00+00:00",
                f"2026-05-26T00:{index:02d}:30+00:00",
                "YES" if correct else "NO",
                int(correct),
                signal_key,
                index,
                start_prediction_id + index,
            ),
        )
        conn.execute(
            "INSERT INTO orders(event_id, side, qty, price) VALUES(?, 'BUY', 5.0, 0.8)",
            (cursor.lastrowid,),
        )
    conn.commit()
    conn.close()


def _slot_row(path: Path, strategy_key: str) -> sqlite3.Row | None:
    conn = _connect(path)
    try:
        return conn.execute(
            "SELECT * FROM auto_trade_strategies WHERE strategy_key = ?",
            (strategy_key,),
        ).fetchone()
    finally:
        conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _stable_metrics() -> dict:
    rows = [
        {"actual_return": 0.01, "prediction_correct": 1}
        for _index in range(30)
    ]
    return high_winrate_metrics(rows)


def _runtime_path(name: str) -> Path:
    path = Path(gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
