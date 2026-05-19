from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pytest

from app.services import forward_validation_service
from app.services.lstm_validation import binary_classification_metrics

def test_lstm_accuracy_tracks_direction_and_win_rate_uses_gross_returns() -> None:
    metrics = binary_classification_metrics(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.6, 0.4, 0.7], dtype=np.float32),
        np.asarray([0.0005, -0.0005, -0.0005], dtype=np.float32),
    )

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["winRate"] == pytest.approx(2 / 3)
    assert metrics["avgReturn"] == pytest.approx(0.0005 / 3)


def test_lstm_metrics_include_confidence_threshold_samples() -> None:
    metrics = binary_classification_metrics(
        np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.54, 0.40, 0.70, 0.30], dtype=np.float32),
        np.asarray([0.01, -0.02, -0.03, -0.04], dtype=np.float32),
    )

    thresholds = {
        row["minConfidence"]: row
        for row in metrics["confidenceThresholds"]
    }
    assert list(thresholds) == [0.55, 0.6, 0.65, 0.7]
    assert thresholds[0.55]["sampleCount"] == 3
    assert thresholds[0.55]["winRate"] == pytest.approx(2 / 3)
    assert thresholds[0.55]["avgReturn"] == pytest.approx(0.01)
    assert thresholds[0.55]["profitFactor"] == pytest.approx(2.0)
    assert thresholds[0.65]["sampleCount"] == 2
    assert thresholds[0.65]["winRate"] == pytest.approx(0.5)
    assert thresholds[0.70]["sampleCount"] == 2


def test_lstm_confidence_threshold_metrics_count_small_gross_wins() -> None:
    metrics = binary_classification_metrics(
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.7, 0.3], dtype=np.float32),
        np.asarray([0.0001, -0.0001], dtype=np.float32),
    )

    threshold = {row["minConfidence"]: row for row in metrics["confidenceThresholds"]}[0.7]
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["winRate"] == pytest.approx(1.0)
    assert threshold["winRate"] == pytest.approx(1.0)
    assert threshold["profitFactor"] == pytest.approx(float("inf"))


def test_lstm_confidence_threshold_metrics_handles_empty_buckets() -> None:
    metrics = binary_classification_metrics(
        np.asarray([1.0], dtype=np.float32),
        np.asarray([0.51], dtype=np.float32),
        np.asarray([0.01], dtype=np.float32),
    )

    rows = {row["minConfidence"]: row for row in metrics["confidenceThresholds"]}
    for row in rows.values():
        assert row["sampleCount"] == 0
        assert row["winRate"] is None
        assert row["avgReturn"] is None
        assert row["profitFactor"] is None


def test_forward_validation_prediction_correct_tracks_direction_without_cost(monkeypatch) -> None:
    db_path = _runtime_path("forward-db") / "predictions.db"
    _create_forward_validation_db(db_path)
    monkeypatch.setattr(forward_validation_service, "get_conn", lambda: _connect(db_path))

    result = forward_validation_service.settle_due_predictions("BTCUSDT", "10m")

    conn = _connect(db_path)
    row = conn.execute(
        "SELECT actual_return, prediction_correct FROM predictions WHERE id = 1"
    ).fetchone()
    conn.close()
    assert result == {"checked": 1, "settled": 1, "pendingData": 0}
    assert row["actual_return"] == pytest.approx(0.0005)
    assert row["prediction_correct"] == 1


def _create_forward_validation_db(path: Path) -> None:
    conn = _connect(path)
    conn.executescript(
        """
        CREATE TABLE klines (
          symbol TEXT NOT NULL,
          interval TEXT NOT NULL,
          open_time INTEGER NOT NULL,
          close REAL NOT NULL
        );
        CREATE TABLE predictions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          open_time INTEGER NOT NULL,
          direction TEXT NOT NULL,
          entry_price REAL,
          exit_price REAL,
          actual_return REAL,
          prediction_correct INTEGER,
          settled_at TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO klines VALUES(?, ?, ?, ?)",
        [
            ("BTCUSDT", "1m", 0, 100.0),
            ("BTCUSDT", "1m", 600_000, 100.05),
            ("BTCUSDT", "1m", 1_200_000, 100.10),
        ],
    )
    conn.execute(
        """
        INSERT INTO predictions(symbol, duration, open_time, direction, entry_price)
        VALUES(?, ?, ?, ?, ?)
        """,
        ("BTCUSDT", "10m", 0, "up", 100.0),
    )
    conn.commit()
    conn.close()


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _runtime_path(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
