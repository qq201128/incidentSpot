from __future__ import annotations

import uuid
import tempfile
from pathlib import Path

import pytest

from app.services.lstm_candidate_progress import (
    complete_lstm_candidate_progress,
    finish_lstm_candidate_progress,
    queue_lstm_candidate_progress,
    read_lstm_candidate_progress,
    start_lstm_candidate_progress,
)
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_status_service import lstm_model_status


def test_lstm_candidate_progress_tracks_completion() -> None:
    artifact_root = _runtime_path("candidate-progress")
    start_lstm_candidate_progress(
        symbol="BTCUSDT",
        duration="10m",
        profile="full",
        total=2,
        search_space_total=225,
        parallel_workers=2,
        artifact_root=artifact_root,
    )
    complete_lstm_candidate_progress(
        config=LstmTrainingConfig(symbol="BTCUSDT", duration="10m", feature_window=24),
        profile="full",
        report=_report("shadow_active"),
        completed=1,
        total=2,
        artifact_root=artifact_root,
    )
    finished = finish_lstm_candidate_progress(
        symbol="BTCUSDT",
        duration="10m",
        status="shadow_active",
        artifact_root=artifact_root,
    )

    assert finished["status"] == "shadow_active"
    assert finished["completed"] == 1
    assert finished["total"] == 2
    assert finished["percent"] == pytest.approx(0.5)
    assert finished["counts"]["shadowActive"] == 1
    assert finished["latestCompleted"]["config"]["featureWindow"] == 24


def test_lstm_status_includes_candidate_search_progress() -> None:
    artifact_root = _runtime_path("candidate-progress-status")
    start_lstm_candidate_progress(
        symbol="BTCUSDT",
        duration="10m",
        profile="full",
        total=3,
        search_space_total=225,
        parallel_workers=2,
        artifact_root=artifact_root,
    )

    status = lstm_model_status("BTCUSDT", "10m", artifact_root=artifact_root)

    assert status["candidateSearchProgress"]["status"] == "running"
    assert status["candidateSearchProgress"]["total"] == 3
    assert status["candidateSearchProgress"]["parallelWorkers"] == 2


def test_lstm_candidate_progress_tracks_queue_state() -> None:
    artifact_root = _runtime_path("candidate-progress-queued")

    queued = queue_lstm_candidate_progress(
        symbol="BTCUSDT",
        duration="10m",
        profile="full",
        total=225,
        search_space_total=225,
        parallel_workers=1,
        artifact_root=artifact_root,
    )
    status = read_lstm_candidate_progress("BTCUSDT", "10m", artifact_root=artifact_root)

    assert queued["status"] == "queued"
    assert queued["startedAt"] is None
    assert status["status"] == "queued"
    assert status["total"] == 225
    assert status["parallelWorkers"] == 1


def _report(status: str) -> dict:
    return {
        "status": status,
        "candidateStatus": f"candidate_{status}",
        "modelVersion": "lstm_test",
        "validation": {"winRate": 0.55, "profitFactor": 1.1, "sampleCount": 50},
        "test": {"winRate": 0.54, "profitFactor": 1.05, "sampleCount": 50},
    }


def _runtime_path(name: str) -> Path:
    path = Path(tempfile.gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
