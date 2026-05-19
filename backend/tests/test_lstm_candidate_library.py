from __future__ import annotations

import uuid
from pathlib import Path

from app.services.experiment_profiles import EXPERIMENT_PROFILE_FULL
from app.services.lstm_candidate_keys import search_key_for_config
from app.services.lstm_candidate_library import (
    attempted_search_keys,
    lstm_candidate_library_summary,
    record_lstm_candidate,
)
from app.services.lstm_config import LstmTrainingConfig


def test_candidate_library_records_and_replaces_search_key() -> None:
    artifact_root = _runtime_path("candidate-library")
    config = LstmTrainingConfig(
        symbol="BTCUSDT",
        duration="10m",
        feature_window=24,
        epochs=8,
        min_move_bps=8,
        seed=20260513,
    )

    first = record_lstm_candidate(
        config,
        EXPERIMENT_PROFILE_FULL,
        _report("shadow_active", "lstm_shadow", 0.62),
        artifact_root=artifact_root,
    )
    second = record_lstm_candidate(
        config,
        EXPERIMENT_PROFILE_FULL,
        _report("trade_active", "lstm_trade", 0.72),
        artifact_root=artifact_root,
    )
    summary = lstm_candidate_library_summary("BTCUSDT", "10m", artifact_root=artifact_root)

    assert first["searchKey"] == search_key_for_config(config, EXPERIMENT_PROFILE_FULL)
    assert second["searchKey"] in attempted_search_keys("BTCUSDT", "10m", artifact_root=artifact_root)
    assert summary["total"] == 1
    assert summary["latest"]["modelVersion"] == "lstm_trade"
    assert summary["bestTradeCandidate"]["modelVersion"] == "lstm_trade"


def _report(status: str, model_version: str, win_rate: float) -> dict:
    return {
        "status": status,
        "candidateStatus": f"candidate_{status}",
        "modelVersion": model_version,
        "sampleCounts": {"train": 100, "validation": 50, "test": 50},
        "validation": _metrics(win_rate),
        "test": _metrics(win_rate),
    }


def _metrics(win_rate: float) -> dict:
    return {
        "winRate": win_rate,
        "profitFactor": 1.2,
        "avgReturn": 0.001,
        "sampleCount": 50,
        "confidenceThresholds": [],
    }


def _runtime_path(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
