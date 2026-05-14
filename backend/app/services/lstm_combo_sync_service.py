from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.services.lstm_artifacts import artifact_paths, read_json, required_artifacts_exist
from app.services.lstm_combo_snapshot import combo_snapshot_status
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_training_service import train_lstm_model

LSTM_SYNC_UP_TO_DATE = "up_to_date"
LSTM_SYNC_TRAINED = "trained"

Trainer = Callable[[LstmTrainingConfig], dict[str, Any]]


def sync_lstm_model_to_combo_ranking(
    symbol: str,
    duration: str,
    *,
    ranking_report: dict[str, Any],
    artifact_root: Path | None = None,
    trainer: Trainer | None = None,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    if _trained_artifacts_match(sym, duration, ranking_report, artifact_root):
        return {"status": LSTM_SYNC_UP_TO_DATE, "symbol": sym, "duration": duration}
    report = _train(sym, duration, artifact_root=artifact_root, trainer=trainer)
    return {
        "status": LSTM_SYNC_TRAINED,
        "symbol": sym,
        "duration": duration,
        "modelVersion": report.get("modelVersion"),
    }


def _trained_artifacts_match(
    symbol: str,
    duration: str,
    ranking_report: dict[str, Any],
    artifact_root: Path | None,
) -> bool:
    paths = artifact_paths(symbol, duration, artifact_root)
    status = read_json(paths.status) or {}
    snapshot = combo_snapshot_status(
        symbol,
        duration,
        ranking_report=ranking_report,
        artifact_root=artifact_root,
    )
    return status.get("status") == "trained" and required_artifacts_exist(paths) and snapshot["matches"]


def _train(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None,
    trainer: Trainer | None,
) -> dict[str, Any]:
    config = LstmTrainingConfig(symbol=symbol, duration=duration)
    if trainer is not None:
        return trainer(config)
    return train_lstm_model(config, artifact_root=artifact_root)
