from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.services.lstm_artifacts import artifact_paths, read_json, required_artifacts_exist
from app.services.lstm_combo_ranking import resolve_lstm_combo_ranking
from app.services.lstm_combo_snapshot import combo_snapshot_status
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_feature_builder import build_lstm_training_dataset
from app.services.lstm_prediction_service import active_lstm_status, lstm_validation_block_reason
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
    ranking = _resolved_ranking(sym, duration, ranking_report)
    if _trained_artifacts_match(sym, duration, ranking, artifact_root):
        return {"status": LSTM_SYNC_UP_TO_DATE, "symbol": sym, "duration": duration}
    report = _train(sym, duration, ranking, artifact_root=artifact_root, trainer=trainer)
    return {
        "status": str(report.get("status") or LSTM_SYNC_TRAINED),
        "symbol": sym,
        "duration": duration,
        "modelVersion": report.get("modelVersion"),
        "validationFailureReason": report.get("validationFailureReason"),
    }


def _resolved_ranking(symbol: str, duration: str, ranking_report: dict[str, Any]) -> dict[str, Any]:
    if _has_ranking(ranking_report):
        return ranking_report
    resolved = resolve_lstm_combo_ranking(
        symbol,
        duration,
        primary_loader=lambda *_args: ranking_report,
    )
    return ranking_report if resolved is None else resolved


def _trained_artifacts_match(
    symbol: str,
    duration: str,
    ranking_report: dict[str, Any],
    artifact_root: Path | None,
) -> bool:
    paths = artifact_paths(symbol, duration, artifact_root)
    status = active_lstm_status(symbol, duration, artifact_root=artifact_root)
    version = read_json(paths.version) or {}
    report = read_json(paths.report) or {}
    snapshot = combo_snapshot_status(
        symbol,
        duration,
        ranking_report=ranking_report,
        artifact_root=artifact_root,
    )
    return (
        status.get("status") == "trained"
        and required_artifacts_exist(paths)
        and lstm_validation_block_reason(status, version, report) == "passed"
        and snapshot["matches"]
    )


def _train(
    symbol: str,
    duration: str,
    ranking_report: dict[str, Any],
    *,
    artifact_root: Path | None,
    trainer: Trainer | None,
) -> dict[str, Any]:
    config = LstmTrainingConfig(symbol=symbol, duration=duration)
    if trainer is not None:
        return trainer(config)
    return train_lstm_model(
        config,
        artifact_root=artifact_root,
        dataset_builder=_dataset_builder_for_ranking(ranking_report),
    )


def _dataset_builder_for_ranking(ranking_report: dict[str, Any]):
    def build(config: LstmTrainingConfig):
        return build_lstm_training_dataset(
            config,
            ranking_loader=lambda _symbol, _duration: ranking_report,
        )

    return build


def _has_ranking(ranking_report: dict[str, Any]) -> bool:
    rows = ranking_report.get("ranking")
    return bool(isinstance(rows, list) and rows)
