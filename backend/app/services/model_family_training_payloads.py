from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any

from app.services.lstm_artifacts import write_json
from app.services.lstm_feature_builder import LstmDataset
from app.services.lstm_lifecycle import candidate_status
from app.services.lstm_validation import apply_standardizer
from app.services.model_family_config import JOBLIB_MODEL_FAMILIES, ModelFamilyTrainingConfig
from app.services.model_family_joblib_backend import JoblibModelOptions
from app.services.model_family_torch_backend import TorchSequenceOptions


def write_training_artifacts(paths, cfg, dataset, scaler: dict, report: dict[str, Any]) -> None:
    write_json(paths.scaler, scaler)
    write_json(paths.features, feature_payload(cfg, dataset))
    write_json(paths.version, version_payload(report))
    write_json(paths.report, report)
    write_json(paths.status, status_payload(report["status"], cfg, report.get("validationFailureReason")))


def feature_payload(cfg, dataset: LstmDataset) -> dict[str, Any]:
    return {
        "modelFamily": cfg.family,
        "symbol": cfg.symbol,
        "duration": cfg.duration,
        "featureWindow": cfg.feature_window,
        "minMoveBps": cfg.min_move_bps,
        "columns": dataset.feature_columns,
        "comboSnapshot": dataset.combo_snapshot,
        "learningContext": dataset.learning_context or {},
        "count": len(dataset.feature_columns),
    }


def version_payload(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "modelFamily", "modelVersion", "trainedAt", "returnStats", "minMoveBps",
        "validationGate", "selectedConfidenceThreshold", "candidateStatus", "promotionReason",
        "probabilityCalibration", "probabilitySource",
    )
    return {key: report.get(key) for key in keys}


def backend_options(cfg: ModelFamilyTrainingConfig, input_size: int):
    if cfg.family in JOBLIB_MODEL_FAMILIES:
        return JoblibModelOptions(cfg.family, cfg.seed, cfg.params)
    return TorchSequenceOptions(
        cfg.family, input_size, cfg.hidden_size, cfg.num_layers,
        cfg.learning_rate, cfg.batch_size, cfg.epochs, cfg.seed,
    )


def scaled_split(split, scaler: dict[str, Any]):
    return type(split)(
        apply_standardizer(split.train_x, scaler), split.train_y, split.train_returns,
        apply_standardizer(split.val_x, scaler), split.val_y, split.val_returns,
        apply_standardizer(split.test_x, scaler), split.test_y, split.test_returns,
    )


def attempt_payload_from_report(cfg, report, version, combo_snapshot=None) -> dict[str, Any]:
    return attempt_payload(str(report["status"]), cfg, version, combo_snapshot, report.get("validationFailureReason"))


def attempt_payload(status: str, cfg, version: str, combo_snapshot=None, reason: str | None = None) -> dict[str, Any]:
    payload = status_payload(status, cfg, reason)
    payload["modelVersion"] = version
    if combo_snapshot is not None:
        payload["comboSnapshot"] = combo_snapshot
    return payload


def status_payload(status: str, cfg, reason: str | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "modelFamily": cfg.family,
        "symbol": cfg.symbol,
        "duration": cfg.duration,
        "featureWindow": cfg.feature_window,
        "minMoveBps": cfg.min_move_bps,
        "updatedAt": utc_now(),
        "candidateStatus": candidate_status(status),
    }
    if reason:
        payload["reason"] = reason
    return payload


def record_failed(paths, staging, cfg, version: str, status: str, exc: Exception, write_attempt: bool) -> None:
    reason = str(exc)
    if write_attempt:
        write_json(paths.attempt, attempt_payload(status, cfg, version, reason=reason))
    write_json(staging.status, status_payload(status, cfg, reason))


def sample_counts(split) -> dict[str, int]:
    return {"train": len(split.train_x), "validation": len(split.val_x), "test": len(split.test_x)}


def model_version(cfg) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    bps = f"{cfg.min_move_bps:g}".replace(".", "p")
    return f"{cfg.family}_{cfg.symbol}_{cfg.duration}_w{cfg.feature_window}_m{bps}_e{cfg.epochs}_s{cfg.seed}_{stamp}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_payload(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
