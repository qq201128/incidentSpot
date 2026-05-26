from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any

import numpy as np

from app.services.lstm_artifacts import write_json
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_feature_builder import LstmDataset
from app.services.lstm_lifecycle import candidate_status
from app.services.lstm_torch_backend import TorchLstmOptions
from app.services.lstm_validation import apply_standardizer


def write_training_artifacts(paths, cfg, dataset: LstmDataset, scaler: dict, report: dict[str, Any]) -> None:
    write_json(paths.scaler, scaler)
    write_json(paths.features, feature_payload(cfg, dataset))
    write_json(paths.version, version_payload(report))
    write_json(paths.report, report)
    write_json(paths.status, status_payload(report["status"], cfg, report.get("validationFailureReason")))


def scaled_split(split, scaler: dict[str, Any]):
    return type(split)(
        apply_standardizer(split.train_x, scaler), split.train_y, split.train_returns,
        apply_standardizer(split.val_x, scaler), split.val_y, split.val_returns,
        apply_standardizer(split.test_x, scaler), split.test_y, split.test_returns,
    )


def torch_options(cfg: LstmTrainingConfig, input_size: int) -> TorchLstmOptions:
    return TorchLstmOptions(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        seed=cfg.seed,
    )


def sample_counts(split) -> dict[str, int]:
    return {"train": len(split.train_x), "validation": len(split.val_x), "test": len(split.test_x)}


def feature_payload(cfg: LstmTrainingConfig, dataset: LstmDataset) -> dict[str, Any]:
    return {
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
    return {
        "modelVersion": report["modelVersion"],
        "trainedAt": report["trainedAt"],
        "returnStats": report["returnStats"],
        "minMoveBps": report.get("minMoveBps"),
        "validationGate": report.get("validationGate"),
        "selectedConfidenceThreshold": report.get("selectedConfidenceThreshold"),
        "candidateStatus": report.get("candidateStatus"),
        "promotionReason": report.get("promotionReason"),
    }


def status_payload(status: str, cfg: LstmTrainingConfig, reason: str | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "symbol": cfg.symbol.strip().upper(),
        "duration": cfg.duration,
        "featureWindow": cfg.feature_window,
        "minMoveBps": cfg.min_move_bps,
        "updatedAt": utc_now(),
    }
    if reason:
        payload["reason"] = reason
    payload["candidateStatus"] = candidate_status(status)
    return payload


def attempt_payload(
    status: str,
    cfg: LstmTrainingConfig,
    model_version: str,
    reason: str | None = None,
    *,
    promotion_reason: str | None = None,
    combo_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = status_payload(status, cfg, reason)
    payload["modelVersion"] = model_version
    if promotion_reason:
        payload["promotionReason"] = promotion_reason
    if combo_snapshot is not None:
        payload["comboSnapshot"] = combo_snapshot
    return payload


def write_failed_staging_status(staging_paths, cfg: LstmTrainingConfig, status: str, reason: str) -> None:
    staging_paths.root.mkdir(parents=True, exist_ok=True)
    write_json(staging_paths.status, status_payload(status, cfg, reason))


def return_stats(returns: np.ndarray) -> dict[str, float]:
    up = returns[returns > 0]
    down = returns[returns <= 0]
    return {"mean": mean(returns), "upMean": mean(up), "downMean": mean(down)}


def model_version(cfg: LstmTrainingConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    bps = f"{cfg.min_move_bps:g}".replace(".", "p")
    return f"lstm_{cfg.symbol}_{cfg.duration}_w{cfg.feature_window}_m{bps}_e{cfg.epochs}_s{cfg.seed}_{stamp}"


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


def mean(values: np.ndarray) -> float:
    return 0.0 if len(values) == 0 else float(np.mean(values))
