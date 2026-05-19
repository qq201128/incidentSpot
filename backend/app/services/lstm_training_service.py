from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.services.lstm_artifacts import (
    artifact_paths,
    artifact_paths_for_root,
    publish_artifacts,
    require_json,
    write_json,
)
from app.services.lstm_config import LSTM_RULE_NAME, LstmTrainingConfig, validated_lstm_config
from app.services.lstm_feature_builder import LstmDataError, LstmDataset, build_lstm_training_dataset
from app.services.lstm_lifecycle import (
    LSTM_STATUS_SHADOW_ACTIVE,
    candidate_status,
    lifecycle_status,
    promotion_reason,
    publishes_active_artifacts,
)
from app.services.lstm_torch_backend import TorchLstmBackend, TorchLstmOptions
from app.services.lstm_training_gate import validation_failure_reason, validation_gate
from app.services.lstm_validation import (
    apply_standardizer,
    binary_classification_metrics,
    chronological_split,
    fit_standardizer,
)

DatasetBuilder = Callable[[LstmTrainingConfig], LstmDataset]


def train_lstm_model(
    config: LstmTrainingConfig,
    *,
    artifact_root: Path | None = None,
    backend: Any | None = None,
    dataset_builder: DatasetBuilder = build_lstm_training_dataset,
    publish_shadow_active: bool = True,
    publish_trade_active: bool = True,
    write_attempt: bool = True,
) -> dict[str, Any]:
    cfg = validated_lstm_config(config)
    paths = artifact_paths(cfg.symbol, cfg.duration, artifact_root)
    version = _model_version(cfg)
    staging = artifact_paths_for_root(paths.root / "_staging" / version)
    if write_attempt:
        write_json(paths.attempt, _attempt_payload("training", cfg, version))
    dataset: LstmDataset | None = None
    try:
        dataset = dataset_builder(cfg)
        if write_attempt:
            write_json(
                paths.attempt,
                _attempt_payload("training", cfg, version, combo_snapshot=dataset.combo_snapshot),
            )
    except LstmDataError as exc:
        reason = str(exc)
        if write_attempt:
            write_json(paths.attempt, _attempt_payload("insufficient_samples", cfg, version, reason))
        _write_failed_staging_status(staging, cfg, "insufficient_samples", reason)
        raise
    except Exception as exc:
        reason = str(exc)
        if write_attempt:
            write_json(paths.attempt, _attempt_payload("failed", cfg, version, reason))
        _write_failed_staging_status(staging, cfg, "failed", reason)
        raise
    try:
        return _train_with_dataset(
            cfg,
            dataset,
            paths,
            staging,
            backend or TorchLstmBackend(),
            version,
            publish_shadow_active=publish_shadow_active,
            publish_trade_active=publish_trade_active,
            write_attempt=write_attempt,
        )
    except Exception as exc:
        reason = str(exc)
        if write_attempt:
            write_json(
                paths.attempt,
                _attempt_payload(
                    "failed",
                    cfg,
                    version,
                    reason,
                    combo_snapshot=dataset.combo_snapshot if dataset else None,
                ),
            )
        _write_failed_staging_status(staging, cfg, "failed", reason)
        raise


def _train_with_dataset(
    cfg: LstmTrainingConfig,
    dataset: LstmDataset,
    active_paths,
    staging_paths,
    trainer: Any,
    version: str,
    *,
    publish_shadow_active: bool,
    publish_trade_active: bool,
    write_attempt: bool,
) -> dict[str, Any]:
    split = chronological_split(dataset.x, dataset.y, dataset.future_returns, cfg.train_ratio, cfg.val_ratio)
    scaler = fit_standardizer(split.train_x)
    scaled = _scaled_split(split, scaler)
    losses = trainer.train(
        scaled.train_x,
        scaled.train_y,
        scaled.val_x,
        scaled.val_y,
        options=_torch_options(cfg, len(dataset.feature_columns)),
        model_path=staging_paths.model,
    )
    report = _training_report(cfg, dataset, scaled, trainer, staging_paths.model, losses, version)
    _write_training_artifacts(staging_paths, cfg, dataset, scaler, report)
    if _should_publish_active_artifacts(report["status"], publish_shadow_active, publish_trade_active):
        publish_artifacts(staging_paths, active_paths)
    if write_attempt:
        _write_attempt_from_report(active_paths, cfg, dataset, report, version)
    return report


def publish_lstm_staged_model(
    config: LstmTrainingConfig,
    report: dict[str, Any],
    *,
    artifact_root: Path | None = None,
) -> None:
    cfg = validated_lstm_config(config)
    paths = artifact_paths(cfg.symbol, cfg.duration, artifact_root)
    version = str(report["modelVersion"])
    staging = artifact_paths_for_root(paths.root / "_staging" / version)
    publish_artifacts(staging, paths)
    staged_report = require_json(staging.report, "LSTM staged training report")
    write_json(paths.attempt, _attempt_payload_from_report(cfg, staged_report, version))


def _should_publish_active_artifacts(
    status: str,
    publish_shadow_active: bool,
    publish_trade_active: bool,
) -> bool:
    if status == LSTM_STATUS_SHADOW_ACTIVE:
        return publish_shadow_active
    if publishes_active_artifacts(status):
        return publish_trade_active
    return False


def _write_attempt_from_report(
    active_paths,
    cfg: LstmTrainingConfig,
    dataset: LstmDataset,
    report: dict[str, Any],
    version: str,
) -> None:
    payload = _attempt_payload_from_report(cfg, report, version, combo_snapshot=dataset.combo_snapshot)
    write_json(active_paths.attempt, payload)


def _attempt_payload_from_report(
    cfg: LstmTrainingConfig,
    report: dict[str, Any],
    version: str,
    *,
    combo_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _attempt_payload(
        str(report["status"]),
        cfg,
        version,
        report.get("validationFailureReason"),
        promotion_reason=report.get("promotionReason"),
        combo_snapshot=combo_snapshot,
    )


def _training_report(
    cfg: LstmTrainingConfig,
    dataset: LstmDataset,
    split,
    backend: Any,
    model_path: Path,
    losses: dict[str, Any],
    version: str,
) -> dict[str, Any]:
    val_prob = backend.predict(model_path, split.val_x)
    test_prob = backend.predict(model_path, split.test_x)
    val_metrics = binary_classification_metrics(split.val_y, val_prob, split.val_returns)
    test_metrics = binary_classification_metrics(split.test_y, test_prob, split.test_returns)
    gate = validation_gate(val_metrics, test_metrics)
    status = lifecycle_status(gate, val_metrics, test_metrics)
    return _finite_payload({
        "status": status,
        "modelVersion": version,
        "ruleName": LSTM_RULE_NAME,
        "symbol": cfg.symbol,
        "duration": cfg.duration,
        "trainedAt": _utc_now(),
        "featureWindow": cfg.feature_window,
        "horizonMinutes": cfg.horizon_minutes,
        "minMoveBps": cfg.min_move_bps,
        "sampleCounts": _sample_counts(split),
        "validation": val_metrics,
        "test": test_metrics,
        "outOfSample": {"validation": val_metrics, "test": test_metrics},
        "validationGate": gate,
        "selectedConfidenceThreshold": gate.get("minConfidence"),
        "validationFailureReason": validation_failure_reason(gate),
        "candidateStatus": candidate_status(status),
        "promotionReason": promotion_reason(status, gate),
        "losses": losses,
        "returnStats": _return_stats(dataset.future_returns),
        "splitPolicy": "chronological_train_validation_test_no_shuffle",
    })


def _write_training_artifacts(paths, cfg, dataset: LstmDataset, scaler: dict, report: dict[str, Any]) -> None:
    write_json(paths.scaler, scaler)
    write_json(paths.features, _feature_payload(cfg, dataset))
    write_json(paths.version, _version_payload(report))
    write_json(paths.report, report)
    write_json(paths.status, _status_payload(report["status"], cfg, report.get("validationFailureReason")))


def _scaled_split(split, scaler: dict[str, Any]):
    return type(split)(
        apply_standardizer(split.train_x, scaler), split.train_y, split.train_returns,
        apply_standardizer(split.val_x, scaler), split.val_y, split.val_returns,
        apply_standardizer(split.test_x, scaler), split.test_y, split.test_returns,
    )


def _torch_options(cfg: LstmTrainingConfig, input_size: int) -> TorchLstmOptions:
    return TorchLstmOptions(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        seed=cfg.seed,
    )


def _sample_counts(split) -> dict[str, int]:
    return {"train": len(split.train_x), "validation": len(split.val_x), "test": len(split.test_x)}


def _feature_payload(cfg: LstmTrainingConfig, dataset: LstmDataset) -> dict[str, Any]:
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


def _version_payload(report: dict[str, Any]) -> dict[str, Any]:
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


def _status_payload(status: str, cfg: LstmTrainingConfig, reason: str | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "symbol": cfg.symbol.strip().upper(),
        "duration": cfg.duration,
        "featureWindow": cfg.feature_window,
        "minMoveBps": cfg.min_move_bps,
        "updatedAt": _utc_now(),
    }
    if reason:
        payload["reason"] = reason
    payload["candidateStatus"] = candidate_status(status)
    return payload


def _attempt_payload(
    status: str,
    cfg: LstmTrainingConfig,
    model_version: str,
    reason: str | None = None,
    *,
    promotion_reason: str | None = None,
    combo_snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = _status_payload(status, cfg, reason)
    payload["modelVersion"] = model_version
    if promotion_reason:
        payload["promotionReason"] = promotion_reason
    if combo_snapshot is not None:
        payload["comboSnapshot"] = combo_snapshot
    return payload


def _write_failed_staging_status(
    staging_paths,
    cfg: LstmTrainingConfig,
    status: str,
    reason: str,
) -> None:
    staging_paths.root.mkdir(parents=True, exist_ok=True)
    write_json(staging_paths.status, _status_payload(status, cfg, reason))


def _return_stats(returns: np.ndarray) -> dict[str, float]:
    up = returns[returns > 0]
    down = returns[returns <= 0]
    return {
        "mean": _mean(returns),
        "upMean": _mean(up),
        "downMean": _mean(down),
    }


def _model_version(cfg: LstmTrainingConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    bps = f"{cfg.min_move_bps:g}".replace(".", "p")
    return f"lstm_{cfg.symbol}_{cfg.duration}_w{cfg.feature_window}_m{bps}_e{cfg.epochs}_s{cfg.seed}_{stamp}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: np.ndarray) -> float:
    return 0.0 if len(values) == 0 else float(np.mean(values))


def _finite_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_payload(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
