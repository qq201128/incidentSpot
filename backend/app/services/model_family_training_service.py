from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.services.lstm_artifacts import (
    artifact_paths,
    artifact_paths_for_family_root,
    publish_artifacts,
    require_json,
    write_json,
)
from app.services.lstm_feature_builder import LstmDataError, LstmDataset, build_lstm_training_dataset
from app.services.lstm_lifecycle import (
    LSTM_STATUS_SHADOW_ACTIVE,
    candidate_status,
    lifecycle_status,
    promotion_reason,
    publishes_active_artifacts,
)
from app.services.lstm_training_gate import validation_failure_reason, validation_gate
from app.services.lstm_validation import (
    apply_standardizer,
    binary_classification_metrics,
    chronological_split,
    fit_standardizer,
)
from app.services.model_family_config import (
    JOBLIB_MODEL_FAMILIES,
    ModelFamilyTrainingConfig,
    model_family_rule_name,
    validated_model_family_config,
)
from app.services.model_family_joblib_backend import JoblibModelBackend, JoblibModelOptions
from app.services.model_family_training_reports import initial_baseline_report, return_stats
from app.services.model_family_torch_backend import TorchSequenceBackend, TorchSequenceOptions

DatasetBuilder = Callable[[ModelFamilyTrainingConfig], LstmDataset]

def train_model_family(
    config: ModelFamilyTrainingConfig,
    *,
    artifact_root: Path | None = None,
    backend: Any | None = None,
    dataset_builder: DatasetBuilder = build_lstm_training_dataset,
    publish_shadow_active: bool = True,
    publish_trade_active: bool = True,
    write_attempt: bool = True,
    persist_artifacts: bool = True,
    publish_initial_baseline: bool = False,
) -> dict[str, Any]:
    cfg = validated_model_family_config(config)
    paths = artifact_paths(cfg.symbol, cfg.duration, artifact_root, family=cfg.family)
    version = _model_version(cfg)
    staging = artifact_paths_for_family_root(paths.root / "_staging" / version, cfg.family)
    if write_attempt:
        write_json(paths.attempt, _attempt_payload("training", cfg, version))
    try:
        dataset = dataset_builder(cfg)
        if write_attempt:
            write_json(paths.attempt, _attempt_payload("training", cfg, version, dataset.combo_snapshot))
    except LstmDataError as exc:
        _record_failed(paths, staging, cfg, version, "insufficient_samples", exc, write_attempt)
        raise
    except Exception as exc:
        _record_failed(paths, staging, cfg, version, "failed", exc, write_attempt)
        raise
    try:
        return _train_with_dataset(
            cfg,
            dataset,
            paths,
            staging,
            backend or _default_backend(cfg),
            version,
            publish_shadow_active=publish_shadow_active,
            publish_trade_active=publish_trade_active,
            write_attempt=write_attempt,
            persist_artifacts=persist_artifacts,
            publish_initial_baseline=publish_initial_baseline,
        )
    except Exception as exc:
        if write_attempt:
            write_json(paths.attempt, _attempt_payload("failed", cfg, version, dataset.combo_snapshot, str(exc)))
        write_json(staging.status, _status_payload("failed", cfg, str(exc)))
        raise

def publish_model_family_staged_model(
    config: ModelFamilyTrainingConfig,
    report: dict[str, Any],
    *,
    artifact_root: Path | None = None,
) -> None:
    cfg = validated_model_family_config(config)
    paths = artifact_paths(cfg.symbol, cfg.duration, artifact_root, family=cfg.family)
    version = str(report["modelVersion"])
    staging = artifact_paths_for_family_root(paths.root / "_staging" / version, cfg.family)
    publish_artifacts(staging, paths)
    staged_report = require_json(staging.report, f"{cfg.family} staged training report")
    write_json(paths.attempt, _attempt_payload_from_report(cfg, staged_report, version))

def _train_with_dataset(
    cfg: ModelFamilyTrainingConfig,
    dataset: LstmDataset,
    active_paths,
    staging_paths,
    trainer: Any,
    version: str,
    *,
    publish_shadow_active: bool,
    publish_trade_active: bool,
    write_attempt: bool,
    persist_artifacts: bool,
    publish_initial_baseline: bool,
) -> dict[str, Any]:
    split = chronological_split(dataset.x, dataset.y, dataset.future_returns, cfg.train_ratio, cfg.val_ratio)
    scaler = fit_standardizer(split.train_x)
    scaled = _scaled_split(split, scaler)
    losses = trainer.train(
        scaled.train_x,
        scaled.train_y,
        scaled.val_x,
        scaled.val_y,
        options=_backend_options(cfg, len(dataset.feature_columns)),
        model_path=staging_paths.model,
        persist_model=persist_artifacts,
    )
    report = _training_report(cfg, dataset, scaled, trainer, staging_paths.model, losses, version)
    report = initial_baseline_report(report, publish_initial_baseline)
    should_publish = _should_publish(report["status"], publish_shadow_active, publish_trade_active)
    if persist_artifacts or should_publish:
        _write_training_artifacts(staging_paths, cfg, dataset, scaler, report)
    if should_publish:
        publish_artifacts(staging_paths, active_paths)
    if write_attempt:
        write_json(active_paths.attempt, _attempt_payload_from_report(cfg, report, version, dataset.combo_snapshot))
    return report

def _training_report(cfg, dataset, split, backend, model_path, losses, version) -> dict[str, Any]:
    val_prob = _predict_backend(backend, model_path, split.val_x)
    test_prob = _predict_backend(backend, model_path, split.test_x)
    val_metrics = binary_classification_metrics(split.val_y, val_prob, split.val_returns)
    test_metrics = binary_classification_metrics(split.test_y, test_prob, split.test_returns)
    gate = validation_gate(val_metrics, test_metrics)
    status = lifecycle_status(gate, val_metrics, test_metrics)
    return _finite_payload({
        "status": status,
        "modelFamily": cfg.family,
        "modelVersion": version,
        "ruleName": model_family_rule_name(cfg.family),
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
        "returnStats": return_stats(dataset.future_returns),
        "splitPolicy": "chronological_train_validation_test_no_shuffle",
        "params": cfg.params,
    })


def _write_training_artifacts(paths, cfg, dataset, scaler: dict, report: dict[str, Any]) -> None:
    write_json(paths.scaler, scaler)
    write_json(paths.features, _feature_payload(cfg, dataset))
    write_json(paths.version, _version_payload(report))
    write_json(paths.report, report)
    write_json(paths.status, _status_payload(report["status"], cfg, report.get("validationFailureReason")))

def _predict_backend(backend, model_path: Path, x: np.ndarray) -> np.ndarray:
    if hasattr(backend, "predict_trained"):
        return backend.predict_trained(x)
    return backend.predict(model_path, x)


def _feature_payload(cfg, dataset: LstmDataset) -> dict[str, Any]:
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


def _version_payload(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "modelFamily", "modelVersion", "trainedAt", "returnStats", "minMoveBps",
        "validationGate", "selectedConfidenceThreshold", "candidateStatus", "promotionReason",
    )
    return {key: report.get(key) for key in keys}


def _backend_options(cfg: ModelFamilyTrainingConfig, input_size: int):
    if cfg.family in JOBLIB_MODEL_FAMILIES:
        return JoblibModelOptions(cfg.family, cfg.seed, cfg.params)
    return TorchSequenceOptions(
        cfg.family, input_size, cfg.hidden_size, cfg.num_layers,
        cfg.learning_rate, cfg.batch_size, cfg.epochs, cfg.seed,
    )


def _default_backend(cfg: ModelFamilyTrainingConfig):
    return JoblibModelBackend() if cfg.family in JOBLIB_MODEL_FAMILIES else TorchSequenceBackend()


def _scaled_split(split, scaler: dict[str, Any]):
    return type(split)(
        apply_standardizer(split.train_x, scaler), split.train_y, split.train_returns,
        apply_standardizer(split.val_x, scaler), split.val_y, split.val_returns,
        apply_standardizer(split.test_x, scaler), split.test_y, split.test_returns,
    )


def _should_publish(status: str, publish_shadow_active: bool, publish_trade_active: bool) -> bool:
    if status == LSTM_STATUS_SHADOW_ACTIVE:
        return publish_shadow_active
    return publish_trade_active and publishes_active_artifacts(status)


def _attempt_payload_from_report(cfg, report, version, combo_snapshot=None) -> dict[str, Any]:
    return _attempt_payload(str(report["status"]), cfg, version, combo_snapshot, report.get("validationFailureReason"))


def _attempt_payload(status: str, cfg, version: str, combo_snapshot=None, reason: str | None = None) -> dict[str, Any]:
    payload = _status_payload(status, cfg, reason)
    payload["modelVersion"] = version
    if combo_snapshot is not None:
        payload["comboSnapshot"] = combo_snapshot
    return payload


def _status_payload(status: str, cfg, reason: str | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "modelFamily": cfg.family,
        "symbol": cfg.symbol,
        "duration": cfg.duration,
        "featureWindow": cfg.feature_window,
        "minMoveBps": cfg.min_move_bps,
        "updatedAt": _utc_now(),
        "candidateStatus": candidate_status(status),
    }
    if reason:
        payload["reason"] = reason
    return payload


def _record_failed(paths, staging, cfg, version: str, status: str, exc: Exception, write_attempt: bool) -> None:
    reason = str(exc)
    if write_attempt:
        write_json(paths.attempt, _attempt_payload(status, cfg, version, reason=reason))
    write_json(staging.status, _status_payload(status, cfg, reason))


def _sample_counts(split) -> dict[str, int]:
    return {"train": len(split.train_x), "validation": len(split.val_x), "test": len(split.test_x)}


def _model_version(cfg) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    bps = f"{cfg.min_move_bps:g}".replace(".", "p")
    return f"{cfg.family}_{cfg.symbol}_{cfg.duration}_w{cfg.feature_window}_m{bps}_e{cfg.epochs}_s{cfg.seed}_{stamp}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_payload(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
