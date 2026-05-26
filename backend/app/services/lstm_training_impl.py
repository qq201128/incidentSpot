from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
from app.services.lstm_torch_backend import TorchLstmBackend
from app.services.lstm_training_payloads import (
    attempt_payload,
    finite_payload,
    model_version,
    return_stats,
    sample_counts,
    scaled_split,
    torch_options,
    utc_now,
    write_failed_staging_status,
    write_training_artifacts,
)
from app.services.lstm_training_gate import validation_failure_reason, validation_gate
from app.services.lstm_validation import (
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
    version = model_version(cfg)
    staging = artifact_paths_for_root(paths.root / "_staging" / version)
    if write_attempt:
        write_json(paths.attempt, attempt_payload("training", cfg, version))
    dataset: LstmDataset | None = None
    try:
        dataset = dataset_builder(cfg)
        if write_attempt:
            write_json(
                paths.attempt,
                attempt_payload("training", cfg, version, combo_snapshot=dataset.combo_snapshot),
            )
    except LstmDataError as exc:
        reason = str(exc)
        if write_attempt:
            write_json(paths.attempt, attempt_payload("insufficient_samples", cfg, version, reason))
        write_failed_staging_status(staging, cfg, "insufficient_samples", reason)
        raise
    except Exception as exc:
        reason = str(exc)
        if write_attempt:
            write_json(paths.attempt, attempt_payload("failed", cfg, version, reason))
        write_failed_staging_status(staging, cfg, "failed", reason)
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
                    attempt_payload(
                    "failed",
                    cfg,
                    version,
                    reason,
                    combo_snapshot=dataset.combo_snapshot if dataset else None,
                ),
            )
        write_failed_staging_status(staging, cfg, "failed", reason)
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
    scaled = scaled_split(split, scaler)
    losses = trainer.train(
        scaled.train_x,
        scaled.train_y,
        scaled.val_x,
        scaled.val_y,
        options=torch_options(cfg, len(dataset.feature_columns)),
        model_path=staging_paths.model,
    )
    report = _training_report(cfg, dataset, scaled, trainer, staging_paths.model, losses, version)
    write_training_artifacts(staging_paths, cfg, dataset, scaler, report)
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
    return attempt_payload(
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
    return finite_payload({
        "status": status,
        "modelVersion": version,
        "ruleName": LSTM_RULE_NAME,
        "symbol": cfg.symbol,
        "duration": cfg.duration,
        "trainedAt": utc_now(),
        "featureWindow": cfg.feature_window,
        "horizonMinutes": cfg.horizon_minutes,
        "minMoveBps": cfg.min_move_bps,
        "sampleCounts": sample_counts(split),
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
    })
