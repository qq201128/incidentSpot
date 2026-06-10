from __future__ import annotations

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
    apply_probability_calibrator,
    binary_classification_metrics,
    calibration_report,
    chronological_split,
    fit_probability_calibrator,
    fit_standardizer,
)
from app.services.model_family_config import (
    JOBLIB_MODEL_FAMILIES,
    ModelFamilyTrainingConfig,
    model_family_rule_name,
    validated_model_family_config,
)
from app.services.model_family_joblib_backend import JoblibModelBackend
from app.services.model_family_paper_live_policy import paper_live_admission_payload
from app.services.model_family_regime_reports import regime_validation_report
from app.services.model_family_relative_promotion import relative_shadow_report
from app.services.model_family_training_payloads import (
    attempt_payload,
    attempt_payload_from_report,
    backend_options,
    finite_payload,
    model_version,
    record_failed,
    sample_counts,
    scaled_split,
    status_payload,
    utc_now,
    write_training_artifacts,
)
from app.services.model_family_training_reports import initial_baseline_report, return_stats
from app.services.model_family_torch_backend import TorchSequenceBackend
from app.services.trading_costs import default_backtest_cost_config, roundtrip_cost_rate

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
    evaluate_test: bool = True,
    active_status_loader: Callable[[str, str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cfg = validated_model_family_config(config)
    paths = artifact_paths(cfg.symbol, cfg.duration, artifact_root, family=cfg.family)
    version = model_version(cfg)
    staging = artifact_paths_for_family_root(paths.root / "_staging" / version, cfg.family)
    if write_attempt:
        write_json(paths.attempt, attempt_payload("training", cfg, version))
    try:
        dataset = dataset_builder(cfg)
        if write_attempt:
            write_json(paths.attempt, attempt_payload("training", cfg, version, dataset.combo_snapshot))
    except LstmDataError as exc:
        record_failed(paths, staging, cfg, version, "insufficient_samples", exc, write_attempt)
        raise
    except Exception as exc:
        record_failed(paths, staging, cfg, version, "failed", exc, write_attempt)
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
            evaluate_test=evaluate_test,
            active_status_loader=active_status_loader,
        )
    except Exception as exc:
        if write_attempt:
            write_json(paths.attempt, attempt_payload("failed", cfg, version, dataset.combo_snapshot, str(exc)))
        write_json(staging.status, status_payload("failed", cfg, str(exc)))
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
    write_json(paths.attempt, attempt_payload_from_report(cfg, staged_report, version))

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
    evaluate_test: bool,
    active_status_loader: Callable[[str, str, str], dict[str, Any]] | None,
) -> dict[str, Any]:
    split = chronological_split(dataset.x, dataset.y, dataset.future_returns, cfg.train_ratio, cfg.val_ratio)
    scaler = fit_standardizer(split.train_x)
    training_split = _training_split(trainer, split, scaler)
    losses = trainer.train(
        training_split.train_x,
        training_split.train_y,
        training_split.val_x,
        training_split.val_y,
        options=backend_options(cfg, len(dataset.feature_columns)),
        model_path=staging_paths.model,
        persist_model=persist_artifacts,
        **_torch_train_kwargs(trainer, training_split, scaler),
    )
    report = _training_report(cfg, dataset, training_split, trainer, staging_paths.model, losses, version, evaluate_test)
    report = initial_baseline_report(report, publish_initial_baseline)
    if active_status_loader is not None:
        report = relative_shadow_report(report, active_status_loader(cfg.family, cfg.symbol, cfg.duration))
    should_publish = _should_publish(report["status"], publish_shadow_active, publish_trade_active)
    if persist_artifacts or should_publish:
        write_training_artifacts(staging_paths, cfg, dataset, scaler, report)
    if should_publish:
        publish_artifacts(staging_paths, active_paths)
    if write_attempt:
        write_json(active_paths.attempt, attempt_payload_from_report(cfg, report, version, dataset.combo_snapshot))
    return report

def _training_report(cfg, dataset, split, backend, model_path, losses, version, evaluate_test: bool) -> dict[str, Any]:
    val_prob = _predict_backend(backend, model_path, split.val_x)
    calibrator = fit_probability_calibrator(split.val_y, val_prob)
    calibrated_val_prob = apply_probability_calibrator(val_prob, calibrator)
    val_metrics = binary_classification_metrics(split.val_y, calibrated_val_prob, split.val_returns)
    test_metrics, test_calibration, raw_test, raw_test_calibration = _test_payloads(
        split, backend, model_path, calibrator, evaluate_test
    )
    gate = validation_gate(val_metrics, test_metrics, require_test=evaluate_test)
    status = lifecycle_status(gate, val_metrics, test_metrics)
    return finite_payload({
        "status": status,
        "modelFamily": cfg.family,
        "modelVersion": version,
        "ruleName": model_family_rule_name(cfg.family),
        "symbol": cfg.symbol,
        "duration": cfg.duration,
        "trainedAt": utc_now(),
        "featureWindow": cfg.feature_window,
        "horizonMinutes": cfg.horizon_minutes,
        "minMoveBps": cfg.min_move_bps,
        "sampleCounts": sample_counts(split),
        "validation": val_metrics,
        "test": test_metrics,
        "rawValidation": binary_classification_metrics(split.val_y, val_prob, split.val_returns),
        "rawTest": raw_test,
        "regimeValidation": regime_validation_report(dataset, split, calibrated_val_prob),
        "probabilityCalibration": {
            "calibrator": calibrator,
            "validation": calibration_report(split.val_y, calibrated_val_prob, split.val_returns),
            "test": test_calibration,
            "rawValidation": calibration_report(split.val_y, val_prob, split.val_returns),
            "rawTest": raw_test_calibration,
        },
        "outOfSample": {"validation": val_metrics, "test": test_metrics},
        "validationGate": gate,
        "paperLiveAdmission": paper_live_admission_payload(status, gate),
        "realTradingEnabled": False,
        "selectedConfidenceThreshold": gate.get("minConfidence"),
        "validationFailureReason": validation_failure_reason(gate),
        "candidateStatus": candidate_status(status),
        "promotionReason": promotion_reason(status, gate),
        "losses": losses,
        "returnStats": return_stats(dataset.future_returns),
        "featureColumns": dataset.feature_columns,
        "explicitFactorComboFeatures": _factor_combo_report(dataset),
        "simFeedback": dataset.sim_feedback_metadata or {"enabled": False},
        "dataQuality": dataset.data_quality_report or {"status": "unknown"},
        "tradingCosts": _trading_cost_report(),
        "splitPolicy": "chronological_train_validation_test_no_shuffle",
        "testEvaluationPolicy": "final_candidate_only" if not evaluate_test else "evaluated",
        "probabilitySource": "calibrated_platt" if calibrator.get("status") == "fitted" else "raw_uncalibrated",
        "params": cfg.params,
    })


def _test_payloads(
    split,
    backend,
    model_path,
    calibrator,
    evaluate_test: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not evaluate_test:
        withheld = {"status": "withheld", "reason": "test_set_reserved_for_final_candidate", "sampleCount": len(split.test_y)}
        return withheld, withheld, withheld, withheld
    test_prob = _predict_backend(backend, model_path, split.test_x)
    calibrated = apply_probability_calibrator(test_prob, calibrator)
    metrics = binary_classification_metrics(split.test_y, calibrated, split.test_returns)
    raw = binary_classification_metrics(split.test_y, test_prob, split.test_returns)
    raw_cal = calibration_report(split.test_y, test_prob, split.test_returns)
    return metrics, calibration_report(split.test_y, calibrated, split.test_returns), raw, raw_cal

def _predict_backend(backend, model_path: Path, x: np.ndarray) -> np.ndarray:
    if hasattr(backend, "predict_trained"):
        return backend.predict_trained(x)
    return backend.predict(model_path, x)


def _training_split(trainer: Any, split, scaler: dict[str, Any]):
    if isinstance(trainer, TorchSequenceBackend):
        return split
    return scaled_split(split, scaler)


def _torch_train_kwargs(trainer: Any, split, scaler: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(trainer, TorchSequenceBackend):
        return {}
    return {"train_returns": split.train_returns, "val_returns": split.val_returns, "scaler": scaler}


def _factor_combo_report(dataset: LstmDataset) -> dict[str, Any]:
    columns = [column for column in dataset.feature_columns if column.startswith("factor_combo_")]
    metadata = dataset.factor_combo_metadata or {"enabled": False}
    return {**metadata, "included": bool(columns), "columns": columns}


def _trading_cost_report() -> dict[str, Any]:
    config = default_backtest_cost_config()
    return {
        "feeRatePerSide": float(config.fee_rate_per_side),
        "slippageRatePerSide": float(config.slippage_rate_per_side),
        "roundtripCostRate": float(roundtrip_cost_rate(config)),
        "minTradeGapMinutes": int(config.min_trade_gap_minutes),
    }


def _default_backend(cfg: ModelFamilyTrainingConfig):
    return JoblibModelBackend() if cfg.family in JOBLIB_MODEL_FAMILIES else TorchSequenceBackend()


def _should_publish(status: str, publish_shadow_active: bool, publish_trade_active: bool) -> bool:
    if status == LSTM_STATUS_SHADOW_ACTIVE:
        return publish_shadow_active
    return publish_trade_active and publishes_active_artifacts(status)
