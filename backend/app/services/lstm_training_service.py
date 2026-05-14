from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.services.lstm_artifacts import artifact_paths, write_json
from app.services.lstm_config import LSTM_RULE_NAME, LstmTrainingConfig, validated_lstm_config
from app.services.lstm_feature_builder import LstmDataError, LstmDataset, build_lstm_training_dataset
from app.services.lstm_torch_backend import TorchLstmBackend, TorchLstmOptions
from app.services.lstm_validation import (
    apply_standardizer,
    binary_classification_metrics,
    chronological_split,
    fit_standardizer,
)

DatasetBuilder = Callable[[LstmTrainingConfig], LstmDataset]
TRADE_GATE_THRESHOLDS = (0.60, 0.65, 0.70)
MIN_VALIDATION_WIN_RATE = 0.5
MIN_VALIDATION_PROFIT_FACTOR = 1.0
MIN_VALIDATION_AVG_RETURN = 0.0


def train_lstm_model(
    config: LstmTrainingConfig,
    *,
    artifact_root: Path | None = None,
    backend: Any | None = None,
    dataset_builder: DatasetBuilder = build_lstm_training_dataset,
) -> dict[str, Any]:
    cfg = validated_lstm_config(config)
    paths = artifact_paths(cfg.symbol, cfg.duration, artifact_root)
    write_json(paths.status, _status_payload("training", cfg))
    try:
        dataset = dataset_builder(cfg)
    except LstmDataError as exc:
        write_json(paths.status, _status_payload("insufficient_samples", cfg, str(exc)))
        raise
    try:
        return _train_with_dataset(cfg, dataset, paths, backend or TorchLstmBackend())
    except Exception as exc:
        write_json(paths.status, _status_payload("failed", cfg, str(exc)))
        raise


def _train_with_dataset(
    cfg: LstmTrainingConfig,
    dataset: LstmDataset,
    paths,
    trainer: Any,
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
        model_path=paths.model,
    )
    report = _training_report(cfg, dataset, scaled, trainer, paths.model, losses)
    _write_training_artifacts(paths, cfg, dataset, scaler, report)
    return report


def _training_report(
    cfg: LstmTrainingConfig,
    dataset: LstmDataset,
    split,
    backend: Any,
    model_path: Path,
    losses: dict[str, Any],
) -> dict[str, Any]:
    val_prob = backend.predict(model_path, split.val_x)
    test_prob = backend.predict(model_path, split.test_x)
    val_metrics = binary_classification_metrics(split.val_y, val_prob, split.val_returns)
    test_metrics = binary_classification_metrics(split.test_y, test_prob, split.test_returns)
    validation_gate = _validation_gate(val_metrics)
    version = _model_version(cfg)
    status = "trained" if validation_gate["status"] == "passed" else "validation_failed"
    return _finite_payload({
        "status": status,
        "modelVersion": version,
        "ruleName": LSTM_RULE_NAME,
        "symbol": cfg.symbol,
        "duration": cfg.duration,
        "trainedAt": _utc_now(),
        "featureWindow": cfg.feature_window,
        "horizonMinutes": cfg.horizon_minutes,
        "sampleCounts": _sample_counts(split),
        "validation": val_metrics,
        "test": test_metrics,
        "outOfSample": {"validation": val_metrics, "test": test_metrics},
        "validationGate": validation_gate,
        "selectedConfidenceThreshold": validation_gate.get("minConfidence"),
        "validationFailureReason": _validation_failure_reason(validation_gate),
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
        "columns": dataset.feature_columns,
        "comboSnapshot": dataset.combo_snapshot,
        "count": len(dataset.feature_columns),
    }


def _version_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "modelVersion": report["modelVersion"],
        "trainedAt": report["trainedAt"],
        "returnStats": report["returnStats"],
        "validationGate": report.get("validationGate"),
        "selectedConfidenceThreshold": report.get("selectedConfidenceThreshold"),
    }


def _status_payload(status: str, cfg: LstmTrainingConfig, reason: str | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "symbol": cfg.symbol.strip().upper(),
        "duration": cfg.duration,
        "featureWindow": cfg.feature_window,
        "updatedAt": _utc_now(),
    }
    if reason:
        payload["reason"] = reason
    return payload


def _return_stats(returns: np.ndarray) -> dict[str, float]:
    up = returns[returns > 0]
    down = returns[returns <= 0]
    return {
        "mean": _mean(returns),
        "upMean": _mean(up),
        "downMean": _mean(down),
    }


def _validation_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    criteria = _validation_gate_criteria()
    candidates = [
        row for row in metrics.get("confidenceThresholds", [])
        if float(row.get("minConfidence") or 0.0) in TRADE_GATE_THRESHOLDS
    ]
    for row in candidates:
        if _threshold_passes(row):
            return {"status": "passed", "criteria": criteria, **row}
    return {
        "status": "failed",
        "reason": "no_validation_confidence_threshold_met",
        "criteria": criteria,
        "candidates": candidates,
    }


def _threshold_passes(row: dict[str, Any]) -> bool:
    return (
        row.get("winRate") is not None
        and float(row["winRate"]) > MIN_VALIDATION_WIN_RATE
        and row.get("profitFactor") is not None
        and float(row["profitFactor"]) > MIN_VALIDATION_PROFIT_FACTOR
        and row.get("avgReturn") is not None
        and float(row["avgReturn"]) > MIN_VALIDATION_AVG_RETURN
    )


def _validation_gate_criteria() -> dict[str, Any]:
    return {
        "thresholds": list(TRADE_GATE_THRESHOLDS),
        "minWinRateExclusive": MIN_VALIDATION_WIN_RATE,
        "minProfitFactorExclusive": MIN_VALIDATION_PROFIT_FACTOR,
        "minAvgReturnExclusive": MIN_VALIDATION_AVG_RETURN,
    }


def _validation_failure_reason(validation_gate: dict[str, Any]) -> str | None:
    if validation_gate["status"] == "passed":
        return None
    return str(validation_gate["reason"])


def _model_version(cfg: LstmTrainingConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"lstm_{cfg.symbol}_{cfg.duration}_w{cfg.feature_window}_{stamp}"


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
        return None if value < 0 else 999999.0
    return value
