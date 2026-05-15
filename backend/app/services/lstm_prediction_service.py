from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.lstm_artifacts import (
    artifact_paths,
    read_json,
    require_json,
    required_artifacts_exist,
)
from app.services.lstm_combo_snapshot import combo_snapshot_status
from app.services.lstm_config import LSTM_RULE_NAME, lstm_shadow_strategy_key
from app.services.lstm_feature_builder import build_live_feature_window
from app.services.lstm_torch_backend import TorchLstmBackend, torch_availability
from app.services.lstm_validation import apply_standardizer


def lstm_model_status(symbol: str, duration: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root)
    status = read_json(paths.status) or _untrained_status(sym, duration)
    attempt = read_json(paths.attempt) or {}
    report = read_json(paths.report) or {}
    version = read_json(paths.version) or {}
    snapshot = combo_snapshot_status(sym, duration, artifact_root=artifact_root)
    artifacts_ready = required_artifacts_exist(paths)
    torch_status = torch_availability()
    torch_available = bool(torch_status["available"])
    ready_reason = _shadow_prediction_ready_reason(
        status,
        version,
        report,
        artifacts_ready,
        snapshot,
        torch_available,
    )
    return {
        **status,
        "strategyKey": lstm_shadow_strategy_key(duration),
        "modelVersion": version.get("modelVersion") or report.get("modelVersion"),
        "trainedAt": version.get("trainedAt") or report.get("trainedAt"),
        "sampleCounts": report.get("sampleCounts") or {},
        "testAccuracy": (report.get("test") or {}).get("accuracy"),
        "testWinRate": (report.get("test") or {}).get("winRate"),
        "validationGate": version.get("validationGate") or report.get("validationGate"),
        "selectedConfidenceThreshold": (
            version.get("selectedConfidenceThreshold")
            or report.get("selectedConfidenceThreshold")
        ),
        "activeModelStatus": status.get("status"),
        "lastAttemptStatus": attempt.get("status"),
        "lastTrainingAttempt": attempt,
        "validationFailureReason": _validation_failure_reason(status, attempt, report),
        "artifactsReady": artifacts_ready,
        "torchAvailable": torch_available,
        "torchStatus": torch_status,
        "comboSnapshotMatches": snapshot["matches"],
        "comboSnapshotReason": snapshot["reason"],
        "comboSnapshotCurrent": snapshot["current"],
        "comboSnapshotTrained": snapshot["trained"],
        "shadowPredictionReady": ready_reason == "passed",
        "shadowPredictionBlockedReason": ready_reason,
    }


def is_lstm_shadow_ready(symbol: str, duration: str, *, artifact_root: Path | None = None) -> bool:
    status = lstm_model_status(symbol, duration, artifact_root=artifact_root)
    return bool(status.get("shadowPredictionReady"))


def predict_lstm_signal(
    symbol: str,
    duration: str,
    *,
    entry_open_time: int | None = None,
    artifact_root: Path | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root)
    _assert_predictable(sym, duration, paths, artifact_root=artifact_root)
    features = require_json(paths.features, "features")
    scaler = require_json(paths.scaler, "scaler")
    version = require_json(paths.version, "version")
    report = require_json(paths.report, "training report")
    status = require_json(paths.status, "status")
    window, meta = build_live_feature_window(
        sym,
        duration,
        list(features["columns"]),
        int(features["featureWindow"]),
        entry_open_time,
        combo_snapshot=features.get("comboSnapshot"),
    )
    scaled = apply_standardizer(window, scaler)
    probability_up = float((backend or TorchLstmBackend()).predict(paths.model, scaled)[0])
    return _signal_payload(
        sym,
        duration,
        probability_up,
        meta,
        features,
        _prediction_version_payload(version, report),
        status,
    )


def predict_lstm_shadow_prediction(
    symbol: str,
    duration: str,
    *,
    entry_open_time: int | None = None,
) -> dict[str, Any]:
    signal = predict_lstm_signal(symbol, duration, entry_open_time=entry_open_time)
    return _prediction_payload(signal)


def _assert_predictable(
    symbol: str,
    duration: str,
    paths,
    *,
    artifact_root: Path | None = None,
) -> None:
    status = read_json(paths.status) or _untrained_status(symbol, duration)
    if status["status"] != "trained":
        reason = status.get("reason") or "model is not trained"
        raise ValueError(f"LSTM model is not ready for {symbol} {duration}: {reason}")
    if not required_artifacts_exist(paths):
        raise ValueError(f"LSTM model artifacts are incomplete for {symbol} {duration}: {paths.root}")
    version = read_json(paths.version) or {}
    report = read_json(paths.report) or {}
    validation_reason = lstm_validation_block_reason(status, version, report)
    if validation_reason != "passed":
        raise ValueError(f"LSTM model is not ready for {symbol} {duration}: {validation_reason}")
    snapshot = combo_snapshot_status(symbol, duration, artifact_root=artifact_root)
    if not snapshot["matches"]:
        raise ValueError(f"LSTM model is not ready for {symbol} {duration}: {snapshot['reason']}")


def _signal_payload(
    symbol: str,
    duration: str,
    probability_up: float,
    meta: dict[str, Any],
    features: dict[str, Any],
    version: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    prob = max(0.0, min(probability_up, 1.0))
    direction = "up" if prob >= 0.5 else "down"
    confidence = prob if direction == "up" else 1.0 - prob
    min_confidence = _selected_confidence_threshold(version)
    gate_passed = confidence >= min_confidence
    return {
        "symbol": symbol,
        "strategyKey": lstm_shadow_strategy_key(duration),
        "direction": direction,
        "probabilityUp": round(prob, 6),
        "confidence": round(confidence, 6),
        "selectedConfidenceThreshold": min_confidence,
        "validationGatePassed": gate_passed,
        "expectedReturn": _expected_return(prob, version),
        "modelVersion": version["modelVersion"],
        "featureWindow": int(features["featureWindow"]),
        "duration": duration,
        "trainedAt": version["trainedAt"],
        "modelStatus": status["status"],
        "openTime": int(meta["entryOpenTime"]),
        "entryPrice": float(meta["entryPrice"]),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
    }


def _prediction_payload(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_key": signal["strategyKey"],
        "symbol": signal["symbol"],
        "duration": signal["duration"],
        "open_time": signal["openTime"],
        "entry_price": signal["entryPrice"],
        "direction": signal["direction"],
        "probability_up": signal["probabilityUp"],
        "confidence": signal["confidence"],
        "certainty_label": "LSTM_SHADOW_SIGNAL",
        "trade_quality_score": signal["confidence"],
        "trade_quality_passed": signal["validationGatePassed"],
        "trade_quality_gate": LSTM_RULE_NAME,
        "high_winrate_gate": LSTM_RULE_NAME,
        "high_winrate_rule": signal["modelVersion"],
        "high_winrate_gate_passed": signal["validationGatePassed"],
        "high_winrate_gate_value": signal["confidence"],
        "high_winrate_gate_min": signal["selectedConfidenceThreshold"],
        "expected_return": signal["expectedReturn"],
        "model_version": signal["modelVersion"],
        "feature_window": signal["featureWindow"],
        "model_duration": signal["duration"],
        "model_trained_at": signal["trainedAt"],
    }


def _expected_return(probability_up: float, version: dict[str, Any]) -> float:
    stats = version.get("returnStats") or {}
    up_mean = float(stats.get("upMean") or 0.0)
    down_mean = float(stats.get("downMean") or 0.0)
    return float(np.round(probability_up * up_mean + (1.0 - probability_up) * down_mean, 8))


def _selected_confidence_threshold(version: dict[str, Any]) -> float:
    threshold = version.get("selectedConfidenceThreshold")
    if threshold is None:
        gate = version.get("validationGate") or {}
        threshold = gate.get("minConfidence")
    if threshold is None:
        raise ValueError("LSTM model version missing selected confidence threshold")
    return float(threshold)


def _prediction_version_payload(version: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    gate = _validation_gate_payload(version, report)
    threshold = _validation_threshold(version, report, gate)
    return {
        **report,
        **version,
        "modelVersion": version.get("modelVersion") or report.get("modelVersion"),
        "trainedAt": version.get("trainedAt") or report.get("trainedAt"),
        "returnStats": version.get("returnStats") or report.get("returnStats") or {},
        "validationGate": gate,
        "selectedConfidenceThreshold": threshold,
    }


def lstm_validation_block_reason(
    status: dict[str, Any],
    version: dict[str, Any],
    report: dict[str, Any],
) -> str:
    if status.get("status") != "trained":
        return status.get("reason") or f"model_status_{status.get('status') or 'unknown'}"
    gate = _validation_gate_payload(version, report)
    if not gate:
        return "validation_gate_missing"
    if gate.get("status") != "passed":
        return str(gate.get("reason") or "validation_gate_failed")
    threshold = _validation_threshold(version, report, gate)
    if threshold is None:
        return "validation_confidence_threshold_missing"
    return "passed"


def _validation_gate_payload(version: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    gate = version.get("validationGate")
    if isinstance(gate, dict):
        return gate
    gate = report.get("validationGate")
    return gate if isinstance(gate, dict) else {}


def _validation_threshold(
    version: dict[str, Any],
    report: dict[str, Any],
    gate: dict[str, Any],
) -> Any:
    threshold = version.get("selectedConfidenceThreshold")
    if threshold is not None:
        return threshold
    threshold = report.get("selectedConfidenceThreshold")
    if threshold is not None:
        return threshold
    return gate.get("minConfidence")


def _untrained_status(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "status": "untrained",
        "symbol": symbol,
        "duration": duration,
        "featureWindow": None,
        "updatedAt": None,
    }


def _validation_failure_reason(
    status: dict[str, Any],
    attempt: dict[str, Any],
    report: dict[str, Any],
) -> str | None:
    for payload in (attempt, status, report):
        reason = payload.get("validationFailureReason") or payload.get("reason")
        if reason:
            return str(reason)
    gate = report.get("validationGate") or {}
    if isinstance(gate, dict) and gate.get("status") != "passed":
        return gate.get("reason")
    return None


def _shadow_prediction_ready_reason(
    status: dict[str, Any],
    version: dict[str, Any],
    report: dict[str, Any],
    artifacts_ready: bool,
    snapshot: dict[str, Any],
    torch_available: bool,
) -> str:
    if not torch_available:
        return "torch_unavailable"
    if status.get("status") != "trained":
        return status.get("reason") or f"model_status_{status.get('status') or 'unknown'}"
    if not artifacts_ready:
        return "artifacts_incomplete"
    validation_reason = lstm_validation_block_reason(status, version, report)
    if validation_reason != "passed":
        return validation_reason
    if not snapshot["matches"]:
        return str(snapshot["reason"])
    return "passed"
