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
from app.services.lstm_status_service import (
    active_lstm_status,
    is_lstm_shadow_ready,
    lstm_model_status,
    lstm_validation_block_reason,
    validation_gate_payload,
    validation_threshold,
)
from app.services.lstm_torch_backend import TorchLstmBackend
from app.services.lstm_validation import apply_standardizer


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
    status = active_lstm_status(sym, duration, artifact_root=artifact_root)
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
    version = read_json(paths.version) or {}
    report = read_json(paths.report) or {}
    status = active_lstm_status(symbol, duration, artifact_root=artifact_root)
    if status["status"] != "trained":
        reason = status.get("reason") or "model is not trained"
        raise ValueError(f"LSTM model is not ready for {symbol} {duration}: {reason}")
    if not required_artifacts_exist(paths):
        raise ValueError(f"LSTM model artifacts are incomplete for {symbol} {duration}: {paths.root}")
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
    gate = validation_gate_payload(version, report)
    threshold = validation_threshold(version, report, gate)
    return {
        **report,
        **version,
        "modelVersion": version.get("modelVersion") or report.get("modelVersion"),
        "trainedAt": version.get("trainedAt") or report.get("trainedAt"),
        "returnStats": version.get("returnStats") or report.get("returnStats") or {},
        "validationGate": gate,
        "selectedConfidenceThreshold": threshold,
    }

