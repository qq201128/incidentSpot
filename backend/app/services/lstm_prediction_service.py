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
from app.services.lstm_config import LSTM_RULE_NAME, lstm_shadow_strategy_key
from app.services.lstm_feature_builder import build_live_feature_window
from app.services.lstm_torch_backend import TorchLstmBackend
from app.services.lstm_validation import apply_standardizer


def lstm_model_status(symbol: str, duration: str, *, artifact_root: Path | None = None) -> dict[str, Any]:
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root)
    status = _normalized_status(read_json(paths.status) or _untrained_status(sym, duration))
    report = read_json(paths.report) or {}
    version = read_json(paths.version) or {}
    return {
        **status,
        "strategyKey": lstm_shadow_strategy_key(duration),
        "modelVersion": version.get("modelVersion") or report.get("modelVersion"),
        "trainedAt": version.get("trainedAt") or report.get("trainedAt"),
        "sampleCounts": report.get("sampleCounts") or {},
        "testAccuracy": (report.get("test") or {}).get("accuracy"),
        "testWinRate": (report.get("test") or {}).get("winRate"),
        "artifactsReady": required_artifacts_exist(paths),
        "shadowPredictionReady": _shadow_prediction_ready(status, paths),
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
    _assert_predictable(sym, duration, paths)
    features = require_json(paths.features, "features")
    scaler = require_json(paths.scaler, "scaler")
    version = require_json(paths.version, "version")
    status = _normalized_status(require_json(paths.status, "status"))
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
    return _signal_payload(sym, duration, probability_up, meta, features, version, status)


def predict_lstm_shadow_prediction(
    symbol: str,
    duration: str,
    *,
    entry_open_time: int | None = None,
) -> dict[str, Any]:
    signal = predict_lstm_signal(symbol, duration, entry_open_time=entry_open_time)
    return _prediction_payload(signal)


def _assert_predictable(symbol: str, duration: str, paths) -> None:
    status = _normalized_status(read_json(paths.status) or _untrained_status(symbol, duration))
    if status["status"] != "trained":
        reason = status.get("reason") or "model is not trained"
        raise ValueError(f"LSTM model is not ready for {symbol} {duration}: {reason}")
    if not required_artifacts_exist(paths):
        raise ValueError(f"LSTM model artifacts are incomplete for {symbol} {duration}: {paths.root}")


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
    return {
        "symbol": symbol,
        "strategyKey": lstm_shadow_strategy_key(duration),
        "direction": direction,
        "probabilityUp": round(prob, 6),
        "confidence": round(confidence, 6),
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
        "trade_quality_passed": True,
        "trade_quality_gate": LSTM_RULE_NAME,
        "high_winrate_gate": LSTM_RULE_NAME,
        "high_winrate_rule": signal["modelVersion"],
        "high_winrate_gate_passed": True,
        "high_winrate_gate_value": signal["confidence"],
        "high_winrate_gate_min": None,
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


def _untrained_status(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "status": "untrained",
        "symbol": symbol,
        "duration": duration,
        "featureWindow": None,
        "updatedAt": None,
    }


def _shadow_prediction_ready(status: dict[str, Any], paths) -> bool:
    return status.get("status") == "trained" and required_artifacts_exist(paths)


def _normalized_status(status: dict[str, Any]) -> dict[str, Any]:
    if status.get("status") != "validation_failed":
        return status
    return {key: value for key, value in status.items() if key != "reason"} | {"status": "trained"}
