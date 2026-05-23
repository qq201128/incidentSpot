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
from app.services.lstm_feature_builder import (
    _assert_columns,
    build_live_feature_window,
    duration_feature_frame,
    sanitize_feature_window,
)
from app.services.lstm_lifecycle import LSTM_STATUS_LEGACY_TRAINED, trade_active_status
from app.services.lstm_market_feature_builder import load_lstm_market_frame
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


def predict_lstm_shadow_predictions(
    symbol: str,
    duration: str,
    entry_open_times: list[int],
    *,
    artifact_root: Path | None = None,
    backend: Any | None = None,
) -> list[dict[str, Any]]:
    entries = sorted({int(item) for item in entry_open_times})
    if not entries:
        return []
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root)
    _assert_predictable(sym, duration, paths, artifact_root=artifact_root)
    features = require_json(paths.features, "features")
    scaler = require_json(paths.scaler, "scaler")
    version = require_json(paths.version, "version")
    report = require_json(paths.report, "training report")
    status = active_lstm_status(sym, duration, artifact_root=artifact_root)
    windows, metas = _live_feature_windows(
        sym,
        duration,
        list(features["columns"]),
        int(features["featureWindow"]),
        entries,
    )
    scaled = apply_standardizer(windows, scaler)
    probabilities = (backend or TorchLstmBackend()).predict(paths.model, scaled)
    version_payload = _prediction_version_payload(version, report)
    return [
        _prediction_payload(_signal_payload(sym, duration, float(prob), meta, features, version_payload, status))
        for prob, meta in zip(probabilities, metas)
    ]


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
    if not _shadow_predictable_status(status):
        reason = status.get("reason") or "model is not trained"
        raise ValueError(f"LSTM model is not ready for {symbol} {duration}: {reason}")
    if not required_artifacts_exist(paths):
        raise ValueError(f"LSTM model artifacts are incomplete for {symbol} {duration}: {paths.root}")
    if status.get("status") != LSTM_STATUS_LEGACY_TRAINED:
        return
    validation_reason = lstm_validation_block_reason(status, version, report)
    if validation_reason != "passed":
        raise ValueError(f"LSTM model is not ready for {symbol} {duration}: {validation_reason}")


def _live_feature_windows(
    symbol: str,
    duration: str,
    feature_columns: list[str],
    feature_window: int,
    entry_open_times: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    sampled = duration_feature_frame(load_lstm_market_frame(symbol, duration), duration)
    _assert_columns(sampled, feature_columns)
    by_entry = {int(row["entry_open_time"]): idx for idx, row in sampled.iterrows()}
    values = sampled[feature_columns].to_numpy(dtype=np.float32)
    windows, metas = [], []
    for entry in entry_open_times:
        idx = by_entry.get(int(entry))
        if idx is None:
            raise ValueError(f"missing completed LSTM feature row for entry_open_time={entry}")
        if idx + 1 < feature_window:
            raise ValueError(f"insufficient LSTM feature rows before entry_open_time={entry}")
        window = sanitize_feature_window(values[idx - feature_window + 1: idx + 1])
        row = sampled.iloc[idx]
        windows.append(window)
        metas.append({"entryOpenTime": int(row["entry_open_time"]), "entryPrice": float(row["close"])})
    return np.asarray(windows, dtype=np.float32), metas


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
    gate_passed = _trade_gate_passed(confidence, min_confidence, status, version)
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
        "signal_key": signal["strategyKey"],
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


def _selected_confidence_threshold(version: dict[str, Any]) -> float | None:
    threshold = version.get("selectedConfidenceThreshold")
    if threshold is None:
        gate = version.get("validationGate") or {}
        threshold = gate.get("minConfidence")
    if threshold is None:
        return None
    return float(threshold)


def _trade_gate_passed(
    confidence: float,
    threshold: float | None,
    status: dict[str, Any],
    version: dict[str, Any],
) -> bool:
    if threshold is None or not trade_active_status(status.get("status")):
        return False
    gate = version.get("validationGate") or {}
    if gate.get("status") != "passed":
        return False
    return bool(confidence >= threshold)


def _shadow_predictable_status(status: dict[str, Any]) -> bool:
    return status.get("status") in {"shadow_active", "trade_active", LSTM_STATUS_LEGACY_TRAINED}


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
