from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.lstm_artifacts import artifact_paths, read_json, require_json, required_artifacts_exist
from app.services.lstm_feature_builder import (
    _assert_columns,
    build_live_feature_window,
    duration_feature_frame,
)
from app.services.lstm_lifecycle import LSTM_STATUS_LEGACY_TRAINED, shadow_predictable_status, trade_active_status
from app.services.lstm_market_feature_builder import load_lstm_market_frame
from app.services.lstm_status_service import validation_gate_payload, validation_threshold
from app.services.lstm_validation import apply_standardizer
from app.services.model_family_config import (
    JOBLIB_MODEL_FAMILIES,
    model_family_rule_name,
    model_family_strategy_key,
    normalize_model_family,
)
from app.services.model_family_joblib_backend import JoblibModelBackend
from app.services.model_family_status_service import (
    active_model_family_status,
    model_family_status,
    model_validation_block_reason,
)
from app.services.model_family_torch_backend import TorchSequenceBackend


def predict_model_family_signal(
    family: str,
    symbol: str,
    duration: str,
    *,
    entry_open_time: int | None = None,
    artifact_root: Path | None = None,
    backend: Any | None = None,
) -> dict[str, Any]:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root, family=selected)
    _assert_predictable(selected, sym, duration, paths, artifact_root=artifact_root)
    features = require_json(paths.features, "features")
    scaler = require_json(paths.scaler, "scaler")
    version = require_json(paths.version, "version")
    report = require_json(paths.report, "training report")
    window, meta = build_live_feature_window(
        sym,
        duration,
        list(features["columns"]),
        int(features["featureWindow"]),
        entry_open_time,
    )
    probability_up = float((backend or _default_backend(selected)).predict(paths.model, apply_standardizer(window, scaler))[0])
    status = active_model_family_status(selected, sym, duration, artifact_root=artifact_root)
    return _signal_payload(selected, sym, duration, probability_up, meta, features, _version_payload(version, report), status)


def predict_model_family_shadow_prediction(family: str, symbol: str, duration: str, *, entry_open_time: int | None = None) -> dict:
    return _prediction_payload(predict_model_family_signal(family, symbol, duration, entry_open_time=entry_open_time))


def predict_model_family_shadow_predictions(
    family: str,
    symbol: str,
    duration: str,
    entry_open_times: list[int],
    *,
    artifact_root: Path | None = None,
    backend: Any | None = None,
) -> list[dict[str, Any]]:
    selected = normalize_model_family(family)
    entries = sorted({int(item) for item in entry_open_times})
    if not entries:
        return []
    sym = symbol.strip().upper()
    paths = artifact_paths(sym, duration, artifact_root, family=selected)
    _assert_predictable(selected, sym, duration, paths, artifact_root=artifact_root)
    features = require_json(paths.features, "features")
    scaler = require_json(paths.scaler, "scaler")
    version = require_json(paths.version, "version")
    report = require_json(paths.report, "training report")
    windows, metas = _live_feature_windows(sym, duration, list(features["columns"]), int(features["featureWindow"]), entries)
    probabilities = (backend or _default_backend(selected)).predict(paths.model, apply_standardizer(windows, scaler))
    status = active_model_family_status(selected, sym, duration, artifact_root=artifact_root)
    version_payload = _version_payload(version, report)
    return [
        _prediction_payload(_signal_payload(selected, sym, duration, float(prob), meta, features, version_payload, status))
        for prob, meta in zip(probabilities, metas)
    ]


def _assert_predictable(family: str, symbol: str, duration: str, paths, *, artifact_root: Path | None) -> None:
    version = read_json(paths.version) or {}
    report = read_json(paths.report) or {}
    status = active_model_family_status(family, symbol, duration, artifact_root=artifact_root)
    if not shadow_predictable_status(status.get("status")):
        raise ValueError(f"{family} model is not ready for {symbol} {duration}: {status.get('reason') or status.get('status')}")
    if not required_artifacts_exist(paths):
        raise ValueError(f"{family} model artifacts are incomplete for {symbol} {duration}: {paths.root}")
    if status.get("status") != LSTM_STATUS_LEGACY_TRAINED:
        return
    reason = model_validation_block_reason(status, version, report)
    if reason != "passed":
        raise ValueError(f"{family} model is not ready for {symbol} {duration}: {reason}")


def _live_feature_windows(symbol: str, duration: str, columns: list[str], feature_window: int, entries: list[int]):
    sampled = duration_feature_frame(load_lstm_market_frame(symbol, duration), duration)
    _assert_columns(sampled, columns)
    by_entry = {int(row["entry_open_time"]): idx for idx, row in sampled.iterrows()}
    values = sampled[columns].to_numpy(dtype=np.float32)
    windows, metas = [], []
    for entry in entries:
        idx = by_entry.get(int(entry))
        if idx is None:
            raise ValueError(f"missing completed model feature row for entry_open_time={entry}")
        if idx + 1 < feature_window:
            raise ValueError(f"insufficient model feature rows before entry_open_time={entry}")
        window = values[idx - feature_window + 1: idx + 1]
        if not np.isfinite(window).all():
            raise ValueError(f"model feature window contains non-finite values at entry_open_time={entry}")
        row = sampled.iloc[idx]
        windows.append(window)
        metas.append({"entryOpenTime": int(row["entry_open_time"]), "entryPrice": float(row["close"])})
    return np.asarray(windows, dtype=np.float32), metas


def _signal_payload(family: str, symbol: str, duration: str, probability_up: float, meta, features, version, status) -> dict:
    prob = max(0.0, min(probability_up, 1.0))
    direction = "up" if prob >= 0.5 else "down"
    confidence = prob if direction == "up" else 1.0 - prob
    threshold = _selected_confidence_threshold(version)
    return {
        "modelFamily": family,
        "symbol": symbol,
        "strategyKey": model_family_strategy_key(family, duration),
        "direction": direction,
        "probabilityUp": round(prob, 6),
        "confidence": round(confidence, 6),
        "selectedConfidenceThreshold": threshold,
        "validationGatePassed": _trade_gate_passed(confidence, threshold, status, version),
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
    rule_name = model_family_rule_name(signal["modelFamily"])
    return {
        "signal_key": signal["strategyKey"],
        "strategy_key": signal["strategyKey"],
        "modelFamily": signal["modelFamily"],
        "symbol": signal["symbol"],
        "duration": signal["duration"],
        "open_time": signal["openTime"],
        "entry_price": signal["entryPrice"],
        "direction": signal["direction"],
        "probability_up": signal["probabilityUp"],
        "confidence": signal["confidence"],
        "certainty_label": f"{signal['modelFamily'].upper()}_SHADOW_SIGNAL",
        "trade_quality_score": signal["confidence"],
        "trade_quality_passed": signal["validationGatePassed"],
        "trade_quality_gate": rule_name,
        "high_winrate_gate": rule_name,
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


def _version_payload(version: dict, report: dict) -> dict:
    gate = validation_gate_payload(version, report)
    return {
        **report,
        **version,
        "modelVersion": version.get("modelVersion") or report.get("modelVersion"),
        "trainedAt": version.get("trainedAt") or report.get("trainedAt"),
        "returnStats": version.get("returnStats") or report.get("returnStats") or {},
        "validationGate": gate,
        "selectedConfidenceThreshold": validation_threshold(version, report, gate),
    }


def _expected_return(probability_up: float, version: dict) -> float:
    stats = version.get("returnStats") or {}
    return float(np.round(probability_up * float(stats.get("upMean") or 0.0) + (1.0 - probability_up) * float(stats.get("downMean") or 0.0), 8))


def _selected_confidence_threshold(version: dict) -> float | None:
    threshold = version.get("selectedConfidenceThreshold")
    return None if threshold is None else float(threshold)


def _trade_gate_passed(confidence: float, threshold: float | None, status: dict, version: dict) -> bool:
    gate = version.get("validationGate") or {}
    return bool(threshold is not None and trade_active_status(status.get("status")) and gate.get("status") == "passed" and confidence >= threshold)


def _default_backend(family: str):
    return JoblibModelBackend() if family in JOBLIB_MODEL_FAMILIES else TorchSequenceBackend()
