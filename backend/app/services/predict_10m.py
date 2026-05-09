from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.services.kline_features import build_feature_frame

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DURATION_TO_FILE = {
    "10m": "10m",
    "30m": "30m",
    "60m": "60m",
    "1d": "1d",
}

_MODEL_CACHE: dict[str, object] = {}
_CALIBRATOR_CACHE: dict[str, object] = {}
_FEATURE_CACHE: dict[str, list[str]] = {}
_THRESHOLD_CACHE: dict[str, float] = {}


def clear_model_cache(duration: str | None = None) -> None:
    targets = [duration] if duration else list(DURATION_TO_FILE)
    for key in targets:
        _MODEL_CACHE.pop(key, None)
        _CALIBRATOR_CACHE.pop(key, None)
        _FEATURE_CACHE.pop(key, None)
        _THRESHOLD_CACHE.pop(key, None)


def _apply_calibrator(raw_prob: float, calibrator: object) -> float:
    clipped = float(np.clip(raw_prob, 1e-6, 1 - 1e-6))
    if isinstance(calibrator, dict):
        method = calibrator.get("method")
        model = calibrator.get("model")
        if method == "platt":
            return float(np.clip(model.predict_proba(np.array([[clipped]]))[0, 1], 1e-6, 1 - 1e-6))
        if method == "isotonic":
            return float(np.clip(model.predict(np.array([clipped]))[0], 1e-6, 1 - 1e-6))
    if hasattr(calibrator, "transform"):
        return float(np.clip(calibrator.transform(np.array([clipped]))[0], 1e-6, 1 - 1e-6))
    return clipped


def _ensure_model_loaded(duration: str) -> None:
    suffix = DURATION_TO_FILE.get(duration)
    if suffix is None:
        raise ValueError(f"unsupported duration: {duration}")
    if (
        duration in _MODEL_CACHE
        and duration in _CALIBRATOR_CACHE
        and duration in _FEATURE_CACHE
        and duration in _THRESHOLD_CACHE
    ):
        return

    model_path = MODEL_DIR / f"model_{suffix}.pkl"
    calibrator_path = MODEL_DIR / f"model_{suffix}_calibrator.pkl"
    meta_path = MODEL_DIR / f"model_{suffix}_meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"{duration} model artifact is missing; run training first")

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("missing dependency: joblib (pip install -r requirements.txt)") from exc

    _MODEL_CACHE[duration] = joblib.load(model_path)
    if calibrator_path.exists():
        _CALIBRATOR_CACHE[duration] = joblib.load(calibrator_path)
    else:
        # Backward compatibility for legacy models trained before calibrator was introduced.
        _CALIBRATOR_CACHE[duration] = {"method": "identity", "model": None}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feats = meta["features"]
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("missing dependency: pandas (pip install -r requirements.txt)") from exc

    # Validate feature names against the current feature builder without needing a long history window.
    probe = pd.DataFrame(
        {
            "open_time": list(range(5000)),
            "open": np.linspace(1.0, 1.2, 5000),
            "high": np.linspace(1.01, 1.21, 5000),
            "low": np.linspace(0.99, 1.19, 5000),
            "close": np.linspace(1.0, 1.2, 5000),
            "volume": np.linspace(10.0, 20.0, 5000),
        }
    )
    # Training uses every column from build_feature_frame except open_time/y; FeatureSpec.columns
    # lists only shifted engineered fields — compare against the actual frame (OHLCV + vol_ma_* etc.).
    feature_df, _spec = build_feature_frame(probe)
    unknown = [c for c in feats if c not in feature_df.columns]
    if unknown:
        raise ValueError(
            "model features are incompatible with current feature builder; retrain models "
            f"(unknown {len(unknown)} feature columns)"
        )

    _FEATURE_CACHE[duration] = feats
    _THRESHOLD_CACHE[duration] = float(meta["best_threshold"])


def predict_direction(rows: list[dict], duration: str) -> dict:
    _ensure_model_loaded(duration)
    features = _FEATURE_CACHE[duration]
    threshold = _THRESHOLD_CACHE[duration]
    model = _MODEL_CACHE[duration]
    calibrator = _CALIBRATOR_CACHE[duration]

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("missing dependency: pandas (pip install -r requirements.txt)") from exc

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no market data")

    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing_cols = sorted(required - set(frame.columns))
    if missing_cols:
        raise ValueError(f"missing ohlcv columns: {', '.join(missing_cols)}")

    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    feature_df, _ = build_feature_frame(frame)
    if feature_df.empty:
        raise ValueError("insufficient candles for feature generation")

    x_latest = feature_df.iloc[[-1]][features]
    raw_prob_up = float(model.predict_proba(x_latest)[0, 1])
    prob_up = _apply_calibrator(raw_prob_up, calibrator)
    direction = "up" if prob_up >= threshold else "down"

    return {
        "duration": duration,
        "direction": direction,
        "probabilityUp": prob_up,
        "rawProbabilityUp": round(raw_prob_up, 6),
        "threshold": threshold,
    }


def predict_10m_direction(rows: list[dict]) -> dict:
    return predict_direction(rows, "10m")
