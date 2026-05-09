from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.db.session import get_conn
from app.services.enhanced_features import build_enhanced_feature_frame
from app.services.high_winrate_gate import evaluate_high_winrate_gate
from app.services.prediction_policy import trade_confidence_threshold_for_duration
from app.services.trade_quality_gate import evaluate_trade_quality

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
PREDICTION_WINDOW_ROWS = 5000

# --------------------------------------------------------------------------- #
# Caches
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, Any] = {}
_CALIB_CACHE: dict[str, Any] = {}
_FEAT_CACHE: dict[str, list[str]] = {}
_THR_CACHE: dict[str, float] = {}


def clear_model_cache(duration: str | None = None) -> None:
    targets = [duration] if duration else list(_MODEL_CACHE)
    for key in targets:
        _MODEL_CACHE.pop(key, None)
        _CALIB_CACHE.pop(key, None)
        _FEAT_CACHE.pop(key, None)
        _THR_CACHE.pop(key, None)


def _apply_calibrator(raw_prob: float, calibrator: object) -> float:
    clipped = float(np.clip(raw_prob, 1e-6, 1 - 1e-6))
    if isinstance(calibrator, dict):
        method = calibrator.get("method")
        model = calibrator.get("model")
        if method == "platt" and model is not None:
            return float(np.clip(model.predict_proba(np.array([[clipped]]))[0, 1], 1e-6, 1 - 1e-6))
        if method == "isotonic" and model is not None:
            return float(np.clip(model.predict(np.array([clipped]))[0], 1e-6, 1 - 1e-6))
    if hasattr(calibrator, "transform"):
        return float(np.clip(calibrator.transform(np.array([clipped]))[0], 1e-6, 1 - 1e-6))
    return clipped


def load_enhanced_model(duration: str) -> None:
    """Load model artifacts for a given prediction duration."""
    suffix = f"{duration}_enhanced"
    model_path = MODEL_DIR / f"model_{suffix}.pkl"
    calib_path = MODEL_DIR / f"model_{suffix}_calibrator.pkl"
    meta_path = MODEL_DIR / f"model_{suffix}_meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Enhanced model for '{duration}' not found. Run training first.")

    if duration in _MODEL_CACHE:
        return

    _MODEL_CACHE[duration] = joblib.load(model_path)
    if calib_path.exists():
        _CALIB_CACHE[duration] = joblib.load(calib_path)
    else:
        _CALIB_CACHE[duration] = {"method": "identity", "model": None}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feats = meta["features"]
    # Verify feature compatibility
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
    feat_df, _ = build_enhanced_feature_frame(probe)
    unknown = [c for c in feats if c not in feat_df.columns]
    if unknown:
        raise ValueError(f"Feature mismatch: {unknown}")

    _FEAT_CACHE[duration] = feats
    _THR_CACHE[duration] = float(meta["best_threshold"])


def predict_direction(
    symbol: str,
    duration: str,
) -> dict[str, Any]:
    """
    Predict 10m direction using top-down enhanced features.
    Returns fixed binary direction: up/down.
    """
    if duration != "10m":
        raise ValueError("enhanced predictor now supports only 10m")
    load_enhanced_model(duration)
    feats = _FEAT_CACHE[duration]
    threshold = _THR_CACHE[duration]
    model = _MODEL_CACHE[duration]
    calibrator = _CALIB_CACHE[duration]

    feat_df = _latest_feature_frame(symbol)
    latest_features = feat_df.iloc[-1].to_dict()
    raw_prob = float(model.predict_proba(feat_df.iloc[[-1]][feats])[0, 1])
    prob_up = _apply_calibrator(raw_prob, calibrator)
    direction = "up" if prob_up >= threshold else "down"
    confidence = max(prob_up, 1.0 - prob_up)
    quality = evaluate_trade_quality(latest_features, direction, duration)
    high_winrate = evaluate_high_winrate_gate(latest_features, direction, confidence, duration)
    return {
        "symbol": symbol.upper(),
        "duration": duration,
        "open_time": int(latest_features["open_time"]),
        "entry_price": round(float(latest_features["close"]), 8),
        "direction": direction,
        "probability_up": round(prob_up, 4),
        "confidence": round(confidence, 4),
        "certainty_label": _certainty_label(confidence),
        "threshold": threshold,
        "trade_confidence_threshold": trade_confidence_threshold_for_duration(duration),
        **quality,
        **high_winrate,
    }


def _latest_feature_frame(symbol: str) -> pd.DataFrame:
    df = _latest_kline_frame(symbol)
    ob_df = _load_orderbook_features(symbol)
    funding_df = _load_funding_features(symbol)
    feat_df, _ = build_enhanced_feature_frame(df, ob_df=ob_df, funding_df=funding_df)
    if feat_df.empty:
        raise ValueError("Insufficient history for enhanced features")
    return feat_df


def _latest_kline_frame(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time, open, high, low, close, volume
            FROM klines
            WHERE symbol = ? AND interval = '1m'
            ORDER BY open_time DESC
            LIMIT ?
            """,
            (symbol.upper(), PREDICTION_WINDOW_ROWS),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        raise ValueError("No 1m klines available")
    rows = list(reversed(rows))
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def _certainty_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "high"
    if confidence >= 0.75:
        return "medium"
    return "low"


def _load_orderbook_features(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(
        "SELECT open_time, imbalance, bid_qty_sum, ask_qty_sum, spread_bps FROM orderbook_features WHERE symbol=? ORDER BY open_time",
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["open_time", "imbalance", "bid_qty_sum", "ask_qty_sum", "spread_bps"])
    df = pd.DataFrame(rows, columns=["open_time", "imbalance", "bid_qty_sum", "ask_qty_sum", "spread_bps"])
    for c in df.columns:
        if c != "open_time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _load_funding_features(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(
        "SELECT open_time, funding_rate FROM funding_features WHERE symbol=? ORDER BY open_time",
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame(columns=["open_time", "funding_rate"])
    df = pd.DataFrame(rows, columns=["open_time", "funding_rate"])
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df
