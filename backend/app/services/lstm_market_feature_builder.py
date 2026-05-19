from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.enhanced_features import (
    build_enhanced_feature_frame,
    load_funding_features,
    load_klines,
    load_orderbook_features,
)
from app.services.factor_learning_controls import load_factor_learning_memory_for
from app.services.factor_learning_retrieval import build_factor_learning_retrieval
from app.services.external_factor_data import load_external_feature_frames
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

LSTM_MARKET_MIN_HISTORY = 240
LEARNING_CONTEXT_PREFIX = "factor_learning_"
REGIME_FEATURE_PREFIX = "lstm_regime_"


def build_lstm_market_feature_frame(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
    *,
    learning_memory: dict[str, Any] | None = None,
    min_history: int = LSTM_MARKET_MIN_HISTORY,
) -> pd.DataFrame:
    _validate_duration(duration)
    orderbook = load_orderbook_features(symbol)
    funding = load_funding_features(symbol)
    external_frames = load_external_feature_frames(symbol)
    feature_frame, _ = build_enhanced_feature_frame(
        frame,
        ob_df=orderbook,
        funding_df=funding,
        external_frames=external_frames,
        min_history=min_history,
    )
    feature_frame = _add_market_regime_features(feature_frame)
    memory = learning_memory if learning_memory is not None else load_factor_learning_memory_for(symbol, duration)
    return _attach_learning_context(feature_frame, memory)


def load_lstm_market_frame(
    symbol: str,
    duration: str,
    *,
    learning_memory: dict[str, Any] | None = None,
    min_history: int = LSTM_MARKET_MIN_HISTORY,
) -> pd.DataFrame:
    raw = load_klines(symbol, duration)
    return build_lstm_market_feature_frame(
        raw,
        symbol,
        duration,
        learning_memory=learning_memory,
        min_history=min_history,
    )


def lstm_learning_context(memory: dict[str, Any] | None) -> dict[str, float]:
    retrieval = build_factor_learning_retrieval(memory)
    summary = retrieval["summary"]
    top_weights = retrieval["topWeights"]
    weights = [float(item["weight"]) for item in top_weights if _finite_float(item.get("weight")) is not None]
    total_weight = sum(weights)
    return {
        f"{LEARNING_CONTEXT_PREFIX}blocked_factor_count": float(len(retrieval["blockedFactorNames"])),
        f"{LEARNING_CONTEXT_PREFIX}mining_excluded_factor_count": float(len(retrieval["miningExcludedFactorNames"])),
        f"{LEARNING_CONTEXT_PREFIX}success_pattern_count": float(summary["successPatternCount"]),
        f"{LEARNING_CONTEXT_PREFIX}forbidden_region_count": float(summary["forbiddenRegionCount"]),
        f"{LEARNING_CONTEXT_PREFIX}loss_pattern_count": float(summary["lossPatternCount"]),
        f"{LEARNING_CONTEXT_PREFIX}weight_count": float(summary["weightCount"]),
        f"{LEARNING_CONTEXT_PREFIX}top_weight_count": float(len(weights)),
        f"{LEARNING_CONTEXT_PREFIX}top_weight_sum": float(total_weight),
        f"{LEARNING_CONTEXT_PREFIX}top_weight_max": float(max(weights) if weights else 0.0),
        f"{LEARNING_CONTEXT_PREFIX}top_weight_mean": float(total_weight / len(weights) if weights else 0.0),
    }


def _attach_learning_context(frame: pd.DataFrame, memory: dict[str, Any] | None) -> pd.DataFrame:
    out = frame.copy()
    for column, value in lstm_learning_context(memory).items():
        out[column] = float(value)
    return out


def _add_market_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    adx = _series(out, "adx_14")
    atr_ratio = _series(out, "atr_ratio")
    chop = _series(out, "chop_14")
    bb_width = _series(out, "bb_width_20")
    volume_z = _series(out, "volume_z_20")
    vol_ratio = _series(out, "vol_ratio_20")
    ema_cross = _series(out, "ema_cross")
    ema_ratio_12 = _series(out, "ema_ratio_12")
    ema_ratio_26 = _series(out, "ema_ratio_26")
    macd_hist = _series(out, "macd_hist")
    rsi_14 = _series(out, "rsi_14")
    sma_slope_20 = _series(out, "sma_slope_20")
    sma_slope_60 = _series(out, "sma_slope_60")
    ret_10_60 = _series(out, "ret_10_60")
    down_vol = _series(out, "downside_vol_20")
    up_vol = _series(out, "upside_vol_20")

    out[f"{REGIME_FEATURE_PREFIX}trend_score"] = _bounded_tanh((adx - 20.0) / 12.0) * _bounded_tanh(ema_cross * 1200.0)
    out[f"{REGIME_FEATURE_PREFIX}range_score"] = _bounded_tanh((60.0 - chop) / 12.0) * (1.0 - _bounded_tanh(bb_width * 12.0))
    out[f"{REGIME_FEATURE_PREFIX}volatility_score"] = _bounded_tanh((atr_ratio - 1.0) * 2.5) + _bounded_tanh(_series(out, "vol_of_vol_20") * 8.0)
    out[f"{REGIME_FEATURE_PREFIX}volume_anomaly_score"] = _bounded_tanh(volume_z / 2.5) + _bounded_tanh((vol_ratio - 1.0) * 1.5)
    out[f"{REGIME_FEATURE_PREFIX}direction_confidence"] = _bounded_tanh(macd_hist.abs() / (out["atr_14"].abs().replace(0, np.nan) + 1e-12))
    out[f"{REGIME_FEATURE_PREFIX}trend_alignment"] = _sign(ema_ratio_12) * _sign(ema_ratio_26) * _sign(sma_slope_20)
    out[f"{REGIME_FEATURE_PREFIX}cross_horizon_momentum"] = _bounded_tanh(ret_10_60 * 18.0) + _bounded_tanh((sma_slope_20 - sma_slope_60) * 30.0)
    out[f"{REGIME_FEATURE_PREFIX}directional_vol_gap"] = _bounded_tanh((up_vol - down_vol) * 25.0)
    out[f"{REGIME_FEATURE_PREFIX}rsi_bias"] = (rsi_14 - 50.0) / 50.0
    out[f"{REGIME_FEATURE_PREFIX}price_pressure"] = _bounded_tanh((_series(out, "donchian_pos_20") - 0.5) * 4.0) * _bounded_tanh((adx - 15.0) / 10.0)
    for column in out.columns:
        if column.startswith(REGIME_FEATURE_PREFIX):
            out[column] = out[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _bounded_tanh(values: pd.Series) -> pd.Series:
    return pd.Series(np.tanh(values.to_numpy(dtype=np.float64)), index=values.index)


def _sign(values: pd.Series) -> pd.Series:
    return pd.Series(np.sign(values.to_numpy(dtype=np.float64)), index=values.index)


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported LSTM duration: {duration}")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
