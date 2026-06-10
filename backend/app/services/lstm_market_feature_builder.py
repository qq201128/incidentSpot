from __future__ import annotations

import os
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
MINUTES_PER_DAY = 1_440
MS_PER_DAY = 86_400_000
TRAINING_LOOKBACK_ENV = "MODEL_FAMILY_TRAINING_LOOKBACK_DAYS"
TRAINING_LOOKBACK_DURATION_ENV_PREFIX = "MODEL_FAMILY_TRAINING_LOOKBACK_DAYS_"
DEFAULT_TRAINING_LOOKBACK_DAYS = {
    "10m": 180,
    "30m": 180,
    "60m": 180,
    "1d": 365,
}
DURATION_MINUTES = {"10m": 10, "30m": 30, "60m": 60, "1d": MINUTES_PER_DAY}


def build_lstm_market_feature_frame(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
    *,
    learning_memory: dict[str, Any] | None = None,
    min_history: int = LSTM_MARKET_MIN_HISTORY,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    _validate_duration(duration)
    orderbook = _load_orderbook(symbol, lookback_days)
    funding = _load_funding(symbol, lookback_days)
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
    return _recent_frame(_attach_learning_context(feature_frame, memory), lookback_days)


def load_lstm_market_frame(
    symbol: str,
    duration: str,
    *,
    learning_memory: dict[str, Any] | None = None,
    min_history: int = LSTM_MARKET_MIN_HISTORY,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    selected_lookback = _selected_training_lookback_days(duration, lookback_days)
    raw_lookback = _raw_lookback_days(duration, selected_lookback, min_history)
    raw = _load_klines(symbol, duration, raw_lookback)
    return build_lstm_market_feature_frame(
        raw,
        symbol,
        duration,
        learning_memory=learning_memory,
        min_history=min_history,
        lookback_days=selected_lookback,
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


def _selected_training_lookback_days(duration: str, requested: int | None) -> int | None:
    if requested is not None:
        return _positive_lookback(requested)
    env_value = _duration_env_value(duration) or os.environ.get(TRAINING_LOOKBACK_ENV)
    if env_value is None:
        return DEFAULT_TRAINING_LOOKBACK_DAYS[duration]
    if env_value.strip().lower() in {"none", "off", "disabled", "0"}:
        return None
    return _positive_lookback(int(env_value))


def _duration_env_value(duration: str) -> str | None:
    key = f"{TRAINING_LOOKBACK_DURATION_ENV_PREFIX}{duration.upper()}"
    return os.environ.get(key)


def _positive_lookback(value: int) -> int:
    selected = int(value)
    if selected <= 0:
        raise ValueError("lookback_days must be greater than 0")
    return selected


def _raw_lookback_days(duration: str, lookback_days: int | None, min_history: int) -> int | None:
    if lookback_days is None:
        return None
    warmup_minutes = min_history * DURATION_MINUTES[duration] * 2
    return lookback_days + _ceil_div(warmup_minutes, MINUTES_PER_DAY)


def _load_klines(symbol: str, duration: str, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return load_klines(symbol, duration)
    return load_klines(symbol, duration, lookback_days=lookback_days)


def _load_orderbook(symbol: str, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return load_orderbook_features(symbol)
    return load_orderbook_features(symbol, lookback_days=lookback_days)


def _load_funding(symbol: str, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return load_funding_features(symbol)
    return load_funding_features(symbol, lookback_days=lookback_days)


def _recent_frame(frame: pd.DataFrame, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return frame
    latest = int(pd.to_numeric(frame["open_time"], errors="raise").max())
    cutoff = latest - int(lookback_days) * MS_PER_DAY
    recent = frame[pd.to_numeric(frame["open_time"], errors="raise") >= cutoff].reset_index(drop=True)
    if recent.empty:
        raise ValueError(f"no LSTM training rows in recent {lookback_days} day window")
    return recent


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
