from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.enhanced_features import load_klines
from app.services.factor_learning_controls import load_factor_learning_memory_for
from app.services.factor_learning_retrieval import build_factor_learning_retrieval
from app.services.kline_features import build_feature_frame
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

LSTM_MARKET_MIN_HISTORY = 240
LEARNING_CONTEXT_PREFIX = "factor_learning_"


def build_lstm_market_feature_frame(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
    *,
    learning_memory: dict[str, Any] | None = None,
    min_history: int = LSTM_MARKET_MIN_HISTORY,
) -> pd.DataFrame:
    _validate_duration(duration)
    feature_frame, _ = build_feature_frame(frame, min_history=min_history)
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
