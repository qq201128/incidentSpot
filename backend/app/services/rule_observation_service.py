from __future__ import annotations

import math
from typing import Any

from app.services.rule_config import (
    BODY_WEIGHT,
    BPS_DIVISOR,
    HIGHER_TIMEFRAME_RULES,
    HIGHER_TIMEFRAME_WEIGHT,
    INTRABAR_CENTER_SCALE,
    MA_BIAS_WEIGHT,
    MAX_SIGNAL_SCORE,
    MIN_SIGNAL_SCORE,
    MOMENTUM_WEIGHT,
    NEUTRAL_PROBABILITY,
    OBSERVATION_MAX_EDGE,
    ORDERBOOK_WEIGHT,
    TEN_MINUTE_RULE,
    TEN_MINUTE_WEIGHT,
    TimeframeRule,
    VEGAS_RESONANCE_WEIGHT,
)


def observation_signal(feature_row: dict[str, Any], orderbook: dict[str, Any]) -> dict[str, Any]:
    ten_minute = _ten_minute_score(feature_row)
    higher = _higher_timeframe_score(feature_row)
    vegas = _vegas_score(feature_row)
    orderbook_score = _clamp(float(orderbook["score"]), MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)
    score = _weighted_observation_score(
        ten_minute=ten_minute,
        higher=higher,
        orderbook_score=orderbook_score,
        vegas=vegas,
    )
    confidence = NEUTRAL_PROBABILITY + abs(score) * OBSERVATION_MAX_EDGE
    return {
        "direction": "up" if score >= 0 else "down",
        "score": round(score, 6),
        "confidence": round(confidence, 6),
        "votes": [
            {"feature": "10m_observation", "score": round(ten_minute, 6)},
            {"feature": "higher_timeframe_observation", "score": round(higher, 6)},
            {"feature": "vegas_fibonacci_resonance", "score": round(vegas, 6)},
            {"feature": "orderbook_ofi_microprice", "score": round(orderbook_score, 6)},
        ],
    }


def _weighted_observation_score(
    *,
    ten_minute: float,
    higher: float,
    orderbook_score: float,
    vegas: float,
) -> float:
    score = (
        ten_minute * TEN_MINUTE_WEIGHT
        + higher * HIGHER_TIMEFRAME_WEIGHT
        + orderbook_score * ORDERBOOK_WEIGHT
        + vegas * VEGAS_RESONANCE_WEIGHT
    )
    return _clamp(score, MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)


def _vegas_score(feature_row: dict[str, Any]) -> float:
    direction = _feature_float(feature_row, "vegas_direction_score")
    resonance = _feature_float(feature_row, "vegas_resonance_score") / 100.0
    return _clamp(direction * abs(resonance), MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)


def _higher_timeframe_score(feature_row: dict[str, Any]) -> float:
    total_weight = sum(rule.weight for rule in HIGHER_TIMEFRAME_RULES)
    score = sum(_timeframe_score(feature_row, rule) * rule.weight for rule in HIGHER_TIMEFRAME_RULES)
    return _clamp(score / total_weight, MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)


def _ten_minute_score(feature_row: dict[str, Any]) -> float:
    return _weighted_feature_score(
        momentum=_feature_scaled(feature_row, "ret_10", TEN_MINUTE_RULE.scale_bps),
        ma_bias=_feature_scaled(feature_row, "ma_ratio_10", TEN_MINUTE_RULE.scale_bps),
        body=_feature_scaled(feature_row, "oc_body", TEN_MINUTE_RULE.scale_bps),
    )


def _timeframe_score(feature_row: dict[str, Any], rule: TimeframeRule) -> float:
    prefix = f"tf_{rule.interval}"
    return _weighted_feature_score(
        momentum=_feature_scaled(feature_row, f"{prefix}_ret_{rule.lookback}", rule.scale_bps),
        ma_bias=_feature_scaled(feature_row, f"{prefix}_ma_ratio_{rule.ma_window}", rule.scale_bps),
        body=_intrabar_position_score(feature_row, f"{prefix}_intrabar_pos"),
    )


def _weighted_feature_score(*, momentum: float, ma_bias: float, body: float) -> float:
    score = momentum * MOMENTUM_WEIGHT + ma_bias * MA_BIAS_WEIGHT + body * BODY_WEIGHT
    return _clamp(score, MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)


def _feature_scaled(feature_row: dict[str, Any], column: str, scale_bps: float) -> float:
    value = _feature_float(feature_row, column)
    scale = float(scale_bps) / BPS_DIVISOR
    return _clamp(value / scale, MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)


def _intrabar_position_score(feature_row: dict[str, Any], column: str) -> float:
    value = _feature_float(feature_row, column)
    score = (value - NEUTRAL_PROBABILITY) * INTRABAR_CENTER_SCALE
    return _clamp(score, MIN_SIGNAL_SCORE, MAX_SIGNAL_SCORE)


def _feature_float(feature_row: dict[str, Any], column: str) -> float:
    if column not in feature_row:
        raise KeyError(f"observation feature missing: {column}")
    value = float(feature_row[column])
    if not math.isfinite(value):
        raise ValueError(f"observation feature is not finite: {column}")
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
