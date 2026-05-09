from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DEFAULT_QUALITY_SCORE_MIN = 0.90
FLOAT_MATCH_EPSILON = 1e-9


def evaluate_trade_quality(
    feature_row: dict[str, Any],
    direction: str,
    duration: str,
) -> dict[str, Any]:
    config = load_quality_gate(duration)
    score = quality_score(feature_row, direction, config)
    score_min = float(config["quality_score_min"])
    return {
        "trade_quality_score": round(score, 4),
        "trade_quality_score_min": score_min,
        "trade_quality_passed": score >= score_min,
        "trade_quality_gate": config["quality_gate_name"],
    }


def quality_gate_policy_payload(duration: str) -> dict[str, Any]:
    config = load_quality_gate(duration)
    backtest, source = _quality_backtest(duration, config)
    return {
        "tradeQualityScoreMin": config["quality_score_min"],
        "tradeQualityGate": config["quality_gate_name"],
        "targetEventWinRate": config.get("target_event_win_rate"),
        "expectedEventWinRate": backtest.get("win_rate"),
        "expectedDirectionHitRate": backtest.get("direction_hit_rate"),
        "expectedTestTrades": backtest.get("test_trades"),
        "expectedTradesPerDay": backtest.get("trades_per_day"),
        "baseConfidenceWinRate": backtest.get("base_confidence_win_rate"),
        "backtestSource": source,
    }


def configured_trade_confidence_threshold(duration: str) -> float:
    return float(load_quality_gate(duration)["trade_confidence_threshold"])


def load_quality_gate(duration: str) -> dict[str, Any]:
    path = MODEL_DIR / f"trade_quality_gate_{duration}.json"
    if not path.exists():
        return {
            "quality_score_min": DEFAULT_QUALITY_SCORE_MIN,
            "quality_gate_name": "disabled_missing_quality_gate",
            "trade_confidence_threshold": 0.90,
            "components": {},
            "backtest": {},
        }
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def quality_score(feature_row: dict[str, Any], direction: str, config: dict[str, Any]) -> float:
    components = config.get("components", {})
    if not components:
        return 1.0
    side = 1.0 if direction == "up" else -1.0
    scores = [_component_percentile(feature_row, side, item) for item in components.values()]
    return float(sum(scores) / len(scores))


def _component_percentile(feature_row: dict[str, Any], side: float, item: dict[str, Any]) -> float:
    value = float(feature_row.get(item["column"], 0.0) or 0.0)
    if item.get("transform") == "signed":
        value *= side
    quantiles = item.get("quantiles") or []
    if not quantiles:
        return 0.5
    idx = bisect_right(quantiles, value)
    return max(0.0, min(1.0, idx / len(quantiles)))


def _quality_backtest(duration: str, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    meta = _load_model_meta(duration)
    profile = _matching_quality_profile(meta, config) if meta else None
    if profile:
        return _quality_policy_backtest(meta, profile, config), "active_model_meta"
    return config.get("backtest", {}), "quality_gate_config"


def _load_model_meta(duration: str) -> dict[str, Any] | None:
    paths = [
        MODEL_DIR / f"model_{duration}_enhanced_meta.json",
        MODEL_DIR / f"model_{duration}_meta.json",
    ]
    for path in paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
    return None


def _matching_quality_profile(
    meta: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    quality_min = float(config["quality_score_min"])
    confidence_min = float(config["trade_confidence_threshold"])
    for profile in meta.get("backtest_quality_profiles", []):
        if _is_matching_quality_profile(profile, quality_min, confidence_min):
            return profile
    return None


def _is_matching_quality_profile(
    profile: dict[str, Any],
    quality_min: float,
    confidence_min: float,
) -> bool:
    quality = float(profile.get("trade_quality_score_threshold", -1.0))
    confidence = float(profile.get("trade_confidence_threshold", -1.0))
    return (
        abs(quality - quality_min) <= FLOAT_MATCH_EPSILON
        and abs(confidence - confidence_min) <= FLOAT_MATCH_EPSILON
    )


def _quality_policy_backtest(
    meta: dict[str, Any],
    profile: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    backtest = dict(profile)
    confidence = _matching_confidence_profile(meta, config)
    if confidence:
        backtest["base_confidence_win_rate"] = confidence.get("win_rate")
    return backtest


def _matching_confidence_profile(
    meta: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    confidence_min = float(config["trade_confidence_threshold"])
    for profile in meta.get("backtest_confidence_profiles", []):
        threshold = float(profile.get("trade_confidence_threshold", -1.0))
        if abs(threshold - confidence_min) <= FLOAT_MATCH_EPSILON:
            return profile
    return None
