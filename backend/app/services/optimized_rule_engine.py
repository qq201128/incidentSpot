from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.enhanced_features import build_enhanced_feature_frame
from app.services.rule_config import OPTIMIZED_EVENT_RULES
from app.services.strategy_registry import strategy_definition


def build_optimized_feature_frame(
    df_1m: pd.DataFrame,
    *,
    min_history: int = 240,
    ob_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    feature_frame, _ = build_enhanced_feature_frame(df_1m, min_history=min_history, ob_df=ob_df)
    timestamps = pd.to_datetime(feature_frame["open_time"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    out = feature_frame.copy()
    out["tod_bucket"] = ((timestamps.dt.hour * 60 + timestamps.dt.minute) // 30).astype(int)
    out["trade_day"] = timestamps.dt.date.astype(str)
    return out


def evaluate_optimized_rules(
    feature_row: dict[str, Any],
    *,
    strategy_key: str | None = None,
) -> dict[str, Any] | None:
    strategy = strategy_definition(strategy_key)
    for rule in _strategy_rules(strategy.rule_names):
        if not _rule_matches(feature_row, rule):
            continue
        return rule
    return None


def _strategy_rules(rule_names: tuple[str, ...] | None) -> tuple[dict[str, Any], ...]:
    if rule_names is None:
        return OPTIMIZED_EVENT_RULES
    allowed = set(rule_names)
    return tuple(rule for rule in OPTIMIZED_EVENT_RULES if rule["name"] in allowed)


def _rule_matches(feature_row: dict[str, Any], rule: dict[str, Any]) -> bool:
    return all(_condition_matches(feature_row, condition) for condition in rule["conditions"])


def _condition_matches(feature_row: dict[str, Any], condition: tuple[str, str, float]) -> bool:
    feature, operator, threshold = condition
    if feature not in feature_row:
        raise KeyError(f"optimized rule feature missing: {feature}")
    value = float(feature_row[feature])
    if operator == "<=":
        return value <= float(threshold)
    if operator == ">=":
        return value >= float(threshold)
    if operator == "==":
        return value == float(threshold)
    raise ValueError(f"unsupported rule operator: {operator}")
