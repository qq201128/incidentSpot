from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

SCORE_DECIMALS = 6


def matched_loss_patterns(frame: pd.DataFrame, index: Any, memory: dict[str, Any]) -> list[dict[str, Any]]:
    patterns = (memory.get("lossMemory") or {}).get("patterns") or []
    matches = []
    for pattern in patterns:
        match = loss_pattern_match(frame, index, pattern)
        if match is not None:
            matches.append(match)
    return matches


def loss_pattern_match(frame: pd.DataFrame, index: Any, pattern: dict[str, Any]) -> dict[str, Any] | None:
    feature = str(pattern.get("feature") or "")
    if not feature or feature not in frame.columns:
        return None
    value = _finite_float(frame.at[index, feature])
    threshold = _finite_float(pattern.get("threshold"))
    if value is None or threshold is None:
        return None
    direction = str(pattern.get("direction"))
    matched = value >= threshold if direction == "high" else value <= threshold
    if not matched:
        return None
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        "value": round(value, SCORE_DECIMALS),
        "lossRate": pattern.get("lossRate"),
        "support": pattern.get("support"),
    }


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
