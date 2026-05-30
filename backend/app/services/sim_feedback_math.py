from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

NEUTRAL_WIN_RATE = 0.5
NEUTRAL_PROFIT_FACTOR = 1.0


def win_rate(wins: int, count: int) -> float:
    if count <= 0:
        return NEUTRAL_WIN_RATE
    return float(wins / count)


def avg(total: float, count: int, *, default: float = 0.0) -> float:
    if count <= 0:
        return default
    return float(total / count)


def profit_factor(win_sum: float, loss_sum: float, count: int) -> float:
    if count <= 0:
        return NEUTRAL_PROFIT_FACTOR
    if loss_sum <= 0.0:
        return float(max(win_sum, NEUTRAL_PROFIT_FACTOR))
    return float(max(win_sum / loss_sum, 0.0))


def recent_win_rate(outcomes: deque[int]) -> float:
    if not outcomes:
        return NEUTRAL_WIN_RATE
    return float(sum(outcomes) / len(outcomes))


def neutral_feature_value(column: str) -> float:
    if column.endswith("_win_rate") or column.endswith("_confidence_mean"):
        return NEUTRAL_WIN_RATE
    if column.endswith("_profit_factor"):
        return NEUTRAL_PROFIT_FACTOR
    return 0.0


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
