from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS

FACTOR_SCORE_CLIP = 4.0


def combination_score(frame: pd.DataFrame, members: list[dict[str, Any]]) -> pd.Series:
    scores = [_member_score(frame, member) for member in members]
    stacked = pd.concat(scores, axis=1)
    weights = np.asarray([float(member.get("weight") or 0.0) for member in members], dtype=float)
    if float(weights.sum()) <= 0:
        return stacked.mean(axis=1).replace([np.inf, -np.inf], np.nan)
    return _weighted_mean_score(stacked, weights).replace([np.inf, -np.inf], np.nan)


def _weighted_mean_score(stacked: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    valid_weights = stacked.notna().mul(weights, axis=1)
    total_weight = valid_weights.sum(axis=1)
    weighted_sum = stacked.mul(weights, axis=1).sum(axis=1)
    return (weighted_sum / total_weight).where(total_weight > 0)


def oriented_zscore(series: pd.Series, orientation: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.expanding(min_periods=BACKTEST_MIN_PERIODS).mean().shift(1)
    std = numeric.expanding(min_periods=BACKTEST_MIN_PERIODS).std().shift(1)
    zscore = (numeric - mean) / std.replace(0.0, np.nan)
    return zscore.clip(-FACTOR_SCORE_CLIP, FACTOR_SCORE_CLIP) * int(orientation)


def _member_score(frame: pd.DataFrame, member: dict[str, Any]) -> pd.Series:
    name = str(member["name"])
    if name not in frame.columns:
        raise ValueError(f"combination score missing factor column: {name}")
    return oriented_zscore(frame[name], int(member.get("orientation") or 1))
