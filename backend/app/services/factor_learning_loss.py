from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_learning_common import (
    EPSILON,
    LOSS_PATTERN_LIMIT,
    MIN_LOSS_LIFT,
    MIN_LOSS_ROWS,
    MIN_MEDIAN_GAP_SCALE,
    MIN_PATTERN_SUPPORT,
    MIN_WIN_ROWS,
    round_metric,
)


def loss_memory(
    frame: pd.DataFrame,
    predictions: list[dict[str, Any]],
    columns: list[str],
) -> dict[str, Any]:
    aligned = _aligned_prediction_frame(frame, predictions, columns)
    if aligned.empty:
        return _empty_loss_memory("insufficient_settled_predictions", 0, 0)
    loss_mask = _loss_mask(aligned)
    loss_count = int(loss_mask.sum())
    win_count = int((~loss_mask).sum())
    if loss_count < MIN_LOSS_ROWS or win_count < MIN_WIN_ROWS:
        return _empty_loss_memory("insufficient_loss_or_win_samples", loss_count, len(aligned))
    patterns = _loss_patterns(aligned, columns, loss_mask)
    status = "learned" if patterns else "no_separable_loss_pattern"
    return {"status": status, "sampleCount": len(aligned), "lossCount": loss_count, "patterns": patterns}


def _aligned_prediction_frame(
    frame: pd.DataFrame,
    predictions: list[dict[str, Any]],
    columns: list[str],
) -> pd.DataFrame:
    if not predictions or "open_time" not in frame.columns or not columns:
        return pd.DataFrame()
    pred_df = pd.DataFrame(predictions)
    if "open_time" not in pred_df.columns or "actual_return" not in pred_df.columns:
        return pd.DataFrame()
    feature_df = frame[["open_time", *columns]].copy()
    return pred_df.merge(feature_df, on="open_time", how="inner")


def _loss_patterns(df: pd.DataFrame, columns: list[str], loss_mask: pd.Series) -> list[dict[str, Any]]:
    overall_loss_rate = float(loss_mask.mean())
    patterns = []
    for column in columns:
        pattern = _loss_pattern_for_column(df[column], column, loss_mask, overall_loss_rate)
        if pattern is not None:
            patterns.append(pattern)
    patterns.sort(key=lambda item: (item["lossLift"], item["support"]), reverse=True)
    return patterns[:LOSS_PATTERN_LIMIT]


def _loss_pattern_for_column(
    series: pd.Series,
    column: str,
    loss_mask: pd.Series,
    overall_loss_rate: float,
) -> dict[str, Any] | None:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.replace([np.inf, -np.inf], np.nan).notna()
    values = values[valid]
    mask = loss_mask[valid]
    losses = values[mask]
    wins = values[~mask]
    if len(losses) < MIN_LOSS_ROWS or len(wins) < MIN_WIN_ROWS:
        return None
    return _loss_pattern_from_samples(column, values, losses, wins, mask, overall_loss_rate)


def _loss_pattern_from_samples(
    column: str,
    values: pd.Series,
    losses: pd.Series,
    wins: pd.Series,
    mask: pd.Series,
    overall_loss_rate: float,
) -> dict[str, Any] | None:
    loss_median = float(losses.median())
    win_median = float(wins.median())
    scale = _robust_scale(values)
    if abs(loss_median - win_median) / scale < MIN_MEDIAN_GAP_SCALE:
        return None
    direction = "high" if loss_median > win_median else "low"
    threshold = (loss_median + win_median) / 2.0
    matched = values >= threshold if direction == "high" else values <= threshold
    support = int(matched.sum())
    if support < MIN_PATTERN_SUPPORT:
        return None
    loss_rate = float(mask[matched].mean())
    if loss_rate - overall_loss_rate < MIN_LOSS_LIFT:
        return None
    return _loss_pattern_payload(column, direction, threshold, support, loss_rate, overall_loss_rate)


def _loss_pattern_payload(
    column: str,
    direction: str,
    threshold: float,
    support: int,
    loss_rate: float,
    overall_loss_rate: float,
) -> dict[str, Any]:
    return {
        "feature": column,
        "direction": direction,
        "threshold": round_metric(threshold, 8),
        "support": support,
        "lossRate": round_metric(loss_rate, 4),
        "lossLift": round_metric(loss_rate - overall_loss_rate, 4),
    }


def _empty_loss_memory(status: str, loss_count: int, sample_count: int) -> dict[str, Any]:
    return {"status": status, "sampleCount": sample_count, "lossCount": loss_count, "patterns": []}


def _loss_mask(df: pd.DataFrame) -> pd.Series:
    returns = pd.to_numeric(df["actual_return"], errors="coerce").fillna(0.0)
    return returns <= 0.0


def _robust_scale(values: pd.Series) -> float:
    q75 = float(values.quantile(0.75))
    q25 = float(values.quantile(0.25))
    std = float(values.std())
    return max(q75 - q25, std, EPSILON)
