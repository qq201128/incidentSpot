from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

EPSILON = 1e-12
CONFIDENCE_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)


@dataclass(frozen=True)
class LstmSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    train_returns: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_returns: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_returns: np.ndarray


def chronological_split(
    x: np.ndarray,
    y: np.ndarray,
    future_returns: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> LstmSplit:
    total = len(x)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    _assert_split_sizes(total, train_end, val_end)
    return LstmSplit(
        x[:train_end], y[:train_end], future_returns[:train_end],
        x[train_end:val_end], y[train_end:val_end], future_returns[train_end:val_end],
        x[val_end:], y[val_end:], future_returns[val_end:],
    )


def fit_standardizer(train_x: np.ndarray) -> dict[str, Any]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < EPSILON, 1.0, std)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def apply_standardizer(x: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if x.shape[-1] != len(mean):
        raise ValueError("LSTM scaler feature size does not match input")
    return ((x - mean) / std).astype(np.float32)


def binary_classification_metrics(
    y_true: np.ndarray,
    probability_up: np.ndarray,
    future_returns: np.ndarray,
) -> dict[str, Any]:
    pred_up = probability_up >= 0.5
    actual_up = y_true >= 0.5
    tp = int(np.logical_and(pred_up, actual_up).sum())
    tn = int(np.logical_and(~pred_up, ~actual_up).sum())
    fp = int(np.logical_and(pred_up, ~actual_up).sum())
    fn = int(np.logical_and(~pred_up, actual_up).sum())
    directional = _directional_returns(pred_up, future_returns)
    return {
        "sampleCount": int(len(y_true)),
        "accuracy": _ratio(tp + tn, len(y_true)),
        "winRate": _ratio(int((directional > 0).sum()), len(directional)),
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "profitFactor": profit_factor(directional),
        "maxDrawdown": max_drawdown(directional),
        "sharpe": sharpe_ratio(directional),
        "avgReturn": _mean_or_none(directional),
        "confidenceThresholds": confidence_threshold_metrics(
            probability_up,
            future_returns,
        ),
        "confusionMatrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def confidence_threshold_metrics(
    probability_up: np.ndarray,
    future_returns: np.ndarray,
) -> list[dict[str, Any]]:
    pred_up = probability_up >= 0.5
    confidence = np.maximum(probability_up, 1.0 - probability_up)
    directional = _directional_returns(pred_up, future_returns)
    return [
        _confidence_threshold_payload(threshold, directional[confidence >= threshold])
        for threshold in CONFIDENCE_THRESHOLDS
    ]


def profit_factor(returns: np.ndarray) -> float | None:
    wins = returns[returns > 0].sum()
    losses = returns[returns <= 0].sum()
    if wins <= 0:
        return 0.0
    if losses == 0:
        return float("inf")
    return float(wins / abs(losses))


def max_drawdown(returns: np.ndarray) -> float:
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return float(drawdown.max()) if len(drawdown) else 0.0


def sharpe_ratio(returns: np.ndarray) -> float | None:
    if len(returns) < 2:
        return None
    std = float(np.std(returns, ddof=1))
    if std < EPSILON:
        return None
    return float(np.mean(returns) / std * sqrt(len(returns)))


def _directional_returns(pred_up: np.ndarray, future_returns: np.ndarray) -> np.ndarray:
    return np.where(pred_up, future_returns, -future_returns)


def _confidence_threshold_payload(threshold: float, returns: np.ndarray) -> dict[str, Any]:
    return {
        "minConfidence": float(threshold),
        "sampleCount": int(len(returns)),
        "winRate": _ratio(int((returns > 0).sum()), len(returns)),
        "avgReturn": _mean_or_none(returns),
        "profitFactor": None if len(returns) == 0 else profit_factor(returns),
    }


def _assert_split_sizes(total: int, train_end: int, val_end: int) -> None:
    if train_end <= 0 or val_end <= train_end or val_end >= total:
        raise ValueError(f"invalid chronological LSTM split for {total} samples")


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator / denominator)


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else float(np.mean(values))
