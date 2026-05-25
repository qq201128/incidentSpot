from __future__ import annotations

import math
from typing import Any

import pandas as pd


def combo_return_metrics(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {"trades": 0, "winRate": 0.0, "profitFactor": 0.0, "avgReturn": 0.0}
    return {
        "trades": len(returns),
        "winRate": float((returns > 0).mean()),
        "profitFactor": profit_factor(returns),
        "avgReturn": float(returns.mean()),
    }


def combo_rejection(
    members: tuple[str, ...],
    threshold: float,
    reason: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {"members": members, "threshold": threshold, "reason": reason, **metrics}


def profit_factor(returns: pd.Series) -> float:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    return gains / losses if losses > 0 else math.inf


def selected_hits(hits: list[Any], target_count: int) -> list[Any]:
    return hits[:target_count]


def nested_frames(frame: pd.DataFrame, train_ratio: float, validation_ratio: float) -> dict[str, pd.DataFrame]:
    train_end = int(len(frame) * train_ratio)
    validation_end = train_end + int(len(frame) * validation_ratio)
    return {
        "train": frame.iloc[:train_end],
        "validation": frame.iloc[train_end:validation_end],
        "test": frame.iloc[validation_end:],
    }


def nested_split_payload(
    frame: pd.DataFrame,
    split: dict[str, pd.DataFrame],
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, Any]:
    return {
        "trainRatio": train_ratio,
        "validationRatio": validation_ratio,
        "testRatio": round(1.0 - train_ratio - validation_ratio, 4),
        "rows": {name: len(value) for name, value in split.items()},
        "totalRows": len(frame),
    }
