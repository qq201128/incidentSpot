from __future__ import annotations

from typing import Any

import numpy as np

from app.services.lstm_dataset_core import LstmDataset
from app.services.lstm_validation import binary_classification_metrics


def regime_validation_report(dataset: LstmDataset, split: Any, val_prob: np.ndarray) -> dict[str, Any]:
    regimes = _dataset_regime_labels(dataset)
    start = len(split.train_y)
    end = start + len(split.val_y)
    grouped = _group_indexes(regimes[start:end])
    return {
        label: binary_classification_metrics(
            split.val_y[indexes],
            val_prob[indexes],
            split.val_returns[indexes],
        )
        for label, indexes in sorted(grouped.items())
    }


def _dataset_regime_labels(dataset: LstmDataset) -> list[str]:
    by_entry = {
        int(row["entry_open_time"]): _row_regime_label(row)
        for _idx, row in dataset.feature_frame.iterrows()
        if "entry_open_time" in row
    }
    return [by_entry.get(int(entry), "unknown") for entry in dataset.entry_open_times]


def _group_indexes(labels: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return groups


def _row_regime_label(row: Any) -> str:
    trend = _trend_label(row)
    vol = _vol_label(row)
    return f"{trend}:{vol}"


def _trend_label(row: Any) -> str:
    if float(row.get("regime_trend_up") or 0) > 0:
        return "trend_up"
    if float(row.get("regime_trend_down") or 0) > 0:
        return "trend_down"
    if float(row.get("regime_range") or 0) > 0:
        return "range"
    return "uncertain"


def _vol_label(row: Any) -> str:
    if float(row.get("regime_high_vol") or 0) > 0:
        return "high_vol"
    if float(row.get("regime_low_vol") or 0) > 0:
        return "low_vol"
    return "normal_vol"
