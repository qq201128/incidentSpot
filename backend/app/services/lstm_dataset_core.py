from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_learning_controls import learning_risk_blocked_factor_names
from app.services.lstm_factor_combo_features import FACTOR_COMBO_FEATURE_PREFIX
from app.services.sim_feedback_features import SIM_FEEDBACK_PREFIX

MIN_FEATURE_FINITE_RATIO = 0.95
RAW_COMBO_PREFIXES = ("combo__", "goal_combo__")
DERIVED_COMBO_PREFIX = "combo_top"
BASE_EXCLUDED_COLUMNS = {
    "open_time", "entry_open_time", "open", "high", "low", "close", "volume",
    "future_return", "future_return_bps", "future_return_abs_bps", "label_threshold_bps",
    "label_up", "y",
}


class LstmDataError(ValueError):
    pass


@dataclass(frozen=True)
class LstmDataset:
    x: np.ndarray
    y: np.ndarray
    future_returns: np.ndarray
    entry_open_times: np.ndarray
    feature_columns: list[str]
    feature_frame: pd.DataFrame
    combo_snapshot: list[dict[str, Any]]
    learning_context: dict[str, Any] | None = None
    data_quality_report: dict[str, Any] | None = None
    sim_feedback_metadata: dict[str, Any] | None = None
    factor_combo_metadata: dict[str, Any] | None = None


def sanitize_feature_window(window: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(window, dtype=np.float32)
    if np.isfinite(cleaned).all():
        return cleaned
    return np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)


def candidate_feature_columns(
    frame: pd.DataFrame,
    learning_memory: dict[str, Any] | None = None,
) -> list[str]:
    blocked = learning_risk_blocked_factor_names(learning_memory)
    columns = []
    for column in frame.columns:
        name = str(column)
        if _candidate_allowed(frame, name, blocked):
            columns.append(name)
    if not columns:
        raise LstmDataError("no numeric LSTM feature columns")
    return columns


def windowed_lstm_dataset(
    frame: pd.DataFrame,
    feature_columns: list[str],
    feature_window: int,
    min_samples: int,
    *,
    combo_snapshot: list[dict[str, Any]] | None = None,
    learning_context: dict[str, Any] | None = None,
    data_quality_report: dict[str, Any] | None = None,
    sim_feedback_metadata: dict[str, Any] | None = None,
    factor_combo_metadata: dict[str, Any] | None = None,
) -> LstmDataset:
    _assert_columns(frame, feature_columns)
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    labels = frame["label_up"].to_numpy(dtype=np.float32)
    returns = frame["future_return"].to_numpy(dtype=np.float32)
    x, y, future_returns, entry_times = _window_arrays(frame, values, labels, returns, feature_window)
    if len(x) < min_samples:
        raise LstmDataError(f"insufficient LSTM samples: {len(x)} < {min_samples}")
    return LstmDataset(
        x,
        y,
        future_returns,
        entry_times,
        feature_columns,
        frame,
        combo_snapshot or [],
        learning_context,
        data_quality_report,
        sim_feedback_metadata,
        factor_combo_metadata,
    )


def _assert_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise LstmDataError(f"LSTM feature columns missing: {', '.join(missing[:12])}")


def _candidate_allowed(frame: pd.DataFrame, name: str, blocked: frozenset[str]) -> bool:
    if not _is_candidate_feature_column(name) or name in blocked:
        return False
    return bool(pd.api.types.is_numeric_dtype(frame[name]) and _column_is_finite_enough(frame[name]))


def _window_arrays(
    frame: pd.DataFrame,
    values: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    feature_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, future_returns, entry_times = [], [], [], []
    for end in range(feature_window - 1, len(frame)):
        window = values[end - feature_window + 1:end + 1]
        if _valid_sample(window, labels[end], returns[end]):
            x.append(window)
            y.append(labels[end])
            future_returns.append(returns[end])
            entry_times.append(int(frame.iloc[end]["entry_open_time"]))
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(future_returns, dtype=np.float32),
        np.asarray(entry_times, dtype=np.int64),
    )


def _column_is_finite_enough(values: pd.Series) -> bool:
    finite = np.isfinite(values.to_numpy(dtype=np.float32)).mean()
    return float(finite) >= MIN_FEATURE_FINITE_RATIO


def _valid_sample(window: np.ndarray, label: float, future_return: float) -> bool:
    return bool(np.isfinite(window).all() and isfinite(float(label)) and isfinite(float(future_return)))


def _is_candidate_feature_column(column: str) -> bool:
    if column.startswith(SIM_FEEDBACK_PREFIX) or column.startswith(FACTOR_COMBO_FEATURE_PREFIX):
        return True
    if column in BASE_EXCLUDED_COLUMNS:
        return False
    if column.startswith(DERIVED_COMBO_PREFIX):
        return False
    return not column.startswith(RAW_COMBO_PREFIXES)
