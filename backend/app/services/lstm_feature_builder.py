from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_duration_alignment import duration_entry_rows, duration_entry_source_open_time
from app.services.factor_learning_controls import (
    learning_risk_blocked_factor_names,
    load_factor_learning_memory_for,
)
from app.services.lstm_combo_ranking import resolve_lstm_combo_ranking
from app.services.lstm_combo_snapshot import combo_snapshot_from_ranking
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_market_feature_builder import (
    build_lstm_market_feature_frame,
    load_lstm_market_frame,
    lstm_learning_context,
)
from app.services.rule_config import horizon_minutes_for_duration

MS_PER_MINUTE = 60_000
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


def build_lstm_training_dataset(
    config: LstmTrainingConfig,
    *,
    frame_loader: Callable[[str, str], pd.DataFrame] | None = None,
    ranking_loader: Callable[[str, str], dict[str, Any] | None] | None = None,
    learning_memory: dict[str, Any] | None = None,
) -> LstmDataset:
    memory = learning_memory if learning_memory is not None else load_factor_learning_memory_for(
        config.symbol,
        config.duration,
    )
    frame = _training_feature_frame(config, frame_loader, memory)
    labeled = duration_labeled_frame(frame, config.duration, _horizon_minutes(config), config.min_move_bps)
    columns = candidate_feature_columns(labeled, learning_memory=memory)
    return windowed_lstm_dataset(
        labeled,
        columns,
        config.feature_window,
        config.min_samples,
        combo_snapshot=_legacy_combo_snapshot(config.symbol, config.duration, ranking_loader),
        learning_context=lstm_learning_context(memory),
    )


def sanitize_feature_window(window: np.ndarray) -> np.ndarray:
    """Replace NaN/inf in a live feature window; pct_change and sparse inputs can leave edge infs."""
    cleaned = np.asarray(window, dtype=np.float32)
    if np.isfinite(cleaned).all():
        return cleaned
    return np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)


def build_live_feature_window(
    symbol: str,
    duration: str,
    feature_columns: list[str],
    feature_window: int,
    entry_open_time: int | None = None,
    combo_snapshot: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    del combo_snapshot
    sym = symbol.strip().upper()
    memory = load_factor_learning_memory_for(sym, duration)
    frame = load_lstm_market_frame(sym, duration, learning_memory=memory)
    sampled = duration_feature_frame(frame, duration, entry_open_time)
    _assert_columns(sampled, feature_columns)
    if len(sampled) < feature_window:
        raise LstmDataError(f"insufficient LSTM feature rows: {len(sampled)} < {feature_window}")
    window = sanitize_feature_window(
        sampled[feature_columns].tail(feature_window).to_numpy(dtype=np.float32),
    )
    last = sampled.iloc[-1]
    meta = {"entryOpenTime": int(last["entry_open_time"]), "entryPrice": float(last["close"])}
    return window.reshape(1, feature_window, len(feature_columns)), meta


def duration_labeled_frame(
    frame: pd.DataFrame,
    duration: str,
    horizon_minutes: int,
    min_move_bps: float,
) -> pd.DataFrame:
    sampled = duration_feature_frame(frame, duration).sort_values("open_time").reset_index(drop=True).copy()
    sampled["future_return"] = sampled["close"].shift(-1) / sampled["close"] - 1.0
    sampled["future_return_bps"] = sampled["future_return"] * 10_000.0
    sampled["future_return_abs_bps"] = sampled["future_return_bps"].abs()
    sampled["label_threshold_bps"] = _label_threshold_bps(sampled, min_move_bps)
    sampled["label_up"] = _label_series(sampled["future_return"], sampled["label_threshold_bps"] / 10_000.0)
    return sampled.dropna(subset=["future_return", "label_up"]).reset_index(drop=True)


def duration_feature_frame(
    frame: pd.DataFrame,
    duration: str,
    entry_open_time: int | None = None,
) -> pd.DataFrame:
    out = frame.sort_values("open_time").reset_index(drop=True).copy()
    open_time = pd.to_numeric(out["open_time"], errors="raise").astype("int64")
    out["entry_open_time"] = open_time + _duration_ms(duration)
    if entry_open_time is not None:
        source_open_time = duration_entry_source_open_time(entry_open_time, duration)
        out = out[open_time <= source_open_time].copy()
    sampled = duration_entry_rows(out, duration).reset_index(drop=True)
    if entry_open_time is not None:
        _assert_live_entry_row(sampled, int(entry_open_time), duration)
    return sampled


def candidate_feature_columns(
    frame: pd.DataFrame,
    learning_memory: dict[str, Any] | None = None,
) -> list[str]:
    return candidate_feature_columns_for_memory(frame, learning_memory)


def candidate_feature_columns_for_memory(
    frame: pd.DataFrame,
    learning_memory: dict[str, Any] | None,
) -> list[str]:
    blocked = learning_risk_blocked_factor_names(learning_memory)
    columns = []
    for column in frame.columns:
        name = str(column)
        if not _is_candidate_feature_column(name):
            continue
        if name in blocked:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) and _column_is_finite_enough(frame[column]):
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
    )


def _training_feature_frame(
    config: LstmTrainingConfig,
    frame_loader: Callable[[str, str], pd.DataFrame] | None,
    learning_memory: dict[str, Any] | None,
) -> pd.DataFrame:
    if frame_loader is None:
        return load_lstm_market_frame(config.symbol, config.duration, learning_memory=learning_memory)
    raw = frame_loader(config.symbol, config.duration)
    return build_lstm_market_feature_frame(
        raw,
        config.symbol,
        config.duration,
        learning_memory=learning_memory,
    )


def _legacy_combo_snapshot(
    symbol: str,
    duration: str,
    ranking_loader: Callable[[str, str], dict[str, Any] | None] | None,
) -> list[dict[str, Any]]:
    ranking = _loaded_ranking(symbol, duration, ranking_loader)
    if not _has_ranking(ranking):
        ranking = resolve_lstm_combo_ranking(symbol, duration)
    return combo_snapshot_from_ranking(ranking) if _has_ranking(ranking) else []


def _loaded_ranking(
    symbol: str,
    duration: str,
    ranking_loader: Callable[[str, str], dict[str, Any] | None] | None,
) -> dict[str, Any] | None:
    if ranking_loader is None:
        return None
    ranking = ranking_loader(symbol.strip().upper(), duration)
    return dict(ranking) if isinstance(ranking, dict) else None


def _has_ranking(ranking: dict[str, Any] | None) -> bool:
    rows = None if ranking is None else ranking.get("ranking")
    return isinstance(rows, list) and bool(rows)


def _assert_live_entry_row(frame: pd.DataFrame, entry_open_time: int, duration: str) -> None:
    source_open_time = duration_entry_source_open_time(entry_open_time, duration)
    if frame.empty or int(frame.iloc[-1]["open_time"]) != source_open_time:
        raise LstmDataError(f"missing completed LSTM source row at open_time={source_open_time}")


def _duration_ms(duration: str) -> int:
    return horizon_minutes_for_duration(duration) * MS_PER_MINUTE


def _horizon_minutes(config: LstmTrainingConfig) -> int:
    return int(config.horizon_minutes or horizon_minutes_for_duration(config.duration))


def _label_series(future_return: pd.Series, threshold: float | pd.Series) -> pd.Series:
    threshold = threshold if isinstance(threshold, pd.Series) else float(threshold)
    labels = pd.Series(np.nan, index=future_return.index)
    labels = labels.mask(future_return > threshold, 1.0)
    labels = labels.mask(future_return <= -threshold, 0.0)
    return labels


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


def _label_threshold_bps(frame: pd.DataFrame, min_move_bps: float) -> pd.Series:
    return pd.Series(float(min_move_bps), index=frame.index, dtype=np.float32)


def _valid_sample(window: np.ndarray, label: float, future_return: float) -> bool:
    return bool(np.isfinite(window).all() and isfinite(float(label)) and isfinite(float(future_return)))


def _assert_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise LstmDataError(f"LSTM feature columns missing: {', '.join(missing[:12])}")


def _is_candidate_feature_column(column: str) -> bool:
    if column in BASE_EXCLUDED_COLUMNS:
        return False
    if column.startswith(DERIVED_COMBO_PREFIX):
        return False
    return not column.startswith(RAW_COMBO_PREFIXES)
