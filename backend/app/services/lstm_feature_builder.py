from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from app.services.factor_duration_alignment import duration_entry_rows, duration_entry_source_open_time
from app.services.factor_learning_controls import load_factor_learning_memory_for
from app.services.lstm_data_quality import (
    feature_column_quality,
    validate_duration_source_frame,
    validate_labeled_frame,
)
from app.services.lstm_factor_combo_features import (
    FACTOR_COMBO_FEATURE_PREFIX,
    attach_live_factor_combo_features,
    attach_training_factor_combo_features,
    load_factor_combo_feature_snapshots,
)
from app.services.lstm_combo_ranking import resolve_lstm_combo_ranking
from app.services.lstm_combo_snapshot import combo_snapshot_from_ranking
from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_dataset_core import (
    LstmDataError,
    LstmDataset,
    _assert_columns,
    candidate_feature_columns,
    sanitize_feature_window,
    windowed_lstm_dataset,
)
from app.services.lstm_market_feature_builder import (
    build_lstm_market_feature_frame,
    load_lstm_market_frame,
    lstm_learning_context,
)
from app.services.rule_config import horizon_minutes_for_duration
from app.services.sim_feedback_features import SIM_FEEDBACK_PREFIX, attach_sim_feedback_features

MS_PER_MINUTE = 60_000

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
    model_family = getattr(config, "family", None)
    frame = _training_feature_frame(config, frame_loader, memory)
    source_quality = validate_duration_source_frame(frame, config.duration)
    labeled = duration_labeled_frame(frame, config.duration, _horizon_minutes(config), config.min_move_bps)
    label_quality = validate_labeled_frame(labeled, config.duration)
    labeled, sim_metadata = _attach_sim_feedback_with_metadata(
        labeled,
        config.symbol,
        config.duration,
        model_family=model_family,
    )
    factor_combo_metadata = None
    if model_family:
        combo_result = attach_training_factor_combo_features(
            labeled,
            config.symbol,
            config.duration,
            snapshots_loader=load_factor_combo_feature_snapshots,
        )
        labeled = combo_result.frame
        factor_combo_metadata = combo_result.metadata
    columns = candidate_feature_columns(labeled, learning_memory=memory)
    quality = _data_quality_report(source_quality, label_quality, columns)
    return windowed_lstm_dataset(
        labeled,
        columns,
        config.feature_window,
        config.min_samples,
        combo_snapshot=_legacy_combo_snapshot(config.symbol, config.duration, ranking_loader),
        learning_context=lstm_learning_context(memory),
        data_quality_report=quality,
        sim_feedback_metadata=sim_metadata,
        factor_combo_metadata=factor_combo_metadata,
    )

def build_live_feature_window(
    symbol: str,
    duration: str,
    feature_columns: list[str],
    feature_window: int,
    entry_open_time: int | None = None,
    combo_snapshot: list[dict[str, Any]] | None = None,
    model_family: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    del combo_snapshot
    sym = symbol.strip().upper()
    memory = load_factor_learning_memory_for(sym, duration)
    frame = load_lstm_market_frame(sym, duration, learning_memory=memory)
    sampled = duration_feature_frame(frame, duration, entry_open_time)
    factor_combo_metadata = None
    if _needs_factor_combo_features(feature_columns):
        combo_result = attach_live_factor_combo_features(sampled, sym, duration)
        sampled = combo_result.frame
        factor_combo_metadata = combo_result.metadata
    sampled, _metadata = _attach_sim_feedback_with_metadata(
        sampled,
        sym,
        duration,
        model_family=model_family,
    )
    _assert_columns(sampled, feature_columns)
    if len(sampled) < feature_window:
        raise LstmDataError(f"insufficient LSTM feature rows: {len(sampled)} < {feature_window}")
    window = sanitize_feature_window(
        sampled[feature_columns].tail(feature_window).to_numpy(dtype=np.float32),
    )
    last = sampled.iloc[-1]
    meta = _live_feature_meta(
        last,
        entry_open_time,
        feature_columns,
        factor_combo_metadata,
        _metadata,
    )
    return window.reshape(1, feature_window, len(feature_columns)), meta

def duration_labeled_frame(
    frame: pd.DataFrame,
    duration: str,
    horizon_minutes: int,
    min_move_bps: float,
) -> pd.DataFrame:
    sampled = duration_feature_frame(frame, duration).sort_values("open_time").reset_index(drop=True).copy()
    horizon_bars = _horizon_bar_count(duration, horizon_minutes)
    sampled["future_return"] = sampled["close"].shift(-horizon_bars) / sampled["close"] - 1.0
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

def _attach_sim_feedback_with_metadata(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
    *,
    model_family: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched = attach_sim_feedback_features(frame, symbol, duration, model_family=model_family)
    return enriched, dict(enriched.attrs.get("simFeedbackMetadata") or {})

def _data_quality_report(
    source_quality: dict[str, Any],
    label_quality: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    return {
        "status": "passed",
        "sourceAlignment": source_quality,
        "labels": label_quality,
        "features": feature_column_quality(columns),
    }

def _needs_factor_combo_features(feature_columns: list[str]) -> bool:
    return any(column.startswith(FACTOR_COMBO_FEATURE_PREFIX) for column in feature_columns)

def _live_feature_meta(
    last: pd.Series,
    requested_entry_open_time: int | None,
    feature_columns: list[str],
    factor_combo_metadata: dict[str, Any] | None,
    sim_feedback_metadata: dict[str, Any],
) -> dict[str, Any]:
    actual_entry = int(last["entry_open_time"])
    return {
        "entryOpenTime": actual_entry,
        "requestedEntryOpenTime": requested_entry_open_time,
        "entryPrice": float(last["close"]),
        "dataFreshnessStatus": _data_freshness_status(actual_entry, requested_entry_open_time),
        "missingFeatureStatus": _missing_feature_status(feature_columns, factor_combo_metadata, sim_feedback_metadata),
        "factorComboFeatureMetadata": factor_combo_metadata,
        "simFeedbackMetadata": sim_feedback_metadata,
    }

def _data_freshness_status(actual_entry: int, requested_entry: int | None) -> str:
    if requested_entry is None:
        return "latest_available"
    if actual_entry == int(requested_entry):
        return "fresh"
    return "entry_mismatch"

def _missing_feature_status(
    feature_columns: list[str],
    factor_combo_metadata: dict[str, Any] | None,
    sim_feedback_metadata: dict[str, Any],
) -> str:
    if _factor_combo_features_missing(feature_columns, factor_combo_metadata):
        return "missing_factor_combo_features"
    if _needs_sim_feedback_features(feature_columns) and sim_feedback_metadata.get("neutralFeaturesUsed"):
        return "neutral_sim_feedback_features"
    return "complete"

def _factor_combo_features_missing(feature_columns: list[str], metadata: dict[str, Any] | None) -> bool:
    if not _needs_factor_combo_features(feature_columns):
        return False
    if not isinstance(metadata, dict):
        return True
    return float(metadata.get("missingRate") or 0.0) > 0.0
def _needs_sim_feedback_features(feature_columns: list[str]) -> bool:
    return any(column.startswith(SIM_FEEDBACK_PREFIX) for column in feature_columns)

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

def _horizon_bar_count(duration: str, horizon_minutes: int) -> int:
    duration_minutes = horizon_minutes_for_duration(duration)
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError("horizon_minutes must be positive")
    if horizon % duration_minutes != 0:
        raise ValueError(
            f"horizon_minutes={horizon} must be a multiple of duration minutes={duration_minutes}"
        )
    return max(1, horizon // duration_minutes)

def _label_series(future_return: pd.Series, threshold: float | pd.Series) -> pd.Series:
    threshold = threshold if isinstance(threshold, pd.Series) else float(threshold)
    labels = pd.Series(np.nan, index=future_return.index)
    labels = labels.mask(future_return > threshold, 1.0)
    labels = labels.mask(future_return <= -threshold, 0.0)
    return labels

def _label_threshold_bps(frame: pd.DataFrame, min_move_bps: float) -> pd.Series:
    return pd.Series(float(min_move_bps), index=frame.index, dtype=np.float32)
