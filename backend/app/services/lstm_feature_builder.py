from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combo_scoring import combination_score
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.lstm_config import LstmTrainingConfig

MS_PER_MINUTE = 60_000
COMBO_RANKS = (1, 2, 3)
BASE_EXCLUDED_COLUMNS = {
    "open_time", "entry_open_time", "open", "high", "low", "close", "volume",
    "future_return", "label_up", "y",
}
MIN_FEATURE_FINITE_RATIO = 0.95
ROLLING_SCORE_WINDOW = 60
EPSILON = 1e-12


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


def build_lstm_training_dataset(
    config: LstmTrainingConfig,
    *,
    frame_loader: Callable[[str], pd.DataFrame] = load_factor_frame,
    ranking_loader: Callable[[str, str], dict[str, Any] | None] = get_cached_combination_ranking,
) -> LstmDataset:
    frame = _load_enriched_factor_frame(config, frame_loader)
    ranking = _ranking_or_raise(config.symbol, config.duration, ranking_loader)
    featured = add_factor_combo_features(frame, ranking)
    labeled = duration_labeled_frame(featured, config.duration, int(config.horizon_minutes), config.min_move_bps)
    columns = candidate_feature_columns(labeled)
    return windowed_lstm_dataset(
        labeled,
        columns,
        config.feature_window,
        config.min_samples,
        combo_snapshot=_combo_snapshot(ranking),
    )


def build_live_feature_window(
    symbol: str,
    duration: str,
    feature_columns: list[str],
    feature_window: int,
    entry_open_time: int | None = None,
    combo_snapshot: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = _load_enriched_factor_frame_symbol(symbol, duration)
    ranking = _ranking_or_raise(symbol, duration, get_cached_combination_ranking)
    _assert_combo_snapshot(ranking, combo_snapshot)
    featured = add_factor_combo_features(frame, ranking)
    sampled = duration_feature_frame(featured, duration, entry_open_time)
    _assert_columns(sampled, feature_columns)
    if len(sampled) < feature_window:
        raise LstmDataError(f"insufficient LSTM feature rows: {len(sampled)} < {feature_window}")
    window = sampled[feature_columns].tail(feature_window).to_numpy(dtype=np.float32)
    if not np.isfinite(window).all():
        raise LstmDataError("LSTM live feature window contains non-finite values")
    last = sampled.iloc[-1]
    meta = {"entryOpenTime": int(last["entry_open_time"]), "entryPrice": float(last["close"])}
    return window.reshape(1, feature_window, len(feature_columns)), meta


def add_factor_combo_features(frame: pd.DataFrame, ranking_report: dict[str, Any]) -> pd.DataFrame:
    ranking = ranking_report.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        raise LstmDataError("factor combination ranking is empty")
    out = frame.copy()
    for rank in COMBO_RANKS:
        if rank > len(ranking):
            raise LstmDataError(f"factor combination ranking has no Top{rank}")
        out = _add_combo_rank_features(out, dict(ranking[rank - 1]), rank)
    return out


def duration_labeled_frame(
    frame: pd.DataFrame,
    duration: str,
    horizon_minutes: int,
    min_move_bps: float,
) -> pd.DataFrame:
    out = frame.sort_values("open_time").reset_index(drop=True).copy()
    out["future_return"] = out["close"].shift(-horizon_minutes) / out["close"] - 1.0
    out["label_up"] = _label_series(out["future_return"], min_move_bps)
    sampled = duration_feature_frame(out, duration)
    return sampled.dropna(subset=["future_return", "label_up"]).reset_index(drop=True)


def duration_feature_frame(
    frame: pd.DataFrame,
    duration: str,
    entry_open_time: int | None = None,
) -> pd.DataFrame:
    out = frame.sort_values("open_time").reset_index(drop=True).copy()
    open_time = pd.to_numeric(out["open_time"], errors="raise").astype("int64")
    out["entry_open_time"] = open_time + MS_PER_MINUTE
    if entry_open_time is not None:
        out = out[open_time <= int(entry_open_time) - MS_PER_MINUTE].copy()
        out["entry_open_time"] = int(entry_open_time)
    return out.reset_index(drop=True)


def candidate_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = []
    for column in frame.columns:
        if column in BASE_EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = frame[column].to_numpy(dtype=np.float32)
            if float(np.isfinite(values).mean()) >= MIN_FEATURE_FINITE_RATIO:
                columns.append(column)
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
) -> LstmDataset:
    _assert_columns(frame, feature_columns)
    values = frame[feature_columns].to_numpy(dtype=np.float32)
    labels = frame["label_up"].to_numpy(dtype=np.float32)
    returns = frame["future_return"].to_numpy(dtype=np.float32)
    x, y, future_returns, entry_times = _window_arrays(frame, values, labels, returns, feature_window)
    if len(x) < min_samples:
        raise LstmDataError(f"insufficient LSTM samples: {len(x)} < {min_samples}")
    return LstmDataset(x, y, future_returns, entry_times, feature_columns, frame, combo_snapshot or [])


def _load_enriched_factor_frame(
    config: LstmTrainingConfig,
    frame_loader: Callable[[str], pd.DataFrame],
) -> pd.DataFrame:
    base = frame_loader(config.symbol)
    return materialize_mined_factor_frame(base, symbol=config.symbol, duration=config.duration).frame


def _load_enriched_factor_frame_symbol(symbol: str, duration: str) -> pd.DataFrame:
    base = load_factor_frame(symbol)
    return materialize_mined_factor_frame(base, symbol=symbol, duration=duration).frame


def _ranking_or_raise(
    symbol: str,
    duration: str,
    ranking_loader: Callable[[str, str], dict[str, Any] | None],
) -> dict[str, Any]:
    ranking = ranking_loader(symbol.strip().upper(), duration)
    if ranking is None:
        raise LstmDataError(f"no cached factor combination ranking for {symbol.upper()} {duration}")
    return ranking


def _add_combo_rank_features(frame: pd.DataFrame, row: dict[str, Any], rank: int) -> pd.DataFrame:
    out = frame.copy()
    score = combination_score(out, _members(row))
    score_col = f"combo_top{rank}_score"
    out[score_col] = score
    _add_combo_rolling_metrics(out, score_col, rank)
    return out


def _add_combo_rolling_metrics(out: pd.DataFrame, score_col: str, rank: int) -> None:
    returns = out["close"].pct_change().fillna(0.0)
    signed = np.where(out[score_col].shift(1).fillna(0.0) >= 0.0, returns, -returns)
    signed_series = pd.Series(signed, index=out.index)
    wins = signed_series.clip(lower=0.0)
    losses = (-signed_series.clip(upper=0.0)).replace(0.0, np.nan)
    window = ROLLING_SCORE_WINDOW
    out[f"combo_top{rank}_rolling_win_rate"] = (signed_series > 0.0).rolling(window).mean()
    out[f"combo_top{rank}_rolling_sharpe"] = signed_series.rolling(window).mean() / signed_series.rolling(window).std()
    out[f"combo_top{rank}_rolling_profit_factor"] = wins.rolling(window).sum() / losses.rolling(window).sum()
    out[f"combo_top{rank}_rolling_contribution"] = out[score_col].abs().rolling(window).mean()
    out[f"combo_top{rank}_rolling_corr"] = out[score_col].shift(1).rolling(window).corr(returns).abs()


def _members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise LstmDataError(f"combination row missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _label_series(future_return: pd.Series, min_move_bps: float) -> pd.Series:
    threshold = float(min_move_bps) / 10_000.0
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


def _valid_sample(window: np.ndarray, label: float, future_return: float) -> bool:
    return bool(np.isfinite(window).all() and isfinite(float(label)) and isfinite(float(future_return)))


def _assert_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise LstmDataError(f"LSTM feature columns missing: {', '.join(missing[:12])}")


def _combo_snapshot(ranking_report: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = ranking_report.get("ranking") or []
    return [_combo_row_snapshot(dict(row), rank) for rank, row in enumerate(ranking[:3], start=1)]


def _combo_row_snapshot(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "factorName": row.get("factorName"),
        "members": [member.get("name") for member in _members(row)],
    }


def _assert_combo_snapshot(
    ranking_report: dict[str, Any],
    expected: list[dict[str, Any]] | None,
) -> None:
    if not expected:
        return
    current = _combo_snapshot(ranking_report)
    if current != expected:
        raise LstmDataError("current factor combo Top1/Top2/Top3 differs from LSTM training snapshot")
