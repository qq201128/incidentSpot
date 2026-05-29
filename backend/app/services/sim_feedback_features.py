from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.db.session import get_conn
from app.services.factor_combo_simulation_keys import factor_combo_event_strategy_filter
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.model_family_config import MODEL_FAMILIES, normalize_model_family

SIM_FEEDBACK_PREFIX = "sim_feedback_"
SIM_FEEDBACK_ROLLING_WINDOW = 20
NEUTRAL_WIN_RATE = 0.5
NEUTRAL_PROFIT_FACTOR = 1.0


def sim_feedback_feature_names(model_family: str | None = None) -> list[str]:
    columns = [
        f"{SIM_FEEDBACK_PREFIX}settled_count",
        f"{SIM_FEEDBACK_PREFIX}win_rate",
        f"{SIM_FEEDBACK_PREFIX}avg_return",
        f"{SIM_FEEDBACK_PREFIX}profit_factor",
        f"{SIM_FEEDBACK_PREFIX}recent_win_rate",
        f"{SIM_FEEDBACK_PREFIX}loss_streak",
        f"{SIM_FEEDBACK_PREFIX}confidence_mean",
    ]
    if model_family:
        family = normalize_model_family(model_family)
        columns.extend(
            [
                f"{SIM_FEEDBACK_PREFIX}family_{family}_settled_count",
                f"{SIM_FEEDBACK_PREFIX}family_{family}_win_rate",
                f"{SIM_FEEDBACK_PREFIX}family_{family}_avg_return",
                f"{SIM_FEEDBACK_PREFIX}family_{family}_loss_streak",
            ]
        )
    return columns


def attach_sim_feedback_features(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
    *,
    model_family: str | None = None,
    predictions_loader: Callable[[str, str], list[dict[str, Any]]] | None = None,
) -> pd.DataFrame:
    if "entry_open_time" not in frame.columns:
        if frame.empty:
            return _with_metadata(_attach_neutral_features(frame, model_family), _metadata(0, "empty_frame"))
        raise ValueError("sim feedback requires entry_open_time column")
    loader = predictions_loader or load_settled_sim_predictions
    predictions = loader(symbol.strip().upper(), duration)
    if not isinstance(predictions, list):
        raise ValueError("settled sim predictions loader must return a list")
    snapshots = _build_sim_feedback_snapshots(predictions, model_family)
    merged = pd.merge_asof(
        frame.sort_values("entry_open_time").reset_index(drop=True),
        snapshots.sort_values("open_time"),
        left_on="entry_open_time",
        right_on="open_time",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.drop(columns=["open_time"], errors="ignore")
    for column in sim_feedback_feature_names(model_family):
        if column not in merged.columns:
            merged[column] = _neutral_feature_value(column)
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(_neutral_feature_value(column))
    status = "neutral_no_settled_predictions" if len(predictions) == 0 else "loaded"
    return _with_metadata(merged.reset_index(drop=True), _metadata(len(predictions), status))


def load_settled_sim_predictions(symbol: str, duration: str) -> list[dict[str, Any]]:
    sym = symbol.strip().upper()
    clause, filter_params = factor_combo_event_strategy_filter()
    conn = get_conn()
    try:
        rows = conn.execute(
            f"""
            SELECT open_time, direction, confidence, trade_quality_score,
                   actual_return, prediction_correct, signal_key, strategy_key,
                   model_family
            FROM predictions
            WHERE symbol = ? AND duration = ? AND settled_at IS NOT NULL
              AND (
                {clause}
                OR signal_key = ?
                OR signal_key LIKE 'factor_%_shadow_%'
              )
            ORDER BY open_time
            """,
            (sym, duration, *filter_params, lstm_shadow_strategy_key(duration)),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _attach_neutral_features(frame: pd.DataFrame, model_family: str | None) -> pd.DataFrame:
    out = frame.copy()
    for column in sim_feedback_feature_names(model_family):
        out[column] = _neutral_feature_value(column)
    return out


def _with_metadata(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    frame.attrs["simFeedbackMetadata"] = metadata
    return frame


def _metadata(settled_count: int, status: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": status,
        "settledCount": int(settled_count),
        "neutralFeaturesUsed": settled_count == 0,
    }


def _build_sim_feedback_snapshots(
    predictions: list[dict[str, Any]],
    model_family: str | None,
) -> pd.DataFrame:
    columns = sim_feedback_feature_names(model_family)
    state = _SimFeedbackState(model_family)
    snapshots = [{"open_time": -1, **state.features()}]
    for prediction in sorted(predictions, key=lambda row: int(row["open_time"])):
        state.update(prediction)
        snapshots.append({"open_time": int(prediction["open_time"]), **state.features()})
    frame = pd.DataFrame(snapshots)
    for column in columns:
        if column not in frame.columns:
            frame[column] = _neutral_feature_value(column)
    return frame[columns + ["open_time"]]


@dataclass
class _FamilyStats:
    settled_count: int = 0
    win_count: int = 0
    return_sum: float = 0.0
    win_sum: float = 0.0
    loss_sum: float = 0.0
    loss_streak: int = 0


@dataclass
class _SimFeedbackState:
    model_family: str | None
    settled_count: int = 0
    win_count: int = 0
    return_sum: float = 0.0
    win_sum: float = 0.0
    loss_sum: float = 0.0
    confidence_sum: float = 0.0
    loss_streak: int = 0
    recent_outcomes: deque[int] = field(default_factory=lambda: deque(maxlen=SIM_FEEDBACK_ROLLING_WINDOW))
    family_stats: dict[str, _FamilyStats] = field(default_factory=dict)

    def update(self, prediction: dict[str, Any]) -> None:
        correct = _prediction_correct(prediction)
        actual_return = _finite_float(prediction.get("actual_return")) or 0.0
        confidence = _prediction_confidence(prediction)
        self.settled_count += 1
        self.return_sum += actual_return
        self.confidence_sum += confidence
        self.recent_outcomes.append(1 if correct else 0)
        if correct:
            self.win_count += 1
            self.win_sum += max(actual_return, 0.0)
            self.loss_streak = 0
        else:
            self.loss_sum += abs(min(actual_return, 0.0))
            self.loss_streak += 1
        family = _prediction_family(prediction)
        if family:
            stats = self.family_stats.setdefault(family, _FamilyStats())
            stats.settled_count += 1
            stats.return_sum += actual_return
            if correct:
                stats.win_count += 1
                stats.win_sum += max(actual_return, 0.0)
                stats.loss_streak = 0
            else:
                stats.loss_sum += abs(min(actual_return, 0.0))
                stats.loss_streak += 1

    def features(self) -> dict[str, float]:
        payload = {
            f"{SIM_FEEDBACK_PREFIX}settled_count": float(self.settled_count),
            f"{SIM_FEEDBACK_PREFIX}win_rate": _win_rate(self.win_count, self.settled_count),
            f"{SIM_FEEDBACK_PREFIX}avg_return": _avg(self.return_sum, self.settled_count),
            f"{SIM_FEEDBACK_PREFIX}profit_factor": _profit_factor(self.win_sum, self.loss_sum, self.settled_count),
            f"{SIM_FEEDBACK_PREFIX}recent_win_rate": _recent_win_rate(self.recent_outcomes),
            f"{SIM_FEEDBACK_PREFIX}loss_streak": float(self.loss_streak),
            f"{SIM_FEEDBACK_PREFIX}confidence_mean": _avg(self.confidence_sum, self.settled_count, default=NEUTRAL_WIN_RATE),
        }
        if self.model_family:
            family = normalize_model_family(self.model_family)
            stats = self.family_stats.get(family, _FamilyStats())
            payload.update(
                {
                    f"{SIM_FEEDBACK_PREFIX}family_{family}_settled_count": float(stats.settled_count),
                    f"{SIM_FEEDBACK_PREFIX}family_{family}_win_rate": _win_rate(stats.win_count, stats.settled_count),
                    f"{SIM_FEEDBACK_PREFIX}family_{family}_avg_return": _avg(stats.return_sum, stats.settled_count),
                    f"{SIM_FEEDBACK_PREFIX}family_{family}_loss_streak": float(stats.loss_streak),
                }
            )
        return payload


def _prediction_family(prediction: dict[str, Any]) -> str | None:
    raw = prediction.get("model_family")
    if raw:
        try:
            return normalize_model_family(str(raw))
        except ValueError:
            pass
    signal_key = str(prediction.get("signal_key") or prediction.get("strategy_key") or "")
    if not signal_key.startswith("factor_") or "_shadow_" not in signal_key:
        return None
    suffix = signal_key.removeprefix("factor_")
    for family in sorted(MODEL_FAMILIES, key=len, reverse=True):
        prefix = f"{family}_shadow_"
        if suffix.startswith(prefix):
            return family
    return None


def _prediction_correct(prediction: dict[str, Any]) -> bool:
    value = prediction.get("prediction_correct")
    if value is not None:
        return bool(value)
    actual_return = _finite_float(prediction.get("actual_return"))
    if actual_return is None:
        return False
    return actual_return > 0.0


def _prediction_confidence(prediction: dict[str, Any]) -> float:
    for key in ("confidence", "trade_quality_score"):
        value = _finite_float(prediction.get(key))
        if value is not None:
            return float(np.clip(value, 0.0, 1.0))
    return NEUTRAL_WIN_RATE


def _win_rate(wins: int, count: int) -> float:
    if count <= 0:
        return NEUTRAL_WIN_RATE
    return float(wins / count)


def _avg(total: float, count: int, *, default: float = 0.0) -> float:
    if count <= 0:
        return default
    return float(total / count)


def _profit_factor(win_sum: float, loss_sum: float, count: int) -> float:
    if count <= 0:
        return NEUTRAL_PROFIT_FACTOR
    if loss_sum <= 0.0:
        return float(max(win_sum, NEUTRAL_PROFIT_FACTOR))
    return float(max(win_sum / loss_sum, 0.0))


def _recent_win_rate(outcomes: deque[int]) -> float:
    if not outcomes:
        return NEUTRAL_WIN_RATE
    return float(sum(outcomes) / len(outcomes))


def _neutral_feature_value(column: str) -> float:
    if column.endswith("_win_rate") or column.endswith("_confidence_mean"):
        return NEUTRAL_WIN_RATE
    if column.endswith("_profit_factor"):
        return NEUTRAL_PROFIT_FACTOR
    return 0.0


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
