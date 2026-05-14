from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.rule_config import MS_PER_MINUTE, horizon_minutes_for_duration


ENTRY_SOURCE_OFFSET_MS = MS_PER_MINUTE


def backtest_duration_frame(frame: pd.DataFrame, factor_name: str, duration: str) -> pd.DataFrame:
    _require_columns(frame, (factor_name, "close", "open_time"))
    horizon = horizon_minutes_for_duration(duration)
    out = frame[[factor_name, "close", "open_time"]].copy()
    out["fwd_ret"] = out["close"].shift(-horizon) / out["close"] - 1.0
    return duration_entry_rows(out, duration)


def duration_entry_rows(frame: pd.DataFrame, duration: str) -> pd.DataFrame:
    _require_columns(frame, ("open_time",))
    mask = _duration_entry_mask(frame["open_time"], duration)
    return frame.loc[mask].copy()


def live_duration_entry_index(
    frame: pd.DataFrame,
    duration: str,
    entry_open_time: int | None = None,
) -> Any:
    _require_columns(frame, ("open_time",))
    if entry_open_time is None:
        return _latest_duration_entry_index(frame, duration)
    return _exact_duration_entry_index(frame, duration_entry_source_open_time(entry_open_time))


def duration_entry_source_open_time(entry_open_time: int) -> int:
    source_open_time = int(entry_open_time) - ENTRY_SOURCE_OFFSET_MS
    if source_open_time < 0:
        raise ValueError(f"entry open time is too early for completed 1m source: {entry_open_time}")
    return source_open_time


def is_duration_entry_source_open_time(open_time: int, duration: str) -> bool:
    interval_ms = horizon_minutes_for_duration(duration) * MS_PER_MINUTE
    return (int(open_time) + ENTRY_SOURCE_OFFSET_MS) % interval_ms == 0


def _latest_duration_entry_index(frame: pd.DataFrame, duration: str) -> Any:
    entries = duration_entry_rows(frame, duration)
    if entries.empty:
        raise ValueError(f"no completed {duration} entry rows in factor frame")
    return entries.index[-1]


def _exact_duration_entry_index(frame: pd.DataFrame, source_open_time: int) -> Any:
    open_times = _open_times(frame)
    matches = frame.index[open_times == int(source_open_time)]
    if len(matches) == 0:
        raise ValueError(f"missing completed factor source row at open_time={source_open_time}")
    return matches[-1]


def _duration_entry_mask(open_time: pd.Series, duration: str) -> pd.Series:
    interval_ms = horizon_minutes_for_duration(duration) * MS_PER_MINUTE
    values = pd.to_numeric(open_time, errors="raise").astype("int64")
    return ((values + ENTRY_SOURCE_OFFSET_MS) % interval_ms) == 0


def _open_times(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["open_time"], errors="raise").astype("int64")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"factor frame missing columns: {', '.join(missing)}")
