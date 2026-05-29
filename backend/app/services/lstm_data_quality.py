from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.services.rule_config import MS_PER_MINUTE, horizon_minutes_for_duration

ALIGNMENT_POLICY = "source_open_time_plus_duration_equals_entry_open_time"


def validate_duration_source_frame(frame: pd.DataFrame, duration: str) -> dict[str, Any]:
    _require_columns(frame, ("open_time",))
    if frame.empty:
        raise ValueError(f"no completed {duration} source rows")
    out = frame.sort_values("open_time").reset_index(drop=True)
    open_time = pd.to_numeric(out["open_time"], errors="raise").astype("int64")
    _assert_unique_open_times(open_time)
    _assert_aligned_open_times(open_time, duration)
    _assert_no_missing_periods(open_time, duration)
    if "close_time" in out.columns:
        _assert_closed_bars(out, open_time, duration)
    return _alignment_report(open_time, duration)


def validate_labeled_frame(frame: pd.DataFrame, duration: str) -> dict[str, Any]:
    _require_columns(frame, ("open_time", "entry_open_time", "future_return", "label_up"))
    source = pd.to_numeric(frame["open_time"], errors="raise").astype("int64")
    entry = pd.to_numeric(frame["entry_open_time"], errors="raise").astype("int64")
    expected = source + _duration_ms(duration)
    if not np.array_equal(entry.to_numpy(), expected.to_numpy()):
        raise ValueError("entry_open_time must equal source open_time plus duration")
    if not np.isfinite(frame["future_return"].to_numpy(dtype=np.float64)).all():
        raise ValueError("future_return contains non-finite values")
    labels = frame["label_up"].to_numpy(dtype=np.float64)
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("label_up must contain only 0/1 labels")
    return {
        "labelPolicy": "future_close_over_source_close_minus_one",
        "labelRows": int(len(frame)),
        "sourceMaxOpenTime": int(source.max()) if len(source) else None,
        "entryMaxOpenTime": int(entry.max()) if len(entry) else None,
    }


def feature_column_quality(columns: list[str]) -> dict[str, Any]:
    return {
        "featureColumnCount": int(len(columns)),
        "featureColumns": list(columns),
        "containsLabelColumns": bool(_label_columns(columns)),
        "labelColumnsExcluded": not bool(_label_columns(columns)),
    }


def _assert_unique_open_times(open_time: pd.Series) -> None:
    duplicated = open_time[open_time.duplicated()]
    if duplicated.empty:
        return
    raise ValueError(f"duplicate source open_time={int(duplicated.iloc[0])}")


def _assert_aligned_open_times(open_time: pd.Series, duration: str) -> None:
    step = _duration_ms(duration)
    misaligned = open_time[open_time % step != 0]
    if misaligned.empty:
        return
    raise ValueError(f"source open_time is not aligned to {duration}: {int(misaligned.iloc[0])}")


def _assert_no_missing_periods(open_time: pd.Series, duration: str) -> None:
    if len(open_time) < 2:
        return
    step = _duration_ms(duration)
    diffs = np.diff(open_time.to_numpy(dtype=np.int64))
    gaps = diffs[diffs != step]
    if len(gaps) == 0:
        return
    index = int(np.where(diffs != step)[0][0])
    raise ValueError(
        f"missing {duration} source period between open_time={int(open_time.iloc[index])} "
        f"and open_time={int(open_time.iloc[index + 1])}"
    )


def _assert_closed_bars(frame: pd.DataFrame, open_time: pd.Series, duration: str) -> None:
    close_time = pd.to_numeric(frame["close_time"], errors="raise").astype("int64")
    expected_max = open_time + _duration_ms(duration)
    invalid = close_time >= expected_max
    if not bool(invalid.any()):
        return
    row = int(np.where(invalid.to_numpy())[0][0])
    raise ValueError(f"unclosed {duration} source kline at open_time={int(open_time.iloc[row])}")


def _alignment_report(open_time: pd.Series, duration: str) -> dict[str, Any]:
    return {
        "alignmentPolicy": ALIGNMENT_POLICY,
        "duration": duration,
        "sourceRowCount": int(len(open_time)),
        "sourceMinOpenTime": int(open_time.min()),
        "sourceMaxOpenTime": int(open_time.max()),
        "durationMs": _duration_ms(duration),
        "missingPeriods": 0,
        "closedKlinesOnly": True,
    }


def _label_columns(columns: list[str]) -> list[str]:
    blocked = {"future_return", "future_return_bps", "future_return_abs_bps", "label_threshold_bps", "label_up", "y"}
    return [column for column in columns if column in blocked]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"training frame missing columns: {', '.join(missing)}")


def _duration_ms(duration: str) -> int:
    return horizon_minutes_for_duration(duration) * MS_PER_MINUTE
