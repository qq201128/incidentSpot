from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.factor_duration_alignment import (
    backtest_duration_frame,
    duration_entry_source_open_time,
    is_duration_entry_source_open_time,
    live_duration_entry_index,
)

ONE_MINUTE_MS = 60_000
TEN_MINUTES = 10
ROWS = 40


def test_backtest_duration_frame_samples_completed_duration_entries() -> None:
    frame = _frame(ROWS)

    out = backtest_duration_frame(frame, "factor_a", "10m")

    assert list(out["open_time"].head(3)) == [9 * ONE_MINUTE_MS, 19 * ONE_MINUTE_MS, 29 * ONE_MINUTE_MS]
    assert all(is_duration_entry_source_open_time(open_time, "10m") for open_time in out["open_time"])
    expected = frame["close"].shift(-TEN_MINUTES) / frame["close"] - 1.0
    assert out["fwd_ret"].iloc[0] == expected.iloc[9]


def test_live_duration_entry_index_uses_entry_previous_completed_minute() -> None:
    frame = _frame(ROWS)
    entry_open_time = 30 * ONE_MINUTE_MS

    index = live_duration_entry_index(frame, "10m", entry_open_time)

    assert frame.at[index, "open_time"] == duration_entry_source_open_time(entry_open_time)


def _frame(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "open_time": np.arange(rows) * ONE_MINUTE_MS,
            "close": 100.0 + index,
            "factor_a": index,
        }
    )
