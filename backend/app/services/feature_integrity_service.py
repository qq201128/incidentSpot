from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from app.services.enhanced_timeframes import add_online_timeframe_features

SAMPLE_ROWS = 6_000
CHECK_CUTOFFS = (1_537, 2_891, 4_327, 5_503)
TOLERANCE = 1e-12


@lru_cache(maxsize=1)
def timeframe_feature_integrity_report() -> dict[str, Any]:
    source = _sample_ohlcv_frame(SAMPLE_ROWS)
    full = add_online_timeframe_features(source.copy(), source)
    leaking = sorted(_leaking_features(source, full))
    return {
        "status": "passed" if not leaking else "failed",
        "checkedFeatureCount": len(_timeframe_columns(full)),
        "sampleRows": SAMPLE_ROWS,
        "leakingFeatures": leaking,
        "checks": [
            {"name": "previous_bucket_only", "passed": not leaking},
            {"name": "intrabar_running_values_only", "passed": not leaking},
            {"name": "no_future_bucket_volume_total", "passed": not leaking},
        ],
    }


def _leaking_features(source: pd.DataFrame, full: pd.DataFrame) -> set[str]:
    leaking: set[str] = set()
    columns = _timeframe_columns(full)
    for cutoff in CHECK_CUTOFFS:
        truncated = source.iloc[: cutoff + 1].copy()
        current = add_online_timeframe_features(truncated.copy(), truncated)
        leaking.update(_changed_columns(full.iloc[cutoff], current.iloc[-1], columns))
    return leaking


def _changed_columns(full_row: pd.Series, current_row: pd.Series, columns: list[str]) -> set[str]:
    changed = set()
    for column in columns:
        left = float(full_row[column])
        right = float(current_row[column])
        if np.isfinite(left) and np.isfinite(right) and abs(left - right) > TOLERANCE:
            changed.add(column)
    return changed


def _timeframe_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("tf_")]


def _sample_ohlcv_frame(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype="float64")
    close = 30_000.0 + index * 0.75 + np.sin(index / 17.0) * 120.0
    open_price = close + np.cos(index / 11.0) * 8.0
    high = np.maximum(open_price, close) + 12.0
    low = np.minimum(open_price, close) - 12.0
    volume = 100.0 + (index % 97.0) * 3.0 + np.sin(index / 23.0) * 10.0
    return pd.DataFrame({
        "open_time": (index.astype("int64") * 60_000),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
