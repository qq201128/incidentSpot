from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.factor_registry import list_factors
from app.services.kline_features import FEATURE_COLUMNS, build_feature_frame

SAMPLE_ROWS = 600
MUTATION_ROW = 420
ONE_MINUTE_MS = 60_000


def test_feature_columns_are_unique_and_computed() -> None:
    frame, spec = build_feature_frame(_sample_ohlcv_frame(), min_history=1)
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    assert spec.columns == FEATURE_COLUMNS
    assert not sorted(set(FEATURE_COLUMNS) - set(frame.columns))


def test_registered_kline_factors_are_shifted_feature_columns() -> None:
    registered = {
        factor.name
        for factor in list_factors()
        if factor.source_file == "kline_features.py"
    }
    missing = sorted(registered - set(FEATURE_COLUMNS))
    assert not missing


def test_shifted_features_ignore_current_bar_mutation() -> None:
    source = _sample_ohlcv_frame()
    mutated = _mutate_current_bar(source, MUTATION_ROW)

    baseline, _ = build_feature_frame(source, min_history=1)
    changed, _ = build_feature_frame(mutated, min_history=1)

    open_time = int(source.loc[MUTATION_ROW, "open_time"])
    baseline_row = _row_for_open_time(baseline, open_time)
    changed_row = _row_for_open_time(changed, open_time)

    np.testing.assert_allclose(
        baseline_row[FEATURE_COLUMNS].astype(float).to_numpy(),
        changed_row[FEATURE_COLUMNS].astype(float).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def _sample_ohlcv_frame(rows: int = SAMPLE_ROWS) -> pd.DataFrame:
    index = np.arange(rows, dtype="float64")
    close = 30_000.0 + index * 0.8 + np.sin(index / 13.0) * 90.0
    open_price = close + np.cos(index / 9.0) * 7.0
    high = np.maximum(open_price, close) + 10.0 + (index % 5)
    low = np.minimum(open_price, close) - 10.0 - (index % 7)
    volume = 100.0 + (index % 83.0) * 2.5 + np.sin(index / 19.0) * 8.0
    return pd.DataFrame({
        "open_time": (index.astype("int64") * ONE_MINUTE_MS),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _mutate_current_bar(source: pd.DataFrame, row_index: int) -> pd.DataFrame:
    mutated = source.copy()
    close = float(mutated.loc[row_index, "close"]) * 1.08
    mutated.loc[row_index, "open"] = close * 0.97
    mutated.loc[row_index, "high"] = close * 1.11
    mutated.loc[row_index, "low"] = close * 0.91
    mutated.loc[row_index, "close"] = close
    mutated.loc[row_index, "volume"] = float(mutated.loc[row_index, "volume"]) * 9.0
    return mutated


def _row_for_open_time(frame: pd.DataFrame, open_time: int) -> pd.Series:
    rows = frame.loc[frame["open_time"] == open_time]
    assert len(rows) == 1
    return rows.iloc[0]
