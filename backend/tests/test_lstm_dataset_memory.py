from __future__ import annotations

import numpy as np

from app.services.lstm_dataset_core import _window_arrays
from app.services.lstm_validation import fit_standardizer


def test_window_arrays_return_view_when_all_samples_are_valid() -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    labels = np.ones(10, dtype=np.float32)
    returns = np.ones(10, dtype=np.float32)

    x, y, future_returns, entry_times = _window_arrays(_window_frame(10), values, labels, returns, 4)

    assert x.shape == (7, 4, 3)
    assert np.shares_memory(x, values)
    assert y.shape == (7,)
    assert future_returns.shape == (7,)
    assert entry_times.tolist() == list(range(3, 10))


def test_window_arrays_filter_invalid_windows() -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    values[4, 1] = np.nan
    labels = np.ones(10, dtype=np.float32)
    returns = np.ones(10, dtype=np.float32)

    x, _y, _future_returns, entry_times = _window_arrays(_window_frame(10), values, labels, returns, 4)

    assert x.shape == (3, 4, 3)
    assert entry_times.tolist() == [3, 8, 9]


def test_standardizer_uses_window_axes_without_flatten_copy() -> None:
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    x, *_ = _window_arrays(_window_frame(10), values, np.ones(10, dtype=np.float32), np.ones(10, dtype=np.float32), 4)
    flat = x.reshape(-1, x.shape[-1])

    scaler = fit_standardizer(x)

    assert scaler["mean"] == flat.mean(axis=0).astype(float).tolist()
    assert scaler["std"] == flat.std(axis=0).astype(float).tolist()


def _window_frame(size: int):
    import pandas as pd

    return pd.DataFrame({"entry_open_time": np.arange(size, dtype=np.int64)})
