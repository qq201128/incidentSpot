from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-12


def _series_window_args(args: list[Any], name: str) -> tuple[pd.Series, int]:
    if len(args) != 2:
        raise ValueError(f"{name} requires series and window arguments")
    return _series_arg(args[0], name), _window_arg(args[1:], name)


def _series_lag_args(args: list[Any], name: str) -> tuple[pd.Series, int]:
    if len(args) != 2:
        raise ValueError(f"{name} requires series and lag arguments")
    return _series_arg(args[0], name), _positive_int_arg(args[1:], name, "lag")


def _series_lag_window_args(args: list[Any], name: str) -> tuple[pd.Series, int, int]:
    if len(args) != 3:
        raise ValueError(f"{name} requires series, lag, and window arguments")
    return _series_arg(args[0], name), _positive_int_arg(args[1:2], name, "lag"), _window_arg(args[2:], name)


def _series_series_window_args(args: list[Any], name: str) -> tuple[pd.Series, pd.Series, int]:
    if len(args) != 3:
        raise ValueError(f"{name} requires two series and window arguments")
    return _series_arg(args[0], name), _series_arg(args[1], name), _window_arg(args[2:], name)


def _window_arg(args: list[Any], name: str) -> int:
    return _positive_int_arg(args, name, "window", min_value=2)


def _int_arg(args: list[Any], name: str) -> int:
    return _window_arg(args, name)


def _float_arg(args: list[Any], name: str) -> float:
    if len(args) != 1 or isinstance(args[0], pd.Series):
        raise ValueError(f"{name} requires a numeric argument")
    return float(args[0])


def _ts_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0.0, np.nan)
    return (series - mean) / std


def _vwap(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    numerator = (price * volume).rolling(window).sum()
    denominator = volume.rolling(window).sum()
    return numerator / _nonzero(denominator)


def _signed_power(series: pd.Series, power: float) -> pd.Series:
    return np.sign(series) * np.power(series.abs(), power)


def _nonzero(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.where(value.abs() > EPSILON, np.nan)
    return value if abs(float(value)) > EPSILON else np.nan


def _positive_int_arg(args: list[Any], name: str, label: str, min_value: int = 1) -> int:
    if len(args) != 1 or isinstance(args[0], pd.Series):
        raise ValueError(f"{name} requires a numeric {label} argument")
    raw = args[0]
    value = int(raw)
    if float(raw) != float(value):
        raise ValueError(f"{name} {label} must be an integer")
    if value < min_value:
        raise ValueError(f"{name} {label} must be >= {min_value}")
    return value


def _series_arg(value: Any, name: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise ValueError(f"{name} requires a series argument")
    return value
