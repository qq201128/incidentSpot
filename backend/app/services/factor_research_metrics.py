from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from numpy.lib.stride_tricks import sliding_window_view

QUINTILE_COUNT = 5
IC_ROLLING_WINDOW = 20


def ic_metrics(ic_series: pd.Series) -> dict[str, float | None]:
    ic_mean = float(ic_series.mean()) if not ic_series.empty else None
    ic_std = float(ic_series.std()) if not ic_series.empty else None
    ir = ic_mean / ic_std if ic_mean is not None and ic_std and ic_std > 0 else None
    positive_rate = float((ic_series > 0).mean()) if not ic_series.empty else None
    return {"mean": ic_mean, "std": ic_std, "ir": ir, "positive_rate": positive_rate}


def compute_rolling_ic(factor: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    values = pd.DataFrame({"factor": factor, "fwd_ret": fwd_ret}).replace([np.inf, -np.inf], np.nan)
    values = values.dropna()
    if len(values) < IC_ROLLING_WINDOW:
        return pd.Series(dtype=float)

    factor_windows = sliding_window_view(values["factor"].to_numpy(dtype=float), IC_ROLLING_WINDOW)
    return_windows = sliding_window_view(values["fwd_ret"].to_numpy(dtype=float), IC_ROLLING_WINDOW)
    factor_ranks = stats.rankdata(factor_windows, axis=1, method="average")
    return_ranks = stats.rankdata(return_windows, axis=1, method="average")
    correlations = _rowwise_correlation(factor_ranks, return_ranks)
    index = values.index[IC_ROLLING_WINDOW - 1 :]
    return pd.Series(correlations, index=index).replace([np.inf, -np.inf], np.nan).dropna()


def _rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def window_spearman_ic(window: pd.DataFrame) -> float | None:
    if len(window) < IC_ROLLING_WINDOW:
        return None
    if window["factor"].nunique(dropna=True) < 2 or window["fwd_ret"].nunique(dropna=True) < 2:
        return None
    corr = window["factor"].rank(method="average").corr(window["fwd_ret"].rank(method="average"))
    return float(corr) if corr is not None and math.isfinite(float(corr)) else None


def compute_ic_ttest(ic_series: pd.Series) -> tuple[float | None, float | None]:
    clean = ic_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None, None
    t_stat, p_value = stats.ttest_1samp(clean, 0)
    return float(t_stat), float(p_value)


def compute_quintile_returns(df: pd.DataFrame, factor_name: str) -> list[float]:
    try:
        data = df.copy()
        data["quintile"] = pd.qcut(data[factor_name], QUINTILE_COUNT, labels=False, duplicates="drop")
        returns = data.groupby("quintile")["fwd_ret"].mean()
        return [float(returns.get(i, 0)) for i in range(QUINTILE_COUNT)]
    except ValueError:
        return []


def compute_turnover(df: pd.DataFrame, factor_name: str) -> float | None:
    try:
        data = df.copy()
        data["quintile"] = pd.qcut(data[factor_name], QUINTILE_COUNT, labels=False, duplicates="drop")
        changes = (data["quintile"] != data["quintile"].shift(1)).sum()
        return float(changes / len(data)) if len(data) > 0 else None
    except ValueError:
        return None
