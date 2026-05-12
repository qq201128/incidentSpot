from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-12
ROLLING_STATS_WINDOW = 20
ROLLING_PERFORMANCE_WINDOW = 60


def add_advanced_kline_features(d):
    features = {}
    features.update(_smc_features(d))
    features.update(_statistic_features(d))
    features.update(_performance_features(d))
    return pd.concat([d, pd.DataFrame(features, index=d.index)], axis=1)


def _smc_features(d) -> dict[str, pd.Series]:
    prior_high = d["high"].shift(1).rolling(20).max()
    prior_low = d["low"].shift(1).rolling(20).min()
    return {
        "fvg_up_3": (d["low"] > d["high"].shift(2)).astype(float),
        "fvg_down_3": (d["high"] < d["low"].shift(2)).astype(float),
        "liquidity_sweep_high_20": ((d["high"] > prior_high) & (d["close"] < prior_high)).astype(float),
        "liquidity_sweep_low_20": ((d["low"] < prior_low) & (d["close"] > prior_low)).astype(float),
        "breakout_high_20": (d["close"] > prior_high).astype(float),
        "breakdown_low_20": (d["close"] < prior_low).astype(float),
    }


def _statistic_features(d) -> dict[str, pd.Series]:
    return {
        "ret_autocorr_20": d["ret_1"].rolling(ROLLING_STATS_WINDOW).corr(d["ret_1"].shift(1)),
        "price_volume_corr_20": d["ret_1"].rolling(ROLLING_STATS_WINDOW).corr(d["vol_chg"]),
        "vol_ret_corr_20": d["vol_chg"].rolling(ROLLING_STATS_WINDOW).corr(d["ret_1"]),
    }


def _performance_features(d) -> dict[str, pd.Series]:
    mean_ret = d["ret_1"].rolling(ROLLING_PERFORMANCE_WINDOW).mean()
    std_ret = d["ret_1"].rolling(ROLLING_PERFORMANCE_WINDOW).std().replace(0, np.nan)
    positive = d["ret_1"].clip(lower=0).rolling(ROLLING_PERFORMANCE_WINDOW).sum()
    negative = d["ret_1"].clip(upper=0).abs().rolling(ROLLING_PERFORMANCE_WINDOW).sum()
    return {
        "rolling_sharpe_60": mean_ret / std_ret * np.sqrt(ROLLING_PERFORMANCE_WINDOW),
        "win_rate_60": (d["ret_1"] > 0).astype(float).rolling(ROLLING_PERFORMANCE_WINDOW).mean(),
        "profit_factor_60": positive / negative.replace(0, EPSILON),
    }
