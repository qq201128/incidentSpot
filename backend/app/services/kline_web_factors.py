from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from app.services.kline_web_factor_specs import WEB_FACTOR_SPECS, WebFactorSpec

EPSILON = 1e-12
SLOPE_PERIOD = 3


def add_web_factor_features(d: pd.DataFrame) -> pd.DataFrame:
    features = {spec.name: _compute_feature(d, spec) for spec in WEB_FACTOR_SPECS}
    return pd.concat([d, pd.DataFrame(features, index=d.index)], axis=1)


def _compute_feature(d: pd.DataFrame, spec: WebFactorSpec) -> pd.Series:
    handlers: dict[str, Callable[[pd.DataFrame, int], pd.Series]] = {
        "ret_ratio": _ret_ratio,
        "log_ret": _log_ret,
        "ret_z": _ret_z,
        "ret_rank": _ret_rank,
        "realized_vol": _realized_vol,
        "parkinson_vol": _parkinson_vol,
        "garman_klass_vol": _garman_klass_vol,
        "atr_norm": _atr_norm,
        "range_z": _range_z,
        "close_pos": _close_pos,
        "sma_ratio": _sma_ratio,
        "ema_ratio": _ema_ratio,
        "sma_slope": _sma_slope,
        "roc_smooth": _roc_smooth,
        "volume_z": _volume_z,
        "volume_ratio": _volume_ratio,
        "dollar_volume_z": _dollar_volume_z,
        "obv_slope": _obv_slope,
        "price_volume_corr": _price_volume_corr,
        "vwap_dev": _vwap_dev,
    }
    return handlers[spec.kind](d, spec.window)


def _ret_ratio(d: pd.DataFrame, window: int) -> pd.Series:
    return d["close"].pct_change(window)


def _log_ret(d: pd.DataFrame, window: int) -> pd.Series:
    return np.log(d["close"] / d["close"].shift(window))


def _ret_z(d: pd.DataFrame, window: int) -> pd.Series:
    return _zscore(d["ret_1"], window)


def _ret_rank(d: pd.DataFrame, window: int) -> pd.Series:
    return _ts_rank(d["ret_1"], window)


def _realized_vol(d: pd.DataFrame, window: int) -> pd.Series:
    return d["ret_1"].rolling(window).std()


def _parkinson_vol(d: pd.DataFrame, window: int) -> pd.Series:
    squared_range = np.log(d["high"] / d["low"].replace(0, EPSILON)).pow(2)
    return np.sqrt(squared_range.rolling(window).mean() / (4.0 * np.log(2.0)))


def _garman_klass_vol(d: pd.DataFrame, window: int) -> pd.Series:
    log_hl = np.log(d["high"] / d["low"].replace(0, EPSILON))
    log_co = np.log(d["close"] / d["open"].replace(0, EPSILON))
    variance = (0.5 * log_hl.pow(2)) - ((2.0 * np.log(2.0) - 1.0) * log_co.pow(2))
    return np.sqrt(variance.clip(lower=0.0).rolling(window).mean())


def _atr_norm(d: pd.DataFrame, window: int) -> pd.Series:
    return _true_range(d).rolling(window).mean() / d["close"].replace(0, EPSILON)


def _range_z(d: pd.DataFrame, window: int) -> pd.Series:
    return _zscore((d["high"] - d["low"]) / d["close"].replace(0, EPSILON), window)


def _close_pos(d: pd.DataFrame, window: int) -> pd.Series:
    low = d["low"].rolling(window).min()
    high = d["high"].rolling(window).max()
    return (d["close"] - low) / (high - low).replace(0, EPSILON)


def _sma_ratio(d: pd.DataFrame, window: int) -> pd.Series:
    return d["close"] / d["close"].rolling(window).mean().replace(0, EPSILON) - 1.0


def _ema_ratio(d: pd.DataFrame, window: int) -> pd.Series:
    return d["close"] / d["close"].ewm(span=window, adjust=False).mean().replace(0, EPSILON) - 1.0


def _sma_slope(d: pd.DataFrame, window: int) -> pd.Series:
    return d["close"].rolling(window).mean().pct_change(SLOPE_PERIOD)


def _roc_smooth(d: pd.DataFrame, window: int) -> pd.Series:
    return d["close"].pct_change(window).rolling(window).mean()


def _volume_z(d: pd.DataFrame, window: int) -> pd.Series:
    return _zscore(d["volume"], window)


def _volume_ratio(d: pd.DataFrame, window: int) -> pd.Series:
    return d["volume"] / d["volume"].rolling(window).mean().replace(0, EPSILON)


def _dollar_volume_z(d: pd.DataFrame, window: int) -> pd.Series:
    return _zscore(d["close"] * d["volume"], window)


def _obv_slope(d: pd.DataFrame, window: int) -> pd.Series:
    obv = (np.sign(d["close"].diff()).fillna(0.0) * d["volume"]).cumsum()
    return obv.diff(window) / d["volume"].rolling(window).sum().replace(0, EPSILON)


def _price_volume_corr(d: pd.DataFrame, window: int) -> pd.Series:
    return d["ret_1"].rolling(window).corr(d["volume"].pct_change())


def _vwap_dev(d: pd.DataFrame, window: int) -> pd.Series:
    typical = (d["high"] + d["low"] + d["close"]) / 3.0
    vwap = (typical * d["volume"]).rolling(window).sum() / d["volume"].rolling(window).sum().replace(0, EPSILON)
    return d["close"] / vwap.replace(0, EPSILON) - 1.0


def _true_range(d: pd.DataFrame) -> pd.Series:
    prev_close = d["close"].shift(1)
    ranges = pd.concat(
        [(d["high"] - d["low"]).abs(), (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, EPSILON)


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(_last_rank_pct, raw=True)


def _last_rank_pct(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    order = np.argsort(np.argsort(values))
    return float(order[-1] / (len(values) - 1))
