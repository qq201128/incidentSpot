from __future__ import annotations

from dataclasses import dataclass

BASE_WINDOWS = (3, 5, 10, 20, 60, 120, 240)
VOLUME_WINDOWS = (20, 60, 120, 240)
VOLUME_MEDIAN_WINDOWS = (10, 20, 60)
BOLLINGER_WINDOW = 20
ADX_PERIOD = 14
CHOP_PERIOD = 14
EPSILON = 1e-12

FEATURE_COLUMNS = [
    "ret_1", "hl_range", "oc_body", "ret_3", "ret_5", "ret_10", "ret_20",
    "ret_60", "ret_120", "ret_240", "vol_std_3", "vol_std_5",
    "vol_std_10", "vol_std_20", "vol_std_60", "vol_std_120", "vol_std_240",
    "ma_ratio_3", "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "ma_ratio_60", "ma_ratio_120", "ma_ratio_240", "vol_chg",
    "vol_ratio_20", "vol_ratio_60", "vol_ratio_120", "vol_ratio_240",
    "rsi_14", "rsi_14_chg_3", "macd", "macd_signal", "macd_hist",
    "macd_hist_chg_3", "atr_14", "atr_ratio", "mom_10_norm", "mom_20_norm",
    "ema_cross", "vol_median_ratio_10", "vol_median_ratio_20",
    "vol_median_ratio_60", "ret_10_60", "upper_shadow", "lower_shadow",
    "wick_imbalance", "bb_width_20", "bb_z_20", "donchian_pos_20",
    "donchian_pos_60", "vwap_dev_20", "vwap_dev_60", "efficiency_ratio_10",
    "efficiency_ratio_20", "adx_14", "chop_14",
]


@dataclass(frozen=True)
class FeatureSpec:
    columns: list[str]


def build_feature_frame(df, min_history: int = 240) -> tuple[object, FeatureSpec]:
    """
    Build shifted feature matrix from 1m OHLCV candles.

    Expects columns: open_time, open, high, low, close, volume (pandas DataFrame).
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("missing dependency: pandas/numpy (pip install -r requirements.txt)") from exc

    d = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    if d.empty:
        raise ValueError("no valid OHLCV rows")

    _add_return_and_range_features(d, np)
    _add_volume_features(d)
    _add_momentum_features(d)
    _add_volatility_features(d, np)
    _add_structure_features(d, np)

    d[FEATURE_COLUMNS] = d[FEATURE_COLUMNS].shift(1)
    out = d.dropna().reset_index(drop=True)
    if len(out) < min_history:
        raise ValueError("insufficient candles for feature generation")

    return out, FeatureSpec(columns=FEATURE_COLUMNS)


def _add_return_and_range_features(d, np) -> None:
    d["ret_1"] = d["close"].pct_change(1)
    d["hl_range"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["oc_body"] = (d["close"] - d["open"]) / d["close"].replace(0, np.nan)
    for window in BASE_WINDOWS:
        d[f"ret_{window}"] = d["close"].pct_change(window)
        d[f"vol_std_{window}"] = d["ret_1"].rolling(window).std()
        d[f"ma_ratio_{window}"] = d["close"] / d["close"].rolling(window).mean() - 1


def _add_volume_features(d) -> None:
    d["vol_chg"] = d["volume"].pct_change(1)
    for window in VOLUME_WINDOWS:
        d[f"vol_ma_{window}"] = d["volume"].rolling(window).mean()
        d[f"vol_ratio_{window}"] = d["volume"] / d[f"vol_ma_{window}"]
    for window in VOLUME_MEDIAN_WINDOWS:
        d[f"vol_median_ratio_{window}"] = d["volume"] / d["volume"].rolling(window).median()


def _add_momentum_features(d) -> None:
    d["rsi_14"] = _rsi(d["close"], 14)
    d["rsi_14_chg_3"] = d["rsi_14"].diff(3)
    macd_line, macd_signal, macd_hist = _macd(d["close"])
    d["macd"] = macd_line
    d["macd_signal"] = macd_signal
    d["macd_hist"] = macd_hist
    d["macd_hist_chg_3"] = d["macd_hist"].diff(3)
    d["mom_10_norm"] = (d["close"] - d["close"].shift(10)) / d["close"]
    d["mom_20_norm"] = (d["close"] - d["close"].shift(20)) / d["close"]
    d["ema_cross"] = _ema_cross(d["close"])
    d["ret_10_60"] = d["close"].pct_change(10) - d["close"].pct_change(60)
    d["efficiency_ratio_10"] = _efficiency_ratio(d["close"], 10)
    d["efficiency_ratio_20"] = _efficiency_ratio(d["close"], 20)


def _add_volatility_features(d, np) -> None:
    atr_abs = _atr(d["high"], d["low"], d["close"], ADX_PERIOD)
    d["atr_14"] = atr_abs / d["close"].replace(0, np.nan)
    d["atr_ratio"] = d["atr_14"] / d["atr_14"].rolling(20).mean()
    d["bb_width_20"], d["bb_z_20"] = _bollinger_features(d["close"])
    d["adx_14"] = _adx(d["high"], d["low"], d["close"], ADX_PERIOD)
    d["chop_14"] = _choppiness(d["high"], d["low"], atr_abs, np)


def _add_structure_features(d, np) -> None:
    d["upper_shadow"] = (d["high"] - d[["close", "open"]].max(axis=1)) / d["close"]
    d["lower_shadow"] = (d[["close", "open"]].min(axis=1) - d["low"]) / d["close"]
    d["wick_imbalance"] = d["lower_shadow"] - d["upper_shadow"]
    d["donchian_pos_20"] = _donchian_position(d["close"], d["high"], d["low"], 20)
    d["donchian_pos_60"] = _donchian_position(d["close"], d["high"], d["low"], 60)
    d["vwap_dev_20"] = _vwap_deviation(d["close"], d["high"], d["low"], d["volume"], 20)
    d["vwap_dev_60"] = _vwap_deviation(d["close"], d["high"], d["low"], d["volume"], 60)


def _rsi(close, period: int):
    import numpy as np
    import pandas as pd

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    import pandas as pd

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _ema_cross(close):
    return (close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()) / close


def _efficiency_ratio(close, period: int):
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return direction / volatility.replace(0, EPSILON)


def _bollinger_features(close):
    mean = close.rolling(BOLLINGER_WINDOW).mean()
    std = close.rolling(BOLLINGER_WINDOW).std()
    width = (std * 4) / mean.replace(0, EPSILON)
    z_score = (close - mean) / std.replace(0, EPSILON)
    return width, z_score


def _donchian_position(close, high, low, period: int):
    rolling_high = high.rolling(period).max()
    rolling_low = low.rolling(period).min()
    channel = (rolling_high - rolling_low).replace(0, EPSILON)
    return (close - rolling_low) / channel


def _vwap_deviation(close, high, low, volume, period: int):
    typical_price = (high + low + close) / 3
    traded_value = (typical_price * volume).rolling(period).sum()
    traded_volume = volume.rolling(period).sum().replace(0, EPSILON)
    vwap = traded_value / traded_volume
    return close / vwap.replace(0, EPSILON) - 1


def _adx(high, low, close, period: int):
    plus_dm = (high.diff()).where(lambda value: value > (-low.diff()), 0.0).clip(lower=0.0)
    minus_dm = (-low.diff()).where(lambda value: value > high.diff(), 0.0).clip(lower=0.0)
    atr = _atr(high, low, close, period).replace(0, EPSILON)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, EPSILON)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _choppiness(high, low, atr_abs, np):
    atr_sum = atr_abs.rolling(CHOP_PERIOD).sum()
    price_range = (high.rolling(CHOP_PERIOD).max() - low.rolling(CHOP_PERIOD).min()).replace(0, EPSILON)
    return 100 * np.log10(atr_sum / price_range) / np.log10(CHOP_PERIOD)


def _atr(high, low, close, period: int):
    import numpy as np
    import pandas as pd

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().fillna(0.0)
