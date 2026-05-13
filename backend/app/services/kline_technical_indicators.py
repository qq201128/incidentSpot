from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-12
AROON_PERIOD = 25
KELTNER_ATR_PERIOD = 10
KELTNER_EMA_PERIOD = 20
TRIX_PERIOD = 15
TSI_FAST_PERIOD = 13
TSI_SLOW_PERIOD = 25
ULTIMATE_FAST_PERIOD = 7
ULTIMATE_MID_PERIOD = 14
ULTIMATE_SLOW_PERIOD = 28


def add_technical_indicator_features(d: pd.DataFrame) -> pd.DataFrame:
    features = {}
    features.update(_trend_features(d))
    features.update(_momentum_features(d))
    return pd.concat([d, pd.DataFrame(features, index=d.index)], axis=1)


def _trend_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    atr = _atr(d, KELTNER_ATR_PERIOD)
    ema = d["close"].ewm(span=KELTNER_EMA_PERIOD, adjust=False).mean()
    aroon_up, aroon_down = _aroon(d)
    plus_di, minus_di = _directional_indicators(d)
    return {
        "aroon_up_25": aroon_up,
        "aroon_down_25": aroon_down,
        "aroon_osc_25": aroon_up - aroon_down,
        "dmi_spread_14": plus_di - minus_di,
        "keltner_width_20": (4.0 * atr) / ema.replace(0, EPSILON),
        "keltner_pos_20": (d["close"] - ema) / (2.0 * atr.replace(0, EPSILON)),
    }


def _momentum_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "trix_15": _trix(d["close"]),
        "tsi_25_13": _tsi(d["close"]),
        "ultimate_osc_7_14_28": _ultimate_oscillator(d),
    }


def _aroon(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    high_age = d["high"].rolling(AROON_PERIOD).apply(_periods_since_high, raw=True)
    low_age = d["low"].rolling(AROON_PERIOD).apply(_periods_since_low, raw=True)
    scale = 100.0 / AROON_PERIOD
    return (AROON_PERIOD - high_age) * scale, (AROON_PERIOD - low_age) * scale


def _directional_indicators(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    plus_dm = d["high"].diff().where(lambda value: value > (-d["low"].diff()), 0.0).clip(lower=0.0)
    minus_dm = (-d["low"].diff()).where(lambda value: value > d["high"].diff(), 0.0).clip(lower=0.0)
    atr = _atr(d, ULTIMATE_MID_PERIOD).replace(0, EPSILON)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / ULTIMATE_MID_PERIOD, adjust=False).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / ULTIMATE_MID_PERIOD, adjust=False).mean() / atr
    return plus_di, minus_di


def _trix(close: pd.Series) -> pd.Series:
    ema1 = close.ewm(span=TRIX_PERIOD, adjust=False).mean()
    ema2 = ema1.ewm(span=TRIX_PERIOD, adjust=False).mean()
    ema3 = ema2.ewm(span=TRIX_PERIOD, adjust=False).mean()
    return ema3.pct_change()


def _tsi(close: pd.Series) -> pd.Series:
    momentum = close.diff()
    smooth = _double_ema(momentum, TSI_SLOW_PERIOD, TSI_FAST_PERIOD)
    abs_smooth = _double_ema(momentum.abs(), TSI_SLOW_PERIOD, TSI_FAST_PERIOD)
    return 100.0 * smooth / abs_smooth.replace(0, EPSILON)


def _ultimate_oscillator(d: pd.DataFrame) -> pd.Series:
    prev_close = d["close"].shift(1)
    low_or_close = pd.concat([d["low"], prev_close], axis=1).min(axis=1)
    high_or_close = pd.concat([d["high"], prev_close], axis=1).max(axis=1)
    buying_pressure = d["close"] - low_or_close
    true_range = (high_or_close - low_or_close).replace(0, EPSILON)
    fast = _average_ratio(buying_pressure, true_range, ULTIMATE_FAST_PERIOD)
    mid = _average_ratio(buying_pressure, true_range, ULTIMATE_MID_PERIOD)
    slow = _average_ratio(buying_pressure, true_range, ULTIMATE_SLOW_PERIOD)
    return 100.0 * ((4.0 * fast) + (2.0 * mid) + slow) / 7.0


def _atr(d: pd.DataFrame, period: int) -> pd.Series:
    prev_close = d["close"].shift(1)
    true_ranges = pd.concat(
        [(d["high"] - d["low"]).abs(), (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()],
        axis=1,
    )
    return true_ranges.max(axis=1).ewm(span=period, adjust=False).mean().fillna(0.0)


def _double_ema(series: pd.Series, slow: int, fast: int) -> pd.Series:
    return series.ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()


def _average_ratio(numerator: pd.Series, denominator: pd.Series, period: int) -> pd.Series:
    top = numerator.rolling(period).sum()
    bottom = denominator.rolling(period).sum().replace(0, EPSILON)
    return top / bottom


def _periods_since_high(values: np.ndarray) -> float:
    return float((len(values) - 1) - int(np.argmax(values)))


def _periods_since_low(values: np.ndarray) -> float:
    return float((len(values) - 1) - int(np.argmin(values)))
