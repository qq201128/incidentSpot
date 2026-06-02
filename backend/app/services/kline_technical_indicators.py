from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.kline_extended_indicators import add_extended_indicator_features

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
VORTEX_PERIOD = 14
MASS_EMA_PERIOD = 9
MASS_SUM_PERIOD = 25
ULCER_PERIOD = 14
CMO_PERIOD = 14
COPPOCK_WMA_PERIOD = 10
COPPOCK_FAST_ROC = 11
COPPOCK_SLOW_ROC = 14
KST_ROC_PERIODS = (10, 15, 20, 30)
KST_SMOOTH_PERIODS = (10, 10, 10, 15)
KST_SIGNAL_PERIOD = 9
FORCE_EMA_PERIOD = 13
EMV_PERIOD = 14
EMV_VOLUME_SCALE = 100_000_000.0
CHAIKIN_FAST_PERIOD = 3
CHAIKIN_SLOW_PERIOD = 10
ADL_SLOPE_PERIOD = 20
PVT_SLOPE_PERIOD = 20
PVO_FAST_PERIOD = 12
PVO_SLOW_PERIOD = 26
VWMA_PERIOD = 20
BOP_PERIOD = 14


def add_technical_indicator_features(d: pd.DataFrame) -> pd.DataFrame:
    features = {}
    features.update(_trend_features(d))
    features.update(_momentum_features(d))
    features.update(_volume_flow_features(d))
    features.update(_structure_features(d))
    return add_extended_indicator_features(pd.concat([d, pd.DataFrame(features, index=d.index)], axis=1))


def _trend_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    atr = _atr(d, KELTNER_ATR_PERIOD)
    ema = d["close"].ewm(span=KELTNER_EMA_PERIOD, adjust=False).mean()
    aroon_up, aroon_down = _aroon(d)
    plus_di, minus_di = _directional_indicators(d)
    vortex_pos, vortex_neg = _vortex(d)
    return {
        "aroon_up_25": aroon_up,
        "aroon_down_25": aroon_down,
        "aroon_osc_25": aroon_up - aroon_down,
        "dmi_spread_14": plus_di - minus_di,
        "keltner_width_20": (4.0 * atr) / ema.replace(0, EPSILON),
        "keltner_pos_20": (d["close"] - ema) / (2.0 * atr.replace(0, EPSILON)),
        "vortex_pos_14": vortex_pos,
        "vortex_neg_14": vortex_neg,
        "vortex_spread_14": vortex_pos - vortex_neg,
        "mass_index_25": _mass_index(d),
        "ulcer_index_14": _ulcer_index(d["close"]),
    }


def _momentum_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    kst, kst_signal = _kst(d["close"])
    return {
        "trix_15": _trix(d["close"]),
        "tsi_25_13": _tsi(d["close"]),
        "ultimate_osc_7_14_28": _ultimate_oscillator(d),
        "cmo_14": _cmo(d["close"]),
        "coppock_10_14_11": _coppock(d["close"]),
        "kst_10_15_20_30": kst,
        "kst_signal_9": kst_signal,
        "kst_diff": kst - kst_signal,
    }


def _volume_flow_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    adl = _accumulation_distribution_line(d)
    pvt = _price_volume_trend(d)
    volume_base = d["volume"].rolling(ADL_SLOPE_PERIOD).sum().replace(0, EPSILON)
    return {
        "force_index_13": (d["close"].diff() * d["volume"]).ewm(span=FORCE_EMA_PERIOD, adjust=False).mean(),
        "emv_14": _ease_of_movement(d),
        "chaikin_osc_3_10": _chaikin_oscillator(adl),
        "adl_slope_20": adl.diff(ADL_SLOPE_PERIOD) / volume_base,
        "pvt_slope_20": pvt.diff(PVT_SLOPE_PERIOD) / d["volume"].rolling(PVT_SLOPE_PERIOD).sum().replace(0, EPSILON),
        "pvo_12_26": _percentage_volume_oscillator(d["volume"]),
        "vwma_ratio_20": _vwma_ratio(d),
    }


def _structure_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    price_range = (d["high"] - d["low"]).replace(0, EPSILON)
    return {
        "bop_14": ((d["close"] - d["open"]) / price_range).rolling(BOP_PERIOD).mean(),
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


def _vortex(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    positive_vm = (d["high"] - d["low"].shift(1)).abs()
    negative_vm = (d["low"] - d["high"].shift(1)).abs()
    true_range_sum = _true_range(d).rolling(VORTEX_PERIOD).sum().replace(0, EPSILON)
    positive = positive_vm.rolling(VORTEX_PERIOD).sum() / true_range_sum
    negative = negative_vm.rolling(VORTEX_PERIOD).sum() / true_range_sum
    return positive, negative


def _mass_index(d: pd.DataFrame) -> pd.Series:
    high_low = d["high"] - d["low"]
    single_ema = high_low.ewm(span=MASS_EMA_PERIOD, adjust=False).mean()
    double_ema = single_ema.ewm(span=MASS_EMA_PERIOD, adjust=False).mean()
    return (single_ema / double_ema.replace(0, EPSILON)).rolling(MASS_SUM_PERIOD).sum()


def _ulcer_index(close: pd.Series) -> pd.Series:
    rolling_high = close.rolling(ULCER_PERIOD).max().replace(0, EPSILON)
    drawdown = 100.0 * (close - rolling_high) / rolling_high
    return np.sqrt(drawdown.pow(2).rolling(ULCER_PERIOD).mean())


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


def _cmo(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(CMO_PERIOD).sum()
    losses = (-delta.clip(upper=0.0)).rolling(CMO_PERIOD).sum()
    return 100.0 * (gains - losses) / (gains + losses).replace(0, EPSILON)


def _coppock(close: pd.Series) -> pd.Series:
    roc_sum = _rate_of_change(close, COPPOCK_FAST_ROC) + _rate_of_change(close, COPPOCK_SLOW_ROC)
    return _weighted_moving_average(roc_sum, COPPOCK_WMA_PERIOD)


def _kst(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    components = [
        _rate_of_change(close, roc_period).rolling(smooth_period).mean() * weight
        for weight, (roc_period, smooth_period) in enumerate(zip(KST_ROC_PERIODS, KST_SMOOTH_PERIODS), start=1)
    ]
    kst = sum(components)
    return kst, kst.rolling(KST_SIGNAL_PERIOD).mean()


def _ease_of_movement(d: pd.DataFrame) -> pd.Series:
    midpoint = (d["high"] + d["low"]) / 2.0
    box_ratio = (d["volume"] / EMV_VOLUME_SCALE) / (d["high"] - d["low"]).replace(0, EPSILON)
    one_period = midpoint.diff() / box_ratio.replace(0, EPSILON)
    return one_period.rolling(EMV_PERIOD).mean()


def _chaikin_oscillator(adl: pd.Series) -> pd.Series:
    fast = adl.ewm(span=CHAIKIN_FAST_PERIOD, adjust=False).mean()
    slow = adl.ewm(span=CHAIKIN_SLOW_PERIOD, adjust=False).mean()
    return fast - slow


def _accumulation_distribution_line(d: pd.DataFrame) -> pd.Series:
    price_range = (d["high"] - d["low"]).replace(0, EPSILON)
    multiplier = ((d["close"] - d["low"]) - (d["high"] - d["close"])) / price_range
    return (multiplier * d["volume"]).cumsum()


def _price_volume_trend(d: pd.DataFrame) -> pd.Series:
    return (d["close"].pct_change().fillna(0.0) * d["volume"]).cumsum()


def _percentage_volume_oscillator(volume: pd.Series) -> pd.Series:
    fast = volume.ewm(span=PVO_FAST_PERIOD, adjust=False).mean()
    slow = volume.ewm(span=PVO_SLOW_PERIOD, adjust=False).mean().replace(0, EPSILON)
    return fast / slow - 1.0


def _vwma_ratio(d: pd.DataFrame) -> pd.Series:
    traded_value = (d["close"] * d["volume"]).rolling(VWMA_PERIOD).sum()
    traded_volume = d["volume"].rolling(VWMA_PERIOD).sum().replace(0, EPSILON)
    vwma = traded_value / traded_volume
    return d["close"] / vwma.replace(0, EPSILON) - 1.0


def _atr(d: pd.DataFrame, period: int) -> pd.Series:
    return _true_range(d).ewm(span=period, adjust=False).mean().fillna(0.0)


def _double_ema(series: pd.Series, slow: int, fast: int) -> pd.Series:
    return series.ewm(span=slow, adjust=False).mean().ewm(span=fast, adjust=False).mean()


def _average_ratio(numerator: pd.Series, denominator: pd.Series, period: int) -> pd.Series:
    top = numerator.rolling(period).sum()
    bottom = denominator.rolling(period).sum().replace(0, EPSILON)
    return top / bottom


def _true_range(d: pd.DataFrame) -> pd.Series:
    prev_close = d["close"].shift(1)
    true_ranges = pd.concat(
        [(d["high"] - d["low"]).abs(), (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()],
        axis=1,
    )
    return true_ranges.max(axis=1)


def _rate_of_change(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period) * 100.0


def _weighted_moving_average(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype="float64")
    return series.rolling(period).apply(lambda values: np.dot(values, weights) / weights.sum(), raw=True)


def _periods_since_high(values: np.ndarray) -> float:
    return float((len(values) - 1) - int(np.argmax(values)))


def _periods_since_low(values: np.ndarray) -> float:
    return float((len(values) - 1) - int(np.argmin(values)))
