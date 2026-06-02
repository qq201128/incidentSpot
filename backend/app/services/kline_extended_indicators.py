from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.kline_web_factors import add_web_factor_features

EPSILON = 1e-12
BB_WINDOW = 20
DPO_PERIOD = 20
PMO_ROC_SMOOTH = 35
PMO_LINE_SMOOTH = 20
PMO_SIGNAL_PERIOD = 10
STOCH_RSI_PERIOD = 14
STOCH_RSI_SIGNAL = 3
AWESOME_FAST = 5
AWESOME_SLOW = 34
FISHER_PERIOD = 10
QSTICK_FAST = 10
QSTICK_SLOW = 20
ELDER_EMA_PERIOD = 13
DONCHIAN_FAST = 20
DONCHIAN_SLOW = 55
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_INDEX_PERIOD = 20
KLINGER_FAST = 34
KLINGER_SLOW = 55
KLINGER_SIGNAL = 13
RVI_PERIOD = 10
RVI_SIGNAL = 4
ZSCORE_WINDOW = 20
MFI_FAST = 7
MFI_SLOW = 21
SMA_FAST = 50
SMA_SLOW = 200


def add_extended_indicator_features(d: pd.DataFrame) -> pd.DataFrame:
    features = {}
    features.update(_momentum_features(d))
    features.update(_trend_structure_features(d))
    features.update(_volume_flow_features(d))
    features.update(_risk_features(d))
    return add_web_factor_features(pd.concat([d, pd.DataFrame(features, index=d.index)], axis=1))


def _momentum_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    pmo, pmo_signal = _pmo(d["close"])
    stoch_rsi = _stoch_rsi(d["rsi_14"])
    fisher = _fisher_transform(d)
    awesome = _awesome_oscillator(d)
    return {
        "bb_percent_b_20": _bollinger_percent_b(d["close"]),
        "dpo_20": _dpo(d["close"], DPO_PERIOD),
        "pmo_35_20": pmo,
        "pmo_signal_10": pmo_signal,
        "pmo_diff": pmo - pmo_signal,
        "stoch_rsi_14": stoch_rsi,
        "stoch_rsi_signal_3": stoch_rsi.rolling(STOCH_RSI_SIGNAL).mean(),
        "rsi_14_sma_5": d["rsi_14"].rolling(5).mean(),
        "rsi_14_slope_3": d["rsi_14"].diff(3),
        "awesome_osc_5_34": awesome,
        "accelerator_osc": awesome - awesome.rolling(AWESOME_FAST).mean(),
        "fisher_10": fisher,
        "fisher_signal_1": fisher.shift(1),
    }


def _trend_structure_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    ichimoku = _ichimoku(d)
    donchian = _donchian(d)
    qstick_fast = _qstick(d, QSTICK_FAST)
    qstick_slow = _qstick(d, QSTICK_SLOW)
    ema_elder = d["close"].ewm(span=ELDER_EMA_PERIOD, adjust=False).mean()
    sma_fast = d["close"].rolling(SMA_FAST).mean()
    sma_slow = d["close"].rolling(SMA_SLOW).mean()
    return {
        **ichimoku,
        **donchian,
        "qstick_10": qstick_fast,
        "qstick_20": qstick_slow,
        "qstick_spread_10_20": qstick_fast - qstick_slow,
        "elder_bull_power_13": d["high"] - ema_elder,
        "elder_bear_power_13": d["low"] - ema_elder,
        "elder_ray_spread_13": d["high"] - d["low"],
        "sma_50_200_spread": sma_fast / sma_slow.replace(0, EPSILON) - 1.0,
        "close_sma_50_ratio": d["close"] / sma_fast.replace(0, EPSILON) - 1.0,
    }


def _volume_flow_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    nvi = _negative_volume_index(d)
    pvi = _positive_volume_index(d)
    klinger, klinger_signal = _klinger(d)
    rvi, rvi_signal = _relative_vigor(d)
    return {
        "nvi_slope_20": nvi.pct_change(VOLUME_INDEX_PERIOD),
        "pvi_slope_20": pvi.pct_change(VOLUME_INDEX_PERIOD),
        "nvi_pvi_spread_20": nvi.pct_change(VOLUME_INDEX_PERIOD) - pvi.pct_change(VOLUME_INDEX_PERIOD),
        "klinger_osc_34_55": klinger,
        "klinger_signal_13": klinger_signal,
        "klinger_diff": klinger - klinger_signal,
        "relative_vigor_10": rvi,
        "relative_vigor_signal_4": rvi_signal,
    }


def _risk_features(d: pd.DataFrame) -> dict[str, pd.Series]:
    mfi_fast = _mfi(d, MFI_FAST)
    mfi_slow = _mfi(d, MFI_SLOW)
    return {
        "bb_percent_b_z_20": _zscore(_bollinger_percent_b(d["close"]), ZSCORE_WINDOW),
        "force_index_z_20": _zscore(d["force_index_13"], ZSCORE_WINDOW),
        "emv_z_20": _zscore(d["emv_14"], ZSCORE_WINDOW),
        "mfi_7": mfi_fast,
        "mfi_21": mfi_slow,
        "mfi_spread_7_21": mfi_fast - mfi_slow,
    }


def _bollinger_percent_b(close: pd.Series) -> pd.Series:
    mean = close.rolling(BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW).std()
    lower = mean - 2.0 * std
    upper = mean + 2.0 * std
    return (close - lower) / (upper - lower).replace(0, EPSILON)


def _dpo(close: pd.Series, period: int) -> pd.Series:
    offset = int(period / 2) + 1
    return close.shift(offset) - close.rolling(period).mean()


def _pmo(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    roc = ((close / close.shift(1)) * 100.0) - 100.0
    first = _custom_smooth(roc, PMO_ROC_SMOOTH)
    pmo = _custom_smooth(10.0 * first, PMO_LINE_SMOOTH)
    return pmo, pmo.ewm(span=PMO_SIGNAL_PERIOD, adjust=False).mean()


def _custom_smooth(series: pd.Series, period: int) -> pd.Series:
    alpha = 2.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def _stoch_rsi(rsi: pd.Series) -> pd.Series:
    low = rsi.rolling(STOCH_RSI_PERIOD).min()
    high = rsi.rolling(STOCH_RSI_PERIOD).max()
    return (rsi - low) / (high - low).replace(0, EPSILON)


def _awesome_oscillator(d: pd.DataFrame) -> pd.Series:
    median = (d["high"] + d["low"]) / 2.0
    return median.rolling(AWESOME_FAST).mean() - median.rolling(AWESOME_SLOW).mean()


def _fisher_transform(d: pd.DataFrame) -> pd.Series:
    low = d["low"].rolling(FISHER_PERIOD).min()
    high = d["high"].rolling(FISHER_PERIOD).max()
    value = (2.0 * ((d["close"] - low) / (high - low).replace(0, EPSILON))) - 1.0
    value = value.clip(lower=-0.999, upper=0.999)
    return 0.5 * np.log((1.0 + value) / (1.0 - value))


def _ichimoku(d: pd.DataFrame) -> dict[str, pd.Series]:
    tenkan = _midrange(d, TENKAN_PERIOD)
    kijun = _midrange(d, KIJUN_PERIOD)
    span_a = (tenkan + kijun) / 2.0
    span_b = _midrange(d, SENKOU_B_PERIOD)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
    return {
        "ichimoku_tenkan_9": tenkan / d["close"].replace(0, EPSILON) - 1.0,
        "ichimoku_kijun_26": kijun / d["close"].replace(0, EPSILON) - 1.0,
        "ichimoku_tenkan_kijun_spread": tenkan / kijun.replace(0, EPSILON) - 1.0,
        "ichimoku_cloud_pos": (d["close"] - cloud_bottom) / (cloud_top - cloud_bottom).replace(0, EPSILON),
        "ichimoku_cloud_width": (cloud_top - cloud_bottom) / d["close"].replace(0, EPSILON),
        "ichimoku_chikou_mom_26": d["close"].pct_change(KIJUN_PERIOD),
    }


def _donchian(d: pd.DataFrame) -> dict[str, pd.Series]:
    fast_high = d["high"].rolling(DONCHIAN_FAST).max()
    fast_low = d["low"].rolling(DONCHIAN_FAST).min()
    slow_high = d["high"].rolling(DONCHIAN_SLOW).max()
    slow_low = d["low"].rolling(DONCHIAN_SLOW).min()
    return {
        "donchian_width_20": (fast_high - fast_low) / d["close"].replace(0, EPSILON),
        "donchian_breakout_20": (d["close"] > fast_high.shift(1)).astype(float),
        "donchian_breakdown_20": (d["close"] < fast_low.shift(1)).astype(float),
        "donchian_width_55": (slow_high - slow_low) / d["close"].replace(0, EPSILON),
    }


def _qstick(d: pd.DataFrame, period: int) -> pd.Series:
    return (d["close"] - d["open"]).rolling(period).mean() / d["close"].replace(0, EPSILON)


def _negative_volume_index(d: pd.DataFrame) -> pd.Series:
    return (1.0 + d["close"].pct_change().where(d["volume"] < d["volume"].shift(1), 0.0)).cumprod()


def _positive_volume_index(d: pd.DataFrame) -> pd.Series:
    return (1.0 + d["close"].pct_change().where(d["volume"] > d["volume"].shift(1), 0.0)).cumprod()


def _klinger(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    trend = np.where((d["high"] + d["low"] + d["close"]) > (d["high"].shift(1) + d["low"].shift(1) + d["close"].shift(1)), 1.0, -1.0)
    dm = d["high"] - d["low"]
    cm = _klinger_cumulative_measure(pd.Series(trend, index=d.index), dm)
    vf = d["volume"] * (2.0 * ((dm / cm.replace(0, EPSILON)) - 1.0)) * trend * 100.0
    klinger = vf.ewm(span=KLINGER_FAST, adjust=False).mean() - vf.ewm(span=KLINGER_SLOW, adjust=False).mean()
    return klinger, klinger.ewm(span=KLINGER_SIGNAL, adjust=False).mean()


def _klinger_cumulative_measure(trend: pd.Series, dm: pd.Series) -> pd.Series:
    values = []
    previous = float(dm.iloc[0] or 0.0)
    for idx, current in enumerate(dm.fillna(0.0)):
        if idx == 0 or trend.iloc[idx] != trend.iloc[idx - 1]:
            previous = float(current + dm.iloc[idx - 1]) if idx > 0 else float(current)
        else:
            previous += float(current)
        values.append(previous)
    return pd.Series(values, index=dm.index)


def _relative_vigor(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    numerator = (d["close"] - d["open"]).rolling(RVI_PERIOD).mean()
    denominator = (d["high"] - d["low"]).rolling(RVI_PERIOD).mean().replace(0, EPSILON)
    rvi = numerator / denominator
    return rvi, rvi.rolling(RVI_SIGNAL).mean()


def _mfi(d: pd.DataFrame, period: int) -> pd.Series:
    typical = (d["high"] + d["low"] + d["close"]) / 3.0
    money_flow = typical * d["volume"]
    positive = money_flow.where(typical.diff() > 0, 0.0).rolling(period).sum()
    negative = money_flow.where(typical.diff() < 0, 0.0).rolling(period).sum().abs()
    return 100.0 - (100.0 / (1.0 + positive / negative.replace(0, EPSILON)))


def _midrange(d: pd.DataFrame, period: int) -> pd.Series:
    return (d["high"].rolling(period).max() + d["low"].rolling(period).min()) / 2.0


def _zscore(series: pd.Series, period: int) -> pd.Series:
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std.replace(0, EPSILON)
