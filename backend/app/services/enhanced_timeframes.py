from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MS_PER_MINUTE = 60_000
FIVE_MINUTES = 5
FIFTEEN_MINUTES = 15
THIRTY_MINUTES = 30
SIXTY_MINUTES = 60
FOUR_HOURS = 240
ONE_DAY = 1440
TF_5M_RETURN_PERIODS = (5, 20)
TF_15M_RETURN_PERIODS = (4, 16)
TF_30M_RETURN_PERIODS = (1, 3)
TF_1H_RETURN_PERIODS = (1, 4)
TF_4H_RETURN_PERIODS = (1, 3, 6)
TF_1D_RETURN_PERIODS = (1, 2, 3)
TF_5M_MA_WINDOW = 20
TF_15M_MA_WINDOW = 16
TF_30M_MA_WINDOW = 8
TF_1H_MA_WINDOW = 4
TF_4H_MA_WINDOW = 6
TF_1D_MA_WINDOW = 5
TF_5M_MA_WINDOWS = (6, 12, 20, 48)
TF_15M_MA_WINDOWS = (4, 8, 16, 32)
TF_30M_MA_WINDOWS = (2, 4, 8)
TF_1H_MA_WINDOWS = (2, 4, 8, 12)
TF_4H_MA_WINDOWS = (2, 3, 6, 12)
TF_1D_MA_WINDOWS = (2, 3, 5, 7)
TF_5M_VOL_WINDOWS = (12, 48)
TF_15M_VOL_WINDOWS = (8, 32)
TF_30M_VOL_WINDOWS = (4, 8)
TF_1H_VOL_WINDOWS = (4, 12)
TF_4H_VOL_WINDOWS = (3, 6, 12)
TF_1D_VOL_WINDOWS = (3, 7)


@dataclass(frozen=True)
class TimeframeState:
    context: pd.DataFrame
    bucket_close: pd.Series
    rule_ms: int


@dataclass(frozen=True)
class OnlineTimeframeSpec:
    minutes: int
    prefix: str
    return_periods: tuple[int, ...]
    ma_window: int
    ma_column: str
    ma_windows: tuple[int, ...]
    vol_windows: tuple[int, ...]


TIMEFRAME_SPECS = (
    OnlineTimeframeSpec(
        FIVE_MINUTES, "tf_5m", TF_5M_RETURN_PERIODS, TF_5M_MA_WINDOW,
        "tf_5m_ma_ratio_20", TF_5M_MA_WINDOWS, TF_5M_VOL_WINDOWS,
    ),
    OnlineTimeframeSpec(
        FIFTEEN_MINUTES, "tf_15m", TF_15M_RETURN_PERIODS, TF_15M_MA_WINDOW,
        "tf_15m_ma_ratio_16", TF_15M_MA_WINDOWS, TF_15M_VOL_WINDOWS,
    ),
    OnlineTimeframeSpec(
        THIRTY_MINUTES, "tf_30m", TF_30M_RETURN_PERIODS, TF_30M_MA_WINDOW,
        "tf_30m_ma_ratio_8", TF_30M_MA_WINDOWS, TF_30M_VOL_WINDOWS,
    ),
    OnlineTimeframeSpec(
        SIXTY_MINUTES, "tf_1h", TF_1H_RETURN_PERIODS, TF_1H_MA_WINDOW,
        "tf_1h_ma_ratio_4", TF_1H_MA_WINDOWS, TF_1H_VOL_WINDOWS,
    ),
    OnlineTimeframeSpec(
        FOUR_HOURS, "tf_4h", TF_4H_RETURN_PERIODS, TF_4H_MA_WINDOW,
        "tf_4h_ma_ratio_6", TF_4H_MA_WINDOWS, TF_4H_VOL_WINDOWS,
    ),
    OnlineTimeframeSpec(
        ONE_DAY, "tf_1d", TF_1D_RETURN_PERIODS, TF_1D_MA_WINDOW,
        "tf_1d_ma_ratio_3", TF_1D_MA_WINDOWS, TF_1D_VOL_WINDOWS,
    ),
)


def add_online_timeframe_features(base_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    for spec in TIMEFRAME_SPECS:
        out = _add_return_features(
            out,
            source_df,
            timeframe_minutes=spec.minutes,
            periods=spec.return_periods,
            prefix=spec.prefix,
        )
        out = _add_ma_ratio_feature(
            out,
            source_df,
            timeframe_minutes=spec.minutes,
            window=spec.ma_window,
            column=spec.ma_column,
        )
        out = _add_regime_features(
            out,
            source_df,
            timeframe_minutes=spec.minutes,
            prefix=spec.prefix,
            ma_windows=spec.ma_windows,
            vol_windows=spec.vol_windows,
        )
    return out


def _add_return_features(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    timeframe_minutes: int,
    periods: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    state = _timeframe_context(base_df, source_df, timeframe_minutes)
    out = base_df.copy()
    for period in periods:
        reference = _previous_bucket_close(state, period)
        value = state.context["close"] / reference.replace(0, np.nan) - 1.0
        out[f"{prefix}_ret_{period}"] = value.fillna(0.0).to_numpy()
    return out


def _add_ma_ratio_feature(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    timeframe_minutes: int,
    window: int,
    column: str,
) -> pd.DataFrame:
    state = _timeframe_context(base_df, source_df, timeframe_minutes)
    closes = [state.context["close"]]
    closes.extend(_previous_bucket_close(state, period) for period in range(1, window))
    close_frame = pd.concat(closes, axis=1)
    mean_close = close_frame.mean(axis=1).replace(0, np.nan)
    out = base_df.copy()
    out[column] = (state.context["close"] / mean_close - 1.0).fillna(0.0).to_numpy()
    return out


def _add_regime_features(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    timeframe_minutes: int,
    prefix: str,
    ma_windows: tuple[int, ...],
    vol_windows: tuple[int, ...],
) -> pd.DataFrame:
    state = _timeframe_context(base_df, source_df, timeframe_minutes)
    out = _add_intrabar_features(base_df, source_df, timeframe_minutes=timeframe_minutes, prefix=prefix)
    for window in ma_windows:
        out[f"{prefix}_ma_ratio_{window}"] = _ma_ratio(state, window).to_numpy()
    for window in vol_windows:
        out[f"{prefix}_ret_vol_{window}"] = _return_volatility(state, window).to_numpy()
    out[f"{prefix}_ret_accel"] = _return_acceleration(state).to_numpy()
    return out


def _add_intrabar_features(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    *,
    timeframe_minutes: int,
    prefix: str,
) -> pd.DataFrame:
    state = _intrabar_state(source_df, timeframe_minutes)
    merged = pd.merge_asof(
        base_df[["open_time"]].sort_values("open_time"),
        state.sort_values("open_time"),
        on="open_time",
        direction="backward",
    ).sort_values("open_time")
    out = base_df.sort_values("open_time").reset_index(drop=True)
    out[f"{prefix}_intrabar_ret"] = merged["intrabar_ret"].fillna(0.0).to_numpy()
    out[f"{prefix}_intrabar_range"] = merged["intrabar_range"].fillna(0.0).to_numpy()
    out[f"{prefix}_intrabar_pos"] = merged["intrabar_pos"].fillna(0.5).to_numpy()
    out[f"{prefix}_volume_share"] = merged["volume_share"].fillna(0.0).to_numpy()
    return out


def _timeframe_context(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    timeframe_minutes: int,
) -> TimeframeState:
    rule_ms = timeframe_minutes * MS_PER_MINUTE
    source = source_df[["open_time", "close"]].copy()
    source["open_time"] = pd.to_numeric(source["open_time"], errors="raise")
    source["close"] = pd.to_numeric(source["close"], errors="coerce")
    source["bucket_start"] = _bucket_start(source["open_time"], rule_ms)
    bucket_close = source.groupby("bucket_start", sort=True)["close"].last()
    context = base_df[["open_time", "close"]].copy()
    context["open_time"] = pd.to_numeric(context["open_time"], errors="raise")
    context["close"] = pd.to_numeric(context["close"], errors="coerce")
    context["bucket_start"] = _bucket_start(context["open_time"], rule_ms)
    return TimeframeState(context, bucket_close, rule_ms)


def _intrabar_state(source_df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    rule_ms = timeframe_minutes * MS_PER_MINUTE
    source = source_df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    source["bucket_start"] = _bucket_start(source["open_time"], rule_ms)
    grouped = source.groupby("bucket_start", sort=False)
    bucket_open = grouped["open"].transform("first").replace(0, np.nan)
    high_so_far = grouped["high"].cummax()
    low_so_far = grouped["low"].cummin()
    elapsed_minutes = ((source["open_time"] - source["bucket_start"]) // MS_PER_MINUTE) + 1
    elapsed_share = elapsed_minutes.clip(lower=1, upper=timeframe_minutes) / float(timeframe_minutes)
    range_so_far = (high_so_far - low_so_far).replace(0, np.nan)
    return pd.DataFrame({
        "open_time": source["open_time"],
        "intrabar_ret": source["close"] / bucket_open - 1.0,
        "intrabar_range": range_so_far / source["close"].replace(0, np.nan),
        "intrabar_pos": (source["close"] - low_so_far) / range_so_far,
        "volume_share": elapsed_share,
    })


def _ma_ratio(state: TimeframeState, window: int) -> pd.Series:
    closes = [state.context["close"]]
    closes.extend(_previous_bucket_close(state, period) for period in range(1, window))
    mean_close = pd.concat(closes, axis=1).mean(axis=1).replace(0, np.nan)
    return (state.context["close"] / mean_close - 1.0).fillna(0.0)


def _return_volatility(state: TimeframeState, window: int) -> pd.Series:
    returns = [_period_return(state, period) for period in range(1, window + 1)]
    return pd.concat(returns, axis=1).std(axis=1).fillna(0.0)


def _return_acceleration(state: TimeframeState) -> pd.Series:
    return (_period_return(state, 1) - _period_return(state, 2)).fillna(0.0)


def _period_return(state: TimeframeState, period: int) -> pd.Series:
    reference = _previous_bucket_close(state, period)
    return (state.context["close"] / reference.replace(0, np.nan) - 1.0).fillna(0.0)


def _previous_bucket_close(state: TimeframeState, period: int) -> pd.Series:
    target_bucket = state.context["bucket_start"] - int(period) * state.rule_ms
    return target_bucket.map(state.bucket_close)


def _bucket_start(open_time: pd.Series, rule_ms: int) -> pd.Series:
    return (open_time.astype("int64") // int(rule_ms)) * int(rule_ms)
