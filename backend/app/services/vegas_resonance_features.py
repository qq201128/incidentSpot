from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EMA_PERIODS = (12, 13, 144, 169, 576, 676)
FIB_LEVELS = (0.236, 0.382, 0.5, 0.618, 0.786)
FIB_LOOKBACK = 144
CHANNEL_NEAR_RATIO = 0.005
RESONANCE_TOLERANCE = 0.01
PARALLEL_WIDTH_THRESHOLD = 0.0005
MS_PER_MINUTE = 60_000


@dataclass(frozen=True)
class VegasTimeframe:
    prefix: str
    minutes: int
    weight: float


VEGAS_TIMEFRAMES = (
    VegasTimeframe("vegas_5m", 5, 0.30),
    VegasTimeframe("vegas_15m", 15, 0.25),
    VegasTimeframe("vegas_1h", 60, 0.25),
    VegasTimeframe("vegas_4h", 240, 0.15),
    VegasTimeframe("vegas_1d", 1440, 0.05),
)


def add_vegas_resonance_features(base_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df.copy()
    weighted_scores = []
    weighted_direction = []
    for timeframe in VEGAS_TIMEFRAMES:
        aligned = _aligned_timeframe_features(base_df, source_df, timeframe)
        out = out.join(aligned)
        weighted_scores.append(out[f"{timeframe.prefix}_score"] * timeframe.weight)
        weighted_direction.append(out[f"{timeframe.prefix}_direction_score"] * timeframe.weight)
    out["vegas_resonance_score"] = pd.concat(weighted_scores, axis=1).sum(axis=1).fillna(0.0)
    out["vegas_direction_score"] = pd.concat(weighted_direction, axis=1).sum(axis=1).fillna(0.0)
    out["vegas_bull_score"] = out["vegas_resonance_score"].where(out["vegas_direction_score"] > 0, 0.0)
    out["vegas_bear_score"] = out["vegas_resonance_score"].where(out["vegas_direction_score"] < 0, 0.0)
    return out


def _aligned_timeframe_features(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    timeframe: VegasTimeframe,
) -> pd.DataFrame:
    bars = _timeframe_bars(source_df, timeframe.minutes)
    scored = _score_timeframe_bars(bars, timeframe.prefix)
    key = _previous_bucket_key(base_df["open_time"], timeframe.minutes)
    aligned = scored.reindex(key.to_numpy()).reset_index(drop=True)
    aligned = aligned.fillna(0.0)
    aligned.index = base_df.index
    return aligned


def _timeframe_bars(source_df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule_ms = minutes * MS_PER_MINUTE
    source = source_df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    source["bucket_start"] = (source["open_time"].astype("int64") // rule_ms) * rule_ms
    grouped = source.groupby("bucket_start", sort=True)
    return grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )


def _score_timeframe_bars(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    data = _with_ema_columns(bars.copy())
    alignment = _alignment_score(data)
    position = _position_score(data)
    width = _width_score(data, alignment)
    fib = _fib_resonance_score(data)
    momentum = _momentum_score(data, alignment)
    total = (alignment.abs() + position.abs() + width + fib + momentum).clip(0.0, 100.0)
    direction = _direction_score(alignment, position)
    return pd.DataFrame({
        f"{prefix}_score": total,
        f"{prefix}_direction_score": direction,
        f"{prefix}_fib_score": fib,
        f"{prefix}_alignment_score": alignment,
    }, index=bars.index)


def _with_ema_columns(data: pd.DataFrame) -> pd.DataFrame:
    for period in EMA_PERIODS:
        data[f"ema_{period}"] = data["close"].ewm(span=period, adjust=False).mean()
    return data


def _alignment_score(data: pd.DataFrame) -> pd.Series:
    close = data["close"]
    bull = (close > data["ema_12"]) & (data["ema_13"] > data["ema_144"]) & (data["ema_169"] > data["ema_576"])
    bear = (close < data["ema_13"]) & (data["ema_12"] < data["ema_169"]) & (data["ema_144"] < data["ema_676"])
    near_inner = _near_channel(close, data["ema_12"], data["ema_13"])
    bull_tangle = near_inner & (data["ema_144"] > data["ema_676"])
    bear_tangle = near_inner & (data["ema_169"] < data["ema_576"])
    score = pd.Series(0.0, index=data.index)
    score = score.mask(bull, 30.0).mask(bear, -30.0)
    return score.mask(bull_tangle & ~bull, 15.0).mask(bear_tangle & ~bear, -15.0)


def _position_score(data: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=data.index)
    score += _channel_position(data["close"], data["ema_12"], data["ema_13"]) * 5.0
    score += _channel_position(data["close"], data["ema_144"], data["ema_169"]) * 10.0
    score += _channel_position(data["close"], data["ema_576"], data["ema_676"]) * 15.0
    return score


def _width_score(data: pd.DataFrame, alignment: pd.Series) -> pd.Series:
    width = (data["ema_144"] - data["ema_169"]).abs() / data["close"].replace(0, np.nan)
    width_change = width.diff().fillna(0.0)
    score = pd.Series(10.0, index=data.index)
    score = score.mask(width_change > PARALLEL_WIDTH_THRESHOLD, 15.0)
    score = score.mask(width_change < -PARALLEL_WIDTH_THRESHOLD, 5.0)
    return score.where(alignment != 0, score * 0.5).fillna(0.0)


def _fib_resonance_score(data: pd.DataFrame) -> pd.Series:
    high = data["high"].rolling(FIB_LOOKBACK, min_periods=2).max()
    low = data["low"].rolling(FIB_LOOKBACK, min_periods=2).min()
    channel = (high - low).replace(0, np.nan)
    levels = [high - channel * level for level in FIB_LEVELS]
    ema_columns = [data[f"ema_{period}"] for period in EMA_PERIODS]
    scores = [_single_fib_score(level, ratio, ema_columns) for level, ratio in zip(levels, FIB_LEVELS)]
    return pd.concat(scores, axis=1).max(axis=1).fillna(0.0)


def _single_fib_score(level: pd.Series, ratio: float, ema_columns: list[pd.Series]) -> pd.Series:
    tolerance = level.abs() * RESONANCE_TOLERANCE
    matches = [(level - ema).abs() <= tolerance for ema in ema_columns]
    matched = pd.concat(matches, axis=1).any(axis=1)
    if ratio == 0.618:
        mid_match = ((level - ema_columns[2]).abs() <= tolerance) | ((level - ema_columns[3]).abs() <= tolerance)
        return pd.Series(np.where(mid_match, 20.0, np.where(matched, 15.0, 0.0)), index=level.index)
    if ratio in (0.382, 0.5):
        return pd.Series(np.where(matched, 15.0, 0.0), index=level.index)
    return pd.Series(np.where(matched, 10.0, 0.0), index=level.index)


def _momentum_score(data: pd.DataFrame, alignment: pd.Series) -> pd.Series:
    direction = np.sign(alignment).replace(0, np.nan).ffill().fillna(0.0)
    inner_mid = (data["ema_12"] + data["ema_13"]) / 2.0
    breakout = np.sign((data["close"] - inner_mid) / data["close"].replace(0, np.nan))
    volume_ma = data["volume"].rolling(20, min_periods=1).mean()
    body = np.sign(data["close"] - data["open"])
    score = (breakout == direction).astype(float) * 10.0
    score += (data["volume"] > volume_ma).astype(float) * 5.0
    score += (body == direction).astype(float) * 5.0
    return score.fillna(0.0)


def _direction_score(alignment: pd.Series, position: pd.Series) -> pd.Series:
    score = (alignment + position).clip(-100.0, 100.0)
    return (score / 100.0).fillna(0.0)


def _channel_position(close: pd.Series, fast: pd.Series, slow: pd.Series) -> pd.Series:
    upper = pd.concat([fast, slow], axis=1).max(axis=1)
    lower = pd.concat([fast, slow], axis=1).min(axis=1)
    return pd.Series(np.where(close > upper, 1.0, np.where(close < lower, -1.0, 0.0)), index=close.index)


def _near_channel(close: pd.Series, fast: pd.Series, slow: pd.Series) -> pd.Series:
    mid = (fast + slow) / 2.0
    distance = (close - mid).abs() / close.replace(0, np.nan)
    return distance <= CHANNEL_NEAR_RATIO


def _previous_bucket_key(open_time: pd.Series, minutes: int) -> pd.Series:
    rule_ms = minutes * MS_PER_MINUTE
    bucket = (open_time.astype("int64") // rule_ms) * rule_ms
    return bucket - rule_ms
