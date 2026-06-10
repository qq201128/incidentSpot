from __future__ import annotations

import pandas as pd

from app.services.enhanced_features import (
    build_enhanced_feature_frame,
    load_funding_features,
    load_klines,
    load_orderbook_features,
)
from app.services.external_factor_data import load_external_feature_frames
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

FACTOR_FRAME_MIN_HISTORY = 240
MINUTES_PER_DAY = 1_440
MS_PER_DAY = 86_400_000
FACTOR_FRAME_WARMUP_MULTIPLIER = 2
DURATION_MINUTES = {"10m": 10, "30m": 30, "60m": 60, "1d": MINUTES_PER_DAY}


def load_factor_frame(
    symbol: str,
    duration: str = "10m",
    *,
    min_history: int = FACTOR_FRAME_MIN_HISTORY,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported factor frame duration: {duration}")
    raw_lookback_days = _raw_lookback_days(duration, lookback_days, min_history)
    df = _load_duration_klines(symbol, duration, raw_lookback_days)
    orderbook = _load_orderbook(symbol, raw_lookback_days)
    funding = _load_funding(symbol, raw_lookback_days)
    frame, _ = build_enhanced_feature_frame(
        df,
        ob_df=orderbook,
        funding_df=funding,
        external_frames=load_external_feature_frames(symbol),
        min_history=min_history,
    )
    return _recent_frame(frame, lookback_days)


def _raw_lookback_days(duration: str, lookback_days: int | None, min_history: int) -> int | None:
    if lookback_days is None:
        return None
    if lookback_days <= 0:
        raise ValueError("lookback_days must be greater than 0")
    warmup_minutes = min_history * DURATION_MINUTES[duration] * FACTOR_FRAME_WARMUP_MULTIPLIER
    return lookback_days + _ceil_div(warmup_minutes, MINUTES_PER_DAY)


def _load_duration_klines(symbol: str, duration: str, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return load_klines(symbol, duration)
    return load_klines(symbol, duration, lookback_days=lookback_days)


def _load_orderbook(symbol: str, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return load_orderbook_features(symbol)
    return load_orderbook_features(symbol, lookback_days=lookback_days)


def _load_funding(symbol: str, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return load_funding_features(symbol)
    return load_funding_features(symbol, lookback_days=lookback_days)


def _recent_frame(frame: pd.DataFrame, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return frame
    latest = int(pd.to_numeric(frame["open_time"], errors="raise").max())
    cutoff = latest - lookback_days * MS_PER_DAY
    recent = frame[pd.to_numeric(frame["open_time"], errors="raise") >= cutoff].reset_index(drop=True)
    if recent.empty:
        raise ValueError(f"no factor frame rows in recent {lookback_days} day window")
    return recent


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def lookback_days_for_bars(duration: str, bars: int) -> int:
    if duration not in DURATION_MINUTES:
        raise ValueError(f"unsupported factor frame duration: {duration}")
    selected_bars = int(bars)
    if selected_bars <= 0:
        raise ValueError("bars must be greater than 0")
    return _ceil_div(selected_bars * DURATION_MINUTES[duration], MINUTES_PER_DAY)
