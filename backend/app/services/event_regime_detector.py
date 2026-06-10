from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.db.session import get_conn
from app.services.rule_config import DURATION_TO_MINUTES

FAST_MA = 8
SLOW_MA = 24
SLOPE_LOOKBACK = 6
ATR_PERIOD = 14
VOL_LOOKBACK = 80
MIN_COMPLETED_BARS = 40
SLOPE_THRESHOLD = 0.0004
RANGE_BIAS_THRESHOLD = 0.003
HIGH_VOL_PERCENTILE = 0.70
LOW_VOL_PERCENTILE = 0.30
REGIME_LOOKBACK_BARS = max(MIN_COMPLETED_BARS, VOL_LOOKBACK + SLOW_MA + SLOPE_LOOKBACK + 1)


class EventRegimeDataError(ValueError):
    pass


@dataclass(frozen=True)
class EventRegime:
    symbol: str
    duration: str
    open_time: int
    trend_state: str
    volatility_state: str
    regime_label: str
    confidence: float
    reason_codes: tuple[str, ...]
    metrics: dict[str, Any]


def add_event_regime_features(frame: pd.DataFrame, duration: str) -> pd.DataFrame:
    _assert_duration(duration)
    out = frame.sort_values("open_time").reset_index(drop=True).copy()
    close = pd.to_numeric(out["close"], errors="raise")
    high = pd.to_numeric(out["high"], errors="raise")
    low = pd.to_numeric(out["low"], errors="raise")
    out["regime_ma_fast"] = close.rolling(FAST_MA, min_periods=FAST_MA).mean()
    out["regime_ma_slow"] = close.rolling(SLOW_MA, min_periods=SLOW_MA).mean()
    out["regime_slow_slope"] = out["regime_ma_slow"].pct_change(SLOPE_LOOKBACK)
    out["regime_atr_ratio"] = _atr_ratio(high, low, close)
    out["regime_atr_percentile"] = _rolling_percentile(out["regime_atr_ratio"], VOL_LOOKBACK)
    return _add_state_columns(out, close)


def detect_event_regime(symbol: str, duration: str, entry_open_time: int) -> EventRegime:
    frame = _completed_kline_frame(symbol, duration, entry_open_time)
    if len(frame) < MIN_COMPLETED_BARS:
        raise EventRegimeDataError("insufficient_regime_data")
    enriched = add_event_regime_features(frame, duration).dropna(subset=["regime_ma_slow"])
    if enriched.empty:
        raise EventRegimeDataError("insufficient_regime_data")
    row = enriched.iloc[-1]
    return _regime_from_row(symbol.strip().upper(), duration, int(entry_open_time), row)


def persist_event_regime(regime: EventRegime) -> None:
    conn = get_conn()
    try:
        upsert_event_regime(conn, regime)
        conn.commit()
    finally:
        conn.close()


def upsert_event_regime(conn: Any, regime: EventRegime) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO event_market_regimes
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            regime.symbol,
            regime.duration,
            regime.open_time,
            regime.trend_state,
            regime.volatility_state,
            regime.regime_label,
            regime.confidence,
            json.dumps(list(regime.reason_codes), ensure_ascii=True),
            json.dumps(regime.metrics, ensure_ascii=True, sort_keys=True),
            _utc_now(),
        ),
    )


def _add_state_columns(frame: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    out = frame.copy()
    slow = out["regime_ma_slow"]
    slope = out["regime_slow_slope"]
    bias = (close / slow - 1.0).replace([np.inf, -np.inf], np.nan)
    out["regime_trend_up"] = ((bias > 0) & (slope > SLOPE_THRESHOLD)).astype(float)
    out["regime_trend_down"] = ((bias < 0) & (slope < -SLOPE_THRESHOLD)).astype(float)
    out["regime_range"] = ((bias.abs() <= RANGE_BIAS_THRESHOLD) & (slope.abs() <= SLOPE_THRESHOLD)).astype(float)
    known = out[["regime_trend_up", "regime_trend_down", "regime_range"]].sum(axis=1)
    out["regime_uncertain"] = (known <= 0).astype(float)
    pct = out["regime_atr_percentile"]
    out["regime_high_vol"] = (pct >= HIGH_VOL_PERCENTILE).astype(float)
    out["regime_low_vol"] = (pct <= LOW_VOL_PERCENTILE).astype(float)
    out["regime_confidence"] = _confidence_series(bias, slope, pct)
    return out


def _atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return (tr / close).replace([np.inf, -np.inf], np.nan)


def _rolling_percentile(values: pd.Series, lookback: int) -> pd.Series:
    return values.rolling(lookback, min_periods=lookback).apply(_last_percentile, raw=True)


def _last_percentile(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    return float((finite <= finite[-1]).mean())


def _confidence_series(bias: pd.Series, slope: pd.Series, pct: pd.Series) -> pd.Series:
    trend_strength = (bias.abs() / RANGE_BIAS_THRESHOLD).clip(0, 1)
    slope_strength = (slope.abs() / (SLOPE_THRESHOLD * 2)).clip(0, 1)
    vol_certainty = (pct - 0.5).abs().mul(2).clip(0, 1).fillna(0)
    return (0.45 * trend_strength + 0.35 * slope_strength + 0.20 * vol_certainty).clip(0, 1)


def _completed_kline_frame(symbol: str, duration: str, entry_open_time: int) -> pd.DataFrame:
    _assert_duration(duration)
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time, open, high, low, close, volume
            FROM (
                SELECT open_time, open, high, low, close, volume
                FROM klines
                WHERE symbol = ? AND interval = ? AND open_time < ?
                ORDER BY open_time DESC
                LIMIT ?
            )
            ORDER BY open_time ASC
            """,
            (symbol.strip().upper(), duration, int(entry_open_time), REGIME_LOOKBACK_BARS),
        ).fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])


def _regime_from_row(symbol: str, duration: str, open_time: int, row: pd.Series) -> EventRegime:
    trend = _trend_state(row)
    vol = _volatility_state(row)
    reasons = _reason_codes(trend, vol)
    return EventRegime(
        symbol=symbol,
        duration=duration,
        open_time=open_time,
        trend_state=trend,
        volatility_state=vol,
        regime_label=f"{trend}:{vol}",
        confidence=round(float(row.get("regime_confidence") or 0.0), 6),
        reason_codes=reasons,
        metrics=_metrics(row),
    )


def _trend_state(row: pd.Series) -> str:
    if float(row.get("regime_trend_up") or 0) > 0:
        return "trend_up"
    if float(row.get("regime_trend_down") or 0) > 0:
        return "trend_down"
    if float(row.get("regime_range") or 0) > 0:
        return "range"
    return "uncertain"


def _volatility_state(row: pd.Series) -> str:
    pct = row.get("regime_atr_percentile")
    if pd.isna(pct):
        return "normal_vol"
    if float(pct) >= HIGH_VOL_PERCENTILE:
        return "high_vol"
    if float(pct) <= LOW_VOL_PERCENTILE:
        return "low_vol"
    return "normal_vol"


def _reason_codes(trend: str, vol: str) -> tuple[str, ...]:
    reasons = [trend, vol]
    if trend == "uncertain":
        reasons.append("trend_uncertain")
    if vol == "high_vol":
        reasons.append("volatility_risk")
    return tuple(reasons)


def _metrics(row: pd.Series) -> dict[str, Any]:
    keys = ("regime_slow_slope", "regime_atr_ratio", "regime_atr_percentile")
    return {key: _finite_or_none(row.get(key)) for key in keys}


def _finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _assert_duration(duration: str) -> None:
    if duration not in DURATION_TO_MINUTES:
        raise ValueError(f"unsupported event regime duration: {duration}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
