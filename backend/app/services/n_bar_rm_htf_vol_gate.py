"""
Higher-timeframe + volatility gates for n-bar index reverse strategies
(three / four / five bar; shared implementation).

- Volatility: last completed bar true range vs Wilder ATR(period); spike → skip.
  Uses the same interval as the trading duration (10m/30m/1h/1d).
- HTF trend: close vs SMA on completed bars; if reversal bet fights HTF trend → skip.
  Uses a higher timeframe relative to the trading duration.
"""

from __future__ import annotations

import os
from typing import Any

from app.services.binance_service import fetch_index_price_klines
from app.services.rule_config import RULE_DURATION

FETCH_KLINE_LIMIT = 80

# duration 到 Binance API interval 的映射（波动率检查使用相同周期）
DURATION_TO_VOL_INTERVAL: dict[str, str] = {
    "10m": "10m",
    "30m": "30m",
    "60m": "1h",
    "1d": "1d",
}

# duration 到更高时间框架的映射（趋势检查使用更高周期）
DURATION_TO_HTF_INTERVAL: dict[str, str] = {
    "10m": "1h",
    "30m": "4h",
    "60m": "4h",
    "1d": "1d",  # 日线使用自身，无更高周期
}
PRICE_DECIMALS_FOR_META = 8

_DEFAULT_ENABLED = True
_DEFAULT_ATR_PERIOD = 14
_DEFAULT_TR_ATR_MULT = 2.0
_DEFAULT_HTF_SMA_PERIOD = 20


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def n_bar_rm_htf_vol_gate_enabled() -> bool:
    return _env_bool("N_BAR_RM_HTF_VOL_GATE_ENABLED", _DEFAULT_ENABLED)


def n_bar_rm_tr_atr_multiplier() -> float:
    return max(0.5, _env_float("N_BAR_RM_TR_ATR_MULT", _DEFAULT_TR_ATR_MULT))


def n_bar_rm_atr_period() -> int:
    return max(2, _env_int("N_BAR_RM_ATR_PERIOD", _DEFAULT_ATR_PERIOD))


def n_bar_rm_htf_sma_period() -> int:
    return max(2, _env_int("N_BAR_RM_HTF_SMA_PERIOD", _DEFAULT_HTF_SMA_PERIOD))


def _true_range(bar: dict[str, Any], prev_close: float | None) -> float:
    h = float(bar["high"])
    low = float(bar["low"])
    if prev_close is None:
        return h - low
    return max(h - low, abs(h - prev_close), abs(low - prev_close))


def _atr_wilder_last(tr_values: list[float], period: int) -> float | None:
    """Wilder ATR value at the last TR sample."""
    n = len(tr_values)
    if n < period:
        return None
    atr = sum(tr_values[:period]) / float(period)
    for i in range(period, n):
        atr = (atr * (period - 1) + tr_values[i]) / float(period)
    return atr


def _completed_before(bars: list[dict[str, Any]], entry_open_time_ms: int) -> list[dict[str, Any]]:
    return [b for b in bars if int(b["closeTime"]) < int(entry_open_time_ms)]


def evaluate_volatility_spike_suppress(
    pair: str,
    entry_open_time_ms: int,
    *,
    duration: str = RULE_DURATION,
    atr_period: int | None = None,
    tr_atr_mult: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    True → suppress trade (last completed bar TR > mult * ATR(period)).
    Uses the same interval as the trading duration.
    If insufficient bars for ATR, do not suppress (fail-open).
    """
    period = atr_period if atr_period is not None else n_bar_rm_atr_period()
    mult = tr_atr_mult if tr_atr_mult is not None else n_bar_rm_tr_atr_multiplier()
    interval = DURATION_TO_VOL_INTERVAL.get(duration, "10m")
    meta: dict[str, Any] = {
        "volGateEvaluated": True,
        "volSuppress": False,
        "volInterval": interval,
        "atrPeriod": period,
        "trAtrMult": mult,
        "lastBarTr": None,
        "lastBarAtr": None,
        "reason": None,
    }
    bars = fetch_index_price_klines(pair, interval, limit=FETCH_KLINE_LIMIT)
    completed = _completed_before(bars, entry_open_time_ms)
    if len(completed) < period:
        meta["reason"] = f"insufficient_{interval}_bars_for_atr"
        meta["volGateEvaluated"] = False
        return False, meta

    tr_vals: list[float] = []
    for i, b in enumerate(completed):
        pc = float(completed[i - 1]["close"]) if i > 0 else None
        tr_vals.append(_true_range(b, pc))

    atr_last = _atr_wilder_last(tr_vals, period)
    tr_last = tr_vals[-1]
    meta["lastBarTr"] = round(tr_last, 8)
    if atr_last is None or atr_last <= 0:
        meta["reason"] = "atr_unavailable"
        return False, meta
    meta["lastBarAtr"] = round(atr_last, 8)
    threshold = mult * atr_last
    if tr_last > threshold:
        meta["volSuppress"] = True
        meta["reason"] = "tr_exceeds_k_times_atr"
        meta["thresholdTr"] = round(threshold, 8)
        return True, meta
    meta["reason"] = "ok"
    meta["thresholdTr"] = round(threshold, 8)
    return False, meta


def evaluate_htf_counter_trend_suppress(
    pair: str,
    entry_open_time_ms: int,
    predicted_direction: str,
    *,
    duration: str = RULE_DURATION,
    sma_period: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Suppress when reversal bet fights HTF trend (close vs SMA on last closed bar).
    Uses a higher timeframe relative to the trading duration.

    - predicted_direction 'up' after bear streak: skip if HTF bearish (close < SMA).
    - predicted_direction 'down' after bull streak: skip if HTF bullish (close > SMA).
    """
    p = sma_period if sma_period is not None else n_bar_rm_htf_sma_period()
    htf_interval = DURATION_TO_HTF_INTERVAL.get(duration, "1h")
    meta: dict[str, Any] = {
        "htfGateEvaluated": True,
        "htfSuppress": False,
        "htfInterval": htf_interval,
        "htfSmaPeriod": p,
        "htfLastClose": None,
        "htfSma": None,
        "htfTrend": None,
        "reason": None,
    }
    bars_htf = fetch_index_price_klines(pair, htf_interval, limit=FETCH_KLINE_LIMIT)
    completed = _completed_before(bars_htf, entry_open_time_ms)
    if len(completed) < p + 1:
        meta["reason"] = f"insufficient_{htf_interval}_bars_for_sma"
        meta["htfGateEvaluated"] = False
        return False, meta

    tail = completed[-p:]
    last_bar = completed[-1]
    sma = sum(float(b["close"]) for b in tail) / float(p)
    last_close = float(last_bar["close"])
    meta["htfLastClose"] = round(last_close, PRICE_DECIMALS_FOR_META)
    meta["htfSma"] = round(sma, PRICE_DECIMALS_FOR_META)
    if last_close > sma:
        trend = "bull"
    elif last_close < sma:
        trend = "bear"
    else:
        trend = "neutral"
    meta["htfTrend"] = trend

    if trend == "neutral":
        meta["reason"] = "htf_neutral_no_skip"
        return False, meta

    d = predicted_direction.lower()
    if d == "up" and trend == "bear":
        meta["htfSuppress"] = True
        meta["reason"] = f"counter_trend_long_vs_bearish_{htf_interval}"
        return True, meta
    if d == "down" and trend == "bull":
        meta["htfSuppress"] = True
        meta["reason"] = f"counter_trend_short_vs_bullish_{htf_interval}"
        return True, meta
    meta["reason"] = "aligned_or_no_conflict"
    return False, meta
