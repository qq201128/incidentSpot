from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import pandas as pd

from app.services.event_regime_detector import add_event_regime_features
from app.services.factor_learning_common import SUCCESS_WIN_RATE_MIN
from app.services.return_metric_policy import ReturnMetricPolicy, rounded_metrics

REQUIRED_COLUMNS = ("open_time", "high", "low", "close")
HIGH_RISK_WIN_RATE_BONUS = 0.05
METRIC_DECIMALS = 6


@dataclass(frozen=True)
class FactorRegimeGate:
    passed: bool
    reason: str
    regime: dict[str, Any]
    min_win_rate: float | None


def factor_regime_report(
    frame: pd.DataFrame,
    returns: pd.Series,
    duration: str,
) -> dict[str, Any]:
    regimes = regime_frame(frame, duration)
    joined = pd.DataFrame({"returns": returns}).join(regimes, how="inner")
    if joined.empty:
        return _empty_report()
    return {
        "policy": "factor_regime_bucket_v1",
        "byTrend": _grouped_metrics(joined, "trendState"),
        "byVolatility": _grouped_metrics(joined, "volatilityState"),
        "byRegime": _grouped_metrics(joined, "regimeLabel"),
    }


def current_factor_regime(frame: pd.DataFrame, index: Any, duration: str) -> dict[str, Any]:
    regimes = regime_frame(frame, duration)
    if index not in regimes.index:
        raise ValueError(f"factor regime missing source row index: {index}")
    row = regimes.loc[index]
    return _regime_payload(row)


def factor_regime_gate(direction: str, win_rate: Any, regime: dict[str, Any]) -> FactorRegimeGate:
    normalized = str(direction).lower()
    trend = str(regime.get("trendState") or "uncertain")
    volatility = str(regime.get("volatilityState") or "normal_vol")
    base = _finite_float(win_rate)
    if normalized == "up" and trend == "trend_down":
        return FactorRegimeGate(False, "regime_counter_trend_long", regime, base)
    if normalized == "down" and trend == "trend_up":
        return FactorRegimeGate(False, "regime_counter_trend_short", regime, base)
    required = _risk_adjusted_min_win_rate(base, trend, volatility)
    if required is not None and (base is None or base < required):
        return FactorRegimeGate(False, "regime_high_risk_win_rate_below_min", regime, required)
    return FactorRegimeGate(True, "passed", regime, required)


def regime_frame(frame: pd.DataFrame, duration: str) -> pd.DataFrame:
    _require_columns(frame)
    source = frame.loc[:, list(dict.fromkeys((*REQUIRED_COLUMNS,)))].copy()
    source["__source_index"] = frame.index
    enriched = add_event_regime_features(source, duration)
    enriched = enriched.set_index("__source_index", drop=True)
    return pd.DataFrame(
        {
            "trendState": enriched.apply(_trend_state, axis=1),
            "volatilityState": enriched.apply(_volatility_state, axis=1),
            "regimeConfidence": enriched["regime_confidence"].map(_finite_float).fillna(0.0),
        },
        index=enriched.index,
    ).assign(regimeLabel=lambda data: data["trendState"] + ":" + data["volatilityState"])


def _grouped_metrics(frame: pd.DataFrame, column: str) -> dict[str, dict[str, Any]]:
    return {
        str(label): _return_metrics(group["returns"])
        for label, group in frame.groupby(column, sort=True)
    }


def _return_metrics(returns: pd.Series) -> dict[str, Any]:
    metrics = rounded_metrics(ReturnMetricPolicy().from_returns(returns.dropna()), METRIC_DECIMALS)
    return {
        "sampleCount": metrics.get("sampleCount"),
        "winRate": metrics.get("winRate"),
        "profitFactor": metrics.get("profitFactor"),
        "avgReturn": metrics.get("avgReturn"),
    }


def _regime_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "trendState": str(row["trendState"]),
        "volatilityState": str(row["volatilityState"]),
        "regimeLabel": str(row["regimeLabel"]),
        "confidence": float(row["regimeConfidence"]),
    }


def _risk_adjusted_min_win_rate(
    base_win_rate: float | None,
    trend: str,
    volatility: str,
) -> float | None:
    if base_win_rate is None:
        return None
    if trend == "uncertain" or volatility == "high_vol":
        return min(SUCCESS_WIN_RATE_MIN + HIGH_RISK_WIN_RATE_BONUS, 0.99)
    return SUCCESS_WIN_RATE_MIN


def _trend_state(row: pd.Series) -> str:
    if float(row.get("regime_trend_up") or 0) > 0:
        return "trend_up"
    if float(row.get("regime_trend_down") or 0) > 0:
        return "trend_down"
    if float(row.get("regime_range") or 0) > 0:
        return "range"
    return "uncertain"


def _volatility_state(row: pd.Series) -> str:
    if float(row.get("regime_high_vol") or 0) > 0:
        return "high_vol"
    if float(row.get("regime_low_vol") or 0) > 0:
        return "low_vol"
    return "normal_vol"


def _empty_report() -> dict[str, Any]:
    return {"policy": "factor_regime_bucket_v1", "byTrend": {}, "byVolatility": {}, "byRegime": {}}


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"factor regime analysis missing columns: {', '.join(missing)}")


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
