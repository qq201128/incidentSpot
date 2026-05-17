from __future__ import annotations

from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS, add_contribution_scores

WIN_RATE_WEIGHT = 35.0
SHARPE_WEIGHT = 20.0
IR_WEIGHT = 18.0
PROFIT_FACTOR_WEIGHT = 12.0
CONTRIBUTION_WEIGHT = 8.0
CORRELATION_WEIGHT = 7.0
SHARPE_SCALE = 3.0
IR_SCALE = 2.0
PROFIT_FACTOR_SCALE = 2.0
SCORE_DECIMALS = 2
METRIC_DECIMALS = 4
INVALID_BACKTEST_SCORE = 0.0


def enrich_factor_results(
    results: list[dict[str, Any]],
    *,
    frame: pd.DataFrame | None = None,
    duration_scoped: bool = False,
) -> None:
    add_contribution_scores(results, duration_scoped=duration_scoped)
    _add_correlation_metrics(results, frame)
    for row in results:
        row["factorScore"] = factor_score(row)


def factor_score(row: dict[str, Any]) -> float:
    validity = backtest_validity(row)
    if not validity["valid"]:
        return INVALID_BACKTEST_SCORE
    raw = (
        _clamp01(row.get("winRate")) * WIN_RATE_WEIGHT
        + _scaled_positive(row.get("sharpe"), SHARPE_SCALE) * SHARPE_WEIGHT
        + _scaled_positive(row.get("ir"), IR_SCALE) * IR_WEIGHT
        + _profit_factor_score(row.get("profitFactor")) * PROFIT_FACTOR_WEIGHT
        + _clamp01(row.get("contribution")) * CONTRIBUTION_WEIGHT
        + _correlation_score(row.get("avgAbsCorrelation")) * CORRELATION_WEIGHT
    )
    return round(raw, SCORE_DECIMALS)


def backtest_validity(row: dict[str, Any]) -> dict[str, Any]:
    periods = int(row.get("totalPeriods") or 0)
    if periods < BACKTEST_MIN_PERIODS:
        return _validity(False, "insufficient_periods", periods)
    if _finite_float(row.get("winRate")) is None:
        return _validity(False, "win_rate_missing", periods)
    return _validity(True, "passed", periods)


def factor_avg_abs_correlations(frame: pd.DataFrame, names: list[str]) -> dict[str, float | None]:
    available = [name for name in dict.fromkeys(names) if name in frame.columns]
    if len(available) < 2:
        return {name: None for name in available}
    numeric = frame[available].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="spearman").abs()
    return {name: _avg_peer_correlation(corr, name) for name in available}


def _add_correlation_metrics(results: list[dict[str, Any]], frame: pd.DataFrame | None) -> None:
    if frame is None:
        _preserve_existing_correlation(results)
        return
    correlations = factor_avg_abs_correlations(frame, [_row_factor_name(row) for row in results])
    for row in results:
        value = correlations.get(_row_factor_name(row))
        row["avgAbsCorrelation"] = _round_or_none(value, METRIC_DECIMALS)
        row["correlationPenalty"] = _round_or_none(value, METRIC_DECIMALS)


def _preserve_existing_correlation(results: list[dict[str, Any]]) -> None:
    for row in results:
        value = _finite_float(row.get("avgAbsCorrelation"))
        row["avgAbsCorrelation"] = _round_or_none(value, METRIC_DECIMALS)
        row["correlationPenalty"] = _round_or_none(value, METRIC_DECIMALS)


def _avg_peer_correlation(corr: pd.DataFrame, name: str) -> float | None:
    values = corr[name].drop(labels=[name], errors="ignore").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _row_factor_name(row: dict[str, Any]) -> str:
    return str(row.get("factorName") or row.get("name") or "")


def _correlation_score(value: Any) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return 1.0 - _clamp01(number)


def _validity(valid: bool, reason: str, periods: int) -> dict[str, Any]:
    return {
        "valid": valid,
        "reason": reason,
        "minPeriods": BACKTEST_MIN_PERIODS,
        "totalPeriods": periods,
    }


def _profit_factor_score(value: Any) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return _clamp01((number - 1.0) / (PROFIT_FACTOR_SCALE - 1.0))


def _scaled_positive(value: Any, scale: float) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return _clamp01(max(number, 0.0) / scale)


def _clamp01(value: Any) -> float:
    number = _finite_float(value)
    if number is None:
        return 0.0
    return max(0.0, min(number, 1.0))


def _round_or_none(value: float | None, decimals: int) -> float | None:
    if value is None or not isfinite(float(value)):
        return None
    return round(float(value), decimals)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None
