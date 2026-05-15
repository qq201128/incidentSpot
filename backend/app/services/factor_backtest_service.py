"""
Factor Backtest Service - Calculate factor performance metrics.

Metrics computed:
- IC (Information Coefficient): Rank correlation between factor values and forward returns
- IR (Information Ratio): IC_mean / IC_std, measures consistency
- Factor Returns: Quintile-based long-short returns
- Turnover: How often factor-based positions change
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.services.factor_duration_alignment import backtest_duration_frame
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_metric_enrichment import backtest_validity, enrich_factor_results
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS, compute_signal_metrics
from app.services.factor_registry import (
    FactorCategory,
    FactorDefinition,
    factor_payload,
    get_factor,
    list_factors,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

QUINTILE_COUNT = 5
IC_ROLLING_WINDOW = 20


@dataclass(frozen=True)
class FactorBacktestResult:
    factor_name: str
    symbol: str
    duration: str
    total_periods: int
    ic_mean: float | None
    ic_std: float | None
    ir: float | None
    ic_positive_rate: float | None
    quintile_returns: list[float]
    long_short_return: float | None
    turnover: float | None
    t_stat: float | None
    p_value: float | None
    sharpe: float | None
    win_rate: float | None
    max_drawdown: float | None
    profit_factor: float | None


def run_factor_backtest(
    factor_name: str,
    symbol: str,
    duration: str = "10m",
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    factor_def = get_factor(factor_name)
    if factor_def is None:
        raise ValueError(f"unknown factor: {factor_name}")

    feature_frame = load_factor_frame(symbol, duration)
    result = run_factor_backtest_on_frame(
        factor_def,
        feature_frame,
        symbol=symbol,
        duration=duration,
    )
    enrich_factor_results([result], frame=feature_frame)
    return result


def run_factor_backtest_on_frame(
    factor_def: FactorDefinition,
    feature_frame: pd.DataFrame,
    *,
    symbol: str,
    duration: str,
) -> dict[str, Any]:
    if factor_def.name not in feature_frame.columns:
        raise ValueError(f"factor {factor_def.name} not found in feature frame")
    result = _compute_factor_metrics(feature_frame, factor_def, symbol, duration)
    return _result_to_dict(result, factor_def)


def run_factor_ranking(
    symbol: str,
    duration: str = "10m",
    category: str | None = None,
) -> list[dict[str, Any]]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")

    feature_frame = load_factor_frame(symbol, duration)

    factors = list_factors()
    if category:
        cat = FactorCategory(category)
        factors = [f for f in factors if f.category == cat]

    results = []
    for factor_def in factors:
        if factor_def.name not in feature_frame.columns:
            continue
        result = run_factor_backtest_on_frame(
            factor_def,
            feature_frame,
            symbol=symbol,
            duration=duration,
        )
        results.append(result)

    enrich_factor_results(results, frame=feature_frame)
    results.sort(key=lambda x: x.get("factorScore") or 0.0, reverse=True)
    return results


def _compute_factor_metrics(
    frame: pd.DataFrame,
    factor_def: FactorDefinition,
    symbol: str,
    duration: str,
) -> FactorBacktestResult:
    factor_name = factor_def.name
    horizon = 1
    df = backtest_duration_frame(frame, factor_name, duration)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[factor_name, "fwd_ret"])

    if len(df) < BACKTEST_MIN_PERIODS:
        return _empty_factor_result(factor_name, symbol, duration, len(df))

    ic_series = _compute_rolling_ic(df[factor_name], df["fwd_ret"])
    t_stat, p_value = _compute_ic_ttest(ic_series)
    quintile_returns = _compute_quintile_returns(df, factor_name)
    long_short = quintile_returns[-1] - quintile_returns[0] if len(quintile_returns) == QUINTILE_COUNT else None
    turnover = _compute_turnover(df, factor_name)
    sharpe, win_rate, max_drawdown, profit_factor = compute_signal_metrics(df, factor_def, horizon)
    ic_metrics = _ic_metrics(ic_series)

    return FactorBacktestResult(
        factor_name=factor_name,
        symbol=symbol.upper(),
        duration=duration,
        total_periods=len(df),
        ic_mean=ic_metrics["mean"],
        ic_std=ic_metrics["std"],
        ir=ic_metrics["ir"],
        ic_positive_rate=ic_metrics["positive_rate"],
        quintile_returns=quintile_returns,
        long_short_return=long_short,
        turnover=turnover,
        t_stat=t_stat,
        p_value=p_value,
        sharpe=sharpe,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
    )


def _empty_factor_result(
    factor_name: str,
    symbol: str,
    duration: str,
    total_periods: int,
) -> FactorBacktestResult:
    return FactorBacktestResult(
        factor_name=factor_name,
        symbol=symbol.upper(),
        duration=duration,
        total_periods=total_periods,
        ic_mean=None,
        ic_std=None,
        ir=None,
        ic_positive_rate=None,
        quintile_returns=[],
        long_short_return=None,
        turnover=None,
        t_stat=None,
        p_value=None,
        sharpe=None,
        win_rate=None,
        max_drawdown=None,
        profit_factor=None,
    )


def _ic_metrics(ic_series: pd.Series) -> dict[str, float | None]:
    ic_mean = float(ic_series.mean()) if not ic_series.empty else None
    ic_std = float(ic_series.std()) if not ic_series.empty else None
    ir = ic_mean / ic_std if ic_mean is not None and ic_std and ic_std > 0 else None
    positive_rate = float((ic_series > 0).mean()) if not ic_series.empty else None
    return {"mean": ic_mean, "std": ic_std, "ir": ir, "positive_rate": positive_rate}


def _compute_rolling_ic(factor: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    ranked = pd.DataFrame({
        "factor": factor.rank(method="average"),
        "fwd_ret": fwd_ret.rank(method="average"),
    })
    x = ranked["factor"]
    y = ranked["fwd_ret"]
    x_mean = x.rolling(IC_ROLLING_WINDOW).mean()
    y_mean = y.rolling(IC_ROLLING_WINDOW).mean()
    covariance = (x * y).rolling(IC_ROLLING_WINDOW).mean() - x_mean * y_mean
    x_var = ((x * x).rolling(IC_ROLLING_WINDOW).mean() - x_mean * x_mean).clip(lower=0.0)
    y_var = ((y * y).rolling(IC_ROLLING_WINDOW).mean() - y_mean * y_mean).clip(lower=0.0)
    denominator = np.sqrt(x_var * y_var).replace(0.0, np.nan)
    return (covariance / denominator).dropna()


def _compute_ic_ttest(ic_series: pd.Series) -> tuple[float | None, float | None]:
    clean = ic_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return None, None
    t_stat, p_value = stats.ttest_1samp(clean, 0)
    return float(t_stat), float(p_value)


def _compute_quintile_returns(df: pd.DataFrame, factor_name: str) -> list[float]:
    try:
        df = df.copy()
        df["quintile"] = pd.qcut(df[factor_name], QUINTILE_COUNT, labels=False, duplicates="drop")
        returns = df.groupby("quintile")["fwd_ret"].mean()
        return [float(returns.get(i, 0)) for i in range(QUINTILE_COUNT)]
    except Exception:
        return []


def _compute_turnover(df: pd.DataFrame, factor_name: str) -> float | None:
    try:
        df = df.copy()
        df["quintile"] = pd.qcut(df[factor_name], QUINTILE_COUNT, labels=False, duplicates="drop")
        changes = (df["quintile"] != df["quintile"].shift(1)).sum()
        return float(changes / len(df)) if len(df) > 0 else None
    except Exception:
        return None


def _result_to_dict(result: FactorBacktestResult, factor_def: FactorDefinition) -> dict[str, Any]:
    validity = backtest_validity({
        "totalPeriods": result.total_periods,
        "winRate": result.win_rate,
    })
    return {
        "factorName": result.factor_name,
        "factorDisplayName": factor_def.description,
        "symbol": result.symbol,
        "duration": result.duration,
        "category": factor_def.category.value,
        "categoryName": factor_payload(factor_def)["categoryName"],
        "description": factor_def.description,
        "formula": factor_def.formula,
        "direction": factor_def.direction.value,
        "totalPeriods": result.total_periods,
        "icMean": _round_or_none(result.ic_mean, 6),
        "icStd": _round_or_none(result.ic_std, 6),
        "ir": _round_or_none(result.ir, 4),
        "icPositiveRate": _round_or_none(result.ic_positive_rate, 4),
        "quintileReturns": [round(r, 6) for r in result.quintile_returns],
        "longShortReturn": _round_or_none(result.long_short_return, 6),
        "turnover": _round_or_none(result.turnover, 4),
        "tStat": _round_or_none(result.t_stat, 4),
        "pValue": _round_or_none(result.p_value, 6),
        "sharpe": _round_or_none(result.sharpe, 4),
        "winRate": _round_or_none(result.win_rate, 4),
        "maxDrawdown": _round_or_none(result.max_drawdown, 6),
        "profitFactor": _round_or_none(result.profit_factor, 4),
        "backtestValid": validity["valid"],
        "backtestInvalidReason": None if validity["valid"] else validity["reason"],
        "backtestMinPeriods": validity["minPeriods"],
    }


def _round_or_none(value: float | None, decimals: int) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(value, decimals)
