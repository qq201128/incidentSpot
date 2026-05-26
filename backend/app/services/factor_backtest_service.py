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

from app.services.factor_duration_alignment import backtest_duration_frame
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_metric_enrichment import backtest_validity, enrich_factor_results
from app.services.factor_catalog import factor_definition_for_backtest
from app.services.factor_out_of_sample import factor_out_of_sample_report
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS, compute_signal_metrics
from app.services.factor_research_metrics import (
    IC_ROLLING_WINDOW,
    QUINTILE_COUNT,
    compute_ic_ttest,
    compute_quintile_returns,
    compute_rolling_ic,
    compute_turnover,
    ic_metrics,
    window_spearman_ic,
)
from app.services.factor_registry import (
    FactorCategory,
    FactorDefinition,
    factor_payload,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


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
    out_of_sample: dict[str, Any]


def run_factor_backtest(
    factor_name: str,
    symbol: str,
    duration: str = "10m",
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    factor_def = factor_definition_for_backtest(factor_name, symbol, duration)

    feature_frame = load_factor_frame(symbol, duration)
    from app.services.factor_backtest_materialization import materialized_frame_for_factor

    feature_frame = materialized_frame_for_factor(feature_frame, factor_def, symbol, duration)
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
    from app.services.factor_ranking_calculation import run_factor_ranking as run_ranking

    return run_ranking(symbol, duration, category)


def run_factor_ranking_report(
    symbol: str,
    duration: str = "10m",
    category: str | None = None,
) -> dict[str, Any]:
    from app.services.factor_ranking_calculation import run_factor_ranking_report as run_report

    return run_report(symbol, duration, category)


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

    research = _research_metrics(df, factor_def)
    quintile_returns = research["quintile_returns"]
    long_short = quintile_returns[-1] - quintile_returns[0] if len(quintile_returns) == QUINTILE_COUNT else None
    sharpe, win_rate, max_drawdown, profit_factor = compute_signal_metrics(df, factor_def, horizon)

    return FactorBacktestResult(
        factor_name=factor_name,
        symbol=symbol.upper(),
        duration=duration,
        total_periods=len(df),
        ic_mean=research["ic"]["mean"],
        ic_std=research["ic"]["std"],
        ir=research["ic"]["ir"],
        ic_positive_rate=research["ic"]["positive_rate"],
        quintile_returns=quintile_returns,
        long_short_return=long_short,
        turnover=research["turnover"],
        t_stat=research["t_stat"],
        p_value=research["p_value"],
        sharpe=sharpe,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        out_of_sample=_out_of_sample_report(df, factor_def),
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
        out_of_sample={},
    )


def _out_of_sample_report(df: pd.DataFrame, factor_def: FactorDefinition) -> dict[str, Any]:
    if factor_def.category == FactorCategory.PERFORMANCE:
        return {}
    return factor_out_of_sample_report(df, factor_def)


def _research_metrics(df: pd.DataFrame, factor_def: FactorDefinition) -> dict[str, Any]:
    if factor_def.category == FactorCategory.PERFORMANCE:
        return _empty_research_metrics()
    ic_series = compute_rolling_ic(df[factor_def.name], df["fwd_ret"])
    t_stat, p_value = compute_ic_ttest(ic_series)
    return {
        "ic": ic_metrics(ic_series),
        "t_stat": t_stat,
        "p_value": p_value,
        "quintile_returns": compute_quintile_returns(df, factor_def.name),
        "turnover": compute_turnover(df, factor_def.name),
    }


def _empty_research_metrics() -> dict[str, Any]:
    return {
        "ic": {"mean": None, "std": None, "ir": None, "positive_rate": None},
        "t_stat": None,
        "p_value": None,
        "quintile_returns": [],
        "turnover": None,
    }


_compute_rolling_ic = compute_rolling_ic
_window_spearman_ic = window_spearman_ic


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
        "sourceFile": factor_def.source_file,
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
        "outOfSample": result.out_of_sample,
        "backtestValid": validity["valid"],
        "backtestInvalidReason": None if validity["valid"] else validity["reason"],
        "backtestMinPeriods": validity["minPeriods"],
    }


def _round_or_none(value: float | None, decimals: int) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(value, decimals)
