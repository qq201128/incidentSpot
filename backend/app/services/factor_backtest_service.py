"""
Factor Backtest Service - Calculate factor performance metrics.

Metrics computed:
- IC (Information Coefficient): Rank correlation between factor values and forward returns
- IR (Information Ratio): IC_mean / IC_std, measures consistency
- Factor Returns: Quintile-based long-short returns
- Turnover: How often factor-based positions change
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.db.session import get_conn
from app.services.factor_registry import (
    FACTOR_BY_NAME,
    FactorDefinition,
    get_factor,
    list_factors,
)
from app.services.kline_features import FEATURE_COLUMNS, build_feature_frame
from app.services.rule_config import DURATION_TO_MINUTES, MS_PER_MINUTE, SUPPORTED_RULE_DURATIONS

BACKTEST_MIN_PERIODS = 100
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

    klines = _load_klines(symbol)
    feature_frame, _ = build_feature_frame(klines)

    if factor_name not in feature_frame.columns:
        raise ValueError(f"factor {factor_name} not found in feature frame")

    result = _compute_factor_metrics(feature_frame, factor_name, symbol, duration)
    return _result_to_dict(result, factor_def)


def run_factor_ranking(
    symbol: str,
    duration: str = "10m",
    category: str | None = None,
) -> list[dict[str, Any]]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")

    klines = _load_klines(symbol)
    feature_frame, _ = build_feature_frame(klines)

    factors = list_factors()
    if category:
        from app.services.factor_registry import FactorCategory
        cat = FactorCategory(category)
        factors = [f for f in factors if f.category == cat]

    results = []
    for factor_def in factors:
        if factor_def.name not in feature_frame.columns:
            continue
        try:
            result = _compute_factor_metrics(feature_frame, factor_def.name, symbol, duration)
            results.append(_result_to_dict(result, factor_def))
        except Exception:
            continue

    results.sort(key=lambda x: abs(x.get("ir") or 0), reverse=True)
    return results


def _load_klines(symbol: str) -> pd.DataFrame:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT open_time, open, high, low, close, volume
        FROM klines
        WHERE symbol = ? AND interval = '1m'
        ORDER BY open_time ASC
        """,
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    if not rows:
        raise ValueError(f"no 1m klines found for {symbol.upper()}")
    frame = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    return frame


def _compute_factor_metrics(
    frame: pd.DataFrame,
    factor_name: str,
    symbol: str,
    duration: str,
) -> FactorBacktestResult:
    horizon = DURATION_TO_MINUTES.get(duration, 10)
    df = frame.copy()
    df["fwd_ret"] = df["close"].pct_change(horizon).shift(-horizon)
    df = df.dropna(subset=[factor_name, "fwd_ret"])

    if len(df) < BACKTEST_MIN_PERIODS:
        return FactorBacktestResult(
            factor_name=factor_name,
            symbol=symbol.upper(),
            duration=duration,
            total_periods=len(df),
            ic_mean=None,
            ic_std=None,
            ir=None,
            ic_positive_rate=None,
            quintile_returns=[],
            long_short_return=None,
            turnover=None,
            t_stat=None,
            p_value=None,
        )

    ic_series = _compute_rolling_ic(df[factor_name], df["fwd_ret"])
    ic_mean = float(ic_series.mean()) if not ic_series.empty else None
    ic_std = float(ic_series.std()) if not ic_series.empty else None
    ir = ic_mean / ic_std if ic_mean is not None and ic_std and ic_std > 0 else None
    ic_positive_rate = float((ic_series > 0).mean()) if not ic_series.empty else None

    t_stat, p_value = _compute_ic_ttest(ic_series)

    quintile_returns = _compute_quintile_returns(df, factor_name)
    long_short = quintile_returns[-1] - quintile_returns[0] if len(quintile_returns) == QUINTILE_COUNT else None

    turnover = _compute_turnover(df, factor_name)

    return FactorBacktestResult(
        factor_name=factor_name,
        symbol=symbol.upper(),
        duration=duration,
        total_periods=len(df),
        ic_mean=ic_mean,
        ic_std=ic_std,
        ir=ir,
        ic_positive_rate=ic_positive_rate,
        quintile_returns=quintile_returns,
        long_short_return=long_short,
        turnover=turnover,
        t_stat=t_stat,
        p_value=p_value,
    )


def _compute_rolling_ic(factor: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    def _rank_corr(sub: pd.DataFrame) -> float:
        if len(sub) < 10:
            return np.nan
        corr, _ = stats.spearmanr(sub["factor"], sub["fwd_ret"])
        return corr

    df = pd.DataFrame({"factor": factor, "fwd_ret": fwd_ret})
    ic_vals = df.rolling(IC_ROLLING_WINDOW).apply(
        lambda x: _rank_corr(df.loc[x.index]), raw=False
    )
    # Simplified: compute per-period rank correlation
    ic_list = []
    for i in range(IC_ROLLING_WINDOW, len(df)):
        sub = df.iloc[i - IC_ROLLING_WINDOW : i]
        corr, _ = stats.spearmanr(sub["factor"], sub["fwd_ret"])
        ic_list.append(corr)
    return pd.Series(ic_list)


def _compute_ic_ttest(ic_series: pd.Series) -> tuple[float | None, float | None]:
    clean = ic_series.dropna()
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
    return {
        "factorName": result.factor_name,
        "symbol": result.symbol,
        "duration": result.duration,
        "category": factor_def.category.value,
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
    }


def _round_or_none(value: float | None, decimals: int) -> float | None:
    return round(value, decimals) if value is not None else None
