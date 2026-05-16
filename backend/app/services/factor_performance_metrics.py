from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.services.factor_registry import FactorDefinition
from app.services.trading_costs import ROUNDTRIP_COST_RATE

BACKTEST_MIN_PERIODS = 100


def compute_signal_metrics(
    df: pd.DataFrame,
    factor_def: FactorDefinition,
    horizon: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    returns = signal_returns(df, factor_def)
    if returns.empty:
        return None, None, None, None
    return (
        _sharpe(returns, horizon),
        float((returns > 0).mean()),
        _max_drawdown(returns),
        _profit_factor(returns),
    )


def signal_returns(df: pd.DataFrame, factor_def: FactorDefinition) -> pd.Series:
    factor = df[factor_def.name].astype(float)
    median = factor.expanding(min_periods=BACKTEST_MIN_PERIODS).median().shift(1)
    signal = pd.Series(np.where(factor >= median, 1.0, -1.0), index=df.index)
    if factor_def.direction.value == "lower_better":
        signal = -signal
    returns = (signal * df["fwd_ret"].astype(float) - ROUNDTRIP_COST_RATE).replace([np.inf, -np.inf], np.nan)
    return returns.loc[returns.index.isin(median.dropna().index)].dropna()


def add_contribution_scores(results: list[dict], *, duration_scoped: bool = False) -> None:
    totals: dict[str, float] = {}
    for row in results:
        key = str(row.get("duration")) if duration_scoped else "all"
        totals[key] = totals.get(key, 0.0) + abs(float(row.get("ir") or 0.0))
    for row in results:
        key = str(row.get("duration")) if duration_scoped else "all"
        total = totals.get(key, 0.0)
        value = abs(float(row.get("ir") or 0.0))
        row["contribution"] = round(value / total, 6) if total > 0 else None


def _sharpe(returns: pd.Series, horizon: int) -> float | None:
    std = float(returns.std())
    if std <= 0:
        return None
    periods_per_year = 365 * 24 * 60 / max(int(horizon), 1)
    return float(returns.mean() / std * math.sqrt(periods_per_year))


def _max_drawdown(returns: pd.Series) -> float | None:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else None


def _profit_factor(returns: pd.Series) -> float | None:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    return gains / losses if losses > 0 else None
