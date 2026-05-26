from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.factor_registry import FactorDefinition
from app.services.return_metric_policy import ReturnMetricPolicy
from app.services.trading_costs import roundtrip_cost_rate

BACKTEST_MIN_PERIODS = 100
MINUTES_PER_YEAR = 365 * 24 * 60


def compute_signal_metrics(
    df: pd.DataFrame,
    factor_def: FactorDefinition,
    horizon: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    returns = signal_returns(df, factor_def)
    if returns.empty:
        return None, None, None, None
    metrics = ReturnMetricPolicy(cost_rate=roundtrip_cost_rate()).from_returns(
        returns,
        annualization=MINUTES_PER_YEAR / max(int(horizon), 1),
    )
    return (
        metrics["sharpe"],
        metrics["winRate"],
        metrics["maxDrawdown"],
        metrics["profitFactor"],
    )


def signal_returns(df: pd.DataFrame, factor_def: FactorDefinition) -> pd.Series:
    factor = df[factor_def.name].astype(float)
    median = factor.expanding(min_periods=BACKTEST_MIN_PERIODS).median().shift(1)
    signal = pd.Series(np.where(factor >= median, 1.0, -1.0), index=df.index)
    if factor_def.direction.value == "lower_better":
        signal = -signal
    returns = (signal * df["fwd_ret"].astype(float) - roundtrip_cost_rate()).replace([np.inf, -np.inf], np.nan)
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
