from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Iterable

import numpy as np
import pandas as pd


METRIC_DECIMALS = 6
EPSILON = 1e-12


@dataclass(frozen=True)
class ReturnMetricPolicy:
    cost_rate: float = 0.0
    sample_count_key: str = "sampleCount"

    def from_returns(self, returns: Iterable[float] | np.ndarray | pd.Series, *, annualization: float | None = None) -> dict[str, Any]:
        values = _clean_returns(returns)
        if len(values) == 0:
            return self.empty_metrics()
        return {
            self.sample_count_key: int(len(values)),
            "winRate": float(np.mean(values > 0)),
            "profitFactor": profit_factor(values),
            "avgReturn": float(np.mean(values)),
            "sharpe": sharpe_ratio(values, annualization),
            "maxDrawdown": max_drawdown(values),
            "totalCost": float(len(values) * self.cost_rate),
        }

    def empty_metrics(self) -> dict[str, Any]:
        return {
            self.sample_count_key: 0,
            "winRate": None,
            "profitFactor": None,
            "avgReturn": None,
            "sharpe": None,
            "maxDrawdown": None,
            "totalCost": 0.0,
        }


@dataclass(frozen=True)
class ValidationGate:
    min_sample_count: int
    min_win_rate: float
    min_profit_factor: float
    min_avg_return: float = 0.0
    strict: bool = True

    def failure_reason(self, metrics: dict[str, Any], *, prefix: str) -> str | None:
        if int(metrics.get("sampleCount") or 0) < self.min_sample_count:
            return f"{prefix}_sample_count_below_min"
        if not _passes(metrics.get("winRate"), self.min_win_rate, self.strict):
            return f"{prefix}_win_rate_below_min"
        if not _passes(metrics.get("profitFactor"), self.min_profit_factor, self.strict):
            return f"{prefix}_profit_factor_below_min"
        if not _passes(metrics.get("avgReturn"), self.min_avg_return, self.strict):
            return f"{prefix}_avg_return_below_min"
        return None


def profit_factor(returns: Iterable[float] | np.ndarray | pd.Series) -> float | None:
    values = _clean_returns(returns)
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values <= 0].sum()))
    if gains <= 0:
        return 0.0
    if losses == 0:
        return float("inf")
    return gains / losses


def max_drawdown(returns: Iterable[float] | np.ndarray | pd.Series) -> float | None:
    values = _clean_returns(returns)
    if len(values) == 0:
        return None
    equity = np.cumprod(1.0 + values)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: Iterable[float] | np.ndarray | pd.Series, annualization: float | None = None) -> float | None:
    values = _clean_returns(returns)
    if len(values) < 2:
        return None
    std = float(np.std(values, ddof=1))
    if std < EPSILON:
        return None
    scale = sqrt(float(annualization)) if annualization is not None else sqrt(len(values))
    return float(np.mean(values) / std * scale)


def rounded_metrics(metrics: dict[str, Any], decimals: int = METRIC_DECIMALS) -> dict[str, Any]:
    return {key: _round_value(value, decimals) for key, value in metrics.items()}


def _clean_returns(returns: Iterable[float] | np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(list(returns) if not isinstance(returns, (np.ndarray, pd.Series)) else returns, dtype=float)
    return values[np.isfinite(values)]


def _passes(value: Any, threshold: float, strict: bool) -> bool:
    if value is None:
        return False
    number = float(value)
    if number == float("inf"):
        return True
    if not isfinite(number):
        return False
    return number > threshold if strict else number >= threshold


def _round_value(value: Any, decimals: int) -> Any:
    if isinstance(value, float) and isfinite(value):
        return round(value, decimals)
    return value
