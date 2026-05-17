from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.factor_metric_enrichment import backtest_validity, factor_score
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS, compute_signal_metrics
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection
from app.services.trading_costs import ROUNDTRIP_COST_RATE


def test_insufficient_backtest_periods_cannot_receive_factor_score() -> None:
    row = {
        "totalPeriods": BACKTEST_MIN_PERIODS - 1,
        "winRate": 1.0,
        "sharpe": 4.0,
        "ir": 3.0,
        "profitFactor": 10.0,
    }

    assert backtest_validity(row)["reason"] == "insufficient_periods"
    assert factor_score(row) == 0.0


def test_missing_correlation_does_not_create_score_by_itself() -> None:
    row = {
        "totalPeriods": BACKTEST_MIN_PERIODS,
        "winRate": 0.0,
        "sharpe": None,
        "ir": None,
        "profitFactor": None,
        "avgAbsCorrelation": None,
    }

    assert backtest_validity(row)["valid"] is True
    assert factor_score(row) == 0.0


def test_negative_sharpe_and_ir_do_not_boost_factor_score() -> None:
    negative = {
        "totalPeriods": BACKTEST_MIN_PERIODS,
        "winRate": 0.18,
        "sharpe": -4.0,
        "ir": -1.2,
        "profitFactor": 0.3,
        "contribution": 0.8,
        "avgAbsCorrelation": 0.2,
    }
    positive = {**negative, "winRate": 0.55, "sharpe": 1.2, "ir": 0.8, "profitFactor": 1.4}

    assert factor_score(negative) < factor_score(positive)
    assert factor_score(negative) < 20.0


def test_factor_signal_metrics_subtract_roundtrip_cost_from_returns() -> None:
    rows = BACKTEST_MIN_PERIODS + 20
    df = pd.DataFrame(
        {
            "factor_a": np.arange(rows, dtype=float),
            "fwd_ret": np.full(rows, ROUNDTRIP_COST_RATE / 2, dtype=float),
        }
    )
    factor = FactorDefinition(
        name="factor_a",
        category=FactorCategory.RETURN,
        description="factor_a",
        formula="factor_a",
        direction=FactorDirection.HIGHER_BETTER,
    )

    _sharpe, win_rate, max_drawdown, profit_factor = compute_signal_metrics(df, factor, horizon=1)

    assert win_rate == pytest.approx(0.0)
    assert profit_factor == pytest.approx(0.0)
    assert max_drawdown is not None
    assert max_drawdown < 0
