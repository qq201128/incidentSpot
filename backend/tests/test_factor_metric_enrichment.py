from __future__ import annotations

from app.services.factor_metric_enrichment import backtest_validity, factor_score
from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS


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
