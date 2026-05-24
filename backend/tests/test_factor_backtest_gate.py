from __future__ import annotations

from app.services.factor_backtest_gate import backtest_gate_thresholds, meets_backtest_gate


def test_meets_backtest_gate_requires_win_rate_profit_factor_and_periods() -> None:
    assert meets_backtest_gate({"winRate": 0.62, "profitFactor": 1.05, "totalPeriods": 100}) is True
    assert meets_backtest_gate({"winRate": 0.61, "profitFactor": 1.05, "totalPeriods": 100}) is False
    assert meets_backtest_gate({"winRate": 0.62, "profitFactor": 1.04, "totalPeriods": 100}) is False
    assert meets_backtest_gate({"winRate": 0.62, "profitFactor": 1.05, "totalPeriods": 99}) is False


def test_backtest_gate_thresholds_match_global_constants() -> None:
    thresholds = backtest_gate_thresholds()
    assert thresholds["minWinRate"] == 0.62
    assert thresholds["minProfitFactor"] == 1.05
    assert thresholds["minTotalPeriods"] == 100
