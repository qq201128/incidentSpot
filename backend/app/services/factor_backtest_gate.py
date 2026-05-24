from __future__ import annotations

from typing import Any

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN, finite


def meets_backtest_gate(row: dict[str, Any]) -> bool:
    win_rate = finite(row.get("winRate"))
    profit_factor = finite(row.get("profitFactor"))
    if win_rate is None or profit_factor is None:
        return False
    total_periods = int(row.get("totalPeriods") or row.get("trades") or 0)
    return (
        win_rate >= SUCCESS_WIN_RATE_MIN
        and profit_factor >= SUCCESS_PROFIT_FACTOR_MIN
        and total_periods >= BACKTEST_MIN_PERIODS
    )


def backtest_gate_thresholds() -> dict[str, float | int]:
    return {
        "minWinRate": SUCCESS_WIN_RATE_MIN,
        "minProfitFactor": SUCCESS_PROFIT_FACTOR_MIN,
        "minTotalPeriods": BACKTEST_MIN_PERIODS,
    }
