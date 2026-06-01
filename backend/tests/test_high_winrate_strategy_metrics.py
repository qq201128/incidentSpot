from __future__ import annotations

import pytest

from app.services.high_winrate_strategy_metrics import (
    empty_high_winrate_metrics,
    high_winrate_decision,
    high_winrate_metrics,
)


def test_high_winrate_metrics_reports_current_and_max_streaks() -> None:
    rows = _pnl_rows([-1, -1, 1, 1, 1, -1, -1, -1, 1])

    metrics = high_winrate_metrics(rows)

    assert metrics["currentConsecutiveWins"] == 0
    assert metrics["currentConsecutiveLosses"] == 2
    assert metrics["consecutiveLosses"] == 2
    assert metrics["maxConsecutiveWins"] == 3
    assert metrics["maxConsecutiveLosses"] == 3


def test_empty_high_winrate_metrics_has_zero_streaks() -> None:
    metrics = empty_high_winrate_metrics()

    assert metrics["currentConsecutiveWins"] == 0
    assert metrics["currentConsecutiveLosses"] == 0
    assert metrics["maxConsecutiveWins"] == 0
    assert metrics["maxConsecutiveLosses"] == 0


def test_positive_paper_live_status_requires_positive_event_pnl() -> None:
    rows = _pnl_rows(([0.8] * 6 + [-1.0] * 4) * 3)

    metrics = high_winrate_metrics(rows)
    decision = high_winrate_decision(metrics)

    assert metrics["winRate"] == 0.6
    assert metrics["profitFactor"] == 1.2
    assert metrics["totalEventPnlU"] == pytest.approx(2.4)
    assert decision["status"] == "paper_stable"


def test_non_positive_event_pnl_blocks_paper_live_stable_status() -> None:
    rows = _return_and_pnl_rows(([0.8] * 6 + [-1.0] * 4) * 3, ([0.1] * 6 + [-1.0] * 4) * 3)

    metrics = high_winrate_metrics(rows)
    decision = high_winrate_decision(metrics)

    assert metrics["winRate"] > 0.58
    assert metrics["profitFactor"] >= 1.0
    assert metrics["totalEventPnlU"] < 0
    assert decision == {"status": "paper_failed", "reason": "paper_live_total_pnl_below_target"}


def _pnl_rows(values: list[float]) -> list[dict[str, float]]:
    return [{"event_pnl": value} for value in values]


def _return_and_pnl_rows(returns: list[float], pnl_values: list[float]) -> list[dict[str, float]]:
    return [{"actual_return": ret, "event_pnl": pnl} for ret, pnl in zip(returns, pnl_values)]
