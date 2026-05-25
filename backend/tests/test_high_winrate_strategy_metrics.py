from __future__ import annotations

from app.services.high_winrate_strategy_metrics import (
    empty_high_winrate_metrics,
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


def _pnl_rows(values: list[float]) -> list[dict[str, float]]:
    return [{"event_pnl": value} for value in values]
