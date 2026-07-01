from __future__ import annotations

from app.services.high_winrate_combo_view import build_high_winrate_combo_view, regular_ranking_view


def test_build_high_winrate_combo_view_adds_daily_trade_metrics() -> None:
    payload = {
        "updatedAt": "2026-05-15T15:40:51.717294+00:00",
        "search": {"entryRows": 438},
        "ranking": [
            {
                "factorName": "goal_combo__alpha__beta",
                "trades": 65,
                "winRate": 0.7538,
            }
        ],
    }

    result = build_high_winrate_combo_view(payload, "10m")

    assert result["highWinrateTotal"] == 1
    assert result["highWinrateSummary"]["sampleDays"] == 3.0417
    assert result["highWinrateSummary"]["topStrategyAvgTradesPerDay"] == 21.37
    assert result["highWinrateRanking"][0]["avgTradesPerDay"] == 21.37
    assert result["highWinrateRanking"][0]["strategyBucket"] == "high_winrate_goal"


def test_regular_ranking_view_adds_daily_trade_metrics_when_trades_exist() -> None:
    payload = {"search": {"entryRows": 438}}
    rows = [{"factorName": "combo__alpha__beta", "trades": 44, "totalPeriods": 438}]

    result = regular_ranking_view(rows, payload, "10m")

    assert result[0]["strategyBucket"] == "regular_combo"
    assert result[0]["sampleDays"] == 3.0417
    assert result[0]["avgTradesPerDay"] == 14.47
