from __future__ import annotations

from app.services.paper_live_candidate_ranking import candidate_rank_key, candidate_robust_score


def test_collecting_candidates_rank_by_robust_score_before_sample_count() -> None:
    fragile = _candidate(
        "sample_rich_but_fragile",
        sample_count=120,
        win_rate=0.58,
        backtest_win_rate=0.75,
        profit_factor=1.02,
        avg_return=-0.002,
        max_consecutive_losses=5,
        recent_rates=(0.50, 0.55, 0.58),
        rolling_rates=(0.40, 0.50, 0.60),
    )
    balanced = _candidate(
        "balanced_lower_sample",
        sample_count=54,
        win_rate=0.62,
        backtest_win_rate=0.63,
        profit_factor=1.45,
        avg_return=0.002,
        max_consecutive_losses=1,
        recent_rates=(0.63, 0.62, 0.62),
        rolling_rates=(0.60, 0.60, 0.70),
    )

    for candidate in (fragile, balanced):
        score = candidate_robust_score(candidate)
        candidate["robustScore"] = score["score"]
        candidate["scoreBreakdown"] = score

    ranked = sorted([fragile, balanced], key=candidate_rank_key, reverse=True)

    assert ranked[0]["candidateKey"] == "balanced_lower_sample"
    assert balanced["robustScore"] > fragile["robustScore"]
    assert fragile["metrics"]["sampleCount"] > balanced["metrics"]["sampleCount"]


def _candidate(
    key: str,
    *,
    sample_count: int,
    win_rate: float,
    backtest_win_rate: float,
    profit_factor: float,
    avg_return: float,
    max_consecutive_losses: int,
    recent_rates: tuple[float, float, float],
    rolling_rates: tuple[float, float, float],
) -> dict:
    return {
        "candidateKey": key,
        "status": "paper_collecting",
        "backtestWinRate": backtest_win_rate,
        "paperLiveWinRate": win_rate,
        "paperLiveSampleCount": sample_count,
        "oosWinRate": 0.6,
        "walkForwardResult": {"status": "passed", "score": 0.5},
        "recentRollingResult": {"paperStability": {"rollingWindows": []}},
        "metrics": {
            "sampleCount": sample_count,
            "winRate": win_rate,
            "profitFactor": profit_factor,
            "avgReturn": avg_return,
            "maxConsecutiveLosses": max_consecutive_losses,
            "paperLiveWindows": {
                "recent30": {"sampleCount": min(sample_count, 30), "winRate": recent_rates[0]},
                "recent60": {"sampleCount": min(sample_count, 60), "winRate": recent_rates[1]},
                "recent100": {"sampleCount": min(sample_count, 100), "winRate": recent_rates[2]},
            },
            "paperStability": {
                "rollingWindows": [
                    {"sampleCount": 10, "winRate": rate}
                    for rate in rolling_rates
                ],
            },
        },
    }
