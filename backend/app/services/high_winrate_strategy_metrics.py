from __future__ import annotations

from typing import Any

ACTIVE_SAMPLE_COUNT = 20
ACTIVE_WIN_RATE_MIN = 0.70
MIN_PROFIT_FACTOR = 1.0
LOSS_STREAK_LIMIT = 5


def high_winrate_thresholds() -> dict[str, Any]:
    return {
        "activeSampleCount": ACTIVE_SAMPLE_COUNT,
        "requiredSampleCount": ACTIVE_SAMPLE_COUNT,
        "activeWinRateMin": ACTIVE_WIN_RATE_MIN,
        "minProfitFactor": MIN_PROFIT_FACTOR,
        "lossStreakLimit": LOSS_STREAK_LIMIT,
    }


def empty_high_winrate_metrics() -> dict[str, Any]:
    return {"sampleCount": 0, "winRate": None, "profitFactor": None, "consecutiveLosses": 0, "latestRule": None}


def high_winrate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(row["actual_return"]) for row in rows if row.get("actual_return") is not None]
    wins = sum(1 for row in rows if bool(row.get("prediction_correct")))
    return {
        "sampleCount": len(rows),
        "winRate": _ratio(wins, len(rows)),
        "profitFactor": _profit_factor(returns),
        "consecutiveLosses": _consecutive_losses(rows),
        "latestRule": str(rows[0].get("high_winrate_rule") or "") if rows else None,
    }


def high_winrate_decision(metrics: dict[str, Any]) -> dict[str, str]:
    if metrics["consecutiveLosses"] >= LOSS_STREAK_LIMIT:
        return {"status": "demoted", "reason": "consecutive_losses"}
    if metrics["sampleCount"] < ACTIVE_SAMPLE_COUNT:
        return {"status": "paper_live_collecting", "reason": "insufficient_settled_samples"}
    if _lt(metrics["winRate"], ACTIVE_WIN_RATE_MIN):
        return {"status": "demoted", "reason": "live_win_rate_below_target"}
    if _lt(metrics["profitFactor"], MIN_PROFIT_FACTOR):
        return {"status": "demoted", "reason": "profit_factor_below_one"}
    return {"status": "paper_live_passed", "reason": "stable_live_target_met"}


def _profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if not values or losses == 0:
        return None
    return round(gains / losses, 4)


def _consecutive_losses(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        if bool(row.get("prediction_correct")):
            break
        count += 1
    return count


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 4)


def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold
