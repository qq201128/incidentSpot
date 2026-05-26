from __future__ import annotations

from typing import Any

ACTIVE_SAMPLE_COUNT = 30
ACTIVE_WIN_RATE_MIN = 0.62
MIN_PROFIT_FACTOR = 1.05
LOSS_STREAK_LIMIT = 5
RECENT_SAMPLE_COUNT = 20
RECENT_WIN_RATE_MIN = 0.58
RECENT_PROFIT_FACTOR_MIN = 1.0
ROLLING_WINDOW_SIZE = 10
ROLLING_WINDOW_COUNT = 3
ROLLING_WINDOW_WIN_RATE_MIN = 0.50


def high_winrate_thresholds() -> dict[str, Any]:
    return {
        "activeSampleCount": ACTIVE_SAMPLE_COUNT,
        "requiredSampleCount": ACTIVE_SAMPLE_COUNT,
        "activeWinRateMin": ACTIVE_WIN_RATE_MIN,
        "minProfitFactor": MIN_PROFIT_FACTOR,
        "lossStreakLimit": LOSS_STREAK_LIMIT,
        "recentSampleCount": RECENT_SAMPLE_COUNT,
        "recentWinRateMin": RECENT_WIN_RATE_MIN,
        "recentProfitFactorMin": RECENT_PROFIT_FACTOR_MIN,
        "rollingWindowSize": ROLLING_WINDOW_SIZE,
        "rollingWindowCount": ROLLING_WINDOW_COUNT,
        "rollingWindowWinRateMin": ROLLING_WINDOW_WIN_RATE_MIN,
    }


def empty_high_winrate_metrics() -> dict[str, Any]:
    return {
        "sampleCount": 0,
        "winRate": None,
        "profitFactor": None,
        "consecutiveLosses": 0,
        "currentConsecutiveWins": 0,
        "currentConsecutiveLosses": 0,
        "maxConsecutiveWins": 0,
        "maxConsecutiveLosses": 0,
        "latestRule": None,
        "metricsSource": "predictions",
        "totalEventPnlU": None,
        "paperStability": _empty_stability(),
    }


def high_winrate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [_return_value(row) for row in rows if _return_value(row) is not None]
    wins = sum(1 for row in rows if _is_win(row))
    streaks = _streak_metrics(rows)
    return {
        "sampleCount": len(rows),
        "winRate": _ratio(wins, len(rows)),
        "profitFactor": _profit_factor(returns),
        "consecutiveLosses": streaks["currentConsecutiveLosses"],
        **streaks,
        "latestRule": str(rows[0].get("high_winrate_rule") or "") if rows else None,
        "metricsSource": _metrics_source(rows),
        "totalEventPnlU": _total_event_pnl(rows),
        "paperStability": _paper_stability(rows),
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
    stability_reason = _stability_failure_reason(metrics["paperStability"])
    if stability_reason is not None:
        return {"status": "demoted", "reason": stability_reason}
    return {"status": "paper_live_passed", "reason": "stable_live_target_met"}


def _paper_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recent = rows[:RECENT_SAMPLE_COUNT]
    windows = _rolling_windows(rows)
    return {
        "recent": _sample_metrics(recent),
        "rollingWindows": [_sample_metrics(window) for window in windows],
        "thresholds": {
            "recentSampleCount": RECENT_SAMPLE_COUNT,
            "recentWinRateMin": RECENT_WIN_RATE_MIN,
            "recentProfitFactorMin": RECENT_PROFIT_FACTOR_MIN,
            "rollingWindowSize": ROLLING_WINDOW_SIZE,
            "rollingWindowCount": ROLLING_WINDOW_COUNT,
            "rollingWindowWinRateMin": ROLLING_WINDOW_WIN_RATE_MIN,
        },
    }


def _stability_failure_reason(stability: dict[str, Any]) -> str | None:
    recent = stability["recent"]
    if int(recent["sampleCount"]) < RECENT_SAMPLE_COUNT:
        return "recent_samples_below_min"
    if _lt(recent["winRate"], RECENT_WIN_RATE_MIN):
        return "recent_win_rate_below_target"
    if _lt(recent["profitFactor"], RECENT_PROFIT_FACTOR_MIN):
        return "recent_profit_factor_below_target"
    return _rolling_failure_reason(stability["rollingWindows"])


def _rolling_failure_reason(windows: list[dict[str, Any]]) -> str | None:
    if len(windows) < ROLLING_WINDOW_COUNT:
        return "rolling_windows_below_min"
    for row in windows:
        if int(row["sampleCount"]) < ROLLING_WINDOW_SIZE:
            return "rolling_window_samples_below_min"
        if _lt(row["winRate"], ROLLING_WINDOW_WIN_RATE_MIN):
            return "rolling_window_win_rate_below_target"
    return None


def _profit_factor(values: list[float]) -> float | None:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if not values or losses == 0:
        return None
    return round(gains / losses, 4)


def _rolling_windows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    limit = ROLLING_WINDOW_SIZE * ROLLING_WINDOW_COUNT
    recent = rows[:limit]
    windows = [recent[start : start + ROLLING_WINDOW_SIZE] for start in range(0, len(recent), ROLLING_WINDOW_SIZE)]
    return windows[:ROLLING_WINDOW_COUNT]


def _sample_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [_return_value(row) for row in rows if _return_value(row) is not None]
    wins = sum(1 for row in rows if _is_win(row))
    return {
        "sampleCount": len(rows),
        "winRate": _ratio(wins, len(rows)),
        "profitFactor": _profit_factor(returns),
    }


def _empty_stability() -> dict[str, Any]:
    return {
        "recent": {"sampleCount": 0, "winRate": None, "profitFactor": None},
        "rollingWindows": [],
        "thresholds": high_winrate_thresholds(),
    }


def _streak_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    current = _current_streak(rows)
    max_streaks = _max_streaks(rows)
    return {
        "currentConsecutiveWins": current["wins"],
        "currentConsecutiveLosses": current["losses"],
        "maxConsecutiveWins": max_streaks["wins"],
        "maxConsecutiveLosses": max_streaks["losses"],
    }


def _current_streak(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        return {"wins": 0, "losses": 0}
    first_is_win = _is_win(rows[0])
    count = _same_outcome_prefix_count(rows, first_is_win)
    return {"wins": count if first_is_win else 0, "losses": 0 if first_is_win else count}


def _same_outcome_prefix_count(rows: list[dict[str, Any]], expected_win: bool) -> int:
    total = 0
    for row in rows:
        if _is_win(row) != expected_win:
            break
        total += 1
    return total


def _max_streaks(rows: list[dict[str, Any]]) -> dict[str, int]:
    best_wins = 0
    best_losses = 0
    current_wins = 0
    current_losses = 0
    for row in rows:
        if _is_win(row):
            current_wins += 1
            current_losses = 0
        else:
            current_losses += 1
            current_wins = 0
        best_wins = max(best_wins, current_wins)
        best_losses = max(best_losses, current_losses)
    return {"wins": best_wins, "losses": best_losses}


def _is_win(row: dict[str, Any]) -> bool:
    if row.get("event_pnl") is not None:
        return float(row["event_pnl"]) > 0
    return bool(row.get("prediction_correct"))


def _return_value(row: dict[str, Any]) -> float | None:
    if row.get("actual_return") is not None:
        return float(row["actual_return"])
    pnl = row.get("event_pnl")
    qty = row.get("order_qty") or 1
    if pnl is None:
        return None
    try:
        return float(pnl) / float(qty)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _metrics_source(rows: list[dict[str, Any]]) -> str:
    if any(row.get("event_pnl") is not None for row in rows):
        return "events"
    return "predictions"


def _total_event_pnl(rows: list[dict[str, Any]]) -> float | None:
    values = [float(row["event_pnl"]) for row in rows if row.get("event_pnl") is not None]
    if not values:
        return None
    return round(sum(values), 6)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 4)


def _lt(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold
