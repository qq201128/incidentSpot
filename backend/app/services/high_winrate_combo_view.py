from __future__ import annotations

from typing import Any

from app.services.rule_config import DURATION_TO_MINUTES

DAY_MINUTES = 1_440
HIGH_WINRATE_BUCKET = "high_winrate_goal"
REGULAR_BUCKET = "regular_combo"


def build_high_winrate_combo_view(payload: dict[str, Any] | None, duration: str) -> dict[str, Any]:
    if payload is None:
        return _empty_high_winrate_view()
    sample_days = _sample_days(payload, duration)
    ranking = [_goal_row_view(row, sample_days) for row in _ranking_rows(payload)]
    top = ranking[0] if ranking else {}
    return {
        "highWinrateRanking": ranking,
        "highWinrateTotal": len(ranking),
        "highWinrateUpdatedAt": payload.get("updatedAt"),
        "highWinrateSummary": {
            "count": len(ranking),
            "sampleDays": sample_days,
            "topStrategyFactorName": top.get("factorName"),
            "topStrategyAvgTradesPerDay": top.get("avgTradesPerDay"),
            "topStrategyTrades": top.get("trades"),
        },
    }


def regular_ranking_view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "strategyBucket": REGULAR_BUCKET} for row in rows]


def _empty_high_winrate_view() -> dict[str, Any]:
    return {
        "highWinrateRanking": [],
        "highWinrateTotal": 0,
        "highWinrateUpdatedAt": None,
        "highWinrateSummary": {
            "count": 0,
            "sampleDays": None,
            "topStrategyFactorName": None,
            "topStrategyAvgTradesPerDay": None,
            "topStrategyTrades": None,
        },
    }


def _goal_row_view(row: dict[str, Any], sample_days: float | None) -> dict[str, Any]:
    trades = _int_or_none(row.get("trades"))
    return {
        **row,
        "strategyBucket": HIGH_WINRATE_BUCKET,
        "sampleDays": sample_days,
        "avgTradesPerDay": _avg_trades_per_day(trades, sample_days),
    }


def _ranking_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = payload.get("ranking")
    return ranking if isinstance(ranking, list) else []


def _sample_days(payload: dict[str, Any], duration: str) -> float | None:
    search = payload.get("search")
    entry_rows = _int_or_none(search.get("entryRows")) if isinstance(search, dict) else None
    minutes = DURATION_TO_MINUTES.get(duration)
    if entry_rows is None or entry_rows <= 0 or minutes is None:
        return None
    return round(entry_rows * minutes / DAY_MINUTES, 4)


def _avg_trades_per_day(trades: int | None, sample_days: float | None) -> float | None:
    if trades is None or sample_days is None or sample_days <= 0:
        return None
    return round(trades / sample_days, 2)


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
