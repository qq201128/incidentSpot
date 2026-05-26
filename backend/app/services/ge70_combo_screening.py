from __future__ import annotations

from typing import Any

from app.services.factor_mined_library import MINED_FACTOR_LIBRARY_PATH, load_mined_factor_library

GE70_MIN_WIN_RATE = 0.62


def load_ge70_mined_combo_rows(*, min_win_rate: float = GE70_MIN_WIN_RATE) -> list[dict[str, Any]]:
    return ge70_mined_combo_screening_report(min_win_rate=min_win_rate)["selected"]


def ge70_mined_combo_screening_report(*, min_win_rate: float = GE70_MIN_WIN_RATE) -> dict[str, Any]:
    library = load_mined_factor_library(MINED_FACTOR_LIBRARY_PATH)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in library.get("factors") or []:
        reason = _rejection_reason(row, min_win_rate)
        if reason:
            rejected.append(_rejected_row(row, reason))
            continue
        selected.append(row)
    selected.sort(key=_sort_key)
    return {
        "policy": "offline_prefilter_only_requires_paper_live_settlement",
        "minBacktestWinRate": min_win_rate,
        "selected": selected,
        "selectedCount": len(selected),
        "rejectedReasons": rejected[:50],
        "rejectedCount": len(rejected),
        "reasonCounts": rejection_reason_counts(rejected),
    }


def _rejection_reason(row: dict[str, Any], min_win_rate: float) -> str | None:
    metrics = row.get("metrics") or {}
    win_rate = metrics.get("winRate")
    if win_rate is None:
        return "backtest_win_rate_missing"
    if float(win_rate) < min_win_rate:
        return "backtest_win_rate_below_paper_live_prefilter"
    if not row.get("members"):
        return "members_missing"
    return None


def _rejected_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    return {
        "factorName": row.get("factorName"),
        "duration": row.get("duration"),
        "reason": reason,
        "backtestWinRate": metrics.get("winRate"),
        "paperLiveStatus": "rejected_offline_prefilter",
    }


def _sort_key(item: dict[str, Any]) -> tuple[str, float]:
    metrics = item.get("metrics") or {}
    return str(item.get("duration") or ""), -float(metrics.get("winRate") or 0.0)


def rejection_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts
