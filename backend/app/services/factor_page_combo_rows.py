from __future__ import annotations

from typing import Any

from app.services.factor_combo_display import combo_display_name
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_ranking_view import regular_ranking_rows
from app.services.factor_mined_library import MINED_FACTOR_SOURCE_FILE
from app.services.factor_page_list import sort_combo_rows_by_score


def combo_cache_total(symbol: str, duration: str) -> int | None:
    cached = get_cached_combination_ranking(symbol, duration)
    if cached is None:
        return None
    return len(regular_ranking_rows(cached))


def combo_list_rows(symbol: str | None, duration: str | None, fallback_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not symbol or not duration:
        return list(fallback_rows)
    cached = get_cached_combination_ranking(symbol, duration)
    if cached is None:
        return list(fallback_rows)
    return [combo_cache_summary(row, symbol, duration) for row in regular_ranking_rows(cached)]


def combo_cache_summary(row: dict[str, Any], symbol: str, duration: str) -> dict[str, Any]:
    name = str(row.get("factorName") or row.get("name") or "")
    display = str(row.get("factorDisplayName") or row.get("displayName") or "")
    if not display:
        members = row.get("members")
        display = combo_display_name(members) if isinstance(members, list) else name
    return {
        "name": name,
        "category": "performance",
        "categoryName": "绩效因子",
        "displayName": display,
        "description": display,
        "formula": str(row.get("formula") or name),
        "sourceFile": MINED_FACTOR_SOURCE_FILE,
        "direction": "higher_better",
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "factorScore": row.get("factorScore"),
        "winRate": row.get("winRate"),
        "profitFactor": row.get("profitFactor"),
        "totalPeriods": row.get("totalPeriods") or row.get("trades"),
        "ir": row.get("ir"),
        "walkForwardPassed": row.get("walkForwardPassed"),
        "walkForwardFailureReason": row.get("walkForwardFailureReason"),
        "paperLiveStatus": _paper_live_status(row),
    }


def sorted_combo_list_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sort_combo_rows_by_score(rows)


def _paper_live_status(row: dict[str, Any]) -> str:
    if row.get("walkForwardPassed") is True:
        return "backtest_passed"
    return "observe_only"
