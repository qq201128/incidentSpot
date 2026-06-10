from __future__ import annotations

from typing import Any

from app.services.factor_combo_simulation_keys import is_high_winrate_combo_name
from app.services.high_winrate_combo_view import regular_ranking_view


def stale_regular_rows(cached: dict[str, Any]) -> list[dict[str, Any]]:
    return regular_ranking_view(regular_ranking_rows(cached))


def regular_ranking_rows(cached: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in display_ranking_rows(cached) if not _has_combo_member(row)]


def ranking_visibility(cached: dict[str, Any], ranking: list[dict[str, Any]]) -> dict[str, int]:
    raw_rows = display_ranking_rows(cached)
    passed_rows = ranking_rows(cached)
    return {
        "rawTotal": len(raw_rows),
        "regularTotal": len(ranking),
        "passedTotal": len(passed_rows),
        "evaluatedTotal": len(raw_rows),
        "nestedComboFilteredCount": max(len(raw_rows) - len(ranking), 0),
    }


def display_ranking_rows(cached: dict[str, Any]) -> list[dict[str, Any]]:
    evaluated = cached.get("evaluatedRanking")
    if isinstance(evaluated, list):
        return list(evaluated)
    return ranking_rows(cached)


def ranking_rows(cached: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = cached.get("ranking")
    return list(ranking) if isinstance(ranking, list) else []


def _has_combo_member(row: dict[str, Any]) -> bool:
    members = row.get("members")
    if not isinstance(members, list):
        return False
    return any(_is_combo_name(member.get("name")) for member in members if isinstance(member, dict))


def _is_combo_name(name: object) -> bool:
    raw = str(name or "")
    return raw.startswith("combo__") or is_high_winrate_combo_name(raw)
