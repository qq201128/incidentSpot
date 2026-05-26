from __future__ import annotations

from typing import Any

from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.high_winrate_combo_view import build_high_winrate_combo_view


def high_winrate_card(symbol: str, duration: str) -> dict[str, Any]:
    cached = get_cached_high_winrate_combo_ranking(symbol, duration)
    view = build_high_winrate_combo_view(cached, duration)
    ranking = list(view.get("highWinrateRanking") or [])
    top = dict(ranking[0]) if ranking else None
    if top is None:
        return {"available": False, "updatedAt": view.get("highWinrateUpdatedAt"), "total": 0}
    members = top.get("members") if isinstance(top.get("members"), list) else []
    return {
        "available": True,
        "updatedAt": view.get("highWinrateUpdatedAt"),
        "total": int(view.get("highWinrateTotal") or len(ranking)),
        "factorName": top.get("factorName"),
        "displayName": top.get("factorDisplayName") or top.get("displayName") or top.get("factorName"),
        "members": member_payloads(members),
        "winRate": top.get("winRate"),
        "avgTradesPerDay": top.get("avgTradesPerDay"),
        "factorScore": top.get("factorScore"),
        "profitFactor": top.get("profitFactor"),
        "maxDrawdown": top.get("maxDrawdown"),
        "totalPeriods": top.get("totalPeriods") or top.get("trades"),
        "sampleDays": top.get("sampleDays"),
    }


def member_payloads(members: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": member.get("name") if isinstance(member, dict) else member,
            "displayName": member.get("displayName") if isinstance(member, dict) else member,
        }
        for member in members
    ]
