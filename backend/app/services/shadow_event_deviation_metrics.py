from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(pairs)
    counts = {"alignedWin": 0, "shadowWinEventLoss": 0, "shadowLossEventWin": 0, "alignedLoss": 0}
    shadow_pnl_estimate = 0.0
    total_event_pnl = 0.0
    for pair in pairs:
        counts[camel(pair["divergenceType"])] += 1
        shadow_return = pair.get("shadowReturn")
        if shadow_return is not None:
            shadow_pnl_estimate += float(shadow_return)
        event_pnl = pair.get("eventPnlU")
        if event_pnl is not None:
            total_event_pnl += float(event_pnl)
    return {
        "pairedCount": total,
        "alignedWinCount": counts["alignedWin"],
        "shadowWinEventLossCount": counts["shadowWinEventLoss"],
        "shadowLossEventWinCount": counts["shadowLossEventWin"],
        "alignedLossCount": counts["alignedLoss"],
        "shadowWinEventLossRate": ratio(counts["shadowWinEventLoss"], total),
        "alignmentRate": ratio(counts["alignedWin"] + counts["alignedLoss"], total),
        "totalEventPnlU": round(total_event_pnl, 6),
        "estimatedShadowReturnSum": round(shadow_pnl_estimate, 6),
    }


def by_strategy(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(str(pair["strategyKey"]), []).append(pair)
    rows = []
    for strategy_key, items in grouped.items():
        item_summary = summary(items)
        rows.append(strategy_row(strategy_key, item_summary))
    rows.sort(key=lambda item: (item["shadowWinEventLossRate"] or 0, item["pairedCount"]), reverse=True)
    return rows


def strategy_row(strategy_key: str, item_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategyKey": strategy_key,
        "pairedCount": item_summary["pairedCount"],
        "shadowWinEventLossCount": item_summary["shadowWinEventLossCount"],
        "shadowWinEventLossRate": item_summary["shadowWinEventLossRate"],
        "totalEventPnlU": item_summary["totalEventPnlU"],
    }


def camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 4)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
