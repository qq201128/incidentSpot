from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MIN_PAIRED_SAMPLES = 5
SYSTEMIC_SHADOW_WIN_EVENT_LOSS_RATE = 0.15
SYSTEMIC_MIN_SHADOW_WIN_EVENT_LOSS = 3


def summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(pairs)
    counts = {"alignedWin": 0, "shadowWinEventLoss": 0, "shadowLossEventWin": 0, "alignedLoss": 0}
    shadow_pnl_estimate = 0.0
    for pair in pairs:
        counts[camel(pair["divergenceType"])] += 1
        shadow_return = pair.get("shadowReturn")
        if shadow_return is not None:
            shadow_pnl_estimate += float(shadow_return)
    return {
        "pairedCount": total,
        "alignedWinCount": counts["alignedWin"],
        "shadowWinEventLossCount": counts["shadowWinEventLoss"],
        "shadowLossEventWinCount": counts["shadowLossEventWin"],
        "alignedLossCount": counts["alignedLoss"],
        "shadowWinEventLossRate": ratio(counts["shadowWinEventLoss"], total),
        "alignmentRate": ratio(counts["alignedWin"] + counts["alignedLoss"], total),
        "totalEventPnlU": round(sum(float(p["eventPnlU"]) for p in pairs if p.get("eventPnlU") is not None), 6),
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


def issues(item_summary: dict[str, Any], strategy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if item_summary["pairedCount"] < MIN_PAIRED_SAMPLES:
        return []
    result = systemic_issues(item_summary)
    result.extend(strategy_issues(strategy_rows))
    return result


def systemic_issues(item_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rate = item_summary.get("shadowWinEventLossRate")
    count = int(item_summary.get("shadowWinEventLossCount") or 0)
    if count < SYSTEMIC_MIN_SHADOW_WIN_EVENT_LOSS or rate is None:
        return []
    if rate < SYSTEMIC_SHADOW_WIN_EVENT_LOSS_RATE:
        return []
    return [{
        "code": "systemic_shadow_win_event_loss",
        "severity": "warning",
        "message": "shadow 预测正确但事件合约模拟亏损的比例偏高，可能存在方向/结算口径偏差",
        "shadowWinEventLossCount": count,
        "shadowWinEventLossRate": rate,
    }]


def strategy_issues(strategy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in strategy_rows:
        issue = strategy_issue(row)
        if issue is not None:
            result.append(issue)
    return result


def strategy_issue(row: dict[str, Any]) -> dict[str, Any] | None:
    if row["pairedCount"] < MIN_PAIRED_SAMPLES:
        return None
    strategy_rate = row.get("shadowWinEventLossRate")
    strategy_count = int(row.get("shadowWinEventLossCount") or 0)
    if strategy_count < 2 or strategy_rate is None or strategy_rate < SYSTEMIC_SHADOW_WIN_EVENT_LOSS_RATE:
        return None
    return {
        "code": "strategy_shadow_win_event_loss",
        "severity": "warning",
        "message": "该策略 shadow 与 event 盈亏口径偏差偏高",
        "strategyKey": row["strategyKey"],
        "shadowWinEventLossCount": strategy_count,
        "shadowWinEventLossRate": strategy_rate,
        "totalEventPnlU": row.get("totalEventPnlU"),
    }


def camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 4)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
