from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.services.agent_formula_dedup import normalize_agent_formula
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN

AGENT_FACTOR_LIBRARY_VERSION = "agent_mined_factor_library_v1"


def merged_rows(existing: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {row_key(row): deepcopy(row) for row in existing}
    for row in candidates:
        previous = by_key.get(row_key(row))
        if previous:
            row["firstSeenAt"] = previous.get("firstSeenAt") or row["firstSeenAt"]
            row["promotionCount"] = int(previous.get("promotionCount") or 0) + (
                1 if row.get("qualityPassed") else 0
            )
        by_key[row_key(row)] = row
    return sorted(by_key.values(), key=lambda row: float(row.get("score") or 0.0), reverse=True)


def library_with_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.services.factor_learning_common import utc_now

    return {
        "version": AGENT_FACTOR_LIBRARY_VERSION,
        "updatedAt": utc_now(),
        "thresholds": threshold_payload(),
        "factors": rows,
    }


def simulation_eligible(metrics: dict[str, Any]) -> bool:
    return num(metrics.get("winRate")) >= SUCCESS_WIN_RATE_MIN and num(metrics.get("profitFactor")) >= SUCCESS_PROFIT_FACTOR_MIN


def usable_metrics(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("totalPeriods") or 0) >= BACKTEST_MIN_PERIODS and metrics.get("winRate") is not None


def orientation(metrics: dict[str, Any]) -> int:
    return 1 if num(metrics.get("ir")) >= 0 else -1


def ingested_agent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row_ingested(row)]


def row_ingested(row: dict[str, Any]) -> bool:
    if str(row.get("candidateStatus") or "") == "promoted":
        return True
    return isinstance(row.get("metrics"), dict) and str(row.get("candidateStatus") or "") not in {
        "failed",
        "duplicate_existing",
    }


def merge_agent_review(llm_agent: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(llm_agent)
    review = updated.get("review") if isinstance(updated.get("review"), dict) else {}
    review = deepcopy(review)
    review["evaluation"] = evaluation
    updated["review"] = review
    return updated


def duplicate_formula(formula: str, existing: list[dict[str, Any]]) -> bool:
    normalized = normalize_agent_formula(formula)
    return any(normalize_agent_formula(str(row.get("formula") or "")) == normalized for row in existing)


def empty_library() -> dict[str, Any]:
    return {"version": AGENT_FACTOR_LIBRARY_VERSION, "thresholds": threshold_payload(), "factors": []}


def threshold_payload() -> dict[str, float]:
    return {"minWinRate": SUCCESS_WIN_RATE_MIN, "minProfitFactor": SUCCESS_PROFIT_FACTOR_MIN}


def failure(row: dict[str, Any], stage: str, exc: Exception) -> dict[str, Any]:
    return {"factorName": row.get("factorName"), "stage": stage, "error": str(exc)}


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row_symbol(row), str(row.get("duration")), str(row.get("factorName")))


def row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or "").strip().upper()


def num(value: Any) -> float:
    return float(value) if value is not None else 0.0
