from __future__ import annotations

from typing import Any

from app.services.factor_learning_common import (
    SUCCESS_PROFIT_FACTOR_MIN,
    SUCCESS_WIN_RATE_MIN,
    edge_score,
    finite,
    round_metric,
)
from app.services.factor_metric_enrichment import factor_score

MINED_FACTOR_LIBRARY_VERSION = "mined_factor_library_v1"


def promotion_report(report: dict[str, Any], *, promoted: int, total: int) -> dict[str, Any]:
    return {
        "symbol": str(report.get("symbol") or "").strip().upper(),
        "duration": str(report.get("duration") or ""),
        "promoted": promoted,
        "libraryTotal": total,
        "thresholds": threshold_payload(),
    }


def threshold_payload() -> dict[str, float]:
    return {
        "minWinRate": SUCCESS_WIN_RATE_MIN,
        "minProfitFactor": SUCCESS_PROFIT_FACTOR_MIN,
    }


def mined_factor_parameters(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "members": [
            str(member["name"])
            for member in row.get("members") or []
            if isinstance(member, dict) and member.get("name")
        ],
        "symbol": row_symbol(row),
        "duration": str(row.get("duration") or ""),
    }


def empty_library() -> dict[str, Any]:
    return {
        "version": MINED_FACTOR_LIBRARY_VERSION,
        "thresholds": threshold_payload(),
        "factors": [],
    }


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row_symbol(row), str(row.get("duration")), str(row.get("factorName")))


def is_regular_combination_name(name: str) -> bool:
    return bool(name) and not name.startswith("goal_combo__")


def library_combination_rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        score_num(row.get("factorScore")),
        score_num(row.get("winRate")),
        score_num(row.get("profitFactor")),
        score_num(row.get("sharpe")),
    )


def validation_passed(validation: Any) -> bool:
    if not isinstance(validation, dict):
        return True
    return str(validation.get("status") or "") == "passed"


def row_symbol(row: dict[str, Any]) -> str:
    return str(row.get("symbol") or "").strip().upper()


def library_score(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or row
    if is_stability_rejected(row):
        return 0.0
    return standard_score(metrics)


def row_score(row: dict[str, Any]) -> float:
    return round_metric(edge_score(row), 6)


def standard_score(row: dict[str, Any]) -> float:
    return factor_score(row)


def display_score(row: dict[str, Any]) -> float:
    if is_stability_rejected(row):
        return 0.0
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else row
    return standard_score(metrics)


def is_stability_rejected(row: dict[str, Any]) -> bool:
    return str(row.get("stabilityStatus") or "") == "rejected"


def score_num(value: Any) -> float:
    number = finite(value)
    return number if number is not None else float("-inf")
